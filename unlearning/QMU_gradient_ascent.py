"""
Edited MU_gradient.py to remove overlap redundancies with HQNN.py, 
micro committed for easy deletion in case we don't want to alter original code too much
"""

import os
import sys
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import random
import pickle

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from HQNN import ConvQNN, calculate_accuracy
from plot_utils import plot_unlearning_curves

def process_continue_training(args):

    # Load state_dict
    checkpoint = torch.load(args.target_checkpoint, weights_only=False)
    model = ConvQNN(checkpoint['args'])
    args.name = f"QMU_{checkpoint['args'].n_layers}_{args.learning_rate}_{args.batch_size}"

    model.load_state_dict(checkpoint["model_state_dict"])

    # Re-seed from the original run so any remaining randomness (dataloader
    # shuffling, etc.) is reproducible and tied to that model.
    seed = checkpoint['args'].seed
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)

    # === 1. Load model, data, and select training set ===
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trainset = datasets.MNIST(root=os.path.join(REPO_ROOT, 'data'), train=True, download=True, transform=transform)
    testset = datasets.MNIST(root=os.path.join(REPO_ROOT, 'data'), train=False, download=True, transform=transform)


    # Use the exact train/test split the original model was trained on, so the
    # forget set unlearned here is the same class-4 subset the MIA scripts
    # evaluate against.
    train_indices = checkpoint['indices'][0]
    test_indices = checkpoint['indices'][1]

    # Split R/U
    def split_indices(dataset, indices, label=4):
        r_idx, u_idx = [], []
        for idx in indices:
            _, lbl = dataset[idx]
            if int(lbl) == label:
                u_idx.append(idx)
            else:
                r_idx.append(idx)
        return r_idx, u_idx

    train_r_idx, train_u_idx = split_indices(trainset, train_indices)
    test_r_idx, test_u_idx = split_indices(testset, test_indices)

    # Select training data: R or U
    train_data_R = Subset(trainset, train_r_idx)
    train_data_U = Subset(trainset, train_u_idx)

    test_r_set = Subset(testset, test_r_idx)
    test_u_set = Subset(testset, test_u_idx)

    trainloader_R = DataLoader(train_data_R, batch_size=args.batch_size, shuffle=True)
    trainloader_U = DataLoader(train_data_U, batch_size=args.batch_size, shuffle=True)

    test_r_loader = DataLoader(test_r_set, batch_size=args.batch_size, shuffle=False)
    test_u_loader = DataLoader(test_u_set, batch_size=args.batch_size, shuffle=False)

    # === 2. Load model + optimizer ===
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)

    acc_r_before = calculate_accuracy(model, test_r_loader)
    acc_u_before = calculate_accuracy(model, test_u_loader)

    print(f"Current model accuracy on Test-R: {acc_r_before:.3f}")
    print(f"Current model accuracy on Test-U: {acc_u_before:.3f}")

    tag_U = True
    tag_R = True
    lists = [[], [], [], [], []]  # combined loss, acc_r, acc_u, ce_u, ce_r


    # === 3. Continue training ===
    for epoch in range(args.n_epochs):
        model.train()

        running_loss = 0.0
        loss_u, loss_r = 0.0, 0.0

        if tag_U:
            for inputs, labels in trainloader_U:
                optimizer.zero_grad()
                outputs = model(inputs)
                ce_loss = torch.nn.functional.cross_entropy(outputs, labels)
                loss = ce_loss * -1
                loss.backward()
                optimizer.step()

                running_loss += ce_loss.item()
                loss_u += ce_loss.item()

        if tag_R:
            for inputs, labels in trainloader_R:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = torch.nn.functional.cross_entropy(outputs, labels)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                loss_r += loss.item()

        acc_r = calculate_accuracy(model, test_r_loader)
        acc_u = calculate_accuracy(model, test_u_loader)

        lists[0].append(running_loss / (tag_U * len(trainloader_U) + tag_R * len(trainloader_R)))
        lists[1].append(acc_r)
        lists[2].append(acc_u)
        lists[3].append(loss_u / len(trainloader_U))
        lists[4].append(loss_r / len(trainloader_R))

        print(f"Epoch {epoch+1}, Loss: {lists[0][-1]:.8f}, CE_U: {lists[3][-1]:.4f}, CE_R: {lists[4][-1]:.4f}, "
              f"Acc_R: {acc_r:.3f}, Acc_U: {acc_u:.3f}")

        if args.plot_intervals > 0 and ((epoch + 1) % args.plot_intervals == 0 or epoch + 1 == args.n_epochs):
            plot_unlearning_curves(lists, f"{args.save_path}/curves_{args.name}_seed{args.seed}.png",
                                   title=f"{args.name} seed{args.seed}",
                                   acc_before=(acc_r_before, acc_u_before))

        if (epoch + 1) % 10 == 0:
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "lists": lists,
                "mu_args": args,
                "args": checkpoint["args"],
                "indices": checkpoint["indices"],
            }, f"{args.save_path}/model_QMU_gradient_ep{epoch + 1}_seed{args.seed}.pth")


    # === 4. Save updated model + data ===
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "lists": lists,
        "mu_args": args,
        "args": checkpoint["args"],
        "indices": checkpoint["indices"],
    }, f"{args.save_path}/model_QMU_gradient_seed{args.seed}.pth")

    with open(f"{args.save_path}/trained_params_{args.name}_seed{args.seed}.pkl", "wb") as f:
        pickle.dump([lists, checkpoint["args"], args], f)



if __name__ == "__main__":


    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-seed', type=int, default=5, help="random seed")
    parser.add_argument('-save_path', type=str, default='checkpoints/unlearned', help="save path")
    parser.add_argument('-time_save', type=str, default='XX', help="save path")
    parser.add_argument('-name', type=str, default='XX', help="name")
    parser.add_argument('-target_checkpoint', type=str, default='checkpoints/original/model_0324_original_5_0.1_8.pth', help="checkpoint to unlearn from")

    parser.add_argument('-n_epochs', type=int, default=6, help="n_epochs")
    parser.add_argument('-batch_size', type=int, default=16, help="batch_size")
    parser.add_argument('-learning_rate', type=float, default=0.01, help="learning_rate")

    parser.add_argument('-plot_intervals', type=int, default=1, help="epochs between curve plots (0 disables)")

    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    process_continue_training(args)
