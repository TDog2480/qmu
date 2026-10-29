import os
import sys
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

import random
import pickle

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from HQNN import ConvQNN, calculate_accuracy  
from plot_utils import plot_training_curves  

DATA_ROOT = os.path.join(REPO_ROOT, "data")


def split_indices_by_label(dataset, indices, target_label=4):
    """Split indices into R (not target_label) and U (== target_label)."""
    r_indices, u_indices = [], []
    for idx in indices:
        _, label = dataset[idx]
        if label == target_label:
            u_indices.append(idx)
        else:
            r_indices.append(idx)
    return r_indices, u_indices


def process(args):
    # Load MNIST data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trainset = datasets.MNIST(root=DATA_ROOT, train=True, download=True, transform=transform)
    testset = datasets.MNIST(root=DATA_ROOT, train=False, download=True, transform=transform)

    # train_indices = random.sample(range(len(trainset)), args.n_train)
    # test_indices = random.sample(range(len(testset)), args.n_test)
    original_ckpt = torch.load(args.original_checkpoint, weights_only=False)
    train_indices, test_indices = original_ckpt["indices"]

    # Training set excludes class 4 (this is the only intended difference from hqnn.py)
    train_r_idx, _ = split_indices_by_label(trainset, train_indices, target_label=4)
    # Test set is kept intact (like hqnn.py), plus split out for R/U diagnostics
    test_r_idx, test_u_idx = split_indices_by_label(testset, test_indices, target_label=4)

    train_subset = Subset(trainset, train_r_idx)
    test_subset = Subset(testset, test_indices)      # full test set, comparable to hqnn.py's test_acc
    test_r_subset = Subset(testset, test_r_idx)       # test set minus class 4
    test_u_subset = Subset(testset, test_u_idx)       # class 4 only

    trainloader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
    testloader = DataLoader(test_subset, batch_size=args.batch_size, shuffle=False)
    test_r_loader = DataLoader(test_r_subset, batch_size=args.batch_size, shuffle=False)
    test_u_loader = DataLoader(test_u_subset, batch_size=args.batch_size, shuffle=False)

    # Initialize model
    model = ConvQNN(args)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    # Train model

    lists = [[], [], [], [], []]  # [loss, train_acc, test_acc, acc_r, acc_u]

    for epoch in range(args.n_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        model.train()
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Backward pass
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_acc = correct / total
        test_acc = calculate_accuracy(model, testloader)
        acc_r = calculate_accuracy(model, test_r_loader)
        acc_u = calculate_accuracy(model, test_u_loader)

        lists[0].append(running_loss / len(trainloader))
        lists[1].append(train_acc)
        lists[2].append(test_acc)
        lists[3].append(acc_r)
        lists[4].append(acc_u)

        if args.plot_intervals > 0 and ((epoch + 1) % args.plot_intervals == 0 or epoch + 1 == args.n_epochs):
            plot_training_curves(lists, f"{args.save_path}/curves{args.name}.png",
                                 title=args.name.lstrip('_'))

        print(
            f"Epoch {epoch + 1}, Loss: {running_loss / len(trainloader):.4f}, "
            f"Train Acc: {train_acc:.2f}, "
            f"Test Acc: {test_acc:.2f}, "
            f"Acc_R: {acc_r:.2f}, "
            f"Acc_U: {acc_u:.2f}")

    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "args": args,
        "lists": lists,
        "indices": [train_indices, test_indices]
    }, f"{args.save_path}/model{args.name}.pth")

    name = f'{args.save_path}/trained_params_{args.name}.pkl'
    f = open(name, 'wb')
    pickle.dump([lists, args], file=f)
    f.close()


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-seed', type=int, default=0, help="random seed")
    parser.add_argument('-save_path', type=str, default=os.path.join(REPO_ROOT, 'checkpoints', 'reference'), help="save path")

    parser.add_argument('-time_save', type=str, default='XX', help="save path")
    parser.add_argument('-name', type=str, default='XX', help="name")

    parser.add_argument('-original_checkpoint', type=str, default=os.path.join(REPO_ROOT, 'checkpoints', 'original', 'model_HQNN_original_5_0.1_8.pth'))

    parser.add_argument('-n_qubit', type=int, default=8, help="n_qubit")
    parser.add_argument('-n_layers', type=int, default=5, help="n_layers")

    parser.add_argument('-n_epochs', type=int, default=10, help="n_epochs")
    parser.add_argument('-batch_size', type=int, default=8, help="batch_size")
    parser.add_argument('-learning_rate', type=float, default=0.1, help="learning_rate")

    parser.add_argument('-plot_intervals', type=int, default=1, help="epochs between curve plots (0 disables)")

    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    args.name = f'_HQNN_target_{args.n_layers}_{args.learning_rate}_{args.batch_size}'

    process(args)