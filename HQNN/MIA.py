import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from mia_common import ATTACK_SPLIT_SEED
import os
import numpy as np
import random
from HQNN import ConvQNN
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from process import get_features

# create dataset group divisions
GROUPS = {2: "member,     non-4 (retain)", 3: "member,     class-4 (forget)",
          0: "non-member, non-4",          1: "non-member, class-4"}

def group_split(indices_train, indices_test, trainset, testset):
    """800/200 attack-train/held-out split, stratified by group, same seed as the other attacks."""
    labels = np.array([int(trainset.targets[i]) for i in indices_train] +
                      [int(testset.targets[i]) for i in indices_test])
    member = np.array([1] * len(indices_train) + [0] * len(indices_test))
    group = 2 * member + (labels == 4)
    tr, te = train_test_split(np.arange(len(member)), test_size=0.2, stratify=group,
                              random_state=ATTACK_SPLIT_SEED)
    return tr, te, group

def report_and_save(role, scores, y, groups, threshold, path):
    fpr, tpr, _ = roc_curve(y, scores)
    auc = roc_auc_score(y, scores)
    print(f"\n[{role}] held-out AUC = {auc:.3f}   (threshold {threshold:.3f}, frozen from original)")
    rates = {}
    for g, name in GROUPS.items():
        m = groups == g
        rates[g] = float((scores[m] >= threshold).mean())
        print(f"  {name:<30} n={m.sum():>3}  called member: {rates[g]*100:5.1f}%")
    print(f"  forget - class-4 non-member gap: {(rates[3] - rates[1])*100:+.1f} pts")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, mlp_fpr=fpr, mlp_tpr=tpr, mlp_auc=auc,
             scores=scores, labels=y, groups=groups, threshold=threshold)

def MIA_train_process_pytorch():

    checkpoint = torch.load("model_0324_original_5_0.1_8.pth", weights_only=False)

    seed = checkpoint['args'].seed
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # === 1. Load original trained model & parameters ===
    model = ConvQNN(checkpoint['args'])
    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()

    # === 2. Load data (500 train + 500 test samples) ===
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    indices_train = checkpoint['indices'][0]
    indices_test = checkpoint['indices'][1]

    loader_train = DataLoader(Subset(trainset, indices_train), batch_size=1, shuffle=False)
    loader_test = DataLoader(Subset(testset, indices_test), batch_size=1, shuffle=False)

    # === 3. Generate MIA attack input data (model output probability + label) ===
    x_data_MIA = []
    y_data_MIA = []

    features_train, features_label4_train, y_label4_train = get_features(loader_train, 1, model)  # Member samples
    features_test, features_label4_test, y_label4_test = get_features(loader_test, 0, model)    # Non-member samples

    # # Label-4 eval subset must include both members and non-members, or accuracy
    # # just measures "always predict member" against an all-member set.
    # features_label4 = features_label4_train + features_label4_test
    # y_data_MIA_label4 = y_label4_train + y_label4_test
    assert len(features_train) == len(indices_train) and len(features_test) == len(indices_test)

    x_data_MIA = np.vstack([features_train, features_test])  # N x 10
    y_data_MIA = np.array([1] * len(features_train) + [0] * len(features_test))

    # # label 4
    # x_attack_tensor = torch.tensor(features_label4, dtype=torch.float32)
    # y_attack_tensor = torch.tensor(y_data_MIA_label4, dtype=torch.float32).unsqueeze(1)


    # === 4. Build attack model (MLP with PyTorch) ===
    class AttackMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(13, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )

        def forward(self, x):
            return self.net(x)

    attack_model = AttackMLP()
    optimizer = torch.optim.Adam(attack_model.parameters(), lr=0.01)
    loss_fn = nn.BCELoss()

    # === 5. Split train/test set ===
    tr, te, group = group_split(indices_train, indices_test, trainset, testset)
    x_train, y_train = x_data_MIA[tr], y_data_MIA[tr]
    x_test, y_test = x_data_MIA[te], y_data_MIA[te]

    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    # === 6. Train attack model ===
    for epoch in range(1000):
        attack_model.train()
        outputs = attack_model(x_train_tensor)
        loss = loss_fn(outputs, y_train_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Evaluate
        attack_model.eval()
        with torch.no_grad():

            val_auc = roc_auc_score(y_test, attack_model(x_test_tensor).numpy().ravel())

        if (epoch + 1) % 10 == 0:
            train_acc = accuracy_score(y_train_tensor.numpy(), (outputs.detach().numpy() > 0.5).astype(int))
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}, Train Acc: {train_acc*100:.2f}%, Held-out AUC: {val_auc:.3f}")

    # Threshold calibrated once on the original model's attack-train rows, then frozen
    with torch.no_grad():
        train_scores = attack_model(x_train_tensor).numpy().ravel()
        test_scores = attack_model(x_test_tensor).numpy().ravel()
    fpr, tpr, thr = roc_curve(y_train, train_scores)
    threshold = float(thr[np.argmax(tpr - fpr)])
    print(f"Attack-train AUC (original): {roc_auc_score(y_train, train_scores):.3f}")
    report_and_save("original", test_scores, y_test, group[te], threshold,
                    "roc_curve_output/mlp_attack_roc.npz")

    attack_model = AttackMLP()
    attack_model.load_state_dict(torch.load("attack_model_mia.pth"))
    attack_model.eval()
    print("Loaded MIA attack model")

    # import matplotlib.pyplot as plt
    #
    # member_max_probs = [f[10] for f, y in zip(x_data_MIA, y_data_MIA) if y == 1]
    # nonmember_max_probs = [f[10] for f, y in zip(x_data_MIA, y_data_MIA) if y == 0]
    #
    # plt.hist(member_max_probs, bins=30, alpha=0.5, label='Member')
    # plt.hist(nonmember_max_probs, bins=30, alpha=0.5, label='Non-Member')
    # plt.legend()
    # plt.title("Max Probability Distribution")
    # plt.show()


def MIA_attack():

    checkpoint = torch.load("model_0324_original_5_0.1_8.pth", weights_only=False)
    # checkpoint = torch.load("result/seed/model_MU_gradient_U8R1.pth", weights_only=False)

    model = ConvQNN(checkpoint['args'])

    seed = checkpoint['args'].seed
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # === 2. Load data (500 train + 500 test samples) ===
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    indices_train = checkpoint['indices'][0]
    indices_test = checkpoint['indices'][1]

    loader_train = DataLoader(Subset(trainset, indices_train), batch_size=1, shuffle=False)
    loader_test = DataLoader(Subset(testset, indices_test), batch_size=1, shuffle=False)


    # checkpoint = torch.load(f"result/seed/model_MU_gradient_U8R1.pth", weights_only=False)
    checkpoint = torch.load("model_0324_original_5_0.1_8.pth", weights_only=False)
    # === 1. Load original trained model & parameters ===
    # model = ConvQNN(checkpoint['args'])
    model.load_state_dict(checkpoint["model_state_dict"])
    orig_sd = torch.load("model_0324_original_5_0.1_8.pth", weights_only=False)["model_state_dict"]
    print("max |w_unlearned - w_original|:",
          max((checkpoint["model_state_dict"][k].float() - orig_sd[k].float()).abs().max().item() for k in orig_sd))


    model.eval()

    # === 3. Generate MIA attack input data (model output probability + label) ===
    features_train, _, _ = get_features(loader_train, 1, model)  # Member samples
    features_test, _, _ = get_features(loader_test, 0, model)    # Non-member samples
    x_data_MIA = np.vstack([features_train, features_test])
    y_data_MIA = np.array([1] * len(features_train) + [0] * len(features_test))
    _, te, group = group_split(indices_train, indices_test, trainset, testset)


    # === 4. Build attack model (MLP with PyTorch) ===
    class AttackMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(13, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )

        def forward(self, x):
            return self.net(x)


    attack_model = AttackMLP()
    attack_model.load_state_dict(torch.load("attack_model_mia.pth"))
    attack_model.eval()
    # print("Loaded MIA attack model")
    threshold = float(np.load("roc_curve_output/mlp_attack_roc.npz")["threshold"])
    with torch.no_grad():
        test_scores = attack_model(torch.tensor(x_data_MIA[te], dtype=torch.float32)).numpy().ravel()
    report_and_save("unlearned", test_scores, y_data_MIA[te], group[te], threshold,
                    "roc_curve_output/mlp_attack_roc_unlearned.npz")


if __name__ == "__main__":
    MIA_train_process_pytorch()
    MIA_attack()
