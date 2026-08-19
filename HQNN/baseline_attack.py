"""
Classical reference baselines for the HQNN membership inference attacks.

MIA.py's AttackMLP and vqc_attack.py's AttackVQC both plateau around ~65% train
accuracy after 1000 epochs, despite being very different architectures (dense net
vs. variational quantum circuit). That could mean either (a) ~65% is close to the
real information ceiling in the 13-dim feature vector against this target model, or
(b) both models are simply under-converged after 1000 epochs of gradient descent.

This script settles that question by removing "did we train long enough" as a
variable entirely: Logistic Regression (a convex problem, solved to its true
optimum by LBFGS in seconds) and Random Forest (no gradient descent, no epoch
budget, captures nonlinear signal a linear model can't) are both fit once, with no
tuning, on the exact same feature vectors / train-test split as MIA.py.

  - If these baselines also cap around ~65-70% on the held-out split, that's clean
    evidence of a real ceiling: the MLP/VQC numbers aren't undertrained, they're
    close to what this feature set can actually support.
  - If either baseline does meaningfully better, that shows more signal is
    available and the MLP/VQC results are a training artifact, not a data limit.

Unlike MIA.py / vqc_attack.py / qsvm_attack.py, nothing here is trained iteratively
or persisted to disk -- both baselines fit in well under a second, so there's no
reason to save/reload them.

Usage:
    python baseline_attack.py
    python baseline_attack.py --original path/to/orig.pth --unlearned path/to/unl.pth
"""

import argparse
import random

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from HQNN import ConvQNN
from process import get_features


def _load_seeded_model_and_loaders(original_ckpt_path):
    checkpoint = torch.load(original_ckpt_path, weights_only=False)

    seed = checkpoint["args"].seed
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)

    model = ConvQNN(checkpoint["args"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trainset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    testset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    indices_train = checkpoint["indices"][0][0:500]
    indices_test = checkpoint["indices"][1]

    loader_train = DataLoader(Subset(trainset, indices_train), batch_size=1, shuffle=False)
    loader_test = DataLoader(Subset(testset, indices_test), batch_size=1, shuffle=False)

    return checkpoint, model, loader_train, loader_test


def run_baseline_attacks(original_ckpt_path, unlearned_ckpt_path):
    checkpoint, model, loader_train, loader_test = _load_seeded_model_and_loaders(original_ckpt_path)

    # === Generate attack input data (model output probability + label) -- same
    #     pipeline as MIA.py / vqc_attack.py / qsvm_attack.py ===
    features_train, features_label4, y_label4 = get_features(loader_train, 1, model)  # Member samples
    features_test, _, _ = get_features(loader_test, 0, model)  # Non-member samples

    x_data_MIA = np.vstack([features_train, features_test])  # N x 13
    y_data_MIA = np.array([1] * len(features_train) + [0] * len(features_test))

    # === Split train/test set (identical seeded shuffle to the other attacks) ===
    indices = np.random.permutation(len(x_data_MIA))
    x_data_MIA, y_data_MIA = x_data_MIA[indices], y_data_MIA[indices]

    x_train, y_train = x_data_MIA[:800], y_data_MIA[:800]
    x_test, y_test = x_data_MIA[800:], y_data_MIA[800:]

    x_label4_orig = np.array(features_label4)
    y_label4_orig = np.array(y_label4)

    scaler = StandardScaler().fit(x_train)
    x_train_s = scaler.transform(x_train)
    x_test_s = scaler.transform(x_test)
    x_label4_orig_s = scaler.transform(x_label4_orig)

    seed = checkpoint["args"].seed
    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=seed),
    }

    results = {}
    for name, clf in classifiers.items():
        clf.fit(x_train_s, y_train)
        results[name] = {
            "clf": clf,
            "train_acc": accuracy_score(y_train, clf.predict(x_train_s)),
            "val_acc": accuracy_score(y_test, clf.predict(x_test_s)),
            "label4_orig": accuracy_score(y_label4_orig, clf.predict(x_label4_orig_s)),
        }

    # === Re-extract label-4 features from the unlearned model ===
    unlearned_ckpt = torch.load(unlearned_ckpt_path, weights_only=False)
    model.load_state_dict(unlearned_ckpt["model_state_dict"])
    model.eval()

    _, features_label4_unl, y_label4_unl = get_features(loader_train, 1, model)
    x_label4_unl_s = scaler.transform(np.array(features_label4_unl))
    y_label4_unl = np.array(y_label4_unl)

    for name, clf in classifiers.items():
        results[name]["label4_unl"] = accuracy_score(y_label4_unl, clf.predict(x_label4_unl_s))

    W = 88
    print(f"\n{'─'*W}")
    print("Classical baseline attacks -- fit once, no epoch/optimizer tuning")
    print(f"{'─'*W}")
    print(f"{'Model':<22} {'Train Acc':>10} {'Held-out Acc':>13} {'Label4 (orig)':>14} {'Label4 (unl.)':>14}")
    print(f"{'─'*W}")
    for name, r in results.items():
        print(
            f"{name:<22} {r['train_acc']*100:>9.2f}% {r['val_acc']*100:>12.2f}% "
            f"{r['label4_orig']*100:>13.2f}% {r['label4_unl']*100:>13.2f}%"
        )
    print(f"{'─'*W}")
    print("Held-out Acc is the number that matters for the ceiling question: it's a genuine")
    print("member/non-member split, unlike Label4 Acc which is single-class and easily biased.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classical baseline MIA attacks for HQNN unlearning evaluation")
    parser.add_argument("--original", default="model_0324_original_5_0.1_8.pth", help="Path to original trained model checkpoint")
    parser.add_argument("--unlearned", default="result/seed/model_MU_gradient_U8R1.pth", help="Path to unlearned model checkpoint")
    args = parser.parse_args()

    run_baseline_attacks(args.original, args.unlearned)
