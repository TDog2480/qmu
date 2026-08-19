"""
Quantum-kernel SVM (QSVM) membership inference attack for evaluating HQNN unlearning.

Same protocol as MIA.py's AttackMLP / vqc_attack.py's AttackVQC, but instead of a
trained quantum circuit, membership is classified by a classical SVM running on a
quantum kernel:

  - Input: the same 13-dim feature vector as MIA.py / vqc_attack.py, produced by
    process.get_features, standardized (scaler fit on the attack-train split) and
    squashed with pi * tanh(bandwidth * z) into rotation-angle range -- smooth and
    non-saturating, unlike a hard clip -- then embedded on 8 qubits with the same
    compact two-axis AngleEmbedding used by vqc_attack.py, followed by one fixed
    (non-trainable) CNOT ring so the feature map produces genuine multi-qubit
    entanglement rather than a classically separable product-state kernel.
  - Kernel: fidelity |<phi(x_i)|phi(x_j)>|^2 between feature-mapped states, built
    with qml.kernels.kernel_matrix.
  - Classifier: sklearn.svm.SVC(kernel='precomputed') on the resulting Gram matrix.

The kernel costs O(N^2) circuit evaluations, so training uses a random ~200-sample
subsample of the 800-sample attack-train split (configurable via --n-samples) rather
than the full 800 used by the MLP/VQC attacks.

Hyperparameter selection: quantum kernels built from many-qubit angle embeddings are
prone to "concentration" -- most pairs of inputs end up looking equally similar to
the circuit, so the SVM degenerates toward a majority-class guess. The embedding
bandwidth (how far z-scores get pushed toward +-pi before tanh saturates) directly
controls this, so a small bandwidth grid is swept and scored against a genuine
held-out member/non-member split (x_data_MIA[800:], computed but never evaluated on
in the original MIA.py) rather than tuned blind. SVC's C is swept in the same loop
at near-zero extra cost, since it only requires refitting on an already-computed
kernel matrix.

Usage:
    python qsvm_attack.py
    python qsvm_attack.py --original path/to/orig.pth --unlearned path/to/unl.pth --n-samples 200
"""

import argparse
import pickle
import random

import numpy as np
import pennylane as qml
import torch
from pennylane.templates.embeddings import AngleEmbedding
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from HQNN import ConvQNN
from process import get_features

N_QUBITS = 8
WIRES = range(N_QUBITS)

# Embedding-bandwidth candidates for pi * tanh(bandwidth * z). Small bandwidth keeps
# angles close to linear in z (mirrors the old ANGLE_SCALE=pi/3 behavior without the
# hard-clip saturation); large bandwidth pushes more mass toward +-pi, which is more
# expressive per feature but concentrates the kernel faster. Swept, not guessed.
BANDWIDTHS = [0.1, 0.3, 1.0]
C_GRID = [0.1, 1.0, 10.0]
VAL_SIZE = 100  # cap on the held-out validation split used for the sweep, for runtime

dev = qml.device("default.qubit", wires=N_QUBITS)


def feature_map(x, wires):
    """Compact two-axis angle embedding + one fixed entangling CNOT ring."""
    AngleEmbedding(x[:8], wires=wires, rotation="X")
    AngleEmbedding(x[8:16], wires=wires, rotation="Y")
    wires = list(wires)
    for i in range(len(wires) - 1):
        qml.CNOT(wires=[wires[i], wires[i + 1]])
    qml.CNOT(wires=[wires[-1], wires[0]])


adjoint_feature_map = qml.adjoint(feature_map)


@qml.qnode(dev)
def kernel_circuit(x1, x2):
    feature_map(x1, wires=WIRES)
    adjoint_feature_map(x2, wires=WIRES)
    return qml.probs(wires=WIRES)


def quantum_kernel(x1, x2):
    return kernel_circuit(x1, x2)[0]


def prepare_inputs(x_raw, scaler, bandwidth):
    """Standardize with a pre-fit scaler, smoothly squash into rotation range, zero-pad 13 -> 16."""
    x_std = scaler.transform(x_raw)
    angles = np.pi * np.tanh(bandwidth * x_std)
    pad = np.zeros((angles.shape[0], 16 - angles.shape[1]))
    return np.hstack([angles, pad]).astype(np.float32)


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


def QSVM_train_process(original_ckpt_path, n_samples=200):
    checkpoint, model, loader_train, loader_test = _load_seeded_model_and_loaders(original_ckpt_path)

    # === Generate attack input data (model output probability + label) ===
    features_train, features_label4, y_data_MIA_label4 = get_features(loader_train, 1, model)  # Member samples
    features_test, _, _ = get_features(loader_test, 0, model)  # Non-member samples

    x_data_MIA = np.vstack([features_train, features_test])  # N x 13
    y_data_MIA = np.array([1] * len(features_train) + [0] * len(features_test))

    # === Split train/test set (matches MIA.py / vqc_attack.py) ===
    indices = np.random.permutation(len(x_data_MIA))
    x_data_MIA, y_data_MIA = x_data_MIA[indices], y_data_MIA[indices]

    x_train, y_train = x_data_MIA[:800], y_data_MIA[:800]
    x_test, y_test = x_data_MIA[800:], y_data_MIA[800:]

    # === Fit scaler on the full 800-sample attack-train split ===
    scaler = StandardScaler().fit(x_train)

    # === Subsample for the O(N^2) quantum kernel ===
    n_samples = min(n_samples, len(x_train))
    sub_idx = np.random.choice(len(x_train), size=n_samples, replace=False)
    x_train_sub, y_train_sub = x_train[sub_idx], y_train[sub_idx]

    # Genuine held-out member/non-member split, used to pick bandwidth & C instead
    # of guessing -- capped since it's re-embedded and re-kernelized per bandwidth.
    n_val = min(VAL_SIZE, len(x_test))
    x_val, y_val = x_test[:n_val], y_test[:n_val]

    # === Sweep embedding bandwidth (expensive: quantum kernel per candidate) and
    #     SVC's C (cheap: refit on an already-computed kernel) against x_val ===
    best_acc, best_bandwidth, best_C = -1.0, None, None
    best_svc, best_train_sub_q = None, None

    print(f"Sweeping bandwidth {BANDWIDTHS} x C {C_GRID} on a {n_samples}-train / {n_val}-val split …")
    for bandwidth in BANDWIDTHS:
        x_train_sub_q = prepare_inputs(x_train_sub, scaler, bandwidth)
        x_val_q = prepare_inputs(x_val, scaler, bandwidth)

        K_train = qml.kernels.kernel_matrix(x_train_sub_q, x_train_sub_q, quantum_kernel)
        K_val = qml.kernels.kernel_matrix(x_val_q, x_train_sub_q, quantum_kernel)

        for C in C_GRID:
            svc = SVC(kernel="precomputed", C=C).fit(K_train, y_train_sub)
            val_acc = accuracy_score(y_val, svc.predict(K_val))
            print(f"  bandwidth={bandwidth:.2f}  C={C:<6g}  val_acc={val_acc*100:.2f}%")

            if val_acc > best_acc:
                best_acc = val_acc
                best_bandwidth, best_C = bandwidth, C
                best_svc, best_train_sub_q = svc, x_train_sub_q

    print(f"Best config: bandwidth={best_bandwidth}, C={best_C}, held-out val_acc={best_acc*100:.2f}%")

    # === Evaluate the selected config on label-4 subset (original model) ===
    x_attack_q = prepare_inputs(np.array(features_label4), scaler, best_bandwidth)
    K_eval = qml.kernels.kernel_matrix(x_attack_q, best_train_sub_q, quantum_kernel)
    y_pred = best_svc.predict(K_eval)
    acc = accuracy_score(y_data_MIA_label4, y_pred)
    print(f"Final QSVM MIA attack accuracy: {acc*100:.2f}%")

    with open("attack_model_qsvm.pkl", "wb") as f:
        pickle.dump(
            {
                "svc": best_svc,
                "scaler": scaler,
                "reference_inputs": best_train_sub_q,
                "bandwidth": best_bandwidth,
            },
            f,
        )
    print("Saved QSVM attack model")


def QSVM_attack(original_ckpt_path, unlearned_ckpt_path):
    checkpoint, model, loader_train, _ = _load_seeded_model_and_loaders(original_ckpt_path)

    unlearned_ckpt = torch.load(unlearned_ckpt_path, weights_only=False)
    model.load_state_dict(unlearned_ckpt["model_state_dict"])
    model.eval()

    _, features_label4, y_data_MIA_label4 = get_features(loader_train, 1, model)

    with open("attack_model_qsvm.pkl", "rb") as f:
        saved = pickle.load(f)
    svc, scaler = saved["svc"], saved["scaler"]
    reference_inputs, bandwidth = saved["reference_inputs"], saved["bandwidth"]

    x_attack_q = prepare_inputs(np.array(features_label4), scaler, bandwidth)
    K_eval = qml.kernels.kernel_matrix(x_attack_q, reference_inputs, quantum_kernel)
    y_pred = svc.predict(K_eval)
    acc = accuracy_score(y_data_MIA_label4, y_pred)
    print(f"QSVM MIA Acc: {acc * 100:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QSVM-based MIA attack for HQNN unlearning evaluation")
    parser.add_argument("--original", default="model_0324_original_5_0.1_8.pth", help="Path to original trained model checkpoint")
    parser.add_argument("--unlearned", default="result/seed/model_MU_gradient_U8R1.pth", help="Path to unlearned model checkpoint")
    parser.add_argument("--n-samples", type=int, default=200, help="Number of attack-train samples used for the quantum kernel")
    args = parser.parse_args()

    QSVM_train_process(args.original, n_samples=args.n_samples)
    QSVM_attack(args.original, args.unlearned)
