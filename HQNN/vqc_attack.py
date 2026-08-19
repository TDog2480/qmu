"""
VQC-based membership inference attack for evaluating HQNN unlearning.

Same protocol as MIA.py's AttackMLP, but the learned attack model is a
variational quantum circuit (VQC) instead of a classical MLP:

  - Input: the same 13-dim feature vector as MIA.py (10 softmax probs +
    max_prob + correct + cross-entropy loss), produced by process.get_features.
  - Encoding: features are standardized (scaler fit on the attack-train split
    only) and squashed with pi * tanh(bandwidth * z) into rotation-angle range
    -- smooth and non-saturating, unlike a hard clip -- zero-padded to 16, and
    data-re-uploaded onto just 4 qubits as four AngleEmbedding passes of 4
    values each (X/Y/Z/X rotations), with a CNOT ring between passes. Fewer
    qubits than the 8 used elsewhere in this codebase -- barren-plateau
    severity scales unfavorably with qubit count for entangling circuits like
    this one, and 4 qubits with data re-uploading (Perez-Salinas et al.,
    "Data re-uploading for a universal quantum classifier") keeps expressivity
    while cutting that scaling.
  - Ansatz: 2 layers of per-qubit U3 rotations + a CNOT ring, matching the
    block HQNN.py's own qnode already uses for the target model (just on 4
    wires instead of 8). Parameters start near-identity (small Gaussian
    noise, not full uniform[-pi, pi]) to avoid landing directly on a barren
    plateau at initialization -- see Grant et al., "An initialization
    strategy for addressing barren plateaus in parametrized quantum
    circuits".
  - Readout: a single expectation value (no classical head), rescaled from
    [-1, 1] to a [0, 1] "member probability" trained with BCELoss.

Hyperparameter selection: same motivation as qsvm_attack.py -- a badly-scaled
embedding can make out-of-distribution inputs (e.g. the unlearned model's
inflated loss on the forget set) collapse toward the same encoded state,
producing a degenerate classifier that isn't actually measuring anything. The
embedding bandwidth is swept with a short proxy training budget and scored
against a genuine held-out member/non-member split (x_data_MIA[800:], computed
but never evaluated on in the original MIA.py); the winning bandwidth is then
used for the full 1000-epoch run. There's no VQC analogue of qsvm_attack.py's
C sweep -- a VQC has no separate regularization knob, its "capacity" tuning
happens through the 1000-epoch gradient training itself.

Usage:
    python vqc_attack.py
    python vqc_attack.py --original path/to/orig.pth --unlearned path/to/unl.pth
"""

import argparse
import pickle
import random

import numpy as np
import pennylane as qml
import torch
import torch.nn as nn
from pennylane.templates.embeddings import AngleEmbedding
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from HQNN import ConvQNN
from process import get_features

N_QUBITS = 4
N_LAYERS = 2

# Embedding-bandwidth candidates for pi * tanh(bandwidth * z), same grid as
# qsvm_attack.py for comparability. Picked via a short proxy training run
# rather than guessed -- see module docstring.
BANDWIDTHS = [0.1, 0.3, 1.0]
SWEEP_EPOCHS = 100

# Near-identity initialization std for vqc_params, to avoid barren plateaus that
# full uniform[-pi, pi] init tends to land on for an entangling ansatz like this one.
INIT_STD = 0.01

dev = qml.device("default.qubit", wires=N_QUBITS)


CHUNK_ROTATIONS = ("X", "Y", "Z", "X")  # 4 chunks x N_QUBITS(4) values = 16 padded features


def attack_embedding(inputs):
    """Data re-uploading: 16 padded values as 4 chunks of N_QUBITS, entangled between chunks."""
    for i, rotation in enumerate(CHUNK_ROTATIONS):
        chunk = inputs[..., i * N_QUBITS : (i + 1) * N_QUBITS]
        AngleEmbedding(chunk, wires=range(N_QUBITS), rotation=rotation)
        if i < len(CHUNK_ROTATIONS) - 1:
            for j in range(N_QUBITS - 1):
                qml.CNOT(wires=[j, j + 1])
            qml.CNOT(wires=[N_QUBITS - 1, 0])


@qml.qnode(dev, interface="torch")
def vqc_qnode(inputs, vqc_params):
    attack_embedding(inputs)

    for k in range(vqc_params.shape[0]):
        for j in range(N_QUBITS):
            qml.U3(*vqc_params[k][j], wires=[j])
        for j in range(N_QUBITS - 1):
            qml.CNOT(wires=[j, j + 1])
        qml.CNOT(wires=[N_QUBITS - 1, 0])

    return qml.expval(qml.PauliZ(0))


class AttackVQC(nn.Module):
    def __init__(self, n_layers=N_LAYERS):
        super().__init__()
        weight_shapes = {"vqc_params": (n_layers, N_QUBITS, 3)}
        self.qlayer = qml.qnn.TorchLayer(vqc_qnode, weight_shapes)
        with torch.no_grad():
            self.qlayer.qnode_weights["vqc_params"].normal_(mean=0.0, std=INIT_STD)

    def forward(self, x):
        expval = self.qlayer(x)
        prob = (expval + 1.0) / 2.0
        prob = torch.clamp(prob, 1e-6, 1 - 1e-6)
        return prob.unsqueeze(-1)


def prepare_inputs(x_raw, scaler, bandwidth):
    """Standardize with a pre-fit scaler, smoothly squash into rotation range, zero-pad 13 -> 16."""
    x_std = scaler.transform(x_raw)
    angles = np.pi * np.tanh(bandwidth * x_std)
    pad = np.zeros((angles.shape[0], 16 - angles.shape[1]))
    return np.hstack([angles, pad]).astype(np.float32)


def _train_vqc(x_train_tensor, y_train_tensor, n_epochs, track_tensor, track_y, print_every=None):
    """Train a fresh AttackVQC, tracking accuracy on (track_tensor, track_y) each epoch.

    track_tensor is often a small subset that heavily overlaps with x_train_tensor
    (e.g. the ~54-sample label-4 set, most of which is also inside the 800-sample
    training split) -- printed alone it reads like a generalization metric but
    mostly reflects training-set fit at low, easily-misleading resolution. When
    print_every is set, full 800-row training accuracy is also printed (reusing
    the same forward pass already computed for the loss, so it's free) to make
    that distinction explicit rather than implicit.
    """
    attack_model = AttackVQC()
    optimizer = torch.optim.Adam(attack_model.parameters(), lr=0.01)
    loss_fn = nn.BCELoss()

    acc = None
    for epoch in range(n_epochs):
        attack_model.train()
        outputs = attack_model(x_train_tensor)
        loss = loss_fn(outputs, y_train_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        attack_model.eval()
        with torch.no_grad():
            y_pred = attack_model(track_tensor)
            acc = accuracy_score(track_y.numpy(), (y_pred.numpy() > 0.5).astype(int))

        if print_every and (epoch + 1) % print_every == 0:
            train_acc = accuracy_score(
                y_train_tensor.numpy(), (outputs.detach().numpy() > 0.5).astype(int)
            )
            print(
                f"Epoch {epoch+1}, Loss: {loss.item():.4f}, "
                f"Train Acc: {train_acc*100:.2f}%, Label4 Acc: {acc*100:.2f}%"
            )

    return attack_model, acc


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


def VQC_train_process(original_ckpt_path):
    checkpoint, model, loader_train, loader_test = _load_seeded_model_and_loaders(original_ckpt_path)

    # === Generate attack input data (model output probability + label) ===
    features_train, features_label4, y_data_MIA_label4 = get_features(loader_train, 1, model)  # Member samples
    features_test, _, _ = get_features(loader_test, 0, model)  # Non-member samples

    x_data_MIA = np.vstack([features_train, features_test])  # N x 13
    y_data_MIA = np.array([1] * len(features_train) + [0] * len(features_test))

    # === Split train/test set ===
    indices = np.random.permutation(len(x_data_MIA))
    x_data_MIA, y_data_MIA = x_data_MIA[indices], y_data_MIA[indices]

    x_train, y_train = x_data_MIA[:800], y_data_MIA[:800]
    x_test, y_test = x_data_MIA[800:], y_data_MIA[800:]

    # === Fit scaler on the attack-train split only ===
    scaler = StandardScaler().fit(x_train)

    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    y_val_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    # === Sweep embedding bandwidth with a short proxy training budget, scored
    #     against the genuine held-out member/non-member split (x_test) ===
    best_acc, best_bandwidth = -1.0, None
    print(f"Sweeping bandwidth {BANDWIDTHS} with {SWEEP_EPOCHS}-epoch proxy runs "
          f"on a {len(x_train)}-train / {len(x_test)}-val split …")
    for bandwidth in BANDWIDTHS:
        x_train_tensor = torch.tensor(prepare_inputs(x_train, scaler, bandwidth), dtype=torch.float32)
        x_val_tensor = torch.tensor(prepare_inputs(x_test, scaler, bandwidth), dtype=torch.float32)

        _, val_acc = _train_vqc(x_train_tensor, y_train_tensor, SWEEP_EPOCHS, x_val_tensor, y_val_tensor)
        print(f"  bandwidth={bandwidth:.2f}  val_acc={val_acc*100:.2f}%")

        if val_acc > best_acc:
            best_acc, best_bandwidth = val_acc, bandwidth

    print(f"Best bandwidth: {best_bandwidth} (proxy held-out val_acc={best_acc*100:.2f}%)")

    # === Full 1000-epoch training run at the selected bandwidth ===
    x_train_tensor = torch.tensor(prepare_inputs(x_train, scaler, best_bandwidth), dtype=torch.float32)
    x_attack_tensor = torch.tensor(
        prepare_inputs(np.array(features_label4), scaler, best_bandwidth), dtype=torch.float32
    )
    y_attack_tensor = torch.tensor(y_data_MIA_label4, dtype=torch.float32).unsqueeze(1)

    attack_model, acc = _train_vqc(
        x_train_tensor, y_train_tensor, 1000, x_attack_tensor, y_attack_tensor, print_every=10
    )

    print(f"Final VQC MIA attack accuracy: {acc*100:.2f}%")

    torch.save(attack_model.state_dict(), "attack_model_vqc.pth")
    with open("attack_scaler_vqc.pkl", "wb") as f:
        pickle.dump({"scaler": scaler, "bandwidth": best_bandwidth}, f)
    print("Saved VQC attack model and scaler")


def VQC_attack(original_ckpt_path, unlearned_ckpt_path):
    checkpoint, model, loader_train, _ = _load_seeded_model_and_loaders(original_ckpt_path)

    unlearned_ckpt = torch.load(unlearned_ckpt_path, weights_only=False)
    model.load_state_dict(unlearned_ckpt["model_state_dict"])
    model.eval()

    _, features_label4, y_data_MIA_label4 = get_features(loader_train, 1, model)

    with open("attack_scaler_vqc.pkl", "rb") as f:
        saved = pickle.load(f)
    scaler, bandwidth = saved["scaler"], saved["bandwidth"]

    x_attack_tensor = torch.tensor(
        prepare_inputs(np.array(features_label4), scaler, bandwidth), dtype=torch.float32
    )
    y_attack_tensor = torch.tensor(y_data_MIA_label4, dtype=torch.float32).unsqueeze(1)

    attack_model = AttackVQC()
    attack_model.load_state_dict(torch.load("attack_model_vqc.pth"))
    attack_model.eval()

    with torch.no_grad():
        y_pred = attack_model(x_attack_tensor)
    acc = accuracy_score(y_attack_tensor.numpy(), (y_pred.numpy() > 0.5).astype(int))
    print(f"VQC MIA Acc: {acc * 100:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VQC-based MIA attack for HQNN unlearning evaluation")
    parser.add_argument("--original", default="model_0324_original_5_0.1_8.pth", help="Path to original trained model checkpoint")
    parser.add_argument("--unlearned", default="result/seed/model_MU_gradient_U8R1.pth", help="Path to unlearned model checkpoint")
    args = parser.parse_args()

    VQC_train_process(args.original)
    VQC_attack(args.original, args.unlearned)
