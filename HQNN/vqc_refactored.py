"""
VQC-based membership inference attack for evaluating HQNN unlearning.

Same attack model as before -- a 4-qubit variational circuit with data re-uploading,
trained with BCE on the 13-dim MIA feature vector -- but rebuilt on the shared
evaluation pool (Part 4) so its numbers line up with the threshold attacks, the MLP,
and the QSVM attack.

What changed from the previous version:

  * The member side is no longer capped at indices[0][:500] and no longer filtered
    to exclude class 4. Members and non-members are the target's full saved train /
    test indices (500/500 after the Part 1 retrain), class 4 present on both sides.

  * The single accuracy number on the ~88-sample label-4 mix is gone, and with it
    the misleading per-epoch "Label4 Acc" -- that set overlapped the training split
    almost entirely, so it read as a generalization metric while mostly reporting
    training fit. Progress is now tracked as training accuracy on the fit rows and
    AUC on the held-out rows, which are actually different things.

  * Results are the 4-group table: member/non-4 (retain), member/class-4 (forget),
    non-member/non-4, non-member/class-4, for the original and the unlearned model.

  * The 800/200 attack-model split is stratified by group and shared with the other
    attacks (mia_common.ATTACK_SPLIT_SEED). It is the attack model's own
    train/validation split and carries no membership semantics -- both halves
    contain members and non-members.

  * The bandwidth sweep is scored by held-out AUC rather than accuracy, so a
    collapsed circuit that predicts one class cannot win the sweep.

  * The decision threshold is calibrated once on the original model and frozen, then
    applied unchanged to the unlearned model.

Bandwidth is selected on the held-out rows, so the held-out AUC reported at the end
is mildly optimistic (one hyperparameter's worth). Nothing else touches it.

Usage:
    python vqc_attack.py --original model_0324_original_5_0.1_8.pth \
                         --unlearned result/seed/model_MU_gradient_U8R1.pth
"""

import argparse
import pickle

import numpy as np
import pennylane as qml
import torch
import torch.nn as nn
from pennylane.templates.embeddings import AngleEmbedding
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

import mia_common as mc
import roc_utils

N_QUBITS = 4
N_LAYERS = 2

# Embedding-bandwidth candidates for pi * tanh(bandwidth * z), same grid as
# qsvm_attack.py for comparability. Picked via a short proxy training run.
BANDWIDTHS = [0.1, 0.3, 1.0]
SWEEP_EPOCHS = 100
FULL_EPOCHS = 1000

# Near-identity initialization std, to avoid the barren plateau that a full
# uniform[-pi, pi] init tends to land on for an entangling ansatz like this one.
INIT_STD = 0.01

dev = qml.device("default.qubit", wires=N_QUBITS)

CHUNK_ROTATIONS = ("X", "Y", "Z", "X")  # 4 chunks x N_QUBITS(4) values = 16 padded features


def attack_embedding(inputs):
    """Data re-uploading: 16 padded values as 4 chunks of N_QUBITS, entangled between chunks."""
    for i, rotation in enumerate(CHUNK_ROTATIONS):
        chunk = inputs[..., i * N_QUBITS: (i + 1) * N_QUBITS]
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


def as_tensor(x):
    return torch.tensor(x, dtype=torch.float32)


@torch.no_grad()
def vqc_scores(model, x_tensor):
    model.eval()
    return model(x_tensor).numpy().reshape(-1)


def train_vqc(x_fit, y_fit, n_epochs, x_val=None, y_val=None, print_every=None):
    """Train a fresh AttackVQC on the attack-train rows.

    Progress reporting keeps two things apart that the old loop merged: training
    accuracy on the rows being fit (free -- reuses the forward pass already computed
    for the loss) and AUC on the held-out rows, which is the only number that says
    anything about generalization.
    """
    model = AttackVQC()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.BCELoss()

    y_fit_np = y_fit.numpy().reshape(-1)
    for epoch in range(n_epochs):
        model.train()
        outputs = model(x_fit)
        loss = loss_fn(outputs, y_fit)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if print_every and (epoch + 1) % print_every == 0:
            fit_acc = float(((outputs.detach().numpy().reshape(-1) > 0.5).astype(int) == y_fit_np).mean())
            msg = f"Epoch {epoch + 1}, Loss: {loss.item():.4f}, Fit Acc: {fit_acc * 100:.2f}%"
            if x_val is not None:
                val_auc = roc_auc_score(y_val, vqc_scores(model, x_val))
                msg += f", Held-out AUC: {val_auc:.3f}"
            print(msg)

    val_auc = float("nan")
    if x_val is not None and len(np.unique(y_val)) > 1:
        val_auc = float(roc_auc_score(y_val, vqc_scores(model, x_val)))
    return model, val_auc


def main():
    parser = argparse.ArgumentParser(description="VQC-based MIA attack, 4-group evaluation")
    parser.add_argument("--original", default="model_0324_original_5_0.1_8.pth",
                        help="target checkpoint the attack model is fit against")
    parser.add_argument("--unlearned", default="result/seed/model_MU_gradient_U8R1.pth",
                        help="unlearned checkpoint to score with the frozen attack model")
    parser.add_argument("--epochs", type=int, default=FULL_EPOCHS)
    parser.add_argument("--forget-class", type=int, default=mc.FORGET_CLASS)
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--artifact", type=str, default="attack_model_vqc.pth")
    parser.add_argument("--scaler-out", type=str, default="attack_scaler_vqc.pkl")
    parser.add_argument("--scores-out", type=str, default="scores_vqc.npz")
    parser.add_argument("--roc-out", type=str, default="vqc_attack_roc.png",
                        help="path to save the VQC attack's ROC plot; pass '' to skip")
    parser.add_argument("--roc-npz-out", type=str, default=None,
                        help="optional path to save raw fpr/tpr/thresholds (npz)")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Pool + features
    # ------------------------------------------------------------------
    ckpt = torch.load(args.original, weights_only=False)
    mc.set_seed(int(getattr(ckpt["args"], "seed", 0)))
    model = mc.build_model(ckpt)
    print(f"[setup] target: {args.original}  (HQNN.ConvQNN, amplitude embedding)")

    spec = mc.build_pool_spec(ckpt, data_root=args.data_root, forget_class=args.forget_class)
    mc.describe_pool(spec)

    X_original = mc.features_for(model, spec)

    fit_rows = np.flatnonzero(spec.attack_train_mask)
    val_rows = np.flatnonzero(spec.heldout_mask)

    # Scaler is fit on the attack-train rows only, then frozen.
    scaler = StandardScaler().fit(X_original[fit_rows])

    y_fit = torch.tensor(spec.membership[fit_rows], dtype=torch.float32).unsqueeze(1)
    y_val = spec.membership[val_rows]

    # ------------------------------------------------------------------
    # Sweep embedding bandwidth with a short proxy budget, scored by held-out AUC
    # ------------------------------------------------------------------
    best_auc, best_bandwidth = -1.0, None
    print(f"\nSweeping bandwidth {BANDWIDTHS} with {SWEEP_EPOCHS}-epoch proxy runs on "
          f"{len(fit_rows)} fit / {len(val_rows)} held-out rows ...")
    for bandwidth in BANDWIDTHS:
        x_fit_t = as_tensor(prepare_inputs(X_original[fit_rows], scaler, bandwidth))
        x_val_t = as_tensor(prepare_inputs(X_original[val_rows], scaler, bandwidth))
        _, val_auc = train_vqc(x_fit_t, y_fit, SWEEP_EPOCHS, x_val_t, y_val)
        print(f"  bandwidth={bandwidth:.2f}  held-out AUC={val_auc:.3f}")
        if val_auc > best_auc:
            best_auc, best_bandwidth = val_auc, bandwidth

    print(f"Best bandwidth: {best_bandwidth} (proxy held-out AUC={best_auc:.3f})")

    # ------------------------------------------------------------------
    # Full training run at the selected bandwidth
    # ------------------------------------------------------------------
    x_fit_t = as_tensor(prepare_inputs(X_original[fit_rows], scaler, best_bandwidth))
    x_val_t = as_tensor(prepare_inputs(X_original[val_rows], scaler, best_bandwidth))
    attack_model, final_val_auc = train_vqc(
        x_fit_t, y_fit, args.epochs, x_val_t, y_val, print_every=max(1, args.epochs // 10)
    )
    print(f"Final held-out AUC: {final_val_auc:.3f}")

    # ------------------------------------------------------------------
    # Score the original model's whole pool; calibrate and freeze the threshold
    # ------------------------------------------------------------------
    x_all_t = as_tensor(prepare_inputs(X_original, scaler, best_bandwidth))
    scores_original = vqc_scores(attack_model, x_all_t)

    # Calibrated on the attack-train rows only, so the held-out row of the headline
    # table stays genuinely held out.
    threshold = mc.calibrate_threshold(scores_original, spec.membership, spec.attack_train_mask)

    # ------------------------------------------------------------------
    # Apply the frozen attack model + threshold to the unlearned checkpoint
    # ------------------------------------------------------------------
    scores_unlearned = None
    try:
        _, unlearned_model = mc.load_weights_into(args.unlearned, ckpt)
    except FileNotFoundError:
        print(f"\n[warn] unlearned checkpoint not found: {args.unlearned} -- reporting the original model only")
    else:
        print(f"[setup] unlearned: {args.unlearned}")
        X_unlearned = mc.features_for(unlearned_model, spec)
        x_unl_t = as_tensor(prepare_inputs(X_unlearned, scaler, best_bandwidth))
        scores_unlearned = vqc_scores(attack_model, x_unl_t)

    mc.report(spec, scores_original, scores_unlearned, threshold, "VQC attack")

    # --- ROC curve for the VQC attack (original model) ---------------------
    roc_data = roc_utils.compute_roc(scores_original, spec.membership, higher_is_member=True)
    if args.roc_out:
        roc_utils.plot_roc(roc_data, args.roc_out,
                            title="VQC attack ROC (original model)", label="VQC attack")
    if args.roc_npz_out:
        roc_utils.save_roc_data({"vqc": roc_data}, args.roc_npz_out)

    torch.save(attack_model.state_dict(), args.artifact)
    with open(args.scaler_out, "wb") as f:
        pickle.dump({
            "scaler": scaler,
            "bandwidth": best_bandwidth,
            "threshold": threshold,
            "heldout_auc": final_val_auc,
            "target_checkpoint": args.original,
            "attack_split_seed": mc.ATTACK_SPLIT_SEED,
            "forget_class": args.forget_class,
        }, f)
    print(f"\n[save] VQC attack model -> {args.artifact}, scaler/config -> {args.scaler_out}")
    mc.save_scores(args.scores_out, spec, scores_original, scores_unlearned, threshold, "VQC")


if __name__ == "__main__":
    main()