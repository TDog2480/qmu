"""
Classical MLP membership inference attack for evaluating HQNN unlearning.

Same pipeline as vqc_attack.py / qsvm_attack.py ; only the attack
model differs: a classical MLP on the same 13-dim feature vector. This isolates the
attack model as the only variable when comparing against the quantum attacks.

Saves:
    roc_curve_output/mlp_attack_roc.npz            original model (held-out rows)
    roc_curve_output/mlp_attack_roc_unlearned.npz  unlearned model (held-out rows)
    roc_curve_output/scores_mlp.npz
    checkpoints/attack-models/attack_model_mlp.pth, attack_config_mlp.pkl

Usage:
    python attacks/mlp-attack.py --original checkpoints/original/model_HQNN_original_5_0.1_8.pth \
                  --unlearned checkpoints/unlearned/model_QMU_gradient_seed5.pth
"""

import argparse
import os
import pickle
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT) 

import attack_scoring
import roc_utils

DATA_ROOT = os.path.join(REPO_ROOT, "data")

class AttackMLP(nn.Module):
    def __init__(self, input_dim=13):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def as_tensor(x):
    return torch.tensor(x, dtype=torch.float32)


@torch.no_grad()
def mlp_scores(model, x_tensor):
    # Logits (network without the final Sigmoid): same ranking as the probability,
    # but no float32 saturation to exactly 0/1, which created large tied blocks in the ROC.
    model.eval()
    return model.net[:-1](x_tensor).numpy().reshape(-1)


def train_mlp(x_fit, y_fit, n_epochs, x_val=None, y_val=None, print_every=None):
    """Same loop as train_vqc: BCE, Adam lr=0.01, full batch; fit acc vs held-out AUC kept apart."""
    model = AttackMLP(x_fit.shape[1])
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
                val_auc = roc_utils.compute_roc(mlp_scores(model, x_val), y_val)["auc"]
                msg += f", Held-out AUC: {val_auc:.3f}"
            print(msg)

    val_auc = float("nan")
    if x_val is not None and len(np.unique(y_val)) > 1:
        val_auc = float(roc_utils.compute_roc(mlp_scores(model, x_val), y_val)["auc"])
    return model, val_auc


def main():
    parser = argparse.ArgumentParser(description="Classical MLP MIA attack, 4-group evaluation")
    parser.add_argument("--original", default=os.path.join(REPO_ROOT, 'checkpoints', 'original', 'model_HQNN_original_5_0.1_8.pth'))
    parser.add_argument("--unlearned", default=os.path.join(REPO_ROOT, 'checkpoints', 'unlearned', 'model_QMU_gradient_seed5.pth'))
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--forget-class", type=int, default=attack_scoring.FORGET_CLASS)
    parser.add_argument("--data-root", type=str, default=DATA_ROOT)

    parser.add_argument("--artifact", type=str, default=os.path.join(REPO_ROOT, "checkpoints", "attack-models", "attack_model_mlp.pth"))
    parser.add_argument("--config-out", type=str, default=os.path.join(REPO_ROOT, "checkpoints", "attack-models", "attack_scaler_mlp.pkl"))
    parser.add_argument("--scores-out", type=str, default=os.path.join(REPO_ROOT, "checkpoints", "attack-models", "scores_mlp.npz"))

    parser.add_argument("--roc-npz-out", type=str, default=os.path.join(roc_utils.OUT_DIR, "mlp_attack_roc.npz"))
    parser.add_argument("--roc-npz-unlearned-out", type=str, default=os.path.join(roc_utils.OUT_DIR, "mlp_attack_roc_unlearned.npz"))
    
    args = parser.parse_args()
    os.makedirs(roc_utils.OUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(args.artifact), exist_ok=True)
    os.makedirs(os.path.dirname(args.config_out), exist_ok=True)

    # ---- Pool + features ----
    ckpt = torch.load(args.original, weights_only=False)
    attack_scoring.set_seed(int(getattr(ckpt["args"], "seed", 0)))
    model = attack_scoring.build_model(ckpt)
    print(f"[setup] target: {args.original}")

    spec = attack_scoring.build_pool_spec(ckpt, data_root=args.data_root, forget_class=args.forget_class)
    attack_scoring.describe_pool(spec)

    X_original = attack_scoring.features_for(model, spec)
    fit_rows = np.flatnonzero(spec.attack_train_mask)
    val_rows = np.flatnonzero(spec.heldout_mask)

    # Scaler fit on attack-train rows only, then frozen
    scaler = StandardScaler().fit(X_original[fit_rows])
    y_fit = torch.tensor(spec.membership[fit_rows], dtype=torch.float32).unsqueeze(1)
    y_val = spec.membership[val_rows]

    # ---- Train (no bandwidth sweep: the MLP takes standardized features directly) ----
    x_fit_t = as_tensor(scaler.transform(X_original[fit_rows]))
    x_val_t = as_tensor(scaler.transform(X_original[val_rows]))
    attack_model, final_val_auc = train_mlp(
        x_fit_t, y_fit, args.epochs, x_val_t, y_val, print_every=max(1, args.epochs // 10)
    )
    print(f"Final held-out AUC: {final_val_auc:.3f}")

    # ---- Score whole pool; calibrate threshold on attack-train rows and freeze ----
    scores_original = mlp_scores(attack_model, as_tensor(scaler.transform(X_original)))
    threshold = attack_scoring.calibrate_threshold(scores_original, spec.membership, spec.attack_train_mask)

    # ---- Frozen attack model + threshold on the unlearned checkpoint ----
    scores_unlearned = None
    try:
        _, unlearned_model = attack_scoring.load_weights_into(args.unlearned, ckpt)
    except FileNotFoundError:
        print(f"\n[warn] unlearned checkpoint not found: {args.unlearned} -- reporting the original model only")
    else:
        print(f"[setup] unlearned: {args.unlearned}")
        X_unlearned = attack_scoring.features_for(unlearned_model, spec)
        scores_unlearned = mlp_scores(attack_model, as_tensor(scaler.transform(X_unlearned)))

    attack_scoring.report(spec, scores_original, scores_unlearned, threshold, "MLP attack")

    # ROC on held-out rows only: the MLP was trained on the attack-train rows
    rows = spec.heldout_mask
    for scores, path in ((scores_original, args.roc_npz_out), (scores_unlearned, args.roc_npz_unlearned_out)):
        if scores is None:
            continue
        roc_utils.save_roc({"mlp": roc_utils.compute_roc(scores[rows], spec.membership[rows])}, path)


    torch.save(attack_model.state_dict(), args.artifact)
    with open(args.config_out, "wb") as f:
        pickle.dump({
            "attack": "mlp",
            "scaler": scaler,
            "threshold": threshold,
            "score_space": "logit",
            "heldout_auc": final_val_auc,
            "input_dim": x_fit_t.shape[1],
            "epochs": args.epochs,
            "target_checkpoint": args.original,
            "attack_split_seed": attack_scoring.ATTACK_SPLIT_SEED,
            "forget_class": args.forget_class,
        }, f)
    print(f"\n[save] MLP attack model -> {args.artifact}, scaler/config -> {args.config_out}")
    attack_scoring.save_scores(args.scores_out, spec, scores_original, scores_unlearned, threshold, "MLP")


if __name__ == "__main__":
    main()