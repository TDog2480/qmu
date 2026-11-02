"""
Class-4 membership-attack AUC on the reference model, as a "no true signal"
baseline for the unlearned model's class-4 AUC.

The reference model was retrained without class 4, so it never saw ANY class-4
sample. The 95 class-4 "members" (the original model's training samples) and
100 class-4 "non-members" (the original model's test samples) are therefore
equally unseen by it. Scoring them with the metric attacks and computing
member-vs-non-member AUC (using the original model's membership labels) gives
what a model with zero membership signal on class 4 scores: any deviation from
0.5 comes from sampling noise or from the two sample groups differing in
difficulty, not from memorization.

If the unlearned model's class-4 AUC sits clearly above the reference's, the
excess is evidence of residual membership signal. If it sits inside the
reference's range, "above 0.5" can't be distinguished from that baseline.

Reuses the group split and per-sample metrics from attacks/metric_attack.py
(membership comes from the ORIGINAL checkpoint's saved indices), so numbers
line up with the class-4 column of the paper's per-group AUC table.

Each AUC gets a 95% percentile-bootstrap CI (members and non-members resampled
independently). Intervals reflect test-sample uncertainty only, not
training-seed variability, and a single reference is one draw of the baseline.

Usage:
    python reference-comparison/reference_attack_auc.py --unlearned path/to/unl.pth
"""

import argparse
import os
import random
import sys

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from models import build_model
from attacks.metric_attack import ATTACKS, GROUP_LABEL, _build_group_loaders, compute_metrics


def _load_model(ckpt_path):
    ckpt = torch.load(ckpt_path, weights_only=False)
    model = build_model(ckpt["args"])
    model.load_state_dict(ckpt["model_state_dict"])
    return model


def _auc(member_vals, nonmember_vals, higher_is_member):
    vals = np.concatenate([member_vals, nonmember_vals])
    labels = np.array([1] * len(member_vals) + [0] * len(nonmember_vals))
    return float(roc_auc_score(labels, vals if higher_is_member else -vals))


def _auc_with_ci(member_vals, nonmember_vals, higher_is_member, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    point = _auc(member_vals, nonmember_vals, higher_is_member)
    boots = []
    for _ in range(n_boot):
        m = rng.choice(member_vals, size=len(member_vals), replace=True)
        nm = rng.choice(nonmember_vals, size=len(nonmember_vals), replace=True)
        boots.append(_auc(m, nm, higher_is_member))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, float(lo), float(hi)


def run(original_path, reference_path, unlearned_path):
    orig_ckpt = torch.load(original_path, weights_only=False)
    seed = orig_ckpt["args"].seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    loaders, sizes = _build_group_loaders(orig_ckpt)
    print("Group sizes:")
    for g in ("member_c4", "nonmember_c4", "member_non4", "nonmember_non4"):
        print(f"  {GROUP_LABEL[g]:<32} {sizes[g]:>5}")

    models = [
        ("original", original_path),
        ("unlearned", unlearned_path),
        ("reference", reference_path),
    ]
    records = {}
    for name, path in models:
        print(f"Computing metrics on {name.upper()} model ...")
        model = _load_model(path)
        records[name] = {g: compute_metrics(model, loaders[g]) for g in
                         ("member_c4", "nonmember_c4", "member_non4", "nonmember_non4")}

    W = 78
    for title, m_group, nm_group in [
        ("Class-4 AUC: member vs non-member (reference = no-signal baseline)",
         "member_c4", "nonmember_c4"),
        ("Non-4 AUC: member vs non-member (context)",
         "member_non4", "nonmember_non4"),
    ]:
        print(f"\n{'─'*W}")
        print(title)
        print(f"{'─'*W}")
        print(f"{'Attack':<40}{'Model':<11}{'AUC':>7}   95% CI")
        for key, (higher, label) in ATTACKS.items():
            for name, _ in models:
                m = np.array([r[key] for r in records[name][m_group]])
                nm = np.array([r[key] for r in records[name][nm_group]])
                auc, lo, hi = _auc_with_ci(m, nm, higher)
                lbl = label.strip() if name == "original" else ""
                print(f"{lbl:<40}{name:<11}{auc:>7.3f}   [{lo:.3f}, {hi:.3f}]")
    print(
        "\nRead: if the unlearned class-4 AUC is above the reference's (and its CI "
        "does not overlap),\nthe excess is residual membership signal. Overlapping "
        "CIs mean the data cannot separate\nthe unlearned model from a model with "
        "no signal."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Class-4 membership-attack AUC on original / unlearned / reference models"
    )
    parser.add_argument(
        "--original",
        default=os.path.join(REPO_ROOT, "checkpoints", "original", "model_HQNN_original_5_0.1_8.pth"),
        help="Original model checkpoint (defines membership via its saved indices)",
    )
    parser.add_argument(
        "--reference",
        default=os.path.join(REPO_ROOT, "checkpoints", "reference", "model_HQNN_target_5_0.1_8.pth"),
        help="Reference model checkpoint (retrained without the forget class)",
    )
    parser.add_argument(
        "--unlearned",
        default=os.path.join(REPO_ROOT, "checkpoints", "unlearned", "model_QMU_gradient_seed5.pth"),
        help="Unlearned model checkpoint",
    )
    args = parser.parse_args()
    run(args.original, args.reference, args.unlearned)
