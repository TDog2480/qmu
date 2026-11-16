"""
Export per-sample metric values for the threshold (metric-based) attacks so their
score distributions can be plotted.

metric_attack.py recomputes loss / max_conf / entropy / mod_entropy from the
checkpoints on every run and never saves them. This script reuses its functions
unchanged (same groups, same metric code, same threshold calibration) and writes
everything to one small .npz.

Place in attacks/ next to metric_attack.py and run from the repo root:
    python attacks/export_metric_scores.py
    python attacks/export_metric_scores.py --original path/to/orig.pth --unlearned path/to/unl.pth

Output (default roc_curve_output/metric_scores.npz), for each metric key in
{loss, max_conf, entropy, mod_entropy}:
    thr_<key>                      calibrated threshold (original model, all members vs all non-members)
    higher_<key>                   1 if a higher value means "member"
    orig_<key>_<group>             per-sample values, original model
    unl_<key>_<group>              per-sample values, unlearned model
with <group> in {member_non4, member_c4, nonmember_non4, nonmember_c4}.
"""

import argparse
import os
import random
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import metric_attack as ma          # attacks/ is on sys.path when this file is run as a script
import roc_utils
from models import build_model


def main():
    parser = argparse.ArgumentParser(description="Export per-sample metric attack scores")
    parser.add_argument(
        "--original",
        default=os.path.join(REPO_ROOT, "checkpoints", "original", "model_HQNN_original_5_0.1_8.pth"),
    )
    parser.add_argument(
        "--unlearned",
        default=os.path.join(REPO_ROOT, "checkpoints", "unlearned", "model_QMU_gradient_seed5.pth"),
    )
    parser.add_argument("--out", default=os.path.join(roc_utils.OUT_DIR, "metric_scores.npz"))
    args = parser.parse_args()

    orig_ckpt = torch.load(args.original, weights_only=False)
    model = build_model(orig_ckpt["args"])
    model.load_state_dict(orig_ckpt["model_state_dict"])

    seed = orig_ckpt["args"].seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    loaders, sizes = ma._build_group_loaders(orig_ckpt)
    print("Group sizes:", {g: sizes[g] for g in ma.GROUPS})

    print("Computing metrics on ORIGINAL model ...")
    orig = {g: ma.compute_metrics(model, loaders[g]) for g in ma.GROUPS}

    unl_ckpt = torch.load(args.unlearned, weights_only=False)
    model.load_state_dict(unl_ckpt["model_state_dict"])
    print("Computing metrics on UNLEARNED model ...")
    unl = {g: ma.compute_metrics(model, loaders[g]) for g in ma.GROUPS}

    # Same calibration as metric_attack.py: original model, all members vs all non-members.
    members = orig["member_non4"] + orig["member_c4"]
    nonmembers = orig["nonmember_non4"] + orig["nonmember_c4"]

    out = {}
    for key, (higher, _label) in ma.ATTACKS.items():
        thr, acc, auc = ma.calibrate_threshold(members, nonmembers, key, higher)
        print(f"{key:<12} threshold {thr:.5f}  (in-sample acc {acc:.3f}, AUC {auc:.3f})")
        out[f"thr_{key}"] = np.float64(thr)
        out[f"higher_{key}"] = np.int64(int(higher))
        for g in ma.GROUPS:
            out[f"orig_{key}_{g}"] = np.array([r[key] for r in orig[g]], dtype=np.float64)
            out[f"unl_{key}_{g}"] = np.array([r[key] for r in unl[g]], dtype=np.float64)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, **out)
    print(f"[save] per-sample metric scores -> {args.out}")


if __name__ == "__main__":
    main()