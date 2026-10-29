"""
Metric-based membership inference attacks for evaluating HQNN unlearning.

Threshold-based attacks that require no learned attack model — each uses a
single scalar computed from the target model's softmax output:

  Loss attack          (Yeom et al., 2018)    — low loss -> member
  Confidence attack                           — high max-prob -> member
  Entropy attack                              — low entropy -> member
  Modified-entropy     (Song & Mittal, 2021)  — lower mod-entropy -> member

Thresholds are calibrated on the original model with ALL training samples
as members and ALL test samples as non-members — the forget class appears on
both sides, so the threshold cannot use "is this a 4" as a proxy for
membership.  

The frozen thresholds are then applied to four disjoint groups, for both the 
original and the unlearned model:

    member,     non-4   (retain)  — should stay classified "member" (utility)
    member,     class-4 (forget)  — should drop to the class-4 non-member rate
    non-member, non-4             — global false-positive baseline
    non-member, class-4           — class-matched target for the forget row

Successful unlearning: the unlearned model's forget-set membership rate
approaches the class-4 *non-member* rate — not 50%, and not the global FPR,
since class 4 may be intrinsically easier or harder than the average digit.

Usage:
    python attacks/metric_attack.py
    python attacks/metric_attack.py --original path/to/orig.pth --unlearned path/to/unl.pth
"""

import argparse
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from HQNN import ConvQNN
import roc_utils

FORGET_CLASS = 4  # MNIST digit being unlearned — class 4 throughout this codebase
DATA_ROOT = os.path.join(REPO_ROOT, "data")

# Per-sample metric extraction

def compute_metrics(model, loader):
    """Return one dict of scalar metrics per sample in loader."""
    model.eval()
    records = []
    with torch.no_grad():
        for x, y in loader:
            output = model(x)
            prob = F.softmax(output, dim=1).squeeze(0).numpy()
            y_idx = int(y.item())

            loss = F.cross_entropy(output, y).item()
            max_conf = float(prob.max())
            entropy = float(-(prob * np.log(prob + 1e-9)).sum())

            # Modified entropy (Song & Mittal 2021, Eq. 8):
            #   M_entr(F(x), y) = -(1 - p_y) * log(p_y) - sum_{k != y} p_k * log(1 - p_k)
            # This is monotonically Decreasing in p_y and monotonically inccreasing in every p_k (k != y): 
            #   confident+correct -> 0, confident+wrong -> +inf. 
            # Members tend to score LOWER, so lower mod_entropy -> member (paper's Eq. 9).

            p_y = float(prob[y_idx])
            mod_entropy = float(
                -(1.0 - p_y) * np.log(p_y + 1e-9)
                - sum(prob[k] * np.log(1.0 - prob[k] + 1e-9) for k in range(len(prob)) if k != y_idx)
            )

            records.append({
                "loss":        loss,
                "max_conf":    max_conf,
                "entropy":     entropy,
                "mod_entropy": mod_entropy,
                "correct":     int(prob.argmax() == y_idx),
                "label":       y_idx,
            })
    return records


# ---------------------------------------------------------------------------
# Attack configurations
# ---------------------------------------------------------------------------

# key -> (higher_is_member, human-readable label)
ATTACKS = {
    "loss":        (False, "Loss attack          (Yeom 2018)"),
    "max_conf":    (True,  "Confidence attack                "),
    "entropy":     (False, "Entropy attack                   "),
    "mod_entropy": (False, "Mod-entropy attack   (Song 2021) "),
}


# ---------------------------------------------------------------------------
# Threshold calibration
# ---------------------------------------------------------------------------

def calibrate_threshold(member_records, nonmember_records, key, higher_is_member):
    """
    Sweep 200 candidate thresholds over the observed value range;
    return (best_threshold, attack_accuracy, AUC).
    """
    m_vals  = np.array([r[key] for r in member_records])
    nm_vals = np.array([r[key] for r in nonmember_records])

    all_vals   = np.concatenate([m_vals, nm_vals])
    all_labels = np.array([1] * len(m_vals) + [0] * len(nm_vals))

    best_acc, best_thr = 0.0, float(np.median(all_vals))
    for thr in np.percentile(all_vals, np.linspace(0, 100, 200)):
        preds = (all_vals >= thr) if higher_is_member else (all_vals <= thr)
        acc = accuracy_score(all_labels, preds.astype(int))
        if acc > best_acc:
            best_acc, best_thr = acc, float(thr)

    score = all_vals if higher_is_member else -all_vals
    auc = roc_auc_score(all_labels, score)
    return best_thr, best_acc, auc


def group_auc(member_records, nonmember_records, key, higher_is_member):
    """Threshold-free separability of two groups on a single metric."""
    m_vals  = np.array([r[key] for r in member_records])
    nm_vals = np.array([r[key] for r in nonmember_records])
    vals   = np.concatenate([m_vals, nm_vals])
    labels = np.array([1] * len(m_vals) + [0] * len(nm_vals))
    return float(roc_auc_score(labels, vals if higher_is_member else -vals))


def membership_rate(records, key, threshold, higher_is_member):
    """Fraction of records classified as members by the threshold."""
    vals = np.array([r[key] for r in records])
    preds = (vals >= threshold) if higher_is_member else (vals <= threshold)
    return float(preds.mean())


# ---------------------------------------------------------------------------
# Dataset groups: membership x forget-class
# ---------------------------------------------------------------------------

GROUPS = ("member_non4", "member_c4", "nonmember_non4", "nonmember_c4")
GROUP_LABEL = {
    "member_non4":    "member,     non-4   (retain)",
    "member_c4":      "member,     class-4 (forget)",
    "nonmember_non4": "non-member, non-4",
    "nonmember_c4":   "non-member, class-4",
}


def _build_group_loaders(orig_ckpt):
    """Four disjoint groups, split by membership (the original model's
    train/test split, read straight from the checkpoint) and by whether the
    ground-truth label is the forget class."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    trainset = datasets.MNIST(root=DATA_ROOT, train=True,  download=True, transform=transform)
    testset  = datasets.MNIST(root=DATA_ROOT, train=False, download=True, transform=transform)

    def split_by_class(ds, idx):
        non4, c4 = [], []
        for i in idx:
            (c4 if ds[i][1] == FORGET_CLASS else non4).append(i)
        return non4, c4

    m_non4, m_c4   = split_by_class(trainset, orig_ckpt["indices"][0])
    nm_non4, nm_c4 = split_by_class(testset,  orig_ckpt["indices"][1])

    spec = {
        "member_non4":    (trainset, m_non4),
        "member_c4":      (trainset, m_c4),
        "nonmember_non4": (testset,  nm_non4),
        "nonmember_c4":   (testset,  nm_c4),
    }
    loaders = {n: DataLoader(Subset(ds, idx), batch_size=1, shuffle=False)
               for n, (ds, idx) in spec.items()}
    sizes = {n: len(idx) for n, (ds, idx) in spec.items()}
    return loaders, sizes


# ---------------------------------------------------------------------------
# Main evaluation pipeline
# ---------------------------------------------------------------------------

def run_metric_attacks(original_ckpt_path, unlearned_ckpt_path):
    # --- Load original model ---
    orig_ckpt = torch.load(original_ckpt_path, weights_only=False)
    model = ConvQNN(orig_ckpt["args"])
    model.load_state_dict(orig_ckpt["model_state_dict"])

    seed = orig_ckpt["args"].seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    loaders, sizes = _build_group_loaders(orig_ckpt)
    print("Group sizes:")
    for name in GROUPS:
        print(f"  {GROUP_LABEL[name]:<32} {sizes[name]:>5}")

    # --- Per-sample metrics on all four groups, both models ---
    print("\nComputing metrics on ORIGINAL model …")
    orig = {name: compute_metrics(model, loaders[name]) for name in GROUPS}

    unl_ckpt = torch.load(unlearned_ckpt_path, weights_only=False)
    model.load_state_dict(unl_ckpt["model_state_dict"])
    print("Computing metrics on UNLEARNED model …")
    unl = {name: compute_metrics(model, loaders[name]) for name in GROUPS}

    # --- Calibrate thresholds on the ORIGINAL model --------------------------
    # Members = all training samples, non-members = all test samples.  Class 4
    # sits on BOTH sides (member_c4 is the forget set; nonmember_c4 are true
    # class-4 non-members), so the threshold sweep cannot latch onto
    # "class-4-ness" as a stand-in for membership.  The forget set does enter
    # the member pool here, but a single scalar threshold can't overfit ~100
    # points among ~1000, and it is calibrated on the original model then
    # frozen before being applied to the unlearned model.
    members_orig    = orig["member_non4"]    + orig["member_c4"]
    nonmembers_orig = orig["nonmember_non4"] + orig["nonmember_c4"]

    thresholds = {}
    print(f"\n{'─'*72}")
    print("Threshold calibration — all members vs all non-members, original model")
    print("  (Acc/threshold are in-sample; AUC is the honest separability number)")
    print(f"{'─'*72}")
    print(f"{'Attack':<48} {'AUC':>6}  {'Acc':>6}  {'Threshold':>12}")
    print(f"{'─'*72}")
    for key, (higher, label) in ATTACKS.items():
        thr, acc, auc = calibrate_threshold(members_orig, nonmembers_orig, key, higher)
        thresholds[key] = thr
        print(f"{label:<48} {auc:>6.3f}  {acc:>6.3f}  {thr:>12.5f}")

    # --- ROC data, original model: all members vs all non-members ---------
    labels = [1] * len(members_orig) + [0] * len(nonmembers_orig)
    roc = {key: roc_utils.compute_roc([r[key] for r in members_orig + nonmembers_orig], labels, higher)
           for key, (higher, _) in ATTACKS.items()}
    roc_utils.save_roc(roc, os.path.join(roc_utils.OUT_DIR, "metric_attack_roc.npz"))

    # --- 4-group membership rates, per attack ------------------------------
    #   member,     non-4    -> want it to STAY high      (utility preserved)
    #   member,     class-4  -> want it to DROP to nonmember_c4   (forgotten)
    #   non-member, non-4    -> global false-positive baseline
    #   non-member, class-4  -> class-matched target for the forget row
    def rates(records_by_group, key, higher):
        return {g: membership_rate(records_by_group[g], key, thresholds[key], higher)
                for g in GROUPS}

    print(f"\n{'='*76}")
    print("Membership rate by group  (fraction the frozen threshold calls 'member')")
    print(f"{'='*76}")
    # Class-matched AUCs: member vs non-member within the same class slice.
    #   non-4   -> want it to STAY at the original level (utility preserved)
    #   class-4 -> want it to DROP to ~0.5            (forgotten)
    def class_aucs(records_by_group, key, higher):
        return {
            "non4": group_auc(records_by_group["member_non4"],
                              records_by_group["nonmember_non4"], key, higher),
            "c4":   group_auc(records_by_group["member_c4"],
                              records_by_group["nonmember_c4"], key, higher),
        }

    summary = {}
    auc_summary = {}
    for key, (higher, label) in ATTACKS.items():
        o = rates(orig, key, higher)
        u = rates(unl,  key, higher)
        print(f"\n{label.strip()}")
        print(f"  {'group':<32} {'n':>5}  {'orig':>8}  {'unlearned':>10}")
        print(f"  {'-'*60}")
        for g in GROUPS:
            print(f"  {GROUP_LABEL[g]:<32} {sizes[g]:>5}  "
                  f"{o[g]*100:>7.1f}%  {u[g]*100:>9.1f}%")
        gap = u["member_c4"] - u["nonmember_c4"]
        summary[key] = (o["member_c4"], u["member_c4"], u["nonmember_c4"], gap)
        print(f"  -> forget rate {o['member_c4']*100:.1f}% -> {u['member_c4']*100:.1f}%   "
              f"(class-4 non-member baseline {u['nonmember_c4']*100:.1f}%,  "
              f"residual gap {gap*100:+.1f} pts)")

        o_auc = class_aucs(orig, key, higher)
        u_auc = class_aucs(unl,  key, higher)
        auc_summary[key] = (o_auc, u_auc)
        print(f"  {'AUC (member vs non-member)':<38} {'orig':>8}  {'unlearned':>10}")
        print(f"  {'-'*60}")
        print(f"  {'non-4   (member vs non-member)':<38} {o_auc['non4']:>8.3f}  {u_auc['non4']:>10.3f}")
        print(f"  {'class-4 (member vs non-member)':<38} {o_auc['c4']:>8.3f}  {u_auc['c4']:>10.3f}")

    # --- Compact cross-attack summary -------------------------------------
    print(f"\n{'='*76}")
    print("Forgetting summary — unlearned forget-set rate vs class-4 non-member baseline")
    print("  gap ~ 0  -> forget set indistinguishable from a true class-4 non-member")
    print("  gap > 0  -> residual membership signal (under-forgotten)")
    print(f"{'='*76}")
    print(f"{'Attack':<40} {'forget orig->unl':>18} {'c4 nonmem':>11} {'gap':>8}")
    print(f"{'-'*76}")
    for key, (higher, label) in ATTACKS.items():
        o_f, u_f, u_nm, gap = summary[key]
        print(f"{label.strip():<40} {o_f*100:>6.1f}% ->{u_f*100:>5.1f}%   "
              f"{u_nm*100:>9.1f}%  {gap*100:>+7.1f}")

    # --- Class-matched AUC summary ----------------------------------------
    print(f"\n{'='*76}")
    print("Class-matched AUC — member vs non-member within each class slice")
    print("  non-4:   should stay near its original value (utility preserved)")
    print("  class-4: should fall to ~0.5 after unlearning (forgotten)")
    print(f"{'='*76}")
    print(f"{'Attack':<40} {'non-4 orig->unl':>17} {'class-4 orig->unl':>19}")
    print(f"{'-'*76}")
    for key, (higher, label) in ATTACKS.items():
        o_auc, u_auc = auc_summary[key]
        print(f"{label.strip():<40} {o_auc['non4']:>7.3f} ->{u_auc['non4']:>6.3f}  "
              f"{o_auc['c4']:>9.3f} ->{u_auc['c4']:>6.3f}")

    # --- Metric distribution shift on the forget set ---------------------
    print(f"\n{'─'*62}")
    print("Forget-set metric shift (mean over class-4 training samples)")
    print(f"{'─'*62}")
    print(f"{'Metric':<14} {'orig':>10} {'unlearned':>11} {'Δ':>10} {'c4 nonmem':>12}")
    print(f"{'─'*62}")
    for key in ATTACKS:
        before = np.mean([r[key] for r in orig["member_c4"]])
        after  = np.mean([r[key] for r in unl["member_c4"]])
        baseln = np.mean([r[key] for r in unl["nonmember_c4"]])
        print(f"{key:<14} {before:>10.4f} {after:>11.4f} {after - before:>+10.4f} {baseln:>12.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Metric-based MIA attacks for HQNN unlearning evaluation")
    parser.add_argument(
        "--original",
        default=os.path.join(REPO_ROOT, "checkpoints", "original", "model_HQNN_original_5_0.1_8.pth"),
        help="Path to original trained model checkpoint",
    )
    parser.add_argument(
        "--unlearned",
        default=os.path.join(REPO_ROOT, "checkpoints", "unlearned", "model_QMU_gradient_seed5.pth"),
        help="Path to unlearned model checkpoint",
    )
    args = parser.parse_args()
    run_metric_attacks(args.original, args.unlearned)