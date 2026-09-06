"""
Metric-based membership inference attacks for evaluating HQNN unlearning.

Threshold-based attacks that require no learned attack model — each uses a
single scalar computed from the target model's softmax output:

  Loss attack          (Yeom et al., 2018)    — low loss → member
  Confidence attack                            — high max-prob → member
  Entropy attack                               — low entropy → member
  Modified-entropy     (Song & Mittal, 2021)   — lower mod-entropy → member

Thresholds are calibrated on the *original* model with ALL training samples
as members and ALL test samples as non-members — the forget class appears on
both sides, so the threshold cannot use "is this a 4" as a proxy for
membership.  The frozen thresholds are then applied to four disjoint groups,
for both the original and the unlearned model:

    member,     non-4   (retain)  — should stay classified "member" (utility)
    member,     class-4 (forget)  — should drop to the class-4 non-member rate
    non-member, non-4             — global false-positive baseline
    non-member, class-4           — class-matched target for the forget row

Successful unlearning: the unlearned model's forget-set membership rate
approaches the class-4 *non-member* rate — not 50%, and not the global FPR,
since class 4 may be intrinsically easier or harder than the average digit.

Usage:
    python metric_attack.py
    python metric_attack.py --original path/to/orig.pth --unlearned path/to/unl.pth
"""

import argparse
import random

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats as scipy_stats
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from HQNN import ConvQNN


FORGET_CLASS = 4  # MNIST digit being unlearned — class 4 throughout this codebase


# ---------------------------------------------------------------------------
# Per-sample metric extraction
# ---------------------------------------------------------------------------

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

            # Modified entropy (Song & Mittal 2021):
            # sum_{k != y} -p_k log p_k  +  (1 - p_y) log p_y
            # Members tend to score lower because p_y is large.
            p_y = float(prob[y_idx])
            mod_entropy = float(
                sum(-prob[k] * np.log(prob[k] + 1e-9) for k in range(len(prob)) if k != y_idx)
                + (1.0 - p_y) * np.log(p_y + 1e-9)
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

# key → (higher_is_member, human-readable label)
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
    trainset = datasets.MNIST(root="./data", train=True,  download=True, transform=transform)
    testset  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

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
    summary = {}
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


# ---------------------------------------------------------------------------
# Reference model comparison
# ---------------------------------------------------------------------------

def _get_softmax_outputs(model, loader):
    """Return (N, C) array of softmax probabilities, one row per sample."""
    model.eval()
    probs = []
    with torch.no_grad():
        for x, _ in loader:
            p = F.softmax(model(x), dim=1).squeeze(0).numpy()
            probs.append(p)
    return np.array(probs)


def _kl_div(p, q, eps=1e-9):
    """KL(p || q)."""
    p, q = p + eps, q + eps
    return float(np.sum(p * np.log(p / q)))


def _js_div(p, q, eps=1e-9):
    """Jensen–Shannon divergence (symmetric, in [0, log 2])."""
    m = 0.5 * (p + q) + eps
    return float(
        0.5 * np.sum((p + eps) * np.log((p + eps) / m))
        + 0.5 * np.sum((q + eps) * np.log((q + eps) / m))
    )


def run_reference_comparison(original_ckpt_path, reference_ckpt_path, unlearned_ckpt_path):
    """
    Compare unlearned model to reference model (retrained without forget class)
    on the forget set.  Statistically indistinguishable outputs → unlearning
    is as good as true forgetting.
    """
    # Load original model only to recover dataset indices and seed
    orig_ckpt = torch.load(original_ckpt_path, weights_only=False)
    seed = orig_ckpt["args"].seed

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    trainset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)

    train_indices = orig_ckpt["indices"][0]
    forget_idx = [i for i in train_indices if trainset[i][1] == FORGET_CLASS]
    loader_forget = DataLoader(Subset(trainset, forget_idx), batch_size=1, shuffle=False)

    # Load reference model (retrained from scratch without class 4)
    ref_ckpt = torch.load(reference_ckpt_path, weights_only=False)
    ref_model = ConvQNN(ref_ckpt["args"])
    ref_model.load_state_dict(ref_ckpt["model_state_dict"])

    # Load unlearned model
    unl_ckpt = torch.load(unlearned_ckpt_path, weights_only=False)
    unl_model = ConvQNN(orig_ckpt["args"])
    unl_model.load_state_dict(unl_ckpt["model_state_dict"])

    print("Computing softmax outputs on forget set …")
    ref_probs = _get_softmax_outputs(ref_model, loader_forget)   # (N, 10)
    unl_probs = _get_softmax_outputs(unl_model, loader_forget)   # (N, 10)

    n = len(ref_probs)

    # Per-sample distances
    kl_unl_ref  = np.array([_kl_div(unl_probs[i], ref_probs[i]) for i in range(n)])
    kl_ref_unl  = np.array([_kl_div(ref_probs[i], unl_probs[i]) for i in range(n)])
    js_divs     = np.array([_js_div(unl_probs[i], ref_probs[i]) for i in range(n)])
    l1_dists    = np.abs(unl_probs - ref_probs).sum(axis=1)
    cosine_sims = np.array([
        float(
            np.dot(unl_probs[i], ref_probs[i])
            / (np.linalg.norm(unl_probs[i]) * np.linalg.norm(ref_probs[i]) + 1e-9)
        )
        for i in range(n)
    ])

    # Scalar distributions for statistical tests
    ref_conf    = ref_probs.max(axis=1)
    unl_conf    = unl_probs.max(axis=1)
    ref_entropy = -(ref_probs * np.log(ref_probs + 1e-9)).sum(axis=1)
    unl_entropy = -(unl_probs * np.log(unl_probs + 1e-9)).sum(axis=1)

    ks_conf    = scipy_stats.ks_2samp(ref_conf,    unl_conf)
    ks_entropy = scipy_stats.ks_2samp(ref_entropy, unl_entropy)
    t_conf     = scipy_stats.ttest_ind(ref_conf,   unl_conf)

    ref_preds  = ref_probs.argmax(axis=1)
    unl_preds  = unl_probs.argmax(axis=1)
    agree_pct  = float((ref_preds == unl_preds).mean()) * 100

    W = 72
    print(f"\n{'─'*W}")
    print("Reference model comparison — forget set (class 4)")
    print(f"  Reference model : {reference_ckpt_path}")
    print(f"  Unlearned model : {unlearned_ckpt_path}")
    print(f"  Forget samples  : {n}")
    print(f"{'─'*W}")

    print(f"\n{'Distribution distance (per sample)':<40} {'Mean':>10} {'Std':>10} {'Median':>10}")
    print(f"{'─'*W}")
    rows = [
        ("KL(unlearned || reference)", kl_unl_ref),
        ("KL(reference || unlearned)", kl_ref_unl),
        ("JS divergence (symmetric)",  js_divs),
        ("L1 distance",               l1_dists),
        ("Cosine similarity",          cosine_sims),
    ]
    for label, vals in rows:
        print(f"{label:<40} {vals.mean():>10.4f} {vals.std():>10.4f} {np.median(vals):>10.4f}")

    print(f"\nPrediction agreement (unlearned vs reference): {agree_pct:.1f}%")

    # Per-sample breakdown sorted by KL divergence descending
    order = np.argsort(kl_unl_ref)[::-1]
    print(f"\n{'Per-sample breakdown (sorted by KL, worst first)'}")
    print(f"{'─'*W}")
    print(f"  {'#':>3}  {'DataIdx':>7}  {'KL(u‖r)':>8}  {'JS':>6}  {'L1':>6}  {'CosSim':>7}  {'Ref→':>5}  {'Unl→':>5}  {'Match':>5}")
    print(f"{'─'*W}")
    for rank, i in enumerate(order):
        match = "  ok" if ref_preds[i] == unl_preds[i] else "DIFF"
        print(
            f"  {rank+1:>3}  {forget_idx[i]:>7}  "
            f"{kl_unl_ref[i]:>8.4f}  {js_divs[i]:>6.4f}  {l1_dists[i]:>6.4f}  "
            f"{cosine_sims[i]:>7.4f}  {ref_preds[i]:>5}  {unl_preds[i]:>5}  {match:>5}"
        )

    def verdict(p):
        return "similar" if p > 0.05 else "differ"

    print(f"\n{'Statistical tests  (p > 0.05 → distributions indistinguishable)'}")
    print(f"{'─'*W}")
    print(f"{'Test':<45} {'Statistic':>10} {'p-value':>10} {'Verdict':>9}")
    print(f"{'─'*W}")
    tests = [
        ("KS test: max confidence",    ks_conf.statistic,  ks_conf.pvalue),
        ("KS test: entropy",           ks_entropy.statistic, ks_entropy.pvalue),
        ("Welch t-test: max confidence", abs(t_conf.statistic), t_conf.pvalue),
    ]
    n_pass = 0
    for label, stat, pval in tests:
        v = verdict(pval)
        n_pass += pval > 0.05
        print(f"{label:<45} {stat:>10.4f} {pval:>10.4f} {v:>9}")

    print(f"\n{'─'*W}")
    if n_pass == 3:
        print("VERDICT: All tests pass — outputs statistically indistinguishable.")
        print("         Unlearning quality: EXCELLENT (matches reference retraining).")
    elif n_pass == 2:
        print(f"VERDICT: 2/3 tests pass — outputs largely similar to reference.")
        print("         Unlearning quality: GOOD.")
    elif n_pass == 1:
        print(f"VERDICT: 1/3 tests pass — outputs partially resemble reference.")
        print("         Unlearning quality: PARTIAL.")
    else:
        print("VERDICT: No tests pass — outputs diverge from reference on forget set.")
        print("         Unlearning quality: POOR.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Metric-based MIA attacks for HQNN unlearning evaluation"
    )
    parser.add_argument(
        "--original",
        default="model_0324_original_5_0.1_8.pth",
        help="Path to original trained model checkpoint",
    )
    parser.add_argument(
        "--unlearned",
        default="result/seed/model_MU_gradient_U8R1.pth",
        help="Path to unlearned model checkpoint",
    )
    parser.add_argument(
        "--reference",
        default=None,
        help="Path to reference model (retrained without forget class). "
             "When provided, runs the reference comparison instead of metric attacks.",
    )
    args = parser.parse_args()
    if args.reference is not None:
        run_reference_comparison(args.original, args.reference, args.unlearned)
    else:
        run_metric_attacks(args.original, args.unlearned)