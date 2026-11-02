"""
Prediction-class agreement between the reference model (retrained from
scratch without the forget class) and the unlearned model, on the reference
model's own test-set subset (checkpoint["indices"][1]) — not the raw
10,000-sample MNIST test split.

This subset is never touched during unlearning (only train_data_U /
train_data_R subsets of the training set receive gradient updates — see
process_continue_training in unlearning/QMU_gradient_ascent.py), so it's 
safe to use here for evaluation.

This only compares argmax predicted class, not softmax distributions.
For the softmax-distribution comparison on the forget set itself, see
+forget_set_distribution.py.

IMPORTANT — interpretation is the OPPOSITE of comparing to the original
model. The reference model was never trained on the forget class, so it
represents the "gold standard" of what a perfectly-unlearned model should
look like. That means we want HIGH agreement with it everywhere, including
on the forget class:
  - High retain agreement  -> the unlearned model still behaves like a
    normal model on classes it's supposed to remember.
  - High forget-class agreement -> the unlearned model's behavior on the
    forgotten class matches a model that never saw it at all, i.e.
    unlearning worked.
Low forget-class agreement here means the unlearned model still behaves
differently from "never having seen class 4" -- i.e. imperfect unlearning.
(Contrast with comparing against the *original* model, where you'd want the
forget-class agreement to be LOW, since divergence from the pre-unlearning
model is the signal that something changed.)

Reports three views:
  1. Per-class agreement: for each digit 0-9, what fraction of samples get
     the same predicted class from both models.
  2. Retain vs. forget split: aggregate agreement on non-4 samples (want
     high -> utility preserved) vs. class-4 samples (also want high -> the
     unlearned model matches the reference model's true-forgetting behavior).
  3. Class-4 zoom-in: every single class-4 sample accounted for (not just
     disagreements) -- an aggregated (ref_pred, unl_pred) count table plus a
     full per-sample list, so agreement and disagreement patterns are both
     visible without a full 10x10 confusion matrix.

Usage:
    python reference_comparison/unlearned_vs_reference.py --reference path/to/ref.pth --unlearned path/to/unl.pth
"""

import argparse
import math
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import wilcoxon
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from models import build_model

FORGET_CLASS = 4  # MNIST digit being unlearned — class 4 throughout this codebase
DATA_ROOT = os.path.join(REPO_ROOT, "data")


def _get_probs_and_labels(model, loader):
    """Return (probs, labels) arrays, one softmax row / label per sample in loader."""
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x)
            probs.append(F.softmax(logits, dim=1).squeeze(0).numpy())
            labels.append(int(y.item()))
    return np.array(probs), np.array(labels)


def _per_class_agreement(ref_preds, unl_preds, true_labels):
    rows = []
    for c in range(10):
        mask = true_labels == c
        n = int(mask.sum())
        agree = float((ref_preds[mask] == unl_preds[mask]).mean()) * 100 if n else float("nan")
        rows.append((c, n, agree))
    return rows


def _class4_full_breakdown(ref_preds, unl_preds, true_labels, data_indices):
    """
    Zoom in on class 4 only: every class-4 sample accounted for, agreements
    and disagreements alike. Returns:
      - per-sample list of (data_idx, ref_pred, unl_pred, match)
      - aggregated (ref_pred, unl_pred) -> count table, sorted by count,
        which sums to the full class-4 sample count (unlike the disagreement-
        only breakdown, agreement pairs like (4 -> 4) are included too)
    """
    mask = true_labels == FORGET_CLASS
    idx_arr = np.array(data_indices)[mask]
    ref4 = ref_preds[mask]
    unl4 = unl_preds[mask]

    per_sample = [
        (int(idx_arr[i]), int(ref4[i]), int(unl4[i]), bool(ref4[i] == unl4[i]))
        for i in range(len(idx_arr))
    ]

    pairs = {}
    for r, u in zip(ref4, unl4):
        pairs[(int(r), int(u))] = pairs.get((int(r), int(u)), 0) + 1
    sorted_pairs = sorted(pairs.items(), key=lambda kv: kv[1], reverse=True)

    return per_sample, sorted_pairs



def _retain_forget_split(ref_preds, unl_preds, true_labels):
    retain_mask = true_labels != FORGET_CLASS
    forget_mask = true_labels == FORGET_CLASS

    retain_agree = float((ref_preds[retain_mask] == unl_preds[retain_mask]).mean()) * 100
    forget_agree = float((ref_preds[forget_mask] == unl_preds[forget_mask]).mean()) * 100

    return retain_agree, forget_agree, int(retain_mask.sum()), int(forget_mask.sum())


def run_prediction_agreement(reference_ckpt_path, unlearned_ckpt_path, original_ckpt_path=None):
    ref_ckpt = torch.load(reference_ckpt_path, weights_only=False)
    seed = ref_ckpt["args"].seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    testset = datasets.MNIST(root=DATA_ROOT, train=False, download=True, transform=transform)

    # Use the same test-set subset the reference model was evaluated/trained
    # against, rather than the raw 10,000-sample MNIST test split, so results
    # line up with the reference model's own notion of "non-member" data.
    test_indices = ref_ckpt["indices"][1]
    test_subset = Subset(testset, test_indices)
    loader = DataLoader(test_subset, batch_size=1, shuffle=False)

    ref_model = build_model(ref_ckpt["args"])
    ref_model.load_state_dict(ref_ckpt["model_state_dict"])

    unl_ckpt = torch.load(unlearned_ckpt_path, weights_only=False)
    unl_model = build_model(unl_ckpt["args"])
    unl_model.load_state_dict(unl_ckpt["model_state_dict"])

    print(f"Computing predictions on reference model's test subset ({len(test_indices)} samples) …")
    ref_probs, ref_labels = _get_probs_and_labels(ref_model, loader)
    unl_probs, _          = _get_probs_and_labels(unl_model, loader)
    ref_preds = ref_probs.argmax(axis=1)
    unl_preds = unl_probs.argmax(axis=1)
    true_labels = ref_labels  # same dataset/order for both models

    orig_probs = None
    if original_ckpt_path:
        orig_ckpt = torch.load(original_ckpt_path, weights_only=False)
        orig_model = build_model(orig_ckpt["args"])
        orig_model.load_state_dict(orig_ckpt["model_state_dict"])
        orig_probs, _ = _get_probs_and_labels(orig_model, loader)

    W = 60

    # --- View 1: per-class agreement ---------------------------------
    print(f"\n{'─'*W}")
    print("Per-class prediction agreement — reference vs. unlearned")
    print(f"  Reference model : {reference_ckpt_path}")
    print(f"  Unlearned model : {unlearned_ckpt_path}")
    print(f"{'─'*W}")
    print(f"{'Class':>5}  {'n':>6}  {'Agreement':>10}")
    print(f"{'-'*W}")
    for c, n, agree in _per_class_agreement(ref_preds, unl_preds, true_labels):
        marker = "  <- forget class" if c == FORGET_CLASS else ""
        print(f"{c:>5}  {n:>6}  {agree:>9.1f}%{marker}")

    # --- View 2: retain vs. forget aggregate --------------------------
    retain_agree, forget_agree, n_retain, n_forget = _retain_forget_split(
        ref_preds, unl_preds, true_labels
    )
    print(f"\n{'─'*W}")
    print("Retain vs. forget aggregate")
    print(f"{'─'*W}")
    print(f"Retain (non-4, n={n_retain}): {retain_agree:.1f}% agreement  (want: high -> utility preserved)")
    print(f"Forget (class-4, n={n_forget}): {forget_agree:.1f}% agreement  (want: high -> matches reference's true-forgetting behavior)")

    # --- View 3: class-4 zoom-in, every sample accounted for ---------
    per_sample, sorted_pairs = _class4_full_breakdown(
        ref_preds, unl_preds, true_labels, test_indices
    )
    print(f"\n{'─'*W}")
    print(f"Class-4 zoom-in — every class-4 sample, agreements and disagreements")
    print(f"{'─'*W}")
    print("Aggregated (ref_pred -> unl_pred) counts, sums to full class-4 count:")
    for (r, u), count in sorted_pairs:
        tag = "  (agree)" if r == u else "  (disagree)"
        print(f"  ref={r} -> unl={u}   : {count}{tag}")

    print(f"\nPer-sample breakdown ({len(per_sample)} class-4 samples):")
    print(f"  {'data_idx':>8}  {'ref':>4}  {'unl':>4}  {'match':>6}")
    for data_idx, r, u, match in per_sample:
        print(f"  {data_idx:>8}  {r:>4}  {u:>4}  {('yes' if match else 'no'):>6}")

    orig_preds = orig_probs.argmax(axis=1) if orig_probs is not None else None
    agreement_confidence_intervals(ref_preds, unl_preds, orig_preds, true_labels)
    statistical_comparison(ref_preds, unl_preds, true_labels)
    distribution_comparison(ref_probs, unl_probs, orig_probs, true_labels)


def _bootstrap_agreement_ci(agree, n_boot=10000, level=0.95, seed=0):
    """Percentile bootstrap CI (in %) for the mean of a boolean agreement array."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(agree), size=(n_boot, len(agree)))
    means = agree[idx].mean(axis=1) * 100
    lo, hi = np.percentile(means, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(lo), float(hi)


def agreement_confidence_intervals(ref_preds, unl_preds, orig_preds, true_labels):
    """
    Bootstrap 95% CIs on the argmax-agreement rate with the reference, on all
    samples, the retain set (non-4) and the forget set (class-4).

    Resamples test samples with replacement, so the interval reflects
    sampling uncertainty from having only n test samples (not training or
    seed variability). If orig_preds is given, the original model's agreement
    with the reference is shown as a yardstick: it is what "no unlearning"
    looks like, so the unlearned model's agreement is only meaningful read
    against it. Overlapping intervals between rows mean the difference could
    be sampling noise.
    """
    forget_mask = true_labels == FORGET_CLASS
    subsets = [
        ("All", np.ones_like(forget_mask)),
        ("Retain (non-4)", ~forget_mask),
        ("Forget (class-4)", forget_mask),
    ]
    models = [("unlearned", unl_preds)]
    if orig_preds is not None:
        models.append(("original", orig_preds))

    W = 60
    print(f"\n{'─'*W}")
    print("Agreement with reference — bootstrap 95% CI (10,000 resamples)")
    print(f"{'─'*W}")
    print(f"{'Subset':<18}{'Model':<11}{'n':>6}{'Agree':>9}   95% CI")
    for name, mask in subsets:
        for model_name, preds in models:
            agree = preds[mask] == ref_preds[mask]
            lo, hi = _bootstrap_agreement_ci(agree)
            print(f"{name:<18}{model_name:<11}{int(mask.sum()):>6}{agree.mean()*100:>8.1f}%   [{lo:.1f}%, {hi:.1f}%]")


def _mcnemar_exact(b, c):
    """
    Two-sided exact McNemar p-value. b = samples only the reference gets
    right, c = samples only the unlearned model gets right. Under H0 (the
    models have the same error rate) each discordant pair is a fair coin flip,
    so min(b, c) ~ Binomial(b + c, 0.5).
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def statistical_comparison(ref_preds, unl_preds, true_labels, alpha=0.05):
    """
    Paired McNemar test on per-sample correctness: do the reference and
    unlearned models have significantly different accuracy on the same test
    samples? Run on all samples, the retain set (non-4), and the forget set
    (class 4).

    Failing to reject H0 (p >= alpha) means no detectable difference, which is
    the desired outcome for unlearning. It is not proof of equivalence,
    especially on the small forget set where power is low. Only discordant
    samples (one model right, the other wrong) carry information; if there
    are none, the models are identical in correctness and p = 1.
    """
    correct_ref = ref_preds == true_labels
    correct_unl = unl_preds == true_labels
    forget_mask = true_labels == FORGET_CLASS

    subsets = [
        ("All", np.ones_like(forget_mask)),
        ("Retain (non-4)", ~forget_mask),
        ("Forget (class-4)", forget_mask),
    ]

    W = 60
    print(f"\n{'─'*W}")
    print(f"Statistical comparison — McNemar exact test on correctness (alpha={alpha})")
    print(f"{'─'*W}")
    print(f"{'Subset':<18}{'n':>6}{'RefAcc':>8}{'UnlAcc':>8}{'b':>5}{'c':>5}{'p-value':>10}  Result")
    for name, mask in subsets:
        n = int(mask.sum())
        if n == 0:
            continue
        b = int((correct_ref[mask] & ~correct_unl[mask]).sum())
        c = int((~correct_ref[mask] & correct_unl[mask]).sum())
        p = _mcnemar_exact(b, c)
        verdict = "differ" if p < alpha else "no sig. difference"
        print(
            f"{name:<18}{n:>6}{correct_ref[mask].mean()*100:>7.1f}%{correct_unl[mask].mean()*100:>7.1f}%"
            f"{b:>5}{c:>5}{p:>10.4f}  {verdict}"
        )
    print("b = only reference correct, c = only unlearned correct")


def _js_divergence(p, q, eps=1e-12):
    """Per-row Jensen-Shannon divergence in bits, bounded in [0, 1]."""
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    m = 0.5 * (p + q)
    kl_pm = (p * np.log2(p / m)).sum(axis=1)
    kl_qm = (q * np.log2(q / m)).sum(axis=1)
    return 0.5 * (kl_pm + kl_qm)


def _total_variation(p, q):
    """Per-row total variation distance (half the L1 distance), in [0, 1]."""
    return 0.5 * np.abs(p - q).sum(axis=1)


def _paired_wilcoxon(x, y, alternative):
    """Paired Wilcoxon signed-rank p-value; 1.0 if x and y are identical."""
    if np.allclose(x, y):
        return 1.0
    return float(wilcoxon(x, y, alternative=alternative).pvalue)


def distribution_comparison(ref_probs, unl_probs, orig_probs, true_labels, alpha=0.05):
    """
    Softmax-level comparison of the unlearned model to the reference, on the
    retain set (non-4) and forget set (class-4) separately.

    Test 1 (needs orig_probs): per-sample distance to the reference for the
    unlearned model vs. for the original model, using JS divergence and total
    variation. One-sided paired Wilcoxon signed-rank asks whether the
    unlearned model is significantly CLOSER to the reference than the
    original is. The original is the "didn't unlearn" yardstick, so this
    shows unlearning moved the model toward the gold standard. It is a
    superiority test, not an equivalence test: it says "closer than before",
    not "as close as another reference model would be".

    Test 3 (forget set only): paired two-sided Wilcoxon on each model's
    softmax probability of the forget class. Here a NON-significant result is
    the desired outcome, with the usual caveat that failing to reject is not
    proof of equivalence. The mean |difference| is reported alongside so the
    effect size is visible.
    """
    forget_mask = true_labels == FORGET_CLASS
    subsets = [("Retain (non-4)", ~forget_mask), ("Forget (class-4)", forget_mask)]

    W = 60
    print(f"\n{'─'*W}")
    print("Softmax distance to the reference model (per-sample distribution)")
    print(f"{'─'*W}")
    print("JS in bits (0-1), TV = total variation (0-1). Lower = closer to reference.")
    print(f"{'Subset':<18}{'Model':<11}{'Metric':<5}{'mean':>8}{'median':>8}{'p95':>8}")
    metrics = [("JS", _js_divergence), ("TV", _total_variation)]
    dists = {}
    for name, mask in subsets:
        models = [("unlearned", unl_probs)]
        if orig_probs is not None:
            models.append(("original", orig_probs))
        for model_name, probs in models:
            for metric_name, fn in metrics:
                d = fn(probs[mask], ref_probs[mask])
                dists[(name, model_name, metric_name)] = d
                print(
                    f"{name:<18}{model_name:<11}{metric_name:<5}"
                    f"{d.mean():>8.4f}{np.median(d):>8.4f}{np.percentile(d, 95):>8.4f}"
                )

    if orig_probs is None:
        print("\n(No original checkpoint given: skipping unlearned-vs-original test.)")
    else:
        print(f"\n{'─'*W}")
        print(f"Is the unlearned model closer to the reference than the original is?")
        print(f"One-sided paired Wilcoxon, H1: d(unlearned, ref) < d(original, ref), alpha={alpha}")
        print(f"{'─'*W}")
        print(f"{'Subset':<18}{'Metric':<8}{'p-value':>10}  Result")
        for name, _ in subsets:
            for metric_name, _fn in metrics:
                d_unl = dists[(name, "unlearned", metric_name)]
                d_orig = dists[(name, "original", metric_name)]
                p = _paired_wilcoxon(d_unl, d_orig, "less")
                verdict = "unlearned closer" if p < alpha else "no sig. difference"
                print(f"{name:<18}{metric_name:<8}{p:>10.4f}  {verdict}")
        print("Expect: forget set -> unlearned closer. Retain set -> not necessarily")
        print("(original and unlearned may both already be near the reference there).")

    print(f"\n{'─'*W}")
    print(f"Forget-class probability on class-{FORGET_CLASS} samples (paired two-sided Wilcoxon)")
    print(f"{'─'*W}")
    p4_ref = ref_probs[forget_mask, FORGET_CLASS]
    p4_unl = unl_probs[forget_mask, FORGET_CLASS]
    print(f"Mean P(class {FORGET_CLASS}): reference={p4_ref.mean():.4f}  unlearned={p4_unl.mean():.4f}", end="")
    if orig_probs is not None:
        print(f"  original={orig_probs[forget_mask, FORGET_CLASS].mean():.4f}", end="")
    print()
    p = _paired_wilcoxon(p4_unl, p4_ref, "two-sided")
    verdict = "differ" if p < alpha else "no sig. difference"
    print(f"unlearned vs reference: mean |diff|={np.abs(p4_unl - p4_ref).mean():.4f}, p={p:.4f}  {verdict}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare predicted classes of reference vs. unlearned model on the reference model's test subset"
    )
    parser.add_argument(
        "--reference",
        default=os.path.join(REPO_ROOT, "checkpoints", "reference", "model_HQNN_target_5_0.1_8.pth"),
        help="Path to reference model checkpoint (retrained without forget class)",
    )
    parser.add_argument(
        "--unlearned",
        default=os.path.join(REPO_ROOT, "checkpoints", "unlearned", "model_QMU_gradient_seed5.pth"),
        help="Path to unlearned model checkpoint",
    )
    parser.add_argument(
        "--original",
        default=os.path.join(REPO_ROOT, "checkpoints", "original", "model_HQNN_original_5_0.1_8.pth"),
        help="Path to original (pre-unlearning) checkpoint, used as the distance yardstick. Pass '' to skip.",
    )
    args = parser.parse_args()
    run_prediction_agreement(args.reference, args.unlearned, args.original or None)