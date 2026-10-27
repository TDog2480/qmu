"""
Prediction-class agreement between the reference model (retrained from
scratch without the forget class) and the unlearned model, on the reference
model's own test-set subset (checkpoint["indices"][1]) — not the raw
10,000-sample MNIST test split.

This subset is never touched during unlearning (only train_data_U /
train_data_R subsets of the training set receive gradient updates — see
process_continue_training in HQNN.py), so it's safe to use here for
evaluation.

This only compares argmax predicted class, not softmax distributions.

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
    python prediction_agreement.py --reference path/to/ref.pth --unlearned path/to/unl.pth
"""

import argparse
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from HQNN import ConvQNN


FORGET_CLASS = 4  # MNIST digit being unlearned — class 4 throughout this codebase


def _get_predictions_and_labels(model, loader):
    """Return (preds, labels) arrays, one entry per sample in loader."""
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x)
            pred = int(F.softmax(logits, dim=1).argmax(dim=1).item())
            preds.append(pred)
            labels.append(int(y.item()))
    return np.array(preds), np.array(labels)


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


def run_prediction_agreement(reference_ckpt_path, unlearned_ckpt_path):
    ref_ckpt = torch.load(reference_ckpt_path, weights_only=False)
    seed = ref_ckpt["args"].seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    testset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    # Use the same test-set subset the reference model was evaluated/trained
    # against, rather than the raw 10,000-sample MNIST test split, so results
    # line up with the reference model's own notion of "non-member" data.
    test_indices = ref_ckpt["indices"][1]
    test_subset = Subset(testset, test_indices)
    loader = DataLoader(test_subset, batch_size=1, shuffle=False)

    ref_model = ConvQNN(ref_ckpt["args"])
    ref_model.load_state_dict(ref_ckpt["model_state_dict"])

    unl_ckpt = torch.load(unlearned_ckpt_path, weights_only=False)
    unl_model = ConvQNN(unl_ckpt["args"])
    unl_model.load_state_dict(unl_ckpt["model_state_dict"])

    print(f"Computing predictions on reference model's test subset ({len(test_indices)} samples) …")
    ref_preds, ref_labels = _get_predictions_and_labels(ref_model, loader)
    unl_preds, _          = _get_predictions_and_labels(unl_model, loader)
    true_labels = ref_labels  # same dataset/order for both models

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare predicted classes of reference vs. unlearned model on full test set"
    )
    parser.add_argument(
        "--reference",
        default="model_0324_target_5_0.1_8.pth",
        help="Path to reference model checkpoint (retrained without forget class)",
    )
    parser.add_argument(
        "--unlearned",
        default="result/seed/model_MU_gradient_U8R1_seed5.pth",
        help="Path to unlearned model checkpoint",
    )
    args = parser.parse_args()
    run_prediction_agreement(args.reference, args.unlearned)