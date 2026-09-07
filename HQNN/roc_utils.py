"""
Shared ROC-curve (TPR vs. FPR) computation and plotting for the membership-
inference attacks in this repo: metric_attack.py's four threshold attacks,
vqc_refactored.py, qsvm_refactored.py, and anything else that reduces to
"one real-valued score per example + a 0/1 membership label."

That (scores, labels) pair is all every attack here ultimately produces --
this module doesn't know or care whether the score came from a loss value,
an SVM decision_function, or a VQC's sigmoid output.

Typical usage -- single attack, single plot:

    import roc_utils
    roc_data = roc_utils.compute_roc(scores, labels, higher_is_member=True)
    roc_utils.plot_roc(roc_data, "my_attack_roc.png",
                        title="My attack ROC", label="My attack")

Typical usage -- several attacks, one plot per attack (metric_attack.py style):

    roc_data_by_key = {key: roc_utils.compute_roc(scores, labels, higher)
                        for key, (scores, labels, higher) in per_attack.items()}
    roc_utils.plot_roc_curves(roc_data_by_key, labels_by_key, out_prefix="metric_attack_roc")
    roc_utils.save_roc_data(roc_data_by_key, "metric_attack_roc.npz")
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def compute_roc(scores, labels, higher_is_member=True):
    """
    Full ROC curve + AUC for one attack.

    scores: (N,) array of real-valued attack scores, one per example.
    labels: (N,) array of 0/1 ground-truth membership labels, aligned with scores.
    higher_is_member: True if a higher score means "more member-like"
        (confidence, SVM decision_function, VQC membership probability, ...).
        False for "lower is member" attacks (e.g. loss).

    Returns {"fpr": ndarray, "tpr": ndarray, "thresholds": ndarray, "auc": float}.
    The AUC here is computed from the exact same (score, label) pair the curve
    is drawn from, so it always matches whatever roc_auc_score would report
    elsewhere on the same inputs.
    """
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    score = scores if higher_is_member else -scores
    fpr, tpr, thresholds = roc_curve(labels, score)
    auc = roc_auc_score(labels, score)
    return {"fpr": fpr, "tpr": tpr, "thresholds": thresholds, "auc": auc}


def plot_roc(roc_data, out_path, title, label, eps=1e-3):
    """
    Save a single log-log ROC plot for one attack.

    roc_data: dict as returned by compute_roc().
    out_path: PNG path to write.
    title: plot title.
    label: legend label for the curve (its AUC is appended automatically).

    Returns out_path.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    fpr = np.clip(roc_data["fpr"], eps, 1.0)
    tpr = np.clip(roc_data["tpr"], eps, 1.0)
    ax.plot(fpr, tpr, linewidth=1.5, color="C0", label=f"{label} (AUC={roc_data['auc']:.3f})")
    ax.plot([eps, 1], [eps, 1], linestyle="--", color="gray", linewidth=1, label="random guess")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(eps, 1)
    ax.set_ylim(eps, 1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[save] ROC curve plot -> {out_path}")
    return out_path


def plot_roc_curves(roc_data_by_key, labels_by_key, out_prefix, eps=1e-3):
    """
    Save one separate ROC plot per attack key (used when several attacks are
    evaluated together, e.g. metric_attack.py's loss/confidence/entropy/
    mod-entropy attacks).

    roc_data_by_key: {key: roc_data} as returned by compute_roc().
    labels_by_key: {key: human-readable label} for titles/legends.
    out_prefix: path prefix; each file is written to f"{out_prefix}_{key}.png".

    Returns the list of file paths written.
    """
    written = []
    for key, roc_data in roc_data_by_key.items():
        label = labels_by_key.get(key, key)
        out_path = f"{out_prefix}_{key}.png"
        plot_roc(roc_data, out_path, title=f"ROC -- {label}", label=label, eps=eps)
        written.append(out_path)
    return written


def save_roc_data(roc_data_by_key, out_path):
    """
    Persist raw fpr/tpr/thresholds/auc arrays (npz) for one or more attacks,
    for later re-plotting without rerunning the attack itself.

    roc_data_by_key: {key: roc_data} as returned by compute_roc(). For a
        single attack, pass a one-entry dict, e.g. {"vqc": roc_data}.
    """
    payload = {}
    for key, d in roc_data_by_key.items():
        payload[f"{key}_fpr"] = d["fpr"]
        payload[f"{key}_tpr"] = d["tpr"]
        payload[f"{key}_thresholds"] = d["thresholds"]
        payload[f"{key}_auc"] = np.array(d["auc"])
    np.savez(out_path, **payload)
    print(f"[save] raw ROC (fpr/tpr/thresholds) arrays -> {out_path}")