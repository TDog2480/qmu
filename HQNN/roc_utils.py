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

Typical usage -- combining attacks that were run as separate scripts/processes
(e.g. metric_attack.py, vqc_refactored.py, qsvm_refactored.py each saved their
own .npz) onto one overlaid plot:

    roc_data = {}
    roc_data.update(roc_utils.load_roc_data("metric_attack_roc.npz"))
    roc_data.update(roc_utils.load_roc_data("vqc_attack_roc.npz"))
    roc_data.update(roc_utils.load_roc_data("qsvm_attack_roc.npz"))
    roc_utils.plot_roc_overlay(roc_data, labels_by_key, "combined_attack_roc.png")

Or, once metric_attack.py, vqc_refactored.py, and qsvm_refactored.py have each
been run and dropped their default .npz files in the current directory, skip
all of the above and just run this file directly:

    python roc_utils.py --plot_all_curves
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

# Default .npz filenames each attack script saves via save_roc_data(), and
# what plot_all_curves()/--plot_all_curves look for by default.
METRIC_NPZ_DEFAULT = "metric_attack_roc.npz"
VQC_NPZ_DEFAULT = "vqc_attack_roc.npz"
QSVM_NPZ_DEFAULT = "qsvm_attack_roc.npz"

METRIC_LABELS = {
    "loss": "Loss attack (Yeom 2018)",
    "max_conf": "Confidence attack",
    "entropy": "Entropy attack",
    "mod_entropy": "Mod-entropy attack (Song 2021)",
}


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


def load_roc_data(npz_path, keys=None):
    """
    Load back one or more roc_data dicts previously written by
    save_roc_data() -- lets a separate script combine ROC curves from
    independent runs (e.g. metric_attack.py, vqc_refactored.py,
    qsvm_refactored.py) without re-running any of the attacks.

    npz_path: path to an .npz file produced by save_roc_data().
    keys: optional iterable of key prefixes to load; defaults to every key
        found in the file (inferred from its "*_auc" entries).

    Returns {key: {"fpr": ndarray, "tpr": ndarray, "thresholds": ndarray,
                   "auc": float}}, same shape compute_roc() returns.
    """
    with np.load(npz_path) as data:
        if keys is None:
            keys = sorted({name.rsplit("_auc", 1)[0] for name in data.files if name.endswith("_auc")})
        roc_data = {}
        for key in keys:
            roc_data[key] = {
                "fpr": data[f"{key}_fpr"],
                "tpr": data[f"{key}_tpr"],
                "thresholds": data[f"{key}_thresholds"],
                "auc": float(data[f"{key}_auc"]),
            }
    return roc_data


def plot_roc_overlay(roc_data_by_key, labels_by_key, out_path, title="ROC curves", eps=1e-3):
    """
    Overlay several attacks' ROC curves on a single log-log plot -- e.g. the
    four metric attacks plus the VQC and QSVM attacks, all on one figure.

    roc_data_by_key: {key: roc_data}, as returned by compute_roc() or
        load_roc_data(). Any number of entries; each gets its own color.
    labels_by_key: {key: human-readable label} for the legend (AUC is
        appended automatically).
    out_path: PNG path to write.
    title: plot title.

    Returns out_path.
    """
    fig, ax = plt.subplots(figsize=(7, 7))
    cmap = plt.get_cmap("tab10")
    for i, (key, roc_data) in enumerate(roc_data_by_key.items()):
        label = labels_by_key.get(key, key)
        fpr = np.clip(roc_data["fpr"], eps, 1.0)
        tpr = np.clip(roc_data["tpr"], eps, 1.0)
        ax.plot(fpr, tpr, linewidth=1.5, color=cmap(i % 10),
                label=f"{label} (AUC={roc_data['auc']:.3f})")

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
    print(f"[save] combined ROC overlay -> {out_path}")
    return out_path


def plot_all_curves(metric_npz=METRIC_NPZ_DEFAULT, vqc_npz=VQC_NPZ_DEFAULT,
                     qsvm_npz=QSVM_NPZ_DEFAULT, out_path="combined_attack_roc.png",
                     title="Metric + quantum attack ROC comparison"):
    """
    Load the metric attacks' ROC data (metric_attack.py), the VQC attack's
    (vqc_refactored.py), and the QSVM attack's (qsvm_refactored.py) from
    their saved .npz files and overlay all of them onto one combined plot.

    Requires all three .npz files to already exist -- run metric_attack.py,
    vqc_refactored.py, and qsvm_refactored.py first (each saves its own .npz
    by default). Raises FileNotFoundError naming whichever are missing,
    rather than silently plotting a subset.

    Returns out_path.
    """
    sources = {"metric": metric_npz, "vqc": vqc_npz, "qsvm": qsvm_npz}
    missing = {name: path for name, path in sources.items() if not os.path.exists(path)}
    if missing:
        details = ", ".join(f"{name} ({path})" for name, path in missing.items())
        raise FileNotFoundError(
            f"plot_all_curves() needs all three attacks' ROC .npz files; missing: {details}. "
            "Run metric_attack.py, vqc_refactored.py, and qsvm_refactored.py first -- each "
            "saves its own .npz by default -- or pass the correct path explicitly."
        )

    roc_data = {}
    labels = {}

    metric_roc = load_roc_data(metric_npz)
    roc_data.update(metric_roc)
    labels.update({key: METRIC_LABELS.get(key, key) for key in metric_roc})

    vqc_roc = load_roc_data(vqc_npz)
    roc_data.update(vqc_roc)
    labels.update({key: "VQC attack" for key in vqc_roc})

    qsvm_roc = load_roc_data(qsvm_npz)
    roc_data.update(qsvm_roc)
    labels.update({key: "QSVM attack" for key in qsvm_roc})

    return plot_roc_overlay(roc_data, labels, out_path, title=title)


def _main():
    parser = argparse.ArgumentParser(
        description="roc_utils command-line entry point -- combine and plot saved ROC curves"
    )
    parser.add_argument(
        "--plot_all_curves", action="store_true",
        help="Load metric_attack_roc.npz, vqc_attack_roc.npz, and qsvm_attack_roc.npz from "
             "the current directory (or --metric-npz/--vqc-npz/--qsvm-npz if given) and "
             "overlay all their ROC curves onto one combined plot. Errors out naming "
             "whichever .npz files can't be found.",
    )
    parser.add_argument("--metric-npz", default=METRIC_NPZ_DEFAULT,
                         help=f"npz saved by metric_attack.py (default: {METRIC_NPZ_DEFAULT})")
    parser.add_argument("--vqc-npz", default=VQC_NPZ_DEFAULT,
                         help=f"npz saved by vqc_refactored.py (default: {VQC_NPZ_DEFAULT})")
    parser.add_argument("--qsvm-npz", default=QSVM_NPZ_DEFAULT,
                         help=f"npz saved by qsvm_refactored.py (default: {QSVM_NPZ_DEFAULT})")
    parser.add_argument("--out", default="combined_attack_roc.png")
    parser.add_argument("--title", default="Metric + quantum attack ROC comparison")
    args = parser.parse_args()

    if args.plot_all_curves:
        plot_all_curves(args.metric_npz, args.vqc_npz, args.qsvm_npz, args.out, args.title)
    else:
        parser.print_help()


if __name__ == "__main__":
    _main()