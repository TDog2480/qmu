
"""
Shared ROC / AUC utilities for every membership-inference attack in this repo.
Every attack reduces to one real-valued score per example plus a 0/1 membership
label; nothing outside this module should call roc_curve / roc_auc_score directly.

Attack scripts compute and persist curve
    roc = roc_utils.compute_roc(scores, labels, higher_is_member)            # fpr, tpr, thresholds, auc
    auc = roc_utils.auc_score(member_vals, nonmember_vals, higher_is_member)
    res = roc_utils.evaluate(member_scores, nonmember_scores)                # + balanced_acc, tpr_at_fpr
    roc_utils.save_attack_roc("vqc_attack", {"vqc": res}, model="original") # or model="unlearned"

save_attack_roc writes OUT_DIR/<source>_roc.npz (original) or <source>_roc_unlearned.npz.
Plotting discovers every such file in OUT_DIR; no file list to maintain:

    python roc_utils.py [--all | --metric-only | --quantum-only | --model-based-only]
                        [--model original|unlearned|both]

    --metric-only       loss, max_conf, entropy, mod_entropy
    --quantum-only      vqc, qsvm
    --model-based-only  vqc, qsvm, mlp (trained attack models; held-out rows only)
    --all               every key above (default)

Output: OUT_DIR/<model>_<group>.pdf, plus an AUC table on stdout.
    
"""

import argparse
import os
from pathlib import Path
 
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

REPO_ROOT = Path(__file__).resolve().parent
OUT_DIR = str(REPO_ROOT / "roc_curve_output")  # anchored to the repo, not the cwd

METRIC_KEYS = ["loss", "max_conf", "entropy", "mod_entropy"]
QUANTUM_KEYS = ["vqc", "qsvm"]
ATTACK_MODEL_KEYS = ["vqc", "qsvm", "mlp"]
ALL_KEYS = METRIC_KEYS + ATTACK_MODEL_KEYS
GROUPS = {
    "metric": METRIC_KEYS,
    "quantum": QUANTUM_KEYS,
    "model_based": ATTACK_MODEL_KEYS,
    "all": ALL_KEYS,
}
MODEL_SUFFIX = {"original": "_roc.npz", "unlearned": "_roc_unlearned.npz"}
ROC_FIELDS = ("fpr", "tpr", "thresholds", "auc") # array/scalar fields persisted to .npz
LABELS = {
    "loss": "Loss (Yeom 2018)",
    "max_conf": "Confidence",
    "entropy": "Entropy",
    "mod_entropy": "Mod-entropy (Song 2021)",
    "vqc": "VQC (held-out rows)",
    "qsvm": "QSVM (held-out rows)",
    "mlp": "Classical MLP (held-out rows)",
    "shadow": "Shadow model (Shokri 2017)",
}
 
 
def compute_roc(scores, labels, higher_is_member=True):
    scores = np.asarray(scores, dtype=float)
    if not higher_is_member:
        scores = -scores
    fpr, tpr, thresholds = roc_curve(labels, scores)
    return {"fpr": fpr, "tpr": tpr, "thresholds": thresholds, "auc": roc_auc_score(labels, scores)}

def auc_score(member_vals, nonmember_vals, higher_is_member=True):
    """Threshold-free separability of members vs non-members on one scalar."""
    vals = np.concatenate([np.asarray(member_vals, dtype=float), np.asarray(nonmember_vals, dtype=float)])
    labels = np.r_[np.ones(len(member_vals)), np.zeros(len(nonmember_vals))]
    return float(roc_auc_score(labels, vals if higher_is_member else -vals))


def tpr_at_fpr(roc, max_fpr):
    """Best TPR with FPR <= max_fpr (roc_curve always includes fpr=0)."""
    return float(roc["tpr"][roc["fpr"] <= max_fpr].max())


def balanced_acc(roc):
    """Best (TPR + TNR) / 2 over all thresholds."""
    return float(np.max((roc["tpr"] + 1.0 - roc["fpr"]) / 2.0))


def evaluate(member_scores, nonmember_scores, higher_is_member=True, fprs=(0.01, 0.001)):
    """compute_roc on member/non-member scores, plus balanced_acc and tpr_at_fpr[f] for f in fprs."""
    scores = np.concatenate([np.asarray(member_scores, dtype=float), np.asarray(nonmember_scores, dtype=float)])
    labels = np.r_[np.ones(len(member_scores)), np.zeros(len(nonmember_scores))]
    roc = compute_roc(scores, labels, higher_is_member)
    roc["balanced_acc"] = balanced_acc(roc)
    roc["tpr_at_fpr"] = {f: tpr_at_fpr(roc, f) for f in fprs}
    return roc 
 
def save_roc(roc_by_key, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez(path, **{f"{key}_{field}": value
                      for key, d in roc_by_key.items() for field, value in d.items()
                      if field in ROC_FIELDS})
    print(f"[save] ROC data -> {path}")

def roc_path(source, model="original"):
    """OUT_DIR/<source>_roc.npz or OUT_DIR/<source>_roc_unlearned.npz."""
    return os.path.join(OUT_DIR, f"{source}{MODEL_SUFFIX[model]}")


def save_attack_roc(source, roc_by_key, model="original"):
    """Entry point for attack scripts: persist curves where the plotting CLI discovers them."""
    unknown = [k for k in roc_by_key if k not in LABELS]
    if unknown:
        print(f"[warn] {source}: keys {unknown} have no LABELS entry and belong to no plot group")
    save_roc(roc_by_key, roc_path(source, model)) 
 
def load_roc(path):
    data = np.load(path)
    keys = [name[:-len("_auc")] for name in data.files if name.endswith("_auc")]
    return {key: {"fpr": data[f"{key}_fpr"], "tpr": data[f"{key}_tpr"], "auc": float(data[f"{key}_auc"])}
            for key in keys}

def load_model_rocs(model):
    """Merge every OUT_DIR/*<suffix> file for this model into one {key: roc} dict."""
    roc = {}
    if not os.path.isdir(OUT_DIR):
        return roc
    for name in sorted(os.listdir(OUT_DIR)):
        if not name.endswith(MODEL_SUFFIX[model]):
            continue
        for key, d in load_roc(os.path.join(OUT_DIR, name)).items():
            if key in roc:
                print(f"[warn] key '{key}' found in multiple {model} files; using {name}")
            roc[key] = d
    return roc 
 
def plot_roc(roc_by_key, out_path, title=None, labels=None, eps=1e-3):
    labels = {**LABELS, **(labels or {})}
    order = ALL_KEYS + [k for k in roc_by_key if k not in ALL_KEYS]
    fig, ax = plt.subplots(figsize=(7, 7))
    for key in [k for k in order if k in roc_by_key]:
        d = roc_by_key[key]
        ax.plot(np.clip(d["fpr"], eps, 1), np.clip(d["tpr"], eps, 1),
                linestyle="-", linewidth=2.5,
                color=f"C{order.index(key)}", label=f"{labels.get(key, key)} (AUC={d['auc']:.3f})")
    ax.plot([eps, 1], [eps, 1], ":", color="gray", linewidth=2, label="random guess")
    ax.set(xscale="log", yscale="log", xlim=(eps, 1), ylim=(eps, 1))
    if title:
        ax.set_title(title, fontsize=14)
    ax.set_xlabel("False Positive Rate", fontsize=16)
    ax.set_ylabel("True Positive Rate", fontsize=16)
    ax.tick_params(labelsize=12)
    ax.legend(loc="upper left", fontsize=12)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[save] {out_path}")
 
 
def plot_group(roc, group, model):
    """Plot one group's keys for one model to OUT_DIR/<model>_<group>.pdf."""
    keys = [k for k in GROUPS[group] if k in roc]
    missing = [k for k in GROUPS[group] if k not in roc]
    if not keys:
        print(f"\n[skip] {model}/{group}: no saved curves in {OUT_DIR}")
        return
    if missing:
        print(f"\n[note] {model}/{group}: no saved curves for {missing}")

    width = max([len("Attack")] + [len(LABELS.get(k, k)) for k in keys])
    print(f"\nAUC -- {group} attacks, {model} model")
    print(f"{'Attack':<{width}}  {'AUC':>6}")
    print("-" * (width + 8))
    for key in keys:
        print(f"{LABELS.get(key, key):<{width}}  {roc[key]['auc']:>6.3f}")


def main():
    parser = argparse.ArgumentParser(description="Plot MIA ROC curves saved in OUT_DIR by the attack scripts")
    sel = parser.add_mutually_exclusive_group()
    sel.add_argument("--all", action="store_true", help="every attack (default)")
    sel.add_argument("--metric-only", action="store_true", help="loss, confidence, entropy, mod-entropy")
    sel.add_argument("--quantum-only", action="store_true", help="VQC, QSVM")
    sel.add_argument("--model-based-only", action="store_true", help="VQC, QSVM, classical MLP")
    parser.add_argument("--model", choices=["original", "unlearned", "both"], default="both")
    args = parser.parse_args()

    group = ("metric" if args.metric_only else
             "quantum" if args.quantum_only else
             "model_based" if args.model_based_only else
             "all")
    models = ["original", "unlearned"] if args.model == "both" else [args.model]
    for model in models:
        plot_group(load_model_rocs(model), group, model)
 
if __name__ == "__main__":
    main()