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



"""
NOTE from changes 9/27:
ROC curves for the membership-inference attacks, evaluated on the ORIGINAL model.
 
Run from the project root (the folder with mia_common.py, HQNN.py and the checkpoints):
    python metric_attack.py
    python vqc_refactored.py     (also saves the unlearned npz for roc_unlearned.py)
    python qsvm_refactored.py    (also saves the unlearned npz for roc_unlearned.py)
    python MIA.py                (also saves the unlearned npz for roc_unlearned.py)
    python roc_utils.py
 
Each attack script saves its ROC data to roc_curve_output/*_roc.npz. This file loads
them and writes:
    roc_curve_output/original_all.png           metric + attack-model attacks
    roc_curve_output/original_attack_model.png  VQC + QSVM + classical MLP
    roc_curve_output/original_metric.png        loss, confidence, entropy, mod-entropy

The metric attacks have no trained parameters, so their curves use every pool row.
VQC, QSVM and the MLP are trained on the attack-train rows, so their curves use only the
held-out rows. Scoring rows the attack model was fit on would inflate their curves,
and that is what made the original combined plot misleading.
"""

import os
 
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
 
OUT_DIR = "roc_curve_output"
METRIC_KEYS = ["loss", "max_conf", "entropy", "mod_entropy"]
ATTACK_MODEL_KEYS = ["vqc", "qsvm", "mlp"]
ALL_KEYS = METRIC_KEYS + ATTACK_MODEL_KEYS
LABELS = {
    "loss": "Loss (Yeom 2018)",
    "max_conf": "Confidence",
    "entropy": "Entropy",
    "mod_entropy": "Mod-entropy (Song 2021)",
    "vqc": "VQC (held-out rows)",
    "qsvm": "QSVM (held-out rows)",
    "mlp": "Classical MLP (held-out rows)",
}
 
 
def compute_roc(scores, labels, higher_is_member=True):
    scores = np.asarray(scores, dtype=float)
    if not higher_is_member:
        scores = -scores
    fpr, tpr, thresholds = roc_curve(labels, scores)
    return {"fpr": fpr, "tpr": tpr, "thresholds": thresholds, "auc": roc_auc_score(labels, scores)}
 
 
def save_roc(roc_by_key, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez(path, **{f"{key}_{field}": value
                      for key, d in roc_by_key.items() for field, value in d.items()})
    print(f"[save] ROC data -> {path}")
 
 
def load_roc(path):
    data = np.load(path)
    keys = [name[:-len("_auc")] for name in data.files if name.endswith("_auc")]
    return {key: {"fpr": data[f"{key}_fpr"], "tpr": data[f"{key}_tpr"], "auc": float(data[f"{key}_auc"])}
            for key in keys}
 
 
def plot_roc(roc_by_key, out_path, title, eps=1e-3):
    fig, ax = plt.subplots(figsize=(7, 7))
    for key in [k for k in ALL_KEYS if k in roc_by_key]:
        d = roc_by_key[key]
        ax.plot(np.clip(d["fpr"], eps, 1), np.clip(d["tpr"], eps, 1),
                linestyle="-", linewidth=1.5,
                color=f"C{ALL_KEYS.index(key)}", label=f"{LABELS[key]} (AUC={d['auc']:.3f})")
    ax.plot([eps, 1], [eps, 1], ":", color="gray", linewidth=1, label="random guess")
    ax.set(xscale="log", yscale="log", xlim=(eps, 1), ylim=(eps, 1),
           xlabel="False Positive Rate", ylabel="True Positive Rate", title=title)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[save] {out_path}")
 
 
def plot_all(npz_names, model_name):
    """Write the combined, attack-model-only and metric-only plots for one model."""
    roc = {}
    for name in npz_names:
        roc.update(load_roc(os.path.join(OUT_DIR, name)))
 
    groups = {"all": ALL_KEYS, "attack_model": ATTACK_MODEL_KEYS, "metric": METRIC_KEYS}
    for group, keys in groups.items():
        plot_roc({k: roc[k] for k in keys if k in roc},
                 os.path.join(OUT_DIR, f"{model_name}_{group}.png"),
                 f"MIA ROC: {group.replace('_', '-')} attacks, {model_name} model")
 
 
if __name__ == "__main__":
    plot_all(["metric_attack_roc.npz", "vqc_attack_roc.npz", "qsvm_attack_roc.npz",
              "mlp_attack_roc.npz"], "original")
    plot_all(["metric_attack_roc_unlearned.npz", "vqc_attack_roc_unlearned.npz",
              "qsvm_attack_roc_unlearned.npz", "mlp_attack_roc_unlearned.npz"], "unlearned")