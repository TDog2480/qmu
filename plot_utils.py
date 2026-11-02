import os

import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Okabe-Ito (colorblind-safe). Retain/unlearn keep the same color across all figures.
COLOR_NEUTRAL = "#000000"
COLOR_TRAIN = "#E69F00"
COLOR_RETAIN = "#0072B2"
COLOR_UNLEARN = "#D55E00"

POSTER_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 20,
    "axes.labelsize": 22,
    "axes.titlesize": 22,
    "figure.titlesize": 24,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 18,
    "legend.frameon": False,
    "axes.linewidth": 1.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.major.width": 1.5,
    "ytick.major.width": 1.5,
    "xtick.major.size": 6,
    "ytick.major.size": 6,
    "lines.linewidth": 3,
    "lines.markersize": 9,
    "grid.alpha": 0.25,
    "grid.linewidth": 1,
    "pdf.fonttype": 42,  # embed editable TrueType text
    "ps.fonttype": 42,
}

def _style_axis(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, steps=[1, 2, 5, 10]))
    ax.grid(True)
    if len(ax.get_legend_handles_labels()[0]) > 1:
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2,
                  columnspacing=1.2, handlelength=1.8, borderaxespad=0.3)

def _save(fig, out_path, title):
    """Save out_path at 300 dpi, plus a vector .pdf beside it."""
    if title:
        fig.suptitle(title, fontweight="bold")
    fig.savefig(out_path, dpi=300)
    root, ext = os.path.splitext(out_path)
    if ext.lower() != ".pdf":
        fig.savefig(root + ".pdf")
    plt.close(fig)


def _markevery(n_points):
    return max(1, n_points // 10)

def plot_training_curves(lists, out_path, title=None):
    loss, train_acc, test_acc = lists[:3]
    epochs = range(1, len(loss) + 1)
    every = _markevery(len(loss))

    with plt.rc_context(POSTER_RC):
        fig, (ax_l, ax_a) = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
        ax_l.plot(epochs, loss, marker="o", markevery=every, color=COLOR_NEUTRAL)
        _style_axis(ax_l, "Epoch", "Train loss")

        ax_a.plot(epochs, train_acc, marker="o", markevery=every, color=COLOR_TRAIN, label="Train")
        ax_a.plot(epochs, test_acc, marker="s", markevery=every, color=COLOR_NEUTRAL, label="Test")
        if len(lists) >= 5:
            ax_a.plot(epochs, lists[3], marker="^", markevery=every, linestyle="--",
                      color=COLOR_RETAIN, label="Test (retain)")
            ax_a.plot(epochs, lists[4], marker="v", markevery=every, linestyle="--",
                      color=COLOR_UNLEARN, label="Test (unlearn)")
        _style_axis(ax_a, "Epoch", "Accuracy")
        _save(fig, out_path, title)

def plot_unlearning_curves(lists, out_path, title=None, acc_before=None):
    """lists = [combined_loss, acc_r, acc_u, (ce_u, ce_r)]; acc_before = (acc_r0, acc_u0)."""
    epochs = range(1, len(lists[0]) + 1)
    every = _markevery(len(lists[0]))

    with plt.rc_context(POSTER_RC):
        fig, (ax_l, ax_a) = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
        if len(lists) >= 5:
            ax_l.plot(epochs, lists[3], marker="o", markevery=every, color=COLOR_UNLEARN,
                      label="Unlearn (ascent)")
            ax_l.plot(epochs, lists[4], marker="s", markevery=every, color=COLOR_RETAIN,
                      label="Retain (descent)")
            ylabel = "Cross-entropy loss"
        else:
            ax_l.plot(epochs, lists[0], marker="o", markevery=every, color=COLOR_NEUTRAL)
            ylabel = "Combined loss"
        _style_axis(ax_l, "Epoch", ylabel)

        for vals, label, color, marker, i in ((lists[1], "Retain", COLOR_RETAIN, "s", 0),
                                              (lists[2], "Unlearn", COLOR_UNLEARN, "o", 1)):
            ax_a.plot(epochs, vals, marker=marker, markevery=every, color=color, label=label)
            if acc_before is not None:
                ax_a.axhline(acc_before[i], color=color, linestyle="--", linewidth=2.5,
                             alpha=0.7, label=f"{label}, before")
        ax_a.set_ylim(-0.02, 1.02)
        _style_axis(ax_a, "Epoch", "Test accuracy")
        _save(fig, out_path, title)