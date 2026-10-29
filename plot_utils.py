import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt

def plot_training_curves(lists, out_path, title=None):
    loss, train_acc, test_acc = lists[:3]
    epochs = range(1, len(loss) + 1)

    fig, (ax_l, ax_a) = plt.subplots(1, 2, figsize=(10, 4))
    ax_l.plot(epochs, loss, marker="o")
    ax_l.set_xlabel("Epoch")
    ax_l.set_ylabel("Train loss")
    ax_l.grid(alpha=0.3)

    ax_a.plot(epochs, train_acc, marker="o", label="train")
    ax_a.plot(epochs, test_acc, marker="s", label="test")
    if len(lists) >= 5:
        ax_a.plot(epochs, lists[3], marker="^", linestyle="--", color="C2", label="test-R")
        ax_a.plot(epochs, lists[4], marker="v", linestyle="--", color="C3", label="test-U")
    ax_a.set_xlabel("Epoch")
    ax_a.set_ylabel("Accuracy")
    ax_a.legend()
    ax_a.grid(alpha=0.3)

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)

def plot_unlearning_curves(lists, out_path, title=None, acc_before=None):
    """lists = [combined_loss, acc_r, acc_u, (ce_u, ce_r)]; acc_before = (acc_r0, acc_u0)."""
    epochs = range(1, len(lists[0]) + 1)

    fig, (ax_l, ax_a) = plt.subplots(1, 2, figsize=(10, 4))
    if len(lists) >= 5:
        ax_l.plot(epochs, lists[3], marker="o",color="C1", label="CE on U (ascent)")
        ax_l.plot(epochs, lists[4], marker="s",color="C0", label="CE on R (descent)")

    else:
        ax_l.plot(epochs, lists[0], marker="o", label="combined")
    ax_l.set_xlabel("Epoch")
    ax_l.set_ylabel("Cross-entropy Loss")
    ax_l.legend()
    ax_l.grid(alpha=0.3)

    for vals, label, i in ((lists[1], "Test-R", 0), (lists[2], "Test-U", 1)):
        line, = ax_a.plot(epochs, vals, marker="o", label=label)
        if acc_before is not None:
            ax_a.axhline(acc_before[i], color=line.get_color(), linestyle="--", alpha=0.6,
                         label=f"{label} before")
    ax_a.set_xlabel("Epoch")
    ax_a.set_ylabel("Accuracy")
    ax_a.set_ylim(-0.02, 1.02)
    ax_a.legend(fontsize=8)
    ax_a.grid(alpha=0.3)

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)