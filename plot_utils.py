import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt

def plot_training_curves(lists, out_path, title=None):
    loss, train_acc, test_acc = lists
    epochs = range(1, len(loss) + 1)

    fig, (ax_l, ax_a) = plt.subplots(1, 2, figsize=(10, 4))
    ax_l.plot(epochs, loss, marker="o")
    ax_l.set_xlabel("Epoch")
    ax_l.set_ylabel("Train loss")
    ax_l.grid(alpha=0.3)

    ax_a.plot(epochs, train_acc, marker="o", label="train")
    ax_a.plot(epochs, test_acc, marker="s", label="test")
    ax_a.set_xlabel("Epoch")
    ax_a.set_ylabel("Accuracy")
    ax_a.legend()
    ax_a.grid(alpha=0.3)

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)