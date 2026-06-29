import pickle
import os
import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def save_pkl(data, path):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def creat_file(path):
    os.makedirs(path, exist_ok=True)


def read_curr_time():
    now = datetime.datetime.now()
    time_save = now.strftime("%Y%m%d_%H%M%S")
    return now, time_save


def draw_sub_data_n(lists, name1, name2, save_path):
    """
    lists  : training history, e.g. [[loss_values], [acc_values]]
    name1  : metric / legend labels, first entry used as x-axis label ('iter')
    name2  : subplot titles, one per series in lists
    save_path : directory to save the figure
    """
    os.makedirs(save_path, exist_ok=True)
    n = len(name2)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4))
    if n == 1:
        axes = [axes]

    for i, (ax, title) in enumerate(zip(axes, name2)):
        if i < len(lists):
            values = lists[i]
            ax.plot(range(1, len(values) + 1), values, label=title)
        ax.set_title(title)
        ax.set_xlabel(name1[0] if name1 else 'iter')
        ax.set_ylabel(title)
        ax.legend()

    plt.tight_layout()
    save_file = os.path.join(save_path, 'training_curve.png')
    plt.savefig(save_file)
    plt.close(fig)
