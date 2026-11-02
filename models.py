import torch
import torch.nn as nn
import torch.nn.functional as F


MODEL_TYPES = ("hqnn", "mlp")


def build_model(args):
    """Instantiate the model named by args.model_type ('hqnn' or 'mlp').

    Checkpoints saved before model_type existed have no such attribute and are HQNNs,
    so the default keeps them loadable.
    """
    model_type = getattr(args, "model_type", "hqnn")
    if model_type == "hqnn":
        from HQNN import ConvQNN  # imported here because HQNN.py itself imports this module
        return ConvQNN(args)
    if model_type == "mlp":
        return ConvMLP(args)
    raise ValueError(f"unknown model_type {model_type!r}, expected one of {MODEL_TYPES}")


class ConvMLP(nn.Module):
    """Classical baseline for ConvQNN: the quantum layer is replaced by Linear(256, 8) + tanh.

    Everything else (conv, pooling, fc1, normalization, fc2) is identical to ConvQNN in HQNN.py.
    The tanh bounds the block's output to [-1, 1], matching the range of the Pauli-Z expectation values.
    """

    def __init__(self, args=None):
        super(ConvMLP, self).__init__()
        self.conv1 = nn.Conv2d(1, 5, kernel_size=5, stride=1, padding=2)  # Convolutional layer
        self.pool = nn.MaxPool2d(2, 2)  # Pooling layer
        self.fc1 = nn.Linear(980, 256)  # Fully connected layer
        self.fc2 = nn.Linear(8, 10)  # Output layer
        self.softmax = nn.LogSoftmax(dim=1)
        # Takes the place of the quantum layer; named qlayer so parameter-splitting code that looks for 'qlayer' still works
        self.qlayer = nn.Linear(256, 8)

    def forward(self, x):
        # Pre-processing layer CNN
        x = self.pool(F.relu(self.conv1(x)))  # Convolution and pooling
        x = x.view(x.shape[0], -1)  # Flatten
        x = self.fc1(x)  # Fully connected layer, outputs 256
        x = F.leaky_relu(x, negative_slope=0.01)  # Activation layer: does not change dimension

        x = x / (torch.norm(x, dim=1, keepdim=True) + 1e-8)  # L2 normalization, same as the per-sample input to the qnode
        x = torch.tanh(self.qlayer(x))  # Replaces the QNN; output in [-1, 1] like the expectation values
        x = self.fc2(x)  # Fully connected layer for 10-class output, post-processing layer
        return x
