import numpy as np
import pennylane as qml
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
from pennylane.templates.embeddings import AmplitudeEmbedding, AngleEmbedding
import random
import pickle

qubit_num = n_qubits = 8
# Define quantum device
dev = qml.device('default.qubit', wires=qubit_num)


# Calculate test set accuracy
def calculate_accuracy(model, testloader):
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)  # Get predicted label for each sample
            total += labels.size(0)  # Number of samples
            correct += (predicted == labels).sum().item()  # Count correct predictions

    accuracy = correct / total
    return accuracy


def extract_parameters(model):
    # Classical parameters (convolution and fully connected layers)
    classical_params = []
    for name, param in model.named_parameters():
        if 'qlayer' not in name:  # Check if the parameter is not related to the quantum layer
            classical_params.append((name, param))

    # Quantum parameters (parameters used in qnode)
    quantum_params = []
    for name, param in model.named_parameters():
        if 'qlayer' in name:  # Parameters related to the quantum layers
            quantum_params.append((name, param))

    return classical_params, quantum_params


# Define quantum neural network
@qml.qnode(dev,interface="torch")
def qnode(inputs,QNN_param):

    # E(x)
    AmplitudeEmbedding(inputs, wires=range(n_qubits), normalize=True)  # Amplitude encoding; hard to implement on real hardware, encodes classical data into quantum state

    # PQC layer U(theta)
    for k in range(QNN_param.shape[0]):
        for j in range(n_qubits):
            qml.U3(*QNN_param[k][j], wires=[j]) #

        for j in range(n_qubits - 1):
            qml.CNOT(wires=[j, j + 1])
        qml.CNOT(wires=[n_qubits - 1, 0])

    # Measurement layer
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]


class ConvQNN(nn.Module):
    def __init__(self,args):
        super(ConvQNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 5, kernel_size=5, stride=1, padding=2)  # Convolutional layer
        self.pool = nn.MaxPool2d(2, 2)  # Pooling layer
        self.fc1 = nn.Linear(980, 256)  # Fully connected layer 16:3136 5:980
        self.fc2 = nn.Linear(8, 10)  # Output layer
        self.softmax = nn.LogSoftmax(dim=1)
        weight_shapes = {"QNN_param": (args.n_layers, args.n_qubit, 3)}
        self.qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)

        with torch.no_grad():
            self.qlayer.qnode_weights["QNN_param"].uniform_(-np.pi, np.pi)

    def forward(self, x):
        # Pre-processing layer CNN
        x = self.pool(F.relu(self.conv1(x)))  # Convolution and pooling
        x = x.view(x.shape[0], -1)  # Flatten
        x = self.fc1(x)  # Fully connected layer, outputs 256: facilitates encoding into quantum state
        x = F.leaky_relu(x, negative_slope=0.01)  # Activation layer: does not change dimension

        # np.save(x)
        x = torch.stack([self.qlayer(xi / (torch.norm(xi) + 1e-8)) for xi in x])  # QNN target: deploy on real quantum hardware
        # Read real quantum hardware structure x
        x = self.fc2(x)  # Fully connected layer for 10-class output, post-processing layer
        return x

def process(args):
    # Load MNIST data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_indices = random.sample(range(len(trainset)), args.n_train)
    test_indices = random.sample(range(len(testset)), args.n_test)
    train_subset = Subset(trainset, train_indices)
    test_subset = Subset(testset, test_indices)
    trainloader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
    testloader = DataLoader(test_subset, batch_size=args.batch_size, shuffle=False)

    # Initialize model
    model = ConvQNN(args)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    # Train model

    lists = [[], [], []]

    for epoch in range(args.n_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Backward pass
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_acc = correct / total
        test_acc = calculate_accuracy(model, testloader)

        lists[0].append(running_loss / len(trainloader))
        lists[1].append(train_acc)
        lists[2].append(test_acc)

        print(
            f"Epoch {epoch + 1}, Loss: {running_loss / len(trainloader):.4f}, "
            f"Train Acc: {train_acc:.2f}, "
            f"Test Acc: {test_acc:.2f}")

    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "args": args,
        "lists": lists,
        "indices": [train_indices,test_indices]
    }, f"model{args.name}.pth")
    # checkpoint = torch.load("model.pth")
    # model.load_state_dict(checkpoint["model_state_dict"])


    name = f'{args.save_path}/trained_params_{args.name}.pkl'
    f = open(name, 'wb')
    pickle.dump([lists, args], file=f)
    f.close()



if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-seed', type=int, default=0, help="random seed")
    parser.add_argument('-save_path', type=str, default='result/', help="save path")
    parser.add_argument('-time_save', type=str, default='XX', help="save path")
    parser.add_argument('-name', type=str, default='XX', help="name")

    parser.add_argument('-n_qubit', type=int, default=8, help="n_qubit")
    parser.add_argument('-n_layers', type=int, default=5, help="n_layers")

    parser.add_argument('-n_epochs', type=int, default=10, help="n_epochs")
    parser.add_argument('-batch_size', type=int, default=8, help="batch_size")
    parser.add_argument('-learning_rate', type=float, default=0.1, help="learning_rate")

    parser.add_argument('-n_train', type=int, default=1000, help="n_train")
    parser.add_argument('-n_test', type=int, default=1000, help="n_test")

    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    import os
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    args.name = f'_0324_original_{args.n_layers}_{args.learning_rate}_{args.batch_size}'

    process(args)



