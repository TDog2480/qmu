import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
import numpy as np
import random
from HQNN import ConvQNN
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from process import get_features

def MIA_train_process_pytorch():

    checkpoint = torch.load("model_0324_original_5_0.1_8.pth", weights_only=False)

    seed = checkpoint['args'].seed
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # === 1. Load original trained model & parameters ===
    model = ConvQNN(checkpoint['args'])
    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()

    # === 2. Load data (500 train + 500 test samples) ===
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    indices_train = checkpoint['indices'][0][0:500]
    indices_test = checkpoint['indices'][1]

    loader_train = DataLoader(Subset(trainset, indices_train), batch_size=1, shuffle=False)
    loader_test = DataLoader(Subset(testset, indices_test), batch_size=1, shuffle=False)

    # === 3. Generate MIA attack input data (model output probability + label) ===
    x_data_MIA = []
    y_data_MIA = []

    features_train, features_label4, y_data_MIA_label4 = get_features(loader_train, 1, model)  # Member samples
    features_test, _, _ = get_features(loader_test, 0, model)    # Non-member samples

    x_data_MIA = np.vstack([features_train, features_test])  # N x 10
    y_data_MIA = np.array([1] * len(features_train) + [0] * len(features_test))

    # label 4
    x_attack_tensor = torch.tensor(features_label4, dtype=torch.float32)
    y_attack_tensor = torch.tensor(y_data_MIA_label4, dtype=torch.float32).unsqueeze(1)


    # === 4. Build attack model (MLP with PyTorch) ===
    class AttackMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(13, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )

        def forward(self, x):
            return self.net(x)

    attack_model = AttackMLP()
    optimizer = torch.optim.Adam(attack_model.parameters(), lr=0.01)
    loss_fn = nn.BCELoss()

    # === 5. Split train/test set ===
    indices = np.random.permutation(len(x_data_MIA))
    x_data_MIA, y_data_MIA = x_data_MIA[indices], y_data_MIA[indices]

    x_train, y_train = x_data_MIA[:800], y_data_MIA[:800]
    x_test, y_test = x_data_MIA[800:], y_data_MIA[800:]


    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    # === 6. Train attack model ===
    for epoch in range(1000):
        attack_model.train()
        outputs = attack_model(x_train_tensor)
        loss = loss_fn(outputs, y_train_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Evaluate
        attack_model.eval()
        with torch.no_grad():
            y_pred = attack_model(x_attack_tensor)
            acc = accuracy_score(y_attack_tensor.numpy(), (y_pred.numpy() > 0.5).astype(int))

        if (epoch + 1) % 10 == 0:
            train_acc = accuracy_score(y_train_tensor.numpy(), (outputs.detach().numpy() > 0.5).astype(int))
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}, Train Acc: {train_acc*100:.2f}%, MIA Acc: {acc*100:.2f}%")

    print(f"Final MIA attack accuracy: {acc*100:.2f}%")

    attack_model_path = "attack_model_mia.pth"
    torch.save(attack_model.state_dict(), attack_model_path)

    attack_model = AttackMLP()
    attack_model.load_state_dict(torch.load("attack_model_mia.pth"))
    attack_model.eval()
    print("Loaded MIA attack model")

    # import matplotlib.pyplot as plt
    #
    # member_max_probs = [f[10] for f, y in zip(x_data_MIA, y_data_MIA) if y == 1]
    # nonmember_max_probs = [f[10] for f, y in zip(x_data_MIA, y_data_MIA) if y == 0]
    #
    # plt.hist(member_max_probs, bins=30, alpha=0.5, label='Member')
    # plt.hist(nonmember_max_probs, bins=30, alpha=0.5, label='Non-Member')
    # plt.legend()
    # plt.title("Max Probability Distribution")
    # plt.show()


def MIA_attack():

    checkpoint = torch.load("model_0324_original_5_0.1_8.pth", weights_only=False)

    model = ConvQNN(checkpoint['args'])

    seed = checkpoint['args'].seed
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # === 2. Load data (500 train + 500 test samples) ===
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    indices_train = checkpoint['indices'][0][0:500]
    indices_test = checkpoint['indices'][1]

    loader_train = DataLoader(Subset(trainset, indices_train), batch_size=1, shuffle=False)
    loader_test = DataLoader(Subset(testset, indices_test), batch_size=1, shuffle=False)


    checkpoint = torch.load(f"result/seed/model_MU_gradient_U8R1.pth", weights_only=False)
    # === 1. Load original trained model & parameters ===
    # model = ConvQNN(checkpoint['args'])
    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()

    # === 3. Generate MIA attack input data (model output probability + label) ===
    x_data_MIA = []
    y_data_MIA = []

    def get_features(loader, label):
        features = []
        features_label4 = []
        y_data_MIA_label4 = []
        for x, y in loader:
            with torch.no_grad():
                output = model(x)
                prob = F.softmax(output, dim=1).squeeze().numpy()

                max_prob = np.max(prob)
                pred_label = np.argmax(prob)
                correct = int(pred_label == y.item())

                loss = F.cross_entropy(output, y).item()

                feature_vector = list(prob) + [max_prob, correct, loss]
                # feature_vector = [loss]

                features.append(feature_vector)
                y_data_MIA.append(label)
                if y == 4:
                    features_label4.append(feature_vector)
                    y_data_MIA_label4.append(1)
        return features, features_label4, y_data_MIA_label4

    features_train, features_label4, y_data_MIA_label4 = get_features(loader_train, 1)  # Member samples

    x_attack_tensor = torch.tensor(features_label4, dtype=torch.float32)
    y_attack_tensor = torch.tensor(y_data_MIA_label4, dtype=torch.float32).unsqueeze(1)

    # === 4. Build attack model (MLP with PyTorch) ===
    class AttackMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(13, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )

        def forward(self, x):
            return self.net(x)


    attack_model = AttackMLP()
    attack_model.load_state_dict(torch.load("attack_model_mia.pth"))
    attack_model.eval()
    # print("Loaded MIA attack model")
    y_pred = attack_model(x_attack_tensor)
    acc = accuracy_score(y_attack_tensor.numpy(), (y_pred.detach().numpy() > 0.5).astype(int))
    print(f"MIA Acc: {acc * 100:.2f}%")


if __name__ == "__main__":
    MIA_train_process_pytorch()
    MIA_attack()
