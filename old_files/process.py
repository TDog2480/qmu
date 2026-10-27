import torch
import torch.nn.functional as F
import numpy as np


def get_features(loader, label, model):
    """
    Extract MIA feature vectors from a DataLoader using the given model.

    Returns:
        features           : list of 13-dim feature vectors for all samples
        features_label4    : subset of feature vectors for samples with ground-truth label 4
        y_data_MIA_label4  : membership labels (all 1) for the label-4 subset
    """
    features = []
    features_label4 = []
    y_data_MIA_label4 = []

    model.eval()
    for x, y in loader:
        with torch.no_grad():
            output = model(x)
            prob = F.softmax(output, dim=1).squeeze().numpy()

            max_prob = np.max(prob)
            pred_label = np.argmax(prob)
            correct = int(pred_label == y.item())

            loss = F.cross_entropy(output, y).item()

            # 13 features: 10 class probs + max_prob + correct + cross_entropy_loss
            feature_vector = list(prob) + [max_prob, correct, loss]

            features.append(feature_vector)
            if y.item() == 4:
                features_label4.append(feature_vector)
                y_data_MIA_label4.append(label)

    return features, features_label4, y_data_MIA_label4
