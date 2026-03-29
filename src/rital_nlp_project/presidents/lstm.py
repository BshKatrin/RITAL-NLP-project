
import torch
import torch.nn as nn
import numpy as np


class BiLSTMClassifier(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=1, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(-1)

        outputs, _ = self.lstm(x)
        center_idx = outputs.size(1) // 2
        center_hidden = self.dropout(outputs[:, center_idx, :])
        logits = self.fc(center_hidden).squeeze(-1)
        return logits

    def predict_proba(self, x):
        return torch.sigmoid(self.forward(x))


def build_probability_windows(probabilities, k):
    n = len(probabilities)
    offsets = np.arange(-k, k + 1)
    windows = np.empty((n, 2 * k + 1), dtype=np.float32)

    for i in range(n):
        indices = np.clip(i + offsets, 0, n - 1)
        windows[i] = probabilities[indices]

    return windows


def weighted_bce_with_logits(logits, labels, class_weights):
    weights = torch.where(labels > 0.5, class_weights[1], class_weights[0])
    return nn.functional.binary_cross_entropy_with_logits(logits, labels, weight=weights)


def calc_weights(train_dataset, num_labels):
    label_counts = np.bincount(train_dataset["label"], minlength=num_labels)
    safe_counts = np.where(label_counts == 0, 1, label_counts)
    class_weights = torch.tensor(
        len(train_dataset) / (len(safe_counts) * safe_counts),
        dtype=torch.float,
    )
    return class_weights
