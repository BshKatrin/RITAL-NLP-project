
import torch
import torch.nn as nn

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