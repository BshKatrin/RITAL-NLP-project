
import torch.nn as nn

class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_dim * 2, 2)  # *2 because bidirectional

    def forward(self, x):
        x = self.embedding(x)
        _, (h, _) = self.lstm(x)

        # h shape: (2, batch, hidden_dim)
        h_forward = h[0]
        h_backward = h[1]

        h_concat = torch.cat((h_forward, h_backward), dim=1)
        return self.fc(h_concat)