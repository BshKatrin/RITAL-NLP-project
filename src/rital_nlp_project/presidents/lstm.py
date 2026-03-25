from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BiLSTMSequenceTagger(nn.Module):
    """Speech-level BiLSTM that predicts one label per sentence."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        *,
        projection_dim: int | None = None,
        num_layers: int = 1,
        dropout: float = 0.2,
        num_labels: int = 2,
    ):
        super().__init__()

        if input_dim <= 0:
            raise ValueError("input_dim must be strictly positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be strictly positive")
        if num_layers <= 0:
            raise ValueError("num_layers must be strictly positive")
        if num_labels <= 1:
            raise ValueError("num_labels must be greater than 1")

        projection_dim = projection_dim or input_dim

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.projection_dim = projection_dim
        self.num_layers = num_layers
        self.num_labels = num_labels

        self.input_projection = (
            nn.Identity()
            if projection_dim == input_dim
            else nn.Linear(input_dim, projection_dim)
        )
        self.input_dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            input_size=projection_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Linear(hidden_dim * 2, num_labels)

    def forward(self, inputs: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3:
            raise ValueError(
                f"inputs must be [batch, time, dim], got shape {tuple(inputs.shape)}"
            )
        if lengths.ndim != 1:
            raise ValueError(
                f"lengths must be a 1D tensor, got shape {tuple(lengths.shape)}"
            )

        projected = self.input_projection(inputs)
        projected = self.input_dropout(projected)

        packed = pack_padded_sequence(
            projected,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_outputs, _ = self.lstm(packed)
        outputs, _ = pad_packed_sequence(
            packed_outputs,
            batch_first=True,
            total_length=inputs.size(1),
        )
        return self.classifier(outputs)


class BiLSTMClassifier(nn.Module):
    """Legacy whole-sequence classifier kept for notebook compatibility."""

    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = nn.Linear(hidden_dim * 2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        _, (h, _) = self.lstm(x)
        h_forward = h[0]
        h_backward = h[1]
        h_concat = torch.cat((h_forward, h_backward), dim=1)
        return self.fc(h_concat)
