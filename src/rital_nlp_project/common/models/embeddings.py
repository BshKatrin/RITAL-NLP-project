from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


def _require_transformers():
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise ModuleNotFoundError(
            "Transformer embeddings require torch and transformers. "
            "Install the embeddings extra before running `embed`."
        ) from exc
    return torch, AutoModel, AutoTokenizer


def detect_torch_device(preferred: str = "auto") -> str:
    if preferred != "auto":
        return preferred

    torch, _, _ = _require_transformers()
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class TransformerEmbeddingConfig:
    model_name: str
    pooling: str = "mean"
    batch_size: int = 16
    max_length: int = 256
    device: str = "auto"
    normalize: bool = False


class TransformerEmbeddingExtractor:
    def __init__(self, config: TransformerEmbeddingConfig):
        self.config = config
        self.device = detect_torch_device(config.device)
        torch, AutoModel, AutoTokenizer = _require_transformers()
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(config.model_name)
        self.model.to(self.device)
        self.model.eval()

    def _pool(self, outputs, attention_mask):
        if self.config.pooling == "cls":
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                return outputs.pooler_output
            return outputs.last_hidden_state[:, 0, :]

        if self.config.pooling != "mean":
            raise ValueError(f"Unsupported pooling mode: {self.config.pooling}")

        mask = attention_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
        masked = outputs.last_hidden_state * mask
        summed = masked.sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        vectors = []
        with self._torch.inference_mode():
            for start in range(0, len(texts), self.config.batch_size):
                batch = texts[start:start + self.config.batch_size]
                encoded = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_length,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                outputs = self.model(**encoded)
                pooled = self._pool(outputs, encoded["attention_mask"])
                batch_vectors = pooled.detach().cpu().numpy().astype(np.float32)
                vectors.append(batch_vectors)

        embeddings = np.vstack(vectors)
        if self.config.normalize and embeddings.size:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.clip(norms, a_min=1e-12, a_max=None)
            embeddings = embeddings / norms
        return embeddings

