from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


PRESIDENT_LABEL_ORDER = (1, -1)
PAD_LABEL_INDEX = -100


@dataclass(frozen=True)
class SpeechSequence:
    speech_id: int
    sentence_ids: np.ndarray
    embeddings: np.ndarray
    labels: np.ndarray | None = None

    @property
    def length(self):
        return int(self.embeddings.shape[0])


class SpeechSequenceDataset(Dataset):
    def __init__(self, sequences, label_order=PRESIDENT_LABEL_ORDER):
        self.sequences = list(sequences)
        self.label_order = tuple(label_order)
        self.label_to_index = {
            label: idx for idx, label in enumerate(self.label_order)
        }

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        sequence = self.sequences[index]
        item = {
            "speech_id": sequence.speech_id,
            "sentence_ids": torch.from_numpy(sequence.sentence_ids.copy()).long(),
            "inputs": torch.from_numpy(sequence.embeddings.copy()).float(),
            "length": sequence.length,
        }

        if sequence.labels is not None:
            label_indices = np.array(
                [self.label_to_index[int(label)] for label in sequence.labels],
                dtype=np.int64,
            )
            item["labels"] = torch.from_numpy(label_indices)

        return item


def load_embeddings_with_metadata(embeddings_path, metadata_path, require_labels=True):
    embeddings = np.load(Path(embeddings_path)).astype(np.float32, copy=False)
    metadata = pd.read_parquet(Path(metadata_path)).reset_index(drop=True)

    required_columns = {"speech_id", "sentence_id"}
    if require_labels:
        required_columns.add("label")

    missing_columns = required_columns - set(metadata.columns)
    if missing_columns:
        raise ValueError(
            f"Missing required metadata columns: {sorted(missing_columns)}"
        )
    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D embeddings array, got {tuple(embeddings.shape)}")
    if len(embeddings) != len(metadata):
        raise ValueError(
            "Embeddings and metadata must have the same number of rows: "
            f"{len(embeddings)} != {len(metadata)}"
        )

    monotonic = metadata.groupby("speech_id", sort=False)["sentence_id"].apply(
        lambda sentence_ids: sentence_ids.is_monotonic_increasing
    )
    if not bool(monotonic.all()):
        raise ValueError("sentence_id must be monotonic within each speech_id")

    return embeddings, metadata


def build_speech_sequences(embeddings, metadata):
    sequences = []
    for speech_id, group in metadata.groupby("speech_id", sort=False):
        row_indices = group.index.to_numpy()
        labels = None
        if "label" in group.columns:
            labels = group["label"].to_numpy(dtype=np.int64, copy=True)

        sequences.append(
            SpeechSequence(
                speech_id=int(speech_id),
                sentence_ids=group["sentence_id"].to_numpy(dtype=np.int64, copy=True),
                embeddings=np.asarray(embeddings[row_indices], dtype=np.float32),
                labels=labels,
            )
        )

    return sequences


def collate_speech_sequences(batch):
    if not batch:
        raise ValueError("batch must not be empty")

    lengths = torch.tensor([int(item["length"]) for item in batch], dtype=torch.long)
    batch_size = len(batch)
    max_length = int(lengths.max().item())
    input_dim = int(batch[0]["inputs"].shape[-1])

    inputs = torch.zeros((batch_size, max_length, input_dim), dtype=torch.float32)
    sentence_ids = torch.full((batch_size, max_length), -1, dtype=torch.long)
    speech_ids = torch.tensor([int(item["speech_id"]) for item in batch], dtype=torch.long)

    labels = None
    if "labels" in batch[0]:
        labels = torch.full((batch_size, max_length), PAD_LABEL_INDEX, dtype=torch.long)

    for row_idx, item in enumerate(batch):
        length = int(item["length"])
        inputs[row_idx, :length] = item["inputs"]
        sentence_ids[row_idx, :length] = item["sentence_ids"]
        if labels is not None:
            labels[row_idx, :length] = item["labels"]

    collated = {
        "inputs": inputs,
        "lengths": lengths,
        "speech_ids": speech_ids,
        "sentence_ids": sentence_ids,
    }
    if labels is not None:
        collated["labels"] = labels

    return collated


def compute_class_weights(sequences, label_order=PRESIDENT_LABEL_ORDER):
    counts = np.zeros(len(label_order), dtype=np.float64)
    label_to_index = {label: idx for idx, label in enumerate(label_order)}

    for sequence in sequences:
        if sequence.labels is None:
            continue
        for label in sequence.labels:
            counts[label_to_index[int(label)]] += 1.0

    if np.any(counts == 0):
        raise ValueError(f"Every label must appear at least once, got counts={counts}")

    weights = counts.sum() / (len(counts) * counts)
    return torch.tensor(weights, dtype=torch.float32)
