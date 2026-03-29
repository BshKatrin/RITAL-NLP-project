from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import DataCollatorWithPadding

from rital_nlp_project.presidents.sequence import PRESIDENT_LABEL_ORDER


class TokenizedSentenceDataset(Dataset):
    def __init__(self, frame, encodings, label_order=PRESIDENT_LABEL_ORDER):
        self.frame = frame.reset_index(drop=True).copy()
        self.encodings = {
            key: list(values) for key, values in encodings.items()
        }
        self.label_to_index = {
            label: idx for idx, label in enumerate(label_order)
        }

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        item = {
            key: self.encodings[key][index]
            for key in self.encodings
        }
        item["speech_id"] = int(self.frame.at[index, "speech_id"])
        item["sentence_id"] = int(self.frame.at[index, "sentence_id"])
        item["row_index"] = int(self.frame.at[index, "row_index"])
        item["text"] = str(self.frame.at[index, "text"])

        if "label" in self.frame.columns:
            item["labels"] = self.label_to_index[int(self.frame.at[index, "label"])]

        return item


def load_metadata(metadata_path, require_labels=True):
    metadata = pd.read_parquet(Path(metadata_path)).reset_index(drop=True)

    required_columns = {"speech_id", "sentence_id", "text"}
    if require_labels:
        required_columns.add("label")

    missing_columns = required_columns - set(metadata.columns)
    if missing_columns:
        raise ValueError(
            f"Missing required metadata columns: {sorted(missing_columns)}"
        )

    monotonic = metadata.groupby("speech_id", sort=False)["sentence_id"].apply(
        lambda sentence_ids: sentence_ids.is_monotonic_increasing
    )
    if not bool(monotonic.all()):
        raise ValueError("sentence_id must be monotonic within each speech_id")

    return metadata


def build_contextual_text_frame(metadata, context_window, sep_token):
    if context_window < 0:
        raise ValueError("context_window must be non-negative")

    frame = metadata.copy().reset_index(drop=True)
    frame["row_index"] = np.arange(len(frame), dtype=np.int64)
    contextual_texts = [""] * len(frame)

    separator = f" {sep_token} " if sep_token else " "
    for _, group in frame.groupby("speech_id", sort=False):
        texts = group["text"].astype(str).tolist()
        row_indices = group["row_index"].tolist()

        for local_index, row_index in enumerate(row_indices):
            start = max(0, local_index - context_window)
            stop = min(len(texts), local_index + context_window + 1)
            contextual_texts[row_index] = separator.join(texts[start:stop])

    frame["context_text"] = contextual_texts

    columns = ["row_index", "speech_id", "sentence_id", "text", "context_text"]
    if "label" in frame.columns:
        columns.append("label")
    return frame[columns].copy()


def tokenize_contextual_texts(tokenizer, frame, max_length):
    return tokenizer(
        frame["context_text"].tolist(),
        truncation=True,
        max_length=max_length,
    )


def slice_encodings(encodings, mask):
    mask = np.asarray(mask, dtype=bool)
    indices = np.flatnonzero(mask)
    return {
        key: [values[idx] for idx in indices]
        for key, values in encodings.items()
    }


def make_collate_fn(tokenizer):
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

    def collate(batch):
        if not batch:
            raise ValueError("batch must not be empty")

        features = []
        speech_ids = []
        sentence_ids = []
        row_indices = []
        texts = []
        has_labels = "labels" in batch[0]

        for item in batch:
            feature = {
                key: item[key]
                for key in ("input_ids", "attention_mask", "token_type_ids")
                if key in item
            }
            if has_labels:
                feature["labels"] = item["labels"]
            features.append(feature)
            speech_ids.append(item["speech_id"])
            sentence_ids.append(item["sentence_id"])
            row_indices.append(item["row_index"])
            texts.append(item["text"])

        collated = data_collator(features)
        collated["speech_ids"] = torch.tensor(speech_ids, dtype=torch.long)
        collated["sentence_ids"] = torch.tensor(sentence_ids, dtype=torch.long)
        collated["row_indices"] = torch.tensor(row_indices, dtype=torch.long)
        collated["texts"] = texts
        return collated

    return collate
