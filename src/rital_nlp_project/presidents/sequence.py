from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import gaussian_filter1d
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
    def length(self) -> int:
        return int(self.embeddings.shape[0])


@dataclass(frozen=True)
class SingleSpanPrior:
    no_span_log_prob: float
    span_log_prob: float
    span_length_log_probs: np.ndarray
    start_bin_log_probs: np.ndarray | None = None
    end_bin_log_probs: np.ndarray | None = None

    def get_span_length_log_prob(self, span_length: int) -> float:
        if span_length <= 0:
            raise ValueError("span_length must be strictly positive")

        capped = min(span_length, len(self.span_length_log_probs) - 1)
        return float(self.span_length_log_probs[capped])

    def get_position_log_prob(
        self,
        start: int,
        end: int,
        *,
        sequence_length: int,
    ) -> float:
        if self.start_bin_log_probs is None or self.end_bin_log_probs is None:
            return 0.0
        if sequence_length <= 0:
            raise ValueError("sequence_length must be strictly positive")

        num_bins = len(self.start_bin_log_probs)
        start_bin = _fraction_to_bin(start / sequence_length, num_bins)
        end_bin = _fraction_to_bin(end / sequence_length, num_bins)
        return float(self.start_bin_log_probs[start_bin] + self.end_bin_log_probs[end_bin])


class SpeechSequenceDataset(Dataset):
    def __init__(
        self,
        sequences: Sequence[SpeechSequence],
        *,
        label_order: Sequence[int] = PRESIDENT_LABEL_ORDER,
    ):
        self.sequences = list(sequences)
        self.label_order = tuple(label_order)
        self.label_to_index = {
            label: idx for idx, label in enumerate(self.label_order)
        }

        for sequence in self.sequences:
            if sequence.labels is None:
                continue
            unknown_labels = set(np.unique(sequence.labels)) - set(self.label_order)
            if unknown_labels:
                raise ValueError(
                    f"Unknown labels {sorted(unknown_labels)} for speech {sequence.speech_id}"
                )

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int]:
        sequence = self.sequences[index]
        item: dict[str, torch.Tensor | int] = {
            "speech_id": sequence.speech_id,
            "sentence_ids": torch.from_numpy(sequence.sentence_ids.copy()).long(),
            "inputs": torch.from_numpy(sequence.embeddings.copy()).float(),
            "length": sequence.length,
        }

        if sequence.labels is not None:
            label_indices = np.array(
                [self.label_to_index[label] for label in sequence.labels],
                dtype=np.int64,
            )
            item["labels"] = torch.from_numpy(label_indices)

        return item


def _fraction_to_bin(value: float, num_bins: int) -> int:
    if num_bins <= 0:
        raise ValueError("num_bins must be strictly positive")

    clipped = min(max(float(value), 0.0), 1.0)
    return min(int(clipped * num_bins), num_bins - 1)


def build_position_features(length: int) -> np.ndarray:
    if length <= 0:
        raise ValueError("length must be strictly positive")

    denominator = max(length - 1, 1)
    positions = np.arange(length, dtype=np.float32) / denominator
    reverse_positions = positions[::-1].copy()
    centered_positions = positions * 2.0 - 1.0
    inverse_length = np.full(length, 1.0 / length, dtype=np.float32)

    return np.stack(
        [
            positions,
            reverse_positions,
            positions**2,
            reverse_positions**2,
            centered_positions,
            inverse_length,
        ],
        axis=1,
    )


def build_transition_features(embeddings: np.ndarray) -> np.ndarray:
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim != 2:
        raise ValueError(
            f"embeddings must be a 2D array, got shape {tuple(embeddings.shape)}"
        )
    length = embeddings.shape[0]
    if length <= 0:
        raise ValueError("embeddings must contain at least one row")

    previous = np.concatenate([embeddings[:1], embeddings[:-1]], axis=0)
    following = np.concatenate([embeddings[1:], embeddings[-1:]], axis=0)

    current_norms = np.linalg.norm(embeddings, axis=1)
    previous_norms = np.linalg.norm(previous, axis=1)
    following_norms = np.linalg.norm(following, axis=1)
    cosine_denom_prev = np.maximum(current_norms * previous_norms, 1e-6)
    cosine_denom_next = np.maximum(current_norms * following_norms, 1e-6)

    cosine_prev = np.sum(embeddings * previous, axis=1) / cosine_denom_prev
    cosine_next = np.sum(embeddings * following, axis=1) / cosine_denom_next
    l2_prev = np.linalg.norm(embeddings - previous, axis=1)
    l2_next = np.linalg.norm(following - embeddings, axis=1)

    return np.stack(
        [
            cosine_prev.astype(np.float32),
            cosine_next.astype(np.float32),
            l2_prev.astype(np.float32),
            l2_next.astype(np.float32),
        ],
        axis=1,
    )


def augment_sequences_with_position_features(
    sequences: Sequence[SpeechSequence],
) -> list[SpeechSequence]:
    augmented: list[SpeechSequence] = []

    for sequence in sequences:
        position_features = build_position_features(sequence.length)
        augmented.append(
            SpeechSequence(
                speech_id=sequence.speech_id,
                sentence_ids=sequence.sentence_ids.copy(),
                embeddings=np.concatenate(
                    [
                        np.asarray(sequence.embeddings, dtype=np.float32),
                        position_features,
                    ],
                    axis=1,
                ),
                labels=None if sequence.labels is None else sequence.labels.copy(),
            )
        )

    return augmented


def augment_sequences_with_transition_features(
    sequences: Sequence[SpeechSequence],
) -> list[SpeechSequence]:
    augmented: list[SpeechSequence] = []

    for sequence in sequences:
        transition_features = build_transition_features(sequence.embeddings)
        augmented.append(
            SpeechSequence(
                speech_id=sequence.speech_id,
                sentence_ids=sequence.sentence_ids.copy(),
                embeddings=np.concatenate(
                    [
                        np.asarray(sequence.embeddings, dtype=np.float32),
                        transition_features,
                    ],
                    axis=1,
                ),
                labels=None if sequence.labels is None else sequence.labels.copy(),
            )
        )

    return augmented


def load_embeddings_with_metadata(
    embeddings_path: str | Path,
    metadata_path: str | Path,
    *,
    require_labels: bool = True,
) -> tuple[np.ndarray, pd.DataFrame]:
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
        raise ValueError(
            f"Expected 2D embeddings array, got shape {tuple(embeddings.shape)}"
        )
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


def build_speech_sequences(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
) -> list[SpeechSequence]:
    sequences: list[SpeechSequence] = []
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


def speech_train_test_split(
    sequences: Sequence[SpeechSequence],
    *,
    test_size: float = 0.2,
) -> tuple[list[SpeechSequence], list[SpeechSequence]]:
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1")
    if len(sequences) < 2:
        raise ValueError("Need at least two speeches to build a split")

    lengths = np.array([sequence.length for sequence in sequences], dtype=np.int64)
    cumulative_lengths = np.cumsum(lengths[:-1])
    target_train_rows = int(round(lengths.sum() * (1.0 - test_size)))
    split_index = int(
        np.argmin(np.abs(cumulative_lengths - target_train_rows))
    ) + 1
    return list(sequences[:split_index]), list(sequences[split_index:])


def collate_speech_sequences(
    batch: Sequence[dict[str, torch.Tensor | int]]
) -> dict[str, torch.Tensor]:
    if not batch:
        raise ValueError("batch must not be empty")

    lengths = torch.tensor([int(item["length"]) for item in batch], dtype=torch.long)
    batch_size = len(batch)
    max_length = int(lengths.max().item())
    input_dim = int(batch[0]["inputs"].shape[-1])  # type: ignore[index]

    inputs = torch.zeros((batch_size, max_length, input_dim), dtype=torch.float32)
    attention_mask = torch.zeros((batch_size, max_length), dtype=torch.bool)
    sentence_ids = torch.full((batch_size, max_length), -1, dtype=torch.long)
    speech_ids = torch.tensor(
        [int(item["speech_id"]) for item in batch],
        dtype=torch.long,
    )

    has_labels = "labels" in batch[0]
    labels = None
    if has_labels:
        labels = torch.full(
            (batch_size, max_length),
            PAD_LABEL_INDEX,
            dtype=torch.long,
        )

    for row_idx, item in enumerate(batch):
        seq_length = int(item["length"])
        inputs[row_idx, :seq_length] = item["inputs"]  # type: ignore[index]
        attention_mask[row_idx, :seq_length] = True
        sentence_ids[row_idx, :seq_length] = item["sentence_ids"]  # type: ignore[index]
        if labels is not None:
            labels[row_idx, :seq_length] = item["labels"]  # type: ignore[index]

    collated = {
        "inputs": inputs,
        "attention_mask": attention_mask,
        "lengths": lengths,
        "speech_ids": speech_ids,
        "sentence_ids": sentence_ids,
    }
    if labels is not None:
        collated["labels"] = labels

    return collated


def compute_class_weights(
    sequences: Sequence[SpeechSequence],
    *,
    label_order: Sequence[int] = PRESIDENT_LABEL_ORDER,
) -> torch.Tensor:
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


def fit_single_span_prior(
    sequences: Sequence[SpeechSequence],
    *,
    negative_label: int = -1,
    alpha: float = 1.0,
    position_bins: int = 0,
) -> SingleSpanPrior:
    negative_span_lengths: list[int] = []
    start_bin_counts = (
        np.zeros(position_bins, dtype=np.float64) if position_bins > 0 else None
    )
    end_bin_counts = (
        np.zeros(position_bins, dtype=np.float64) if position_bins > 0 else None
    )
    no_span_count = 0

    for sequence in sequences:
        if sequence.labels is None:
            raise ValueError("fit_single_span_prior requires labeled sequences")

        negative_mask = sequence.labels == negative_label
        if not bool(negative_mask.any()):
            no_span_count += 1
            continue

        change_points = np.diff(
            np.concatenate(([0], negative_mask.astype(np.int8), [0]))
        )
        starts = np.where(change_points == 1)[0]
        ends = np.where(change_points == -1)[0]
        lengths = ends - starts
        negative_span_lengths.append(int(lengths.max()))

        if start_bin_counts is not None and end_bin_counts is not None:
            span_start = int(starts[0])
            span_end = int(ends[-1])
            start_bin = _fraction_to_bin(span_start / sequence.length, position_bins)
            end_bin = _fraction_to_bin(span_end / sequence.length, position_bins)
            start_bin_counts[start_bin] += 1.0
            end_bin_counts[end_bin] += 1.0

    span_count = len(negative_span_lengths)
    total_sequences = span_count + no_span_count
    max_length = max(negative_span_lengths, default=1)

    length_counts = np.zeros(max_length + 1, dtype=np.float64)
    for span_length in negative_span_lengths:
        length_counts[span_length] += 1.0

    smoothed_presence_total = total_sequences + 2 * alpha
    no_span_log_prob = np.log((no_span_count + alpha) / smoothed_presence_total)
    span_log_prob = np.log((span_count + alpha) / smoothed_presence_total)

    smoothed_length_total = length_counts[1:].sum() + alpha * max_length
    span_length_log_probs = np.full(max_length + 1, -np.inf, dtype=np.float64)
    for span_length in range(1, max_length + 1):
        span_length_log_probs[span_length] = np.log(
            (length_counts[span_length] + alpha) / smoothed_length_total
        )

    start_bin_log_probs = None
    end_bin_log_probs = None
    if start_bin_counts is not None and end_bin_counts is not None:
        start_bin_log_probs = np.log(
            (start_bin_counts + alpha)
            / (start_bin_counts.sum() + alpha * len(start_bin_counts))
        )
        end_bin_log_probs = np.log(
            (end_bin_counts + alpha)
            / (end_bin_counts.sum() + alpha * len(end_bin_counts))
        )

    return SingleSpanPrior(
        no_span_log_prob=no_span_log_prob,
        span_log_prob=span_log_prob,
        span_length_log_probs=span_length_log_probs,
        start_bin_log_probs=start_bin_log_probs,
        end_bin_log_probs=end_bin_log_probs,
    )


def decode_single_negative_span(
    probabilities: np.ndarray,
    *,
    positive_index: int = 0,
    negative_index: int = 1,
    prior: SingleSpanPrior | None = None,
    min_span_length: int = 1,
    position_prior_weight: float = 0.0,
    no_span_bias: float = 0.0,
) -> np.ndarray:
    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        raise ValueError(
            "probabilities must be a 2D array shaped [time, num_labels>=2]"
        )
    if min_span_length <= 0:
        raise ValueError("min_span_length must be strictly positive")

    log_probs = np.log(np.clip(probabilities, 1e-9, 1.0))
    time_steps = log_probs.shape[0]
    base_positive_score = float(log_probs[:, positive_index].sum())

    best_score = base_positive_score
    if prior is not None:
        best_score += prior.no_span_log_prob + no_span_bias

    best_interval: tuple[int, int] | None = None
    prefix_delta = np.r_[0.0, np.cumsum(
        log_probs[:, negative_index] - log_probs[:, positive_index]
    )]

    for start in range(time_steps):
        for end in range(start + min_span_length, time_steps + 1):
            span_length = end - start
            score = base_positive_score + prefix_delta[end] - prefix_delta[start]
            if prior is not None:
                score += prior.span_log_prob + prior.get_span_length_log_prob(span_length)
                if position_prior_weight:
                    score += position_prior_weight * prior.get_position_log_prob(
                        start,
                        end,
                        sequence_length=time_steps,
                    )
            if score > best_score:
                best_score = score
                best_interval = (start, end)

    predictions = np.full(time_steps, positive_index, dtype=np.int64)
    if best_interval is not None:
        predictions[best_interval[0]:best_interval[1]] = negative_index
    return predictions


def single_negative_span_posteriors(
    probabilities: np.ndarray,
    *,
    positive_index: int = 0,
    negative_index: int = 1,
    prior: SingleSpanPrior | None = None,
    min_span_length: int = 1,
    position_prior_weight: float = 0.0,
    no_span_bias: float = 0.0,
) -> np.ndarray:
    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        raise ValueError(
            "probabilities must be a 2D array shaped [time, num_labels>=2]"
        )
    if min_span_length <= 0:
        raise ValueError("min_span_length must be strictly positive")

    log_probs = np.log(np.clip(probabilities, 1e-9, 1.0))
    time_steps = log_probs.shape[0]
    base_positive_score = float(log_probs[:, positive_index].sum())
    prefix_delta = np.r_[0.0, np.cumsum(
        log_probs[:, negative_index] - log_probs[:, positive_index]
    )]

    configuration_scores = [(
        base_positive_score + (
            0.0 if prior is None else prior.no_span_log_prob + no_span_bias
        ),
        None,
    )]

    for start in range(time_steps):
        for end in range(start + min_span_length, time_steps + 1):
            span_length = end - start
            score = base_positive_score + prefix_delta[end] - prefix_delta[start]
            if prior is not None:
                score += prior.span_log_prob + prior.get_span_length_log_prob(span_length)
                if position_prior_weight:
                    score += position_prior_weight * prior.get_position_log_prob(
                        start,
                        end,
                        sequence_length=time_steps,
                    )
            configuration_scores.append((score, (start, end)))

    scores = np.array([item[0] for item in configuration_scores], dtype=np.float64)
    max_score = float(scores.max())
    weights = np.exp(scores - max_score)
    weights /= weights.sum()

    negative_posteriors = np.zeros(time_steps, dtype=np.float64)
    for weight, (_, interval) in zip(weights, configuration_scores, strict=True):
        if interval is None:
            continue
        start, end = interval
        negative_posteriors[start:end] += weight

    positive_posteriors = 1.0 - negative_posteriors
    posteriors = np.asarray(probabilities, dtype=np.float64).copy()
    posteriors[:, positive_index] = positive_posteriors
    posteriors[:, negative_index] = negative_posteriors
    row_sums = posteriors.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return posteriors / row_sums


def decode_batch(
    sequence_probabilities: Iterable[np.ndarray],
    *,
    decoder: str = "argmax",
    prior: SingleSpanPrior | None = None,
    min_span_length: int = 1,
    position_prior_weight: float = 0.0,
    no_span_bias: float = 0.0,
) -> list[np.ndarray]:
    decoded_sequences: list[np.ndarray] = []

    for probabilities in sequence_probabilities:
        if decoder == "argmax":
            decoded = np.argmax(probabilities, axis=1)
        elif decoder == "single_negative_span":
            decoded = decode_single_negative_span(
                probabilities,
                prior=prior,
                min_span_length=min_span_length,
                position_prior_weight=position_prior_weight,
                no_span_bias=no_span_bias,
            )
        else:
            raise ValueError(f"Unknown decoder: {decoder}")
        decoded_sequences.append(decoded)

    return decoded_sequences


def posterior_probabilities_by_sequence(
    sequence_probabilities: Iterable[np.ndarray],
    *,
    decoder: str = "identity",
    prior: SingleSpanPrior | None = None,
    min_span_length: int = 1,
    position_prior_weight: float = 0.0,
    no_span_bias: float = 0.0,
) -> list[np.ndarray]:
    posterior_sequences: list[np.ndarray] = []

    for probabilities in sequence_probabilities:
        if decoder == "identity":
            posteriors = np.asarray(probabilities, dtype=np.float64).copy()
        elif decoder == "single_negative_span":
            posteriors = single_negative_span_posteriors(
                probabilities,
                prior=prior,
                min_span_length=min_span_length,
                position_prior_weight=position_prior_weight,
                no_span_bias=no_span_bias,
            )
        else:
            raise ValueError(f"Unknown decoder: {decoder}")
        posterior_sequences.append(posteriors)

    return posterior_sequences


def smooth_probabilities_by_sequence(
    sequence_probabilities: Iterable[np.ndarray],
    *,
    sigma: float,
) -> list[np.ndarray]:
    smoothed_sequences: list[np.ndarray] = []

    for probabilities in sequence_probabilities:
        if sigma <= 0:
            smoothed = np.asarray(probabilities, dtype=np.float64).copy()
        else:
            smoothed = gaussian_filter1d(
                np.asarray(probabilities, dtype=np.float64),
                sigma=sigma,
                axis=0,
            )
        row_sums = smoothed.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        smoothed_sequences.append(smoothed / row_sums)

    return smoothed_sequences
