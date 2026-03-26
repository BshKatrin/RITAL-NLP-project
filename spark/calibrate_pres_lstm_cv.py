from __future__ import annotations

import argparse
import copy
import json
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.special import logit
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedShuffleSplit
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader

from rital_nlp_project.presidents import (
    BiLSTMSequenceTagger,
    PAD_LABEL_INDEX,
    PRESIDENT_LABEL_ORDER,
    SpeechSequence,
    SpeechSequenceDataset,
    augment_sequences_with_position_features,
    augment_sequences_with_transition_features,
    build_speech_sequences,
    collate_speech_sequences,
    compute_class_weights,
    decode_batch,
    fit_single_span_prior,
    load_embeddings_with_metadata,
    posterior_probabilities_by_sequence,
)


INDEX_TO_LABEL = np.array(PRESIDENT_LABEL_ORDER, dtype=np.int64)
POSITIVE_INDEX = int(np.where(INDEX_TO_LABEL == 1)[0][0])
NEGATIVE_INDEX = int(np.where(INDEX_TO_LABEL == -1)[0][0])


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if (
        getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available()
    ):
        return "mps"
    return "cpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate presidents LSTM decision threshold on labeled data only, "
            "using grouped speech-level cross-validation."
        )
    )
    parser.add_argument(
        "--embeddings-path",
        default="Dataset/embeddings/presidents_camembert_base_mean.npy",
    )
    parser.add_argument(
        "--metadata-path",
        default="Dataset/clean/presidents_clean_bert.parquet",
    )
    parser.add_argument(
        "--output-dir",
        default="Dataset/out/presidents_lstm_cv_calibration",
    )
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--inner-validation-size", type=float, default=0.15)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--projection-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--min-negative-span", type=int, default=3)
    parser.add_argument(
        "--position-features",
        choices=("none", "basic"),
        default="none",
    )
    parser.add_argument("--transition-features", action="store_true")
    parser.add_argument("--position-bins", type=int, default=5)
    parser.add_argument("--position-prior-weight", type=float, default=0.25)
    parser.add_argument("--no-span-bias", type=float, default=0.0)
    parser.add_argument(
        "--selection-decoder",
        choices=("argmax", "single_negative_span"),
        default="single_negative_span",
    )
    parser.add_argument("--threshold-start", type=float, default=0.05)
    parser.add_argument("--threshold-end", type=float, default=0.95)
    parser.add_argument("--threshold-step", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=default_device())
    parser.add_argument(
        "--max-speeches",
        type=int,
        default=None,
        help="Optional cap used for smoke tests.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def maybe_slice_sequences(
    sequences: list[SpeechSequence], limit: int | None
) -> list[SpeechSequence]:
    if limit is None:
        return list(sequences)
    return list(sequences[:limit])


def create_loader(sequences, *, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = SpeechSequenceDataset(sequences)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_speech_sequences,
    )


def build_model(
    *,
    input_dim: int,
    hidden_dim: int,
    projection_dim: int,
    num_layers: int,
    dropout: float,
    device: torch.device,
) -> BiLSTMSequenceTagger:
    return BiLSTMSequenceTagger(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        projection_dim=projection_dim,
        num_layers=num_layers,
        dropout=dropout,
        num_labels=len(PRESIDENT_LABEL_ORDER),
    ).to(device)


def train_one_epoch(
    model: BiLSTMSequenceTagger,
    dataloader: DataLoader,
    *,
    optimizer: AdamW,
    criterion: nn.CrossEntropyLoss,
    device: torch.device,
    gradient_clip: float,
) -> float:
    model.train()
    total_loss = 0.0
    total_items = 0

    for batch in dataloader:
        inputs = batch["inputs"].to(device)
        lengths = batch["lengths"]
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        logits = model(inputs, lengths)
        loss = criterion(logits.transpose(1, 2), labels)
        loss.backward()
        clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()

        batch_size = inputs.size(0)
        total_loss += float(loss.item()) * batch_size
        total_items += batch_size

    return total_loss / max(total_items, 1)


def collect_predictions(
    model: BiLSTMSequenceTagger,
    dataloader: DataLoader,
    *,
    device: torch.device,
) -> list[dict[str, np.ndarray | int]]:
    predictions = []
    model.eval()

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch["inputs"].to(device)
            lengths = batch["lengths"]
            logits = model(inputs, lengths)
            probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
            labels = batch["labels"].numpy()
            sentence_ids = batch["sentence_ids"].numpy()
            speech_ids = batch["speech_ids"].tolist()

            for row_idx, length in enumerate(lengths.tolist()):
                label_indices = labels[row_idx, :length].copy()
                predictions.append(
                    {
                        "speech_id": int(speech_ids[row_idx]),
                        "sentence_ids": sentence_ids[row_idx, :length].copy(),
                        "label_indices": label_indices,
                        "labels": INDEX_TO_LABEL[label_indices].copy(),
                        "probabilities": probabilities[row_idx, :length].copy(),
                    }
                )

    return predictions


def evaluate_predictions(
    predictions: list[dict[str, np.ndarray | int]],
    *,
    decoder: str,
    prior,
    min_negative_span: int,
    position_prior_weight: float,
    no_span_bias: float,
) -> dict[str, float]:
    sequence_probabilities = [
        item["probabilities"] for item in predictions  # type: ignore[index]
    ]
    decoded = decode_batch(
        sequence_probabilities,
        decoder=decoder,
        prior=prior,
        min_span_length=min_negative_span,
        position_prior_weight=position_prior_weight,
        no_span_bias=no_span_bias,
    )

    y_true_idx = np.concatenate(
        [item["label_indices"] for item in predictions]  # type: ignore[index]
    )
    y_pred_idx = np.concatenate(decoded)
    y_true = INDEX_TO_LABEL[y_true_idx]
    y_pred = INDEX_TO_LABEL[y_pred_idx]

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def speech_targets(sequences: list[SpeechSequence]) -> np.ndarray:
    return np.array(
        [
            int(sequence.labels is not None and np.any(sequence.labels == -1))
            for sequence in sequences
        ],
        dtype=np.int64,
    )


def build_outer_splits(
    sequences: list[SpeechSequence],
    *,
    n_splits: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    targets = speech_targets(sequences)
    min_count = int(np.bincount(targets).min()) if len(targets) else 0

    if min_count >= n_splits:
        splitter = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=seed,
        )
        return list(splitter.split(np.zeros(len(sequences)), targets))

    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(splitter.split(np.zeros(len(sequences))))


def split_inner_train_validation(
    sequences: list[SpeechSequence],
    *,
    validation_size: float,
    seed: int,
) -> tuple[list[SpeechSequence], list[SpeechSequence]]:
    if len(sequences) < 2:
        raise ValueError("Need at least two speeches for inner validation")

    targets = speech_targets(sequences)
    min_count = int(np.bincount(targets).min()) if len(targets) else 0
    indices = np.arange(len(sequences))

    if min_count >= 2:
        splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=validation_size,
            random_state=seed,
        )
        train_idx, val_idx = next(splitter.split(np.zeros(len(sequences)), targets))
    else:
        rng = np.random.default_rng(seed)
        shuffled = rng.permutation(indices)
        val_size = max(1, int(round(len(sequences) * validation_size)))
        val_idx = np.sort(shuffled[:val_size])
        train_idx = np.sort(shuffled[val_size:])

    return (
        [sequences[idx] for idx in train_idx],
        [sequences[idx] for idx in val_idx],
    )


def select_best_epoch(
    train_sequences: list[SpeechSequence],
    validation_sequences: list[SpeechSequence],
    *,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> tuple[int, float]:
    set_seed(seed)
    train_loader = create_loader(
        train_sequences,
        batch_size=args.batch_size,
        shuffle=True,
    )
    validation_loader = create_loader(
        validation_sequences,
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = build_model(
        input_dim=int(train_sequences[0].embeddings.shape[1]),
        hidden_dim=args.hidden_dim,
        projection_dim=args.projection_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        device=device,
    )
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    criterion = nn.CrossEntropyLoss(
        weight=compute_class_weights(train_sequences).to(device),
        ignore_index=PAD_LABEL_INDEX,
    )
    validation_prior = fit_single_span_prior(
        train_sequences,
        position_bins=args.position_bins,
    )

    best_epoch = 1
    best_score = float("-inf")
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(1, args.epochs + 1):
        train_one_epoch(
            model,
            train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            gradient_clip=args.gradient_clip,
        )
        validation_predictions = collect_predictions(
            model,
            validation_loader,
            device=device,
        )
        validation_metrics = evaluate_predictions(
            validation_predictions,
            decoder=args.selection_decoder,
            prior=validation_prior if args.selection_decoder != "argmax" else None,
            min_negative_span=args.min_negative_span,
            position_prior_weight=args.position_prior_weight,
            no_span_bias=args.no_span_bias,
        )
        selection_score = validation_metrics["f1_macro"]
        if selection_score > best_score:
            best_score = selection_score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return best_epoch, best_score


def train_fixed_epochs(
    train_sequences: list[SpeechSequence],
    *,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
    epochs: int,
) -> BiLSTMSequenceTagger:
    set_seed(seed)
    train_loader = create_loader(
        train_sequences,
        batch_size=args.batch_size,
        shuffle=True,
    )
    model = build_model(
        input_dim=int(train_sequences[0].embeddings.shape[1]),
        hidden_dim=args.hidden_dim,
        projection_dim=args.projection_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        device=device,
    )
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    criterion = nn.CrossEntropyLoss(
        weight=compute_class_weights(train_sequences).to(device),
        ignore_index=PAD_LABEL_INDEX,
    )

    for _ in range(epochs):
        train_one_epoch(
            model,
            train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            gradient_clip=args.gradient_clip,
        )

    return model


def predictions_to_frame(
    predictions: list[dict[str, np.ndarray | int]],
    *,
    prior,
    min_negative_span: int,
    position_prior_weight: float,
    no_span_bias: float,
    fold: int,
) -> pd.DataFrame:
    sequence_probabilities = [
        item["probabilities"] for item in predictions  # type: ignore[index]
    ]
    span_posteriors = posterior_probabilities_by_sequence(
        sequence_probabilities,
        decoder="single_negative_span",
        prior=prior,
        min_span_length=min_negative_span,
        position_prior_weight=position_prior_weight,
        no_span_bias=no_span_bias,
    )

    frames = []
    for item, span_probs in zip(predictions, span_posteriors, strict=True):
        frames.append(
            pd.DataFrame(
                {
                    "fold": fold,
                    "speech_id": int(item["speech_id"]),
                    "sentence_id": item["sentence_ids"],  # type: ignore[index]
                    "true_label": item["labels"],  # type: ignore[index]
                    "prob_mitterrand_raw": item["probabilities"][:, NEGATIVE_INDEX],  # type: ignore[index]
                    "prob_mitterrand_single_negative_span": span_probs[
                        :,
                        NEGATIVE_INDEX,
                    ],
                }
            )
        )

    return pd.concat(frames, ignore_index=True)


def threshold_grid(start: float, end: float, step: float) -> np.ndarray:
    if not 0.0 < start < 1.0 or not 0.0 < end < 1.0:
        raise ValueError("threshold range must be strictly between 0 and 1")
    if end <= start:
        raise ValueError("threshold-end must be greater than threshold-start")
    if step <= 0:
        raise ValueError("threshold-step must be strictly positive")

    values = np.arange(start, end + step / 2.0, step, dtype=np.float64)
    return np.clip(values, 1e-6, 1.0 - 1e-6)


def threshold_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    threshold: float,
) -> dict[str, float]:
    y_pred = np.where(scores >= threshold, -1, 1)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def sweep_thresholds(
    frame: pd.DataFrame,
    *,
    score_name: str,
    thresholds: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, float]]:
    y_true = frame["true_label"].to_numpy(dtype=np.int64)
    scores = frame[score_name].to_numpy(dtype=np.float64)
    records = []

    for threshold in thresholds:
        metrics = threshold_metrics(y_true, scores, threshold=float(threshold))
        records.append(
            {
                "score_name": score_name,
                "threshold": float(threshold),
                "best_threshold": float(threshold),
                "equivalent_score_delta": float(logit(float(threshold))),
                **metrics,
                "mitterrand_count": int((scores >= threshold).sum()),
                "chirac_count": int((scores < threshold).sum()),
            }
        )

    sweep_frame = pd.DataFrame(records).sort_values(
        ["f1_macro", "balanced_accuracy", "threshold"],
        ascending=[False, False, True],
    )
    best = sweep_frame.iloc[0].to_dict()
    return pd.DataFrame(records), best


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    started_at = datetime.now().astimezone().isoformat()
    run_start = time.perf_counter()

    embeddings, metadata = load_embeddings_with_metadata(
        args.embeddings_path,
        args.metadata_path,
    )
    sequences = build_speech_sequences(embeddings, metadata)
    sequences = maybe_slice_sequences(sequences, args.max_speeches)
    if args.position_features != "none":
        sequences = augment_sequences_with_position_features(sequences)
    if args.transition_features:
        sequences = augment_sequences_with_transition_features(sequences)

    folds = build_outer_splits(
        sequences,
        n_splits=args.n_splits,
        seed=args.seed,
    )

    fold_records = []
    oof_frames = []

    for fold_idx, (train_idx, test_idx) in enumerate(folds, start=1):
        outer_train_sequences = [sequences[idx] for idx in train_idx]
        outer_test_sequences = [sequences[idx] for idx in test_idx]

        inner_train_sequences, inner_validation_sequences = split_inner_train_validation(
            outer_train_sequences,
            validation_size=args.inner_validation_size,
            seed=args.seed + fold_idx,
        )

        fold_start = time.perf_counter()
        best_epoch, best_selection_score = select_best_epoch(
            inner_train_sequences,
            inner_validation_sequences,
            args=args,
            device=device,
            seed=args.seed + fold_idx,
        )
        model = train_fixed_epochs(
            outer_train_sequences,
            args=args,
            device=device,
            seed=args.seed + 1000 + fold_idx,
            epochs=best_epoch,
        )
        outer_prior = fit_single_span_prior(
            outer_train_sequences,
            position_bins=args.position_bins,
        )
        outer_test_loader = create_loader(
            outer_test_sequences,
            batch_size=args.batch_size,
            shuffle=False,
        )
        outer_predictions = collect_predictions(
            model,
            outer_test_loader,
            device=device,
        )
        oof_frames.append(
            predictions_to_frame(
                outer_predictions,
                prior=outer_prior,
                min_negative_span=args.min_negative_span,
                position_prior_weight=args.position_prior_weight,
                no_span_bias=args.no_span_bias,
                fold=fold_idx,
            )
        )
        fold_records.append(
            {
                "fold": fold_idx,
                "train_speeches": len(outer_train_sequences),
                "test_speeches": len(outer_test_sequences),
                "best_epoch": int(best_epoch),
                "inner_selection_f1_macro": float(best_selection_score),
                "fold_seconds": time.perf_counter() - fold_start,
            }
        )
        print(json.dumps(fold_records[-1], indent=2))

    oof_frame = pd.concat(oof_frames, ignore_index=True).sort_values(
        ["speech_id", "sentence_id"]
    )
    oof_path = output_dir / "oof_predictions.csv"
    oof_frame.to_csv(oof_path, index=False)

    thresholds = threshold_grid(
        args.threshold_start,
        args.threshold_end,
        args.threshold_step,
    )
    sweep_frames = []
    best_records = []
    for score_name in (
        "prob_mitterrand_raw",
        "prob_mitterrand_single_negative_span",
    ):
        sweep_frame, best_record = sweep_thresholds(
            oof_frame,
            score_name=score_name,
            thresholds=thresholds,
        )
        sweep_frames.append(sweep_frame)
        best_records.append(best_record)

    threshold_sweep = pd.concat(sweep_frames, ignore_index=True)
    threshold_sweep_path = output_dir / "threshold_sweep.csv"
    threshold_sweep.to_csv(threshold_sweep_path, index=False)

    best_by_score = {
        record["score_name"]: record
        for record in best_records
    }
    selected_score_name = max(
        best_by_score,
        key=lambda score_name: (
            best_by_score[score_name]["f1_macro"],
            best_by_score[score_name]["balanced_accuracy"],
        ),
    )
    default_threshold_metrics = {}
    y_true = oof_frame["true_label"].to_numpy(dtype=np.int64)
    for score_name in best_by_score:
        metrics = threshold_metrics(
            y_true,
            oof_frame[score_name].to_numpy(dtype=np.float64),
            threshold=0.5,
        )
        default_threshold_metrics[score_name] = metrics

    summary = {
        "config": vars(args),
        "device": str(device),
        "started_at": started_at,
        "ended_at": datetime.now().astimezone().isoformat(),
        "total_seconds": time.perf_counter() - run_start,
        "num_sequences": len(sequences),
        "num_rows": int(sum(sequence.length for sequence in sequences)),
        "folds": fold_records,
        "selected_score_name": selected_score_name,
        "default_threshold_metrics": default_threshold_metrics,
        "best_by_score_name": best_by_score,
        "recommended_calibration": best_by_score[selected_score_name],
        "outputs": {
            "oof_predictions": str(oof_path.resolve()),
            "threshold_sweep": str(threshold_sweep_path.resolve()),
        },
    }

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(summary, indent=2))

    calibration_path = output_dir / "calibration.json"
    calibration_path.write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary["recommended_calibration"], indent=2))


if __name__ == "__main__":
    main()
