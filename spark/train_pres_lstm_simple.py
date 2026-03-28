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
    build_speech_sequences,
    collate_speech_sequences,
    compute_class_weights,
    load_embeddings_with_metadata,
)


INDEX_TO_LABEL = np.array(PRESIDENT_LABEL_ORDER, dtype=np.int64)
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
            "Train a simple speech-aware presidents BiLSTM and learn a clean "
            "labeled-only calibration threshold."
        )
    )
    parser.add_argument(
        "--embeddings-path",
        default="Dataset/embeddings/presidents_camembert_base_mean.npy",
        help="Path to the labeled presidents sentence embeddings.",
    )
    parser.add_argument(
        "--metadata-path",
        default="Dataset/clean/presidents_clean_bert.parquet",
        help="Path to the labeled presidents metadata parquet.",
    )
    parser.add_argument(
        "--output-dir",
        default="Dataset/out/presidents_lstm_simple",
        help="Directory where checkpoint, calibration and metrics are written.",
    )
    parser.add_argument("--validation-size", type=float, default=0.15)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--projection-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
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


def speech_targets(sequences: list[SpeechSequence]) -> np.ndarray:
    return np.array(
        [
            int(sequence.labels is not None and np.any(sequence.labels == -1))
            for sequence in sequences
        ],
        dtype=np.int64,
    )


def stratified_train_validation_split(
    sequences: list[SpeechSequence],
    *,
    validation_size: float,
    seed: int,
) -> tuple[list[SpeechSequence], list[SpeechSequence]]:
    if not 0.0 < validation_size < 1.0:
        raise ValueError("validation-size must be between 0 and 1")
    if len(sequences) < 2:
        raise ValueError("Need at least two speeches to build a validation split")

    targets = speech_targets(sequences)
    min_count = int(np.bincount(targets).min()) if len(targets) else 0

    if min_count >= 2:
        splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=validation_size,
            random_state=seed,
        )
        train_idx, validation_idx = next(
            splitter.split(np.zeros(len(sequences)), targets)
        )
    else:
        rng = np.random.default_rng(seed)
        shuffled = rng.permutation(np.arange(len(sequences)))
        validation_size_int = max(1, int(round(len(sequences) * validation_size)))
        validation_idx = np.sort(shuffled[:validation_size_int])
        train_idx = np.sort(shuffled[validation_size_int:])

    return (
        [sequences[idx] for idx in train_idx],
        [sequences[idx] for idx in validation_idx],
    )


def build_cv_splits(
    sequences: list[SpeechSequence],
    *,
    n_folds: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    targets = speech_targets(sequences)
    min_count = int(np.bincount(targets).min()) if len(targets) else 0

    if min_count >= n_folds:
        splitter = StratifiedKFold(
            n_splits=n_folds,
            shuffle=True,
            random_state=seed,
        )
        return list(splitter.split(np.zeros(len(sequences)), targets))

    splitter = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return list(splitter.split(np.zeros(len(sequences))))


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
                        "prob_mitterrand_raw": probabilities[row_idx, :length, NEGATIVE_INDEX].copy(),
                    }
                )

    return predictions


def predictions_to_frame(
    predictions: list[dict[str, np.ndarray | int]],
    *,
    fold: int | None = None,
) -> pd.DataFrame:
    frames = []
    for item in predictions:
        frame = pd.DataFrame(
            {
                "speech_id": int(item["speech_id"]),
                "sentence_id": item["sentence_ids"],  # type: ignore[index]
                "true_label": item["labels"],  # type: ignore[index]
                "prob_mitterrand_raw": item["prob_mitterrand_raw"],  # type: ignore[index]
            }
        )
        if fold is not None:
            frame.insert(0, "fold", fold)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


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


def threshold_grid(start: float, end: float, step: float) -> np.ndarray:
    if not 0.0 < start < 1.0 or not 0.0 < end < 1.0:
        raise ValueError("threshold range must be strictly between 0 and 1")
    if end <= start:
        raise ValueError("threshold-end must be greater than threshold-start")
    if step <= 0:
        raise ValueError("threshold-step must be strictly positive")
    values = np.arange(start, end + step / 2.0, step, dtype=np.float64)
    return np.clip(values, 1e-6, 1.0 - 1e-6)


def sweep_thresholds(frame: pd.DataFrame, *, thresholds: np.ndarray) -> tuple[pd.DataFrame, dict[str, float]]:
    y_true = frame["true_label"].to_numpy(dtype=np.int64)
    scores = frame["prob_mitterrand_raw"].to_numpy(dtype=np.float64)
    records = []

    for threshold in thresholds:
        metrics = threshold_metrics(y_true, scores, threshold=float(threshold))
        records.append(
            {
                "selected_score_name": "prob_mitterrand_raw",
                "threshold": float(threshold),
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
    return pd.DataFrame(records), sweep_frame.iloc[0].to_dict()


def select_best_epoch(
    train_sequences: list[SpeechSequence],
    validation_sequences: list[SpeechSequence],
    *,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[int, float, list[dict[str, np.ndarray | int]]]:
    set_seed(args.seed)
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

    best_epoch = 1
    best_score = float("-inf")
    best_predictions: list[dict[str, np.ndarray | int]] = []
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
        validation_frame = predictions_to_frame(validation_predictions)
        validation_metrics = threshold_metrics(
            validation_frame["true_label"].to_numpy(dtype=np.int64),
            validation_frame["prob_mitterrand_raw"].to_numpy(dtype=np.float64),
            threshold=0.5,
        )

        if validation_metrics["f1_macro"] > best_score:
            best_score = validation_metrics["f1_macro"]
            best_epoch = epoch
            best_predictions = validation_predictions
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return best_epoch, best_score, best_predictions


def train_fixed_epochs(
    sequences: list[SpeechSequence],
    *,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
    epochs: int,
) -> BiLSTMSequenceTagger:
    set_seed(seed)
    loader = create_loader(
        sequences,
        batch_size=args.batch_size,
        shuffle=True,
    )
    model = build_model(
        input_dim=int(sequences[0].embeddings.shape[1]),
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
        weight=compute_class_weights(sequences).to(device),
        ignore_index=PAD_LABEL_INDEX,
    )

    for _ in range(epochs):
        train_one_epoch(
            model,
            loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            gradient_clip=args.gradient_clip,
        )

    return model


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

    train_sequences, validation_sequences = stratified_train_validation_split(
        sequences,
        validation_size=args.validation_size,
        seed=args.seed,
    )

    best_epoch, best_validation_f1, best_validation_predictions = select_best_epoch(
        train_sequences,
        validation_sequences,
        args=args,
        device=device,
    )
    validation_frame = predictions_to_frame(best_validation_predictions)
    validation_default_metrics = threshold_metrics(
        validation_frame["true_label"].to_numpy(dtype=np.int64),
        validation_frame["prob_mitterrand_raw"].to_numpy(dtype=np.float64),
        threshold=0.5,
    )

    cv_splits = build_cv_splits(
        sequences,
        n_folds=args.n_folds,
        seed=args.seed,
    )
    fold_records = []
    oof_frames = []

    for fold_idx, (train_idx, test_idx) in enumerate(cv_splits, start=1):
        fold_start = time.perf_counter()
        fold_train_sequences = [sequences[idx] for idx in train_idx]
        fold_test_sequences = [sequences[idx] for idx in test_idx]

        model = train_fixed_epochs(
            fold_train_sequences,
            args=args,
            device=device,
            seed=args.seed + 1000 + fold_idx,
            epochs=best_epoch,
        )
        fold_test_loader = create_loader(
            fold_test_sequences,
            batch_size=args.batch_size,
            shuffle=False,
        )
        fold_predictions = collect_predictions(
            model,
            fold_test_loader,
            device=device,
        )
        oof_frames.append(predictions_to_frame(fold_predictions, fold=fold_idx))
        fold_records.append(
            {
                "fold": fold_idx,
                "train_speeches": len(fold_train_sequences),
                "test_speeches": len(fold_test_sequences),
                "epochs": int(best_epoch),
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
    threshold_sweep, best_threshold_record = sweep_thresholds(
        oof_frame,
        thresholds=thresholds,
    )
    threshold_sweep_path = output_dir / "threshold_sweep.csv"
    threshold_sweep.to_csv(threshold_sweep_path, index=False)

    default_oof_metrics = threshold_metrics(
        oof_frame["true_label"].to_numpy(dtype=np.int64),
        oof_frame["prob_mitterrand_raw"].to_numpy(dtype=np.float64),
        threshold=0.5,
    )

    final_model_start = time.perf_counter()
    final_model = train_fixed_epochs(
        sequences,
        args=args,
        device=device,
        seed=args.seed + 9999,
        epochs=best_epoch,
    )
    final_model_seconds = time.perf_counter() - final_model_start

    checkpoint_path = output_dir / "checkpoint.pt"
    torch.save(
        {
            "model_state_dict": final_model.state_dict(),
            "config": {
                "hidden_dim": args.hidden_dim,
                "projection_dim": args.projection_dim,
                "num_layers": args.num_layers,
                "dropout": args.dropout,
                "batch_size": args.batch_size,
            },
            "label_order": PRESIDENT_LABEL_ORDER,
            "best_epoch": int(best_epoch),
        },
        checkpoint_path,
    )

    calibration = {
        "selected_score_name": "prob_mitterrand_raw",
        "best_threshold": float(best_threshold_record["threshold"]),
        "equivalent_score_delta": float(best_threshold_record["equivalent_score_delta"]),
        "oof_metrics_at_0p5": default_oof_metrics,
        "oof_metrics_at_best_threshold": {
            "threshold": float(best_threshold_record["threshold"]),
            "accuracy": float(best_threshold_record["accuracy"]),
            "balanced_accuracy": float(best_threshold_record["balanced_accuracy"]),
            "f1_macro": float(best_threshold_record["f1_macro"]),
            "mitterrand_count": int(best_threshold_record["mitterrand_count"]),
            "chirac_count": int(best_threshold_record["chirac_count"]),
        },
        "best_epoch": int(best_epoch),
        "n_folds": int(args.n_folds),
    }
    calibration_path = output_dir / "calibration.json"
    calibration_path.write_text(json.dumps(calibration, indent=2))

    summary = {
        "config": vars(args),
        "device": str(device),
        "started_at": started_at,
        "ended_at": datetime.now().astimezone().isoformat(),
        "total_seconds": time.perf_counter() - run_start,
        "num_sequences": len(sequences),
        "num_rows": int(sum(sequence.length for sequence in sequences)),
        "train_speeches": len(train_sequences),
        "validation_speeches": len(validation_sequences),
        "best_epoch": int(best_epoch),
        "best_validation_f1_macro": float(best_validation_f1),
        "validation_metrics_at_0p5": validation_default_metrics,
        "default_threshold_metrics": {
            "selected_score_name": "prob_mitterrand_raw",
            **default_oof_metrics,
        },
        "recommended_calibration": calibration,
        "folds": fold_records,
        "timing": {
            "final_model_seconds": final_model_seconds,
        },
        "outputs": {
            "checkpoint": str(checkpoint_path.resolve()),
            "calibration": str(calibration_path.resolve()),
            "oof_predictions": str(oof_path.resolve()),
            "threshold_sweep": str(threshold_sweep_path.resolve()),
        },
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["recommended_calibration"], indent=2))


if __name__ == "__main__":
    main()
