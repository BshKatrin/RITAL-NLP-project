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
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader

from rital_nlp_project.presidents import (
    BiLSTMSequenceTagger,
    PAD_LABEL_INDEX,
    PRESIDENT_LABEL_ORDER,
    SpeechSequenceDataset,
    augment_sequences_with_position_features,
    augment_sequences_with_transition_features,
    build_speech_sequences,
    collate_speech_sequences,
    compute_class_weights,
    decode_batch,
    fit_single_span_prior,
    load_embeddings_with_metadata,
    speech_train_test_split,
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
        description="Train a speech-aware BiLSTM on president sentence embeddings."
    )
    parser.add_argument(
        "--embeddings-path",
        default="Dataset/embeddings/presidents_camembert_base_mean.npy",
        help="Path to the sentence embeddings numpy file.",
    )
    parser.add_argument(
        "--metadata-path",
        default="Dataset/clean/presidents_clean_bert.parquet",
        help="Path to the cleaned metadata parquet file.",
    )
    parser.add_argument(
        "--output-dir",
        default="Dataset/out/presidents_lstm",
        help="Directory where checkpoints, metrics and predictions are saved.",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--validation-size", type=float, default=0.1)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--projection-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--min-negative-span", type=int, default=3)
    parser.add_argument(
        "--position-features",
        choices=("none", "basic"),
        default="none",
        help="Concatenate deterministic position features to each sentence embedding.",
    )
    parser.add_argument(
        "--transition-features",
        action="store_true",
        help="Concatenate local change-point features derived from neighboring sentence embeddings.",
    )
    parser.add_argument(
        "--position-bins",
        type=int,
        default=0,
        help="Number of normalized start/end bins used by the single-span prior.",
    )
    parser.add_argument(
        "--position-prior-weight",
        type=float,
        default=0.0,
        help="Weight applied to the span location prior during constrained decoding.",
    )
    parser.add_argument(
        "--selection-decoder",
        choices=("argmax", "single_negative_span"),
        default="single_negative_span",
        help="Decoder used for model selection on the validation split.",
    )
    parser.add_argument(
        "--device",
        default=default_device(),
        help="Device used for training.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-speeches", type=int, default=None)
    parser.add_argument("--max-val-speeches", type=int, default=None)
    parser.add_argument("--max-test-speeches", type=int, default=None)
    parser.add_argument(
        "--final-train-full-data",
        action="store_true",
        help="Train on all labeled speeches with fixed hyperparameters and skip held-out evaluation.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def maybe_slice_sequences(sequences, limit: int | None):
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

            labels = batch.get("labels")
            sentence_ids = batch["sentence_ids"].numpy()
            speech_ids = batch["speech_ids"].tolist()

            for row_idx, length in enumerate(lengths.tolist()):
                item = {
                    "speech_id": int(speech_ids[row_idx]),
                    "sentence_ids": sentence_ids[row_idx, :length].copy(),
                    "probabilities": probabilities[row_idx, :length].copy(),
                }

                if labels is not None:
                    label_indices = labels[row_idx, :length].numpy()
                    item["label_indices"] = label_indices.copy()
                    item["labels"] = INDEX_TO_LABEL[label_indices].copy()

                predictions.append(item)

    return predictions


def evaluate_predictions(
    predictions: list[dict[str, np.ndarray | int]],
    *,
    decoder: str,
    prior=None,
    min_negative_span: int,
    position_prior_weight: float,
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
    )

    y_true_idx = np.concatenate(
        [item["label_indices"] for item in predictions]  # type: ignore[index]
    )
    y_pred_idx = np.concatenate(decoded)
    y_true = INDEX_TO_LABEL[y_true_idx]
    y_pred = INDEX_TO_LABEL[y_pred_idx]
    positive_scores = np.concatenate(
        [item["probabilities"][:, POSITIVE_INDEX] for item in predictions]  # type: ignore[index]
    )
    roc_auc_positive = None
    if np.unique(y_true).size > 1:
        roc_auc_positive = float(
            roc_auc_score((y_true == 1).astype(np.int64), positive_scores)
        )

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "f1_macro": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "positive_recall": float(recall_score(y_true, y_pred, pos_label=1)),
        "negative_recall": float(recall_score(y_true, y_pred, pos_label=-1)),
        "roc_auc_positive": roc_auc_positive,
    }


def predictions_to_frame(
    predictions: list[dict[str, np.ndarray | int]],
    *,
    prior,
    min_negative_span: int,
    position_prior_weight: float,
) -> pd.DataFrame:
    raw_decoded = decode_batch(
        [item["probabilities"] for item in predictions],  # type: ignore[index]
        decoder="argmax",
    )
    constrained_decoded = decode_batch(
        [item["probabilities"] for item in predictions],  # type: ignore[index]
        decoder="single_negative_span",
        prior=prior,
        min_span_length=min_negative_span,
        position_prior_weight=position_prior_weight,
    )

    frames = []
    for item, raw_pred_idx, constrained_pred_idx in zip(
        predictions,
        raw_decoded,
        constrained_decoded,
        strict=True,
    ):
        frame = pd.DataFrame(
            {
                "speech_id": int(item["speech_id"]),
                "sentence_id": item["sentence_ids"],  # type: ignore[index]
                "prob_positive": item["probabilities"][:, POSITIVE_INDEX],  # type: ignore[index]
                "prob_negative": item["probabilities"][:, NEGATIVE_INDEX],  # type: ignore[index]
                "pred_raw": INDEX_TO_LABEL[raw_pred_idx],
                "pred_single_negative_span": INDEX_TO_LABEL[constrained_pred_idx],
            }
        )
        if "labels" in item:
            frame["true_label"] = item["labels"]  # type: ignore[index]
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "history.jsonl"
    history_path.write_text("")

    started_at = datetime.now().astimezone().isoformat()
    run_start = time.perf_counter()

    embeddings, metadata = load_embeddings_with_metadata(
        args.embeddings_path,
        args.metadata_path,
    )
    sequences = build_speech_sequences(embeddings, metadata)
    if args.position_features != "none":
        sequences = augment_sequences_with_position_features(sequences)
    if args.transition_features:
        sequences = augment_sequences_with_transition_features(sequences)

    if args.final_train_full_data:
        train_sequences = maybe_slice_sequences(sequences, args.max_train_speeches)
        val_sequences = []
        test_sequences = []
    else:
        train_sequences, test_sequences = speech_train_test_split(
            sequences,
            test_size=args.test_size,
        )
        train_sequences, val_sequences = speech_train_test_split(
            train_sequences,
            test_size=args.validation_size,
        )

        train_sequences = maybe_slice_sequences(train_sequences, args.max_train_speeches)
        val_sequences = maybe_slice_sequences(val_sequences, args.max_val_speeches)
        test_sequences = maybe_slice_sequences(test_sequences, args.max_test_speeches)

    class_weights = compute_class_weights(train_sequences).to(device)
    span_prior = fit_single_span_prior(
        train_sequences,
        position_bins=args.position_bins,
    )

    train_loader = create_loader(
        train_sequences,
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = None
    test_loader = None
    if not args.final_train_full_data:
        val_loader = create_loader(
            val_sequences,
            batch_size=args.batch_size,
            shuffle=False,
        )
        test_loader = create_loader(
            test_sequences,
            batch_size=args.batch_size,
            shuffle=False,
        )

    model = BiLSTMSequenceTagger(
        input_dim=int(train_sequences[0].embeddings.shape[1]),
        hidden_dim=args.hidden_dim,
        projection_dim=args.projection_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        num_labels=len(PRESIDENT_LABEL_ORDER),
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        ignore_index=PAD_LABEL_INDEX,
    )

    history = []
    best_epoch = 0
    best_score = float("-inf")
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        epoch_started_at = datetime.now().astimezone().isoformat()

        train_start = time.perf_counter()
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            gradient_clip=args.gradient_clip,
        )
        train_seconds = time.perf_counter() - train_start

        if args.final_train_full_data:
            validation_seconds = 0.0
            val_metrics = {}
            best_epoch = epoch
            best_score = float("nan")
            best_state = copy.deepcopy(model.state_dict())
        else:
            validation_start = time.perf_counter()
            val_predictions = collect_predictions(model, val_loader, device=device)  # type: ignore[arg-type]
            val_metrics = {
                "argmax": evaluate_predictions(
                    val_predictions,
                    decoder="argmax",
                    prior=None,
                    min_negative_span=args.min_negative_span,
                    position_prior_weight=0.0,
                ),
                "single_negative_span": evaluate_predictions(
                    val_predictions,
                    decoder="single_negative_span",
                    prior=span_prior,
                    min_negative_span=args.min_negative_span,
                    position_prior_weight=args.position_prior_weight,
                ),
            }
            validation_seconds = time.perf_counter() - validation_start

            selection_score = val_metrics[args.selection_decoder]["f1_macro"]
            if selection_score > best_score:
                best_score = selection_score
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())

        epoch_record = {
            "epoch": epoch,
            "started_at": epoch_started_at,
            "train_loss": train_loss,
            "train_seconds": train_seconds,
            "validation_seconds": validation_seconds,
            "epoch_seconds": time.perf_counter() - epoch_start,
            "validation": val_metrics,
        }
        history.append(epoch_record)
        with history_path.open("a", encoding="utf-8") as history_file:
            history_file.write(json.dumps(epoch_record) + "\n")
        print(json.dumps(epoch_record, indent=2))

    model.load_state_dict(best_state)

    val_predictions = None
    test_predictions = None
    final_evaluation_seconds = 0.0
    if not args.final_train_full_data:
        final_evaluation_start = time.perf_counter()
        val_predictions = collect_predictions(model, val_loader, device=device)  # type: ignore[arg-type]
        test_predictions = collect_predictions(model, test_loader, device=device)  # type: ignore[arg-type]
        final_evaluation_seconds = time.perf_counter() - final_evaluation_start

    ended_at = datetime.now().astimezone().isoformat()
    total_seconds = time.perf_counter() - run_start

    summary = {
        "config": vars(args),
        "input_dim": int(train_sequences[0].embeddings.shape[1]),
        "device": str(device),
        "mode": "final_full_data" if args.final_train_full_data else "held_out_eval",
        "train_speeches": len(train_sequences),
        "validation_speeches": len(val_sequences),
        "test_speeches": len(test_sequences),
        "train_rows": int(sum(sequence.length for sequence in train_sequences)),
        "validation_rows": int(sum(sequence.length for sequence in val_sequences)),
        "test_rows": int(sum(sequence.length for sequence in test_sequences)),
        "best_epoch": best_epoch,
        "best_selection_score": best_score,
        "timing": {
            "started_at": started_at,
            "ended_at": ended_at,
            "total_seconds": total_seconds,
            "train_seconds": float(sum(item["train_seconds"] for item in history)),
            "validation_seconds": float(sum(item["validation_seconds"] for item in history)),
            "final_evaluation_seconds": final_evaluation_seconds,
        },
        "history": history,
        "validation": None,
        "test": None,
    }
    if not args.final_train_full_data:
        summary["validation"] = {
            "argmax": evaluate_predictions(
                val_predictions,  # type: ignore[arg-type]
                decoder="argmax",
                prior=None,
                min_negative_span=args.min_negative_span,
                position_prior_weight=0.0,
            ),
            "single_negative_span": evaluate_predictions(
                val_predictions,  # type: ignore[arg-type]
                decoder="single_negative_span",
                prior=span_prior,
                min_negative_span=args.min_negative_span,
                position_prior_weight=args.position_prior_weight,
            ),
        }
        summary["test"] = {
            "argmax": evaluate_predictions(
                test_predictions,  # type: ignore[arg-type]
                decoder="argmax",
                prior=None,
                min_negative_span=args.min_negative_span,
                position_prior_weight=0.0,
            ),
            "single_negative_span": evaluate_predictions(
                test_predictions,  # type: ignore[arg-type]
                decoder="single_negative_span",
                prior=span_prior,
                min_negative_span=args.min_negative_span,
                position_prior_weight=args.position_prior_weight,
            ),
        }

    checkpoint_path = output_dir / "presidents_lstm.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": vars(args),
            "label_order": PRESIDENT_LABEL_ORDER,
        },
        checkpoint_path,
    )

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(summary, indent=2))

    if not args.final_train_full_data:
        val_predictions_path = output_dir / "validation_predictions.csv"
        predictions_to_frame(
            val_predictions,  # type: ignore[arg-type]
            prior=span_prior,
            min_negative_span=args.min_negative_span,
            position_prior_weight=args.position_prior_weight,
        ).to_csv(val_predictions_path, index=False)

        test_predictions_path = output_dir / "test_predictions.csv"
        predictions_to_frame(
            test_predictions,  # type: ignore[arg-type]
            prior=span_prior,
            min_negative_span=args.min_negative_span,
            position_prior_weight=args.position_prior_weight,
        ).to_csv(test_predictions_path, index=False)

        print(json.dumps(summary["test"], indent=2))
    else:
        print(json.dumps({"checkpoint": str(checkpoint_path), "timing": summary["timing"]}, indent=2))


if __name__ == "__main__":
    main()
