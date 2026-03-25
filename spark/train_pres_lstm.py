from __future__ import annotations

import argparse
import copy
import json
import random
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
        "--selection-decoder",
        choices=("argmax", "single_negative_span"),
        default="single_negative_span",
        help="Decoder used for model selection on the validation split.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device used for training.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-speeches", type=int, default=None)
    parser.add_argument("--max-val-speeches", type=int, default=None)
    parser.add_argument("--max-test-speeches", type=int, default=None)
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
) -> dict[str, float]:
    sequence_probabilities = [
        item["probabilities"] for item in predictions  # type: ignore[index]
    ]
    decoded = decode_batch(
        sequence_probabilities,
        decoder=decoder,
        prior=prior,
        min_span_length=min_negative_span,
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

    embeddings, metadata = load_embeddings_with_metadata(
        args.embeddings_path,
        args.metadata_path,
    )
    sequences = build_speech_sequences(embeddings, metadata)

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
    span_prior = fit_single_span_prior(train_sequences)

    train_loader = create_loader(
        train_sequences,
        batch_size=args.batch_size,
        shuffle=True,
    )
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
        input_dim=embeddings.shape[1],
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
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            gradient_clip=args.gradient_clip,
        )
        val_predictions = collect_predictions(model, val_loader, device=device)
        val_metrics = {
            "argmax": evaluate_predictions(
                val_predictions,
                decoder="argmax",
                prior=None,
                min_negative_span=args.min_negative_span,
            ),
            "single_negative_span": evaluate_predictions(
                val_predictions,
                decoder="single_negative_span",
                prior=span_prior,
                min_negative_span=args.min_negative_span,
            ),
        }

        selection_score = val_metrics[args.selection_decoder]["f1_macro"]
        if selection_score > best_score:
            best_score = selection_score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation": val_metrics,
        }
        history.append(epoch_record)
        print(json.dumps(epoch_record, indent=2))

    model.load_state_dict(best_state)

    val_predictions = collect_predictions(model, val_loader, device=device)
    test_predictions = collect_predictions(model, test_loader, device=device)

    summary = {
        "config": vars(args),
        "input_dim": int(embeddings.shape[1]),
        "train_speeches": len(train_sequences),
        "validation_speeches": len(val_sequences),
        "test_speeches": len(test_sequences),
        "best_epoch": best_epoch,
        "history": history,
        "validation": {
            "argmax": evaluate_predictions(
                val_predictions,
                decoder="argmax",
                prior=None,
                min_negative_span=args.min_negative_span,
            ),
            "single_negative_span": evaluate_predictions(
                val_predictions,
                decoder="single_negative_span",
                prior=span_prior,
                min_negative_span=args.min_negative_span,
            ),
        },
        "test": {
            "argmax": evaluate_predictions(
                test_predictions,
                decoder="argmax",
                prior=None,
                min_negative_span=args.min_negative_span,
            ),
            "single_negative_span": evaluate_predictions(
                test_predictions,
                decoder="single_negative_span",
                prior=span_prior,
                min_negative_span=args.min_negative_span,
            ),
        },
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

    val_predictions_path = output_dir / "validation_predictions.csv"
    predictions_to_frame(
        val_predictions,
        prior=span_prior,
        min_negative_span=args.min_negative_span,
    ).to_csv(val_predictions_path, index=False)

    test_predictions_path = output_dir / "test_predictions.csv"
    predictions_to_frame(
        test_predictions,
        prior=span_prior,
        min_negative_span=args.min_negative_span,
    ).to_csv(test_predictions_path, index=False)

    print(json.dumps(summary["test"], indent=2))


if __name__ == "__main__":
    main()
