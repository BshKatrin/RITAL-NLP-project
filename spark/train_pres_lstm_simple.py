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

from rital_nlp_project.presidents.lstm import BiLSTMSequenceTagger
from rital_nlp_project.presidents.sequence import (
    PAD_LABEL_INDEX,
    PRESIDENT_LABEL_ORDER,
    SpeechSequenceDataset,
    build_speech_sequences,
    collate_speech_sequences,
    compute_class_weights,
    load_embeddings_with_metadata,
)


VALIDATION_THRESHOLD_START = 0.80
VALIDATION_THRESHOLD_END = 0.95
VALIDATION_THRESHOLD_STEP = 0.005
CALIBRATION_THRESHOLD_START = 0.05
CALIBRATION_THRESHOLD_END = 0.95
CALIBRATION_THRESHOLD_STEP = 0.005

INDEX_TO_LABEL = np.array(PRESIDENT_LABEL_ORDER, dtype=np.int64)
NEGATIVE_INDEX = int(np.where(INDEX_TO_LABEL == -1)[0][0])


def default_device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a speech-level BiLSTM on president sentence embeddings."
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
        default="Dataset/out/presidents_lstm_simple",
    )
    parser.add_argument("--validation-size", type=float, default=0.15)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--projection-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--minority-weight-scale", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=default_device())
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def speech_targets(sequences):
    return np.array(
        [int(sequence.labels is not None and np.any(sequence.labels == -1)) for sequence in sequences],
        dtype=np.int64,
    )


def stratified_train_validation_split(sequences, validation_size, seed):
    if not 0.0 < validation_size < 1.0:
        raise ValueError("validation-size must be between 0 and 1")

    targets = speech_targets(sequences)
    bincount = np.bincount(targets, minlength=2)

    if bincount.min() >= 2:
        splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=validation_size,
            random_state=seed,
        )
        train_idx, validation_idx = next(splitter.split(np.zeros(len(sequences)), targets))
    else:
        rng = np.random.default_rng(seed)
        indices = rng.permutation(np.arange(len(sequences)))
        validation_size_int = max(1, int(round(len(sequences) * validation_size)))
        validation_idx = np.sort(indices[:validation_size_int])
        train_idx = np.sort(indices[validation_size_int:])

    return (
        [sequences[idx] for idx in train_idx],
        [sequences[idx] for idx in validation_idx],
    )


def build_cv_splits(sequences, n_folds, seed):
    targets = speech_targets(sequences)
    bincount = np.bincount(targets, minlength=2)

    if bincount.min() >= n_folds:
        splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        return list(splitter.split(np.zeros(len(sequences)), targets))

    splitter = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return list(splitter.split(np.zeros(len(sequences))))


def create_loader(sequences, batch_size, shuffle):
    dataset = SpeechSequenceDataset(sequences)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_speech_sequences,
    )


def build_model(args, input_dim, device):
    return BiLSTMSequenceTagger(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        projection_dim=args.projection_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        num_labels=len(PRESIDENT_LABEL_ORDER),
    ).to(device)


def build_loss_weights(sequences, minority_weight_scale):
    weights = compute_class_weights(sequences).float()
    weights[NEGATIVE_INDEX] *= float(minority_weight_scale)
    return weights


def train_one_epoch(model, dataloader, optimizer, criterion, device, gradient_clip):
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

        total_loss += float(loss.item()) * inputs.size(0)
        total_items += inputs.size(0)

    return total_loss / max(total_items, 1)


def collect_predictions(model, dataloader, device):
    rows = []
    model.eval()

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch["inputs"].to(device)
            lengths = batch["lengths"]
            logits = model(inputs, lengths)
            probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
            labels = batch["labels"].cpu().numpy()
            sentence_ids = batch["sentence_ids"].cpu().numpy()
            speech_ids = batch["speech_ids"].cpu().tolist()

            for row_idx, length in enumerate(lengths.tolist()):
                label_indices = labels[row_idx, :length].copy()
                rows.append(
                    pd.DataFrame(
                        {
                            "speech_id": int(speech_ids[row_idx]),
                            "sentence_id": sentence_ids[row_idx, :length].copy(),
                            "true_label": INDEX_TO_LABEL[label_indices].copy(),
                            "prob_mitterrand_raw": probabilities[row_idx, :length, NEGATIVE_INDEX].copy(),
                        }
                    )
                )

    return pd.concat(rows, ignore_index=True)


def threshold_grid(start, end, step):
    values = np.arange(start, end + step / 2.0, step, dtype=np.float64)
    return np.clip(values, 1e-6, 1.0 - 1e-6)


def evaluate_threshold(y_true, scores, threshold):
    y_pred = np.where(scores >= threshold, -1, 1)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "mitterrand_count": int((y_pred == -1).sum()),
        "chirac_count": int((y_pred == 1).sum()),
    }


def sweep_thresholds(frame, thresholds):
    y_true = frame["true_label"].to_numpy(dtype=np.int64)
    scores = frame["prob_mitterrand_raw"].to_numpy(dtype=np.float64)
    records = []

    for threshold in thresholds:
        record = evaluate_threshold(y_true, scores, float(threshold))
        record["threshold"] = float(threshold)
        record["equivalent_score_delta"] = float(logit(float(threshold)))
        records.append(record)

    sweep_frame = pd.DataFrame(records).sort_values(
        ["f1_macro", "balanced_accuracy", "threshold"],
        ascending=[False, False, True],
    )
    return pd.DataFrame(records), sweep_frame.iloc[0].to_dict()


def selection_key(record):
    return (
        float(record["f1_macro"]),
        float(record["balanced_accuracy"]),
        -float(record["threshold"]),
    )


def select_best_epoch(train_sequences, validation_sequences, args, device):
    train_loader = create_loader(train_sequences, args.batch_size, shuffle=True)
    validation_loader = create_loader(validation_sequences, args.batch_size, shuffle=False)

    model = build_model(args, train_sequences[0].embeddings.shape[1], device)
    loss_weights = build_loss_weights(train_sequences, args.minority_weight_scale)
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    criterion = nn.CrossEntropyLoss(
        weight=loss_weights.to(device),
        ignore_index=PAD_LABEL_INDEX,
    )

    thresholds = threshold_grid(
        VALIDATION_THRESHOLD_START,
        VALIDATION_THRESHOLD_END,
        VALIDATION_THRESHOLD_STEP,
    )

    best_epoch = 1
    best_record = None
    best_predictions = None
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(1, args.epochs + 1):
        train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            args.gradient_clip,
        )
        validation_frame = collect_predictions(model, validation_loader, device)
        _, record = sweep_thresholds(validation_frame, thresholds)

        if best_record is None or selection_key(record) > selection_key(best_record):
            best_epoch = epoch
            best_record = record
            best_predictions = validation_frame
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return {
        "best_epoch": best_epoch,
        "validation_selection": best_record,
        "validation_frame": best_predictions,
        "class_weights": {
            "chirac": float(loss_weights[0].item()),
            "mitterrand": float(loss_weights[1].item()),
        },
    }


def train_for_epochs(sequences, args, device, epochs, seed):
    set_seed(seed)
    loader = create_loader(sequences, args.batch_size, shuffle=True)
    model = build_model(args, sequences[0].embeddings.shape[1], device)
    loss_weights = build_loss_weights(sequences, args.minority_weight_scale)
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    criterion = nn.CrossEntropyLoss(
        weight=loss_weights.to(device),
        ignore_index=PAD_LABEL_INDEX,
    )

    for _ in range(epochs):
        train_one_epoch(
            model,
            loader,
            optimizer,
            criterion,
            device,
            args.gradient_clip,
        )

    return model, loss_weights


def main():
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    started_at = datetime.now().astimezone().isoformat()
    run_start = time.perf_counter()

    embeddings, metadata = load_embeddings_with_metadata(args.embeddings_path, args.metadata_path)
    sequences = build_speech_sequences(embeddings, metadata)

    train_sequences, validation_sequences = stratified_train_validation_split(
        sequences,
        args.validation_size,
        args.seed,
    )
    selection = select_best_epoch(train_sequences, validation_sequences, args, device)
    best_epoch = int(selection["best_epoch"])

    validation_frame = selection["validation_frame"]
    validation_metrics_0p5 = evaluate_threshold(
        validation_frame["true_label"].to_numpy(dtype=np.int64),
        validation_frame["prob_mitterrand_raw"].to_numpy(dtype=np.float64),
        0.5,
    )

    oof_frames = []
    fold_summaries = []
    for fold_idx, (train_idx, test_idx) in enumerate(build_cv_splits(sequences, args.n_folds, args.seed), start=1):
        fold_train_sequences = [sequences[idx] for idx in train_idx]
        fold_test_sequences = [sequences[idx] for idx in test_idx]
        model, _ = train_for_epochs(
            fold_train_sequences,
            args,
            device,
            best_epoch,
            args.seed + fold_idx,
        )
        fold_loader = create_loader(fold_test_sequences, args.batch_size, shuffle=False)
        fold_frame = collect_predictions(model, fold_loader, device)
        fold_frame.insert(0, "fold", fold_idx)
        oof_frames.append(fold_frame)
        fold_summaries.append(
            {
                "fold": fold_idx,
                "train_speeches": len(fold_train_sequences),
                "test_speeches": len(fold_test_sequences),
            }
        )

    oof_frame = pd.concat(oof_frames, ignore_index=True).sort_values(
        ["speech_id", "sentence_id"]
    )
    oof_path = output_dir / "oof_predictions.csv"
    oof_frame.to_csv(oof_path, index=False)

    threshold_sweep, best_threshold_record = sweep_thresholds(
        oof_frame,
        threshold_grid(
            CALIBRATION_THRESHOLD_START,
            CALIBRATION_THRESHOLD_END,
            CALIBRATION_THRESHOLD_STEP,
        ),
    )
    threshold_sweep_path = output_dir / "threshold_sweep.csv"
    threshold_sweep.to_csv(threshold_sweep_path, index=False)

    best_threshold = float(best_threshold_record["threshold"])
    oof_metrics_0p5 = evaluate_threshold(
        oof_frame["true_label"].to_numpy(dtype=np.int64),
        oof_frame["prob_mitterrand_raw"].to_numpy(dtype=np.float64),
        0.5,
    )
    oof_metrics_best = evaluate_threshold(
        oof_frame["true_label"].to_numpy(dtype=np.int64),
        oof_frame["prob_mitterrand_raw"].to_numpy(dtype=np.float64),
        best_threshold,
    )

    calibration = {
        "threshold": best_threshold,
        "equivalent_score_delta": float(logit(best_threshold)),
        "best_epoch": best_epoch,
    }
    calibration_path = output_dir / "calibration.json"
    calibration_path.write_text(json.dumps(calibration, indent=2))

    final_model, final_loss_weights = train_for_epochs(
        sequences,
        args,
        device,
        best_epoch,
        args.seed + 100,
    )
    checkpoint = {
        "model_state_dict": final_model.state_dict(),
        "label_order": list(PRESIDENT_LABEL_ORDER),
        "config": {
            "hidden_dim": args.hidden_dim,
            "projection_dim": args.projection_dim,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "batch_size": args.batch_size,
        },
    }
    checkpoint_path = output_dir / "checkpoint.pt"
    torch.save(checkpoint, checkpoint_path)

    metrics = {
        "started_at": started_at,
        "ended_at": datetime.now().astimezone().isoformat(),
        "total_seconds": time.perf_counter() - run_start,
        "config": {
            "embeddings_path": args.embeddings_path,
            "metadata_path": args.metadata_path,
            "validation_size": args.validation_size,
            "n_folds": args.n_folds,
            "hidden_dim": args.hidden_dim,
            "projection_dim": args.projection_dim,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "gradient_clip": args.gradient_clip,
            "minority_weight_scale": args.minority_weight_scale,
            "seed": args.seed,
            "device": args.device,
        },
        "best_epoch": best_epoch,
        "validation_threshold_range": {
            "start": VALIDATION_THRESHOLD_START,
            "end": VALIDATION_THRESHOLD_END,
            "step": VALIDATION_THRESHOLD_STEP,
        },
        "calibration_threshold_range": {
            "start": CALIBRATION_THRESHOLD_START,
            "end": CALIBRATION_THRESHOLD_END,
            "step": CALIBRATION_THRESHOLD_STEP,
        },
        "train_speeches": len(train_sequences),
        "validation_speeches": len(validation_sequences),
        "folds": fold_summaries,
        "class_weights": {
            "selection_split": selection["class_weights"],
            "full_train": {
                "chirac": float(final_loss_weights[0].item()),
                "mitterrand": float(final_loss_weights[1].item()),
            },
        },
        "validation_metrics_at_0p5": validation_metrics_0p5,
        "validation_selection": selection["validation_selection"],
        "oof_metrics_at_0p5": oof_metrics_0p5,
        "oof_metrics_at_best_threshold": oof_metrics_best,
        "outputs": {
            "checkpoint": str(checkpoint_path.resolve()),
            "calibration": str(calibration_path.resolve()),
            "metrics": str((output_dir / "metrics.json").resolve()),
            "oof_predictions": str(oof_path.resolve()),
            "threshold_sweep": str(threshold_sweep_path.resolve()),
        },
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
