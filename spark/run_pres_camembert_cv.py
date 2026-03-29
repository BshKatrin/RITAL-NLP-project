from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)

from rital_nlp_project.presidents.eval_utils import (
    apply_score_delta,
    build_grouped_splits,
    compute_boundary_diagnostics,
    speech_target_frame,
    sweep_thresholds,
    threshold_grid,
    threshold_metrics,
)


SUBMISSION_LABELS = {1: "M", 0: "C"}


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
            "Train a contextual CamemBERT sentence classifier for the presidents task "
            "with grouped speech-level CV, OOF threshold calibration, and final test prediction."
        )
    )
    parser.add_argument(
        "--train-metadata-path",
        default="Dataset/clean/presidents_clean_bert.parquet",
    )
    parser.add_argument(
        "--test-metadata-path",
        default="Dataset/clean/presidents_test_clean_bert.parquet",
    )
    parser.add_argument(
        "--proxy-labels-path",
        default=None,
    )
    parser.add_argument(
        "--model-name",
        default="camembert-base",
    )
    parser.add_argument(
        "--output-dir",
        default="Dataset/out/presidents_camembert_cv",
    )
    parser.add_argument("--context-window", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
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


def maybe_slice_speeches(frame: pd.DataFrame, limit: int | None) -> pd.DataFrame:
    if limit is None:
        return frame
    speech_ids = frame["speech_id"].drop_duplicates().iloc[:limit].tolist()
    return frame.loc[frame["speech_id"].isin(speech_ids)].reset_index(drop=True)


def label_to_class_id(label: int) -> int:
    if int(label) == -1:
        return 1
    if int(label) == 1:
        return 0
    raise ValueError(f"Unsupported label value: {label}")


def class_id_to_label(class_id: int) -> int:
    if int(class_id) == 1:
        return -1
    if int(class_id) == 0:
        return 1
    raise ValueError(f"Unsupported class id: {class_id}")


def build_contextual_texts(
    frame: pd.DataFrame,
    *,
    text_column: str,
    context_window: int,
    sep_token: str,
) -> pd.Series:
    if context_window < 0:
        raise ValueError("context_window must be >= 0")

    contextual_texts = np.empty(len(frame), dtype=object)
    ordered = frame.sort_values(["speech_id", "sentence_id"]).reset_index()

    for _, group in ordered.groupby("speech_id", sort=False):
        texts = group[text_column].astype(str).tolist()
        original_indices = group["index"].to_numpy(dtype=np.int64)
        for position, original_index in enumerate(original_indices):
            start = max(0, position - context_window)
            end = min(len(texts), position + context_window + 1)
            contextual_texts[original_index] = f" {sep_token} ".join(texts[start:end])

    return pd.Series(contextual_texts, index=frame.index, name="text_with_context")


@dataclass(frozen=True)
class EncodedRow:
    speech_id: int
    sentence_id: int
    input_ids: list[int]
    attention_mask: list[int]
    label: int | None


class PresidentTextDataset(Dataset):
    def __init__(self, rows: list[EncodedRow]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        item: dict[str, Any] = {
            "speech_id": row.speech_id,
            "sentence_id": row.sentence_id,
            "input_ids": row.input_ids,
            "attention_mask": row.attention_mask,
        }
        if row.label is not None:
            item["labels"] = row.label
        return item


class PresidentBatchCollator:
    def __init__(self, tokenizer) -> None:
        self.inner = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        model_features = []
        speech_ids = []
        sentence_ids = []
        for feature in features:
            model_feature = {
                "input_ids": feature["input_ids"],
                "attention_mask": feature["attention_mask"],
            }
            if "labels" in feature:
                model_feature["labels"] = feature["labels"]
            model_features.append(model_feature)
            speech_ids.append(int(feature["speech_id"]))
            sentence_ids.append(int(feature["sentence_id"]))

        batch = self.inner(model_features)
        batch["speech_id"] = torch.tensor(speech_ids, dtype=torch.long)
        batch["sentence_id"] = torch.tensor(sentence_ids, dtype=torch.long)
        return batch


def encode_frame(
    frame: pd.DataFrame,
    *,
    tokenizer,
    text_column: str,
    max_length: int,
    include_labels: bool,
) -> PresidentTextDataset:
    encodings = tokenizer(
        frame[text_column].astype(str).tolist(),
        truncation=True,
        max_length=max_length,
        padding=False,
    )

    labels = None
    if include_labels:
        labels = [label_to_class_id(label) for label in frame["label"].tolist()]

    rows = []
    for index in range(len(frame)):
        rows.append(
            EncodedRow(
                speech_id=int(frame.iloc[index]["speech_id"]),
                sentence_id=int(frame.iloc[index]["sentence_id"]),
                input_ids=list(encodings["input_ids"][index]),
                attention_mask=list(encodings["attention_mask"][index]),
                label=None if labels is None else int(labels[index]),
            )
        )
    return PresidentTextDataset(rows)


def create_loader(
    dataset: PresidentTextDataset,
    *,
    tokenizer,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=PresidentBatchCollator(tokenizer),
    )


def compute_class_weights(class_ids: np.ndarray) -> torch.Tensor:
    counts = np.bincount(class_ids, minlength=2).astype(np.float64)
    if np.any(counts == 0):
        raise ValueError(f"Every class must appear at least once, got counts={counts}")
    weights = counts.sum() / (len(counts) * counts)
    return torch.tensor(weights, dtype=torch.float32)


def build_model(model_name: str, *, device: torch.device):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
    )
    return model.to(device)


def train_one_epoch(
    model,
    dataloader: DataLoader,
    *,
    optimizer: AdamW,
    scheduler,
    device: torch.device,
    class_weights: torch.Tensor,
    gradient_clip: float,
) -> float:
    model.train()
    total_loss = 0.0
    total_items = 0

    for batch in dataloader:
        labels = batch["labels"].to(device)
        inputs = {
            "input_ids": batch["input_ids"].to(device),
            "attention_mask": batch["attention_mask"].to(device),
        }

        optimizer.zero_grad()
        outputs = model(**inputs)
        loss = F.cross_entropy(outputs.logits, labels, weight=class_weights)
        loss.backward()
        clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()
        scheduler.step()

        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_items += batch_size

    return total_loss / max(total_items, 1)


def predict_frame(
    model,
    dataloader: DataLoader,
    *,
    device: torch.device,
    include_labels: bool,
    sort_output: bool = True,
) -> pd.DataFrame:
    model.eval()
    rows = []

    with torch.no_grad():
        for batch in dataloader:
            inputs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
            }
            outputs = model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
            labels = batch.get("labels")

            for index in range(probabilities.shape[0]):
                row: dict[str, Any] = {
                    "speech_id": int(batch["speech_id"][index].item()),
                    "sentence_id": int(batch["sentence_id"][index].item()),
                    "prob_mitterrand_raw": float(probabilities[index, 1]),
                }
                if include_labels and labels is not None:
                    row["true_label"] = class_id_to_label(int(labels[index].item()))
                rows.append(row)

    frame = pd.DataFrame(rows)
    if sort_output:
        frame = frame.sort_values(["speech_id", "sentence_id"]).reset_index(drop=True)
    return frame


def train_model(
    train_frame: pd.DataFrame,
    *,
    tokenizer,
    args: argparse.Namespace,
    seed: int,
    device: torch.device,
) -> tuple[Any, list[dict[str, Any]]]:
    set_seed(seed)
    train_dataset = encode_frame(
        train_frame,
        tokenizer=tokenizer,
        text_column="text_with_context",
        max_length=args.max_length,
        include_labels=True,
    )
    train_loader = create_loader(
        train_dataset,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        shuffle=True,
    )
    model = build_model(args.model_name, device=device)

    class_weights = compute_class_weights(
        np.asarray(
            [label_to_class_id(label) for label in train_frame["label"].tolist()],
            dtype=np.int64,
        )
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    total_steps = max(len(train_loader) * args.epochs, 1)
    warmup_steps = int(math.ceil(total_steps * args.warmup_ratio))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    history = []
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            class_weights=class_weights,
            gradient_clip=args.gradient_clip,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "epoch_seconds": time.perf_counter() - epoch_start,
            }
        )

    return model, history


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    started_at = datetime.now().astimezone().isoformat()
    run_start = time.perf_counter()

    train_frame = (
        pd.read_parquet(args.train_metadata_path)
        .sort_values(["speech_id", "sentence_id"])
        .reset_index(drop=True)
    )
    test_frame = pd.read_parquet(args.test_metadata_path).reset_index(drop=True)
    train_frame = maybe_slice_speeches(train_frame, args.max_speeches)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    sep_token = tokenizer.sep_token or "</s>"

    train_frame["text_with_context"] = build_contextual_texts(
        train_frame,
        text_column="text",
        context_window=args.context_window,
        sep_token=sep_token,
    )
    test_frame["text_with_context"] = build_contextual_texts(
        test_frame,
        text_column="text",
        context_window=args.context_window,
        sep_token=sep_token,
    )

    grouped_targets = speech_target_frame(train_frame, label_column="label")
    folds = build_grouped_splits(
        grouped_targets,
        n_splits=args.n_folds,
        seed=args.seed,
    )

    fold_records = []
    oof_frames = []
    all_history = []

    for fold_idx, (train_idx, test_idx) in enumerate(folds, start=1):
        train_speech_ids = set(
            grouped_targets.iloc[train_idx]["speech_id"].astype(int).tolist()
        )
        test_speech_ids = set(
            grouped_targets.iloc[test_idx]["speech_id"].astype(int).tolist()
        )

        fold_train_frame = train_frame.loc[
            train_frame["speech_id"].isin(train_speech_ids)
        ].reset_index(drop=True)
        fold_test_frame = train_frame.loc[
            train_frame["speech_id"].isin(test_speech_ids)
        ].reset_index(drop=True)

        fold_start = time.perf_counter()
        model, history = train_model(
            fold_train_frame,
            tokenizer=tokenizer,
            args=args,
            seed=args.seed + 1000 + fold_idx,
            device=device,
        )
        all_history.append(
            {
                "fold": fold_idx,
                "history": history,
            }
        )

        fold_test_dataset = encode_frame(
            fold_test_frame,
            tokenizer=tokenizer,
            text_column="text_with_context",
            max_length=args.max_length,
            include_labels=True,
        )
        fold_test_loader = create_loader(
            fold_test_dataset,
            tokenizer=tokenizer,
            batch_size=args.eval_batch_size,
            shuffle=False,
        )
        fold_predictions = predict_frame(
            model,
            fold_test_loader,
            device=device,
            include_labels=True,
        )
        fold_predictions.insert(0, "fold", fold_idx)
        oof_frames.append(fold_predictions)

        fold_records.append(
            {
                "fold": fold_idx,
                "train_speeches": int(fold_train_frame["speech_id"].nunique()),
                "test_speeches": int(fold_test_frame["speech_id"].nunique()),
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
        score_name="prob_mitterrand_raw",
        thresholds=thresholds,
    )
    threshold_sweep_path = output_dir / "threshold_sweep.csv"
    threshold_sweep.to_csv(threshold_sweep_path, index=False)
    best_threshold = float(best_threshold_record["threshold"])

    default_oof_metrics = threshold_metrics(
        oof_frame["true_label"].to_numpy(dtype=np.int64),
        oof_frame["prob_mitterrand_raw"].to_numpy(dtype=np.float64),
        threshold=0.5,
    )
    default_oof_metrics["mitterrand_count"] = int(
        (oof_frame["prob_mitterrand_raw"].to_numpy(dtype=np.float64) >= 0.5).sum()
    )
    default_oof_metrics["chirac_count"] = int(
        (oof_frame["prob_mitterrand_raw"].to_numpy(dtype=np.float64) < 0.5).sum()
    )

    oof_boundary_default, oof_speech_metrics_default = compute_boundary_diagnostics(
        oof_frame,
        threshold=0.5,
        score_column="prob_mitterrand_raw",
    )
    oof_boundary_tuned, oof_speech_metrics_tuned = compute_boundary_diagnostics(
        oof_frame,
        threshold=best_threshold,
        score_column="prob_mitterrand_raw",
    )
    oof_speech_metrics_default_path = output_dir / "oof_speech_metrics_at_0p5.csv"
    oof_speech_metrics_tuned_path = output_dir / "oof_speech_metrics_at_best_threshold.csv"
    oof_speech_metrics_default.to_csv(oof_speech_metrics_default_path, index=False)
    oof_speech_metrics_tuned.to_csv(oof_speech_metrics_tuned_path, index=False)

    final_model_start = time.perf_counter()
    final_model, final_history = train_model(
        train_frame,
        tokenizer=tokenizer,
        args=args,
        seed=args.seed + 9999,
        device=device,
    )
    final_model_seconds = time.perf_counter() - final_model_start

    model_dir = output_dir / "final_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    final_model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    test_dataset = encode_frame(
        test_frame,
        tokenizer=tokenizer,
        text_column="text_with_context",
        max_length=args.max_length,
        include_labels=False,
    )
    test_loader = create_loader(
        test_dataset,
        tokenizer=tokenizer,
        batch_size=args.eval_batch_size,
        shuffle=False,
    )
    test_predictions = predict_frame(
        final_model,
        test_loader,
        device=device,
        include_labels=False,
        sort_output=False,
    )
    if not test_predictions[["speech_id", "sentence_id"]].equals(
        test_frame[["speech_id", "sentence_id"]].reset_index(drop=True)
    ):
        raise ValueError("Final test predictions are not aligned with the test metadata")

    delta = float(best_threshold_record["equivalent_score_delta"])
    test_predictions["prob_mitterrand_calibrated"] = apply_score_delta(
        test_predictions["prob_mitterrand_raw"].to_numpy(dtype=np.float64),
        delta,
    )
    test_predictions["pred_label_calibrated"] = np.where(
        test_predictions["prob_mitterrand_calibrated"].to_numpy(dtype=np.float64) >= 0.5,
        "M",
        "C",
    )
    test_predictions["text"] = test_frame["text"].to_numpy()

    detailed_path = output_dir / "test_predictions_detailed.csv"
    test_predictions.to_csv(detailed_path, index=False)
    raw_prob_path = output_dir / "submission_prob_mitterrand_raw.csv"
    calibrated_prob_path = output_dir / "submission_prob_mitterrand_calibrated.csv"
    calibrated_label_path = output_dir / "submission_label_calibrated.csv"
    test_predictions[["prob_mitterrand_raw"]].to_csv(
        raw_prob_path,
        index=False,
        header=False,
    )
    test_predictions[["prob_mitterrand_calibrated"]].to_csv(
        calibrated_prob_path,
        index=False,
        header=False,
    )
    test_predictions[["pred_label_calibrated"]].to_csv(
        calibrated_label_path,
        index=False,
        header=False,
    )

    calibration = {
        "selected_score_name": "prob_mitterrand_raw",
        "threshold": best_threshold,
        "best_threshold": best_threshold,
        "delta": delta,
        "equivalent_score_delta": delta,
        "oof_metrics_at_0p5": default_oof_metrics,
        "oof_metrics_at_best_threshold": {
            "threshold": best_threshold,
            "accuracy": float(best_threshold_record["accuracy"]),
            "balanced_accuracy": float(best_threshold_record["balanced_accuracy"]),
            "f1_macro": float(best_threshold_record["f1_macro"]),
            "mitterrand_count": int(best_threshold_record["mitterrand_count"]),
            "chirac_count": int(best_threshold_record["chirac_count"]),
        },
        "oof_boundary_diagnostics_at_0p5": oof_boundary_default,
        "oof_boundary_diagnostics_at_best_threshold": oof_boundary_tuned,
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
        "num_sequences": int(train_frame["speech_id"].nunique()),
        "num_rows": int(len(train_frame)),
        "folds": fold_records,
        "history": all_history,
        "final_history": final_history,
        "default_threshold_metrics": {
            "selected_score_name": "prob_mitterrand_raw",
            **default_oof_metrics,
        },
        "recommended_calibration": calibration,
        "timing": {
            "final_model_seconds": final_model_seconds,
        },
        "outputs": {
            "model_dir": str(model_dir.resolve()),
            "calibration": str(calibration_path.resolve()),
            "oof_predictions": str(oof_path.resolve()),
            "threshold_sweep": str(threshold_sweep_path.resolve()),
            "oof_speech_metrics_at_0p5": str(oof_speech_metrics_default_path.resolve()),
            "oof_speech_metrics_at_best_threshold": str(
                oof_speech_metrics_tuned_path.resolve()
            ),
            "detailed_predictions": str(detailed_path.resolve()),
            "submission_prob_mitterrand_raw": str(raw_prob_path.resolve()),
            "submission_prob_mitterrand_calibrated": str(calibrated_prob_path.resolve()),
            "submission_label_calibrated": str(calibrated_label_path.resolve()),
        },
    }

    if args.proxy_labels_path is not None:
        proxy_output_dir = output_dir / "proxy_eval"
        proxy_output_dir.mkdir(parents=True, exist_ok=True)
        import subprocess
        import sys

        command = [
            sys.executable,
            "scripts/eval_presidents_proxy.py",
            "--prediction-path",
            str(calibrated_label_path),
            "--score-path",
            str(calibrated_prob_path),
            "--proxy-labels-path",
            args.proxy_labels_path,
            "--output-dir",
            str(proxy_output_dir),
        ]
        subprocess.run(command, cwd=Path(__file__).resolve().parents[1], check=True)
        summary["proxy_eval"] = json.loads((proxy_output_dir / "metrics.json").read_text())

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["recommended_calibration"], indent=2))


if __name__ == "__main__":
    main()
