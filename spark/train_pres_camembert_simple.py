import argparse
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
from sklearn.model_selection import KFold, StratifiedKFold
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from rital_nlp_project.presidents.camembert import (
    TokenizedSentenceDataset,
    build_contextual_text_frame,
    load_metadata,
    make_collate_fn,
    slice_encodings,
    tokenize_contextual_texts,
)
from rital_nlp_project.presidents.sequence import PRESIDENT_LABEL_ORDER


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
        description="Train a contextual CamemBERT classifier for the presidents task."
    )
    parser.add_argument(
        "--train-metadata-path",
        default="Dataset/clean/presidents_clean_bert.parquet",
    )
    parser.add_argument(
        "--output-dir",
        default="Dataset/out/presidents_camembert_simple",
    )
    parser.add_argument("--model-name", default="camembert-base")
    parser.add_argument("--context-window", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=default_device())
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def speech_targets(frame):
    grouped = frame.groupby("speech_id", sort=False)["label"].apply(
        lambda labels: int(np.any(labels.to_numpy(dtype=np.int64) == -1))
    )
    return grouped.index.to_numpy(dtype=np.int64), grouped.to_numpy(dtype=np.int64)


def build_cv_splits(frame, n_folds, seed):
    speech_ids, targets = speech_targets(frame)
    bincount = np.bincount(targets, minlength=2)

    if bincount.min() >= n_folds:
        splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        index_splits = list(splitter.split(np.zeros(len(speech_ids)), targets))
    else:
        splitter = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        index_splits = list(splitter.split(np.zeros(len(speech_ids))))

    return [
        (speech_ids[train_idx], speech_ids[test_idx])
        for train_idx, test_idx in index_splits
    ]


def subset_frame_and_encodings(frame, encodings, speech_ids):
    speech_ids = set(int(speech_id) for speech_id in speech_ids)
    mask = frame["speech_id"].isin(speech_ids).to_numpy(dtype=bool)
    subset_frame = frame.loc[mask].reset_index(drop=True)
    subset_encodings = slice_encodings(encodings, mask)
    return subset_frame, subset_encodings


def create_loader(frame, encodings, tokenizer, batch_size, shuffle):
    dataset = TokenizedSentenceDataset(frame, encodings)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=make_collate_fn(tokenizer),
    )


def build_model(model_name, device):
    return AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(PRESIDENT_LABEL_ORDER),
        id2label={0: "Chirac", 1: "Mitterrand"},
        label2id={"Chirac": 0, "Mitterrand": 1},
    ).to(device)


def compute_class_weights(frame):
    labels = frame["label"].to_numpy(dtype=np.int64)
    counts = np.array(
        [(labels == label).sum() for label in PRESIDENT_LABEL_ORDER],
        dtype=np.float64,
    )
    if np.any(counts == 0):
        raise ValueError(f"Every label must appear at least once, got counts={counts}")
    weights = counts.sum() / (len(counts) * counts)
    return torch.tensor(weights, dtype=torch.float32)


def train_model(frame, encodings, tokenizer, args, device, seed):
    set_seed(seed)
    model = build_model(args.model_name, device)
    dataloader = create_loader(
        frame,
        encodings,
        tokenizer,
        args.train_batch_size,
        shuffle=True,
    )
    class_weights = compute_class_weights(frame).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    total_steps = max(len(dataloader) * args.epochs, 1)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    history = []
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        total_loss = 0.0
        total_items = 0

        for batch in dataloader:
            labels = batch["labels"].to(device)
            model_inputs = {
                key: value.to(device)
                for key, value in batch.items()
                if key in {"input_ids", "attention_mask", "token_type_ids"}
            }

            optimizer.zero_grad()
            logits = model(**model_inputs).logits
            loss = criterion(logits, labels)
            loss.backward()
            clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step()
            scheduler.step()

            total_loss += float(loss.item()) * labels.size(0)
            total_items += int(labels.size(0))

        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / max(total_items, 1),
                "epoch_seconds": time.perf_counter() - epoch_start,
            }
        )

    return model, class_weights.cpu(), history


def collect_predictions(model, dataloader, device):
    rows = []
    model.eval()

    with torch.no_grad():
        for batch in dataloader:
            model_inputs = {
                key: value.to(device)
                for key, value in batch.items()
                if key in {"input_ids", "attention_mask", "token_type_ids"}
            }
            probabilities = torch.softmax(model(**model_inputs).logits, dim=-1).cpu().numpy()

            frame = pd.DataFrame(
                {
                    "row_index": batch["row_indices"].cpu().numpy(),
                    "speech_id": batch["speech_ids"].cpu().numpy(),
                    "sentence_id": batch["sentence_ids"].cpu().numpy(),
                    "prob_mitterrand_raw": probabilities[:, NEGATIVE_INDEX].copy(),
                    "text": list(batch["texts"]),
                }
            )
            if "labels" in batch:
                frame["true_label"] = INDEX_TO_LABEL[batch["labels"].cpu().numpy()].copy()
            rows.append(frame)

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


def main():
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    started_at = datetime.now().astimezone().isoformat()
    run_start = time.perf_counter()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    metadata = load_metadata(args.train_metadata_path, require_labels=True)
    frame = build_contextual_text_frame(
        metadata,
        context_window=args.context_window,
        sep_token=tokenizer.sep_token or "</s>",
    )
    encodings = tokenize_contextual_texts(tokenizer, frame, args.max_length)

    oof_frames = []
    fold_summaries = []
    fold_histories = []

    for fold_idx, (train_speech_ids, test_speech_ids) in enumerate(
        build_cv_splits(frame, args.n_folds, args.seed),
        start=1,
    ):
        fold_start = time.perf_counter()
        train_frame, train_encodings = subset_frame_and_encodings(
            frame,
            encodings,
            train_speech_ids,
        )
        test_frame, test_encodings = subset_frame_and_encodings(
            frame,
            encodings,
            test_speech_ids,
        )

        model, _, history = train_model(
            train_frame,
            train_encodings,
            tokenizer,
            args,
            device,
            args.seed + fold_idx,
        )
        fold_loader = create_loader(
            test_frame,
            test_encodings,
            tokenizer,
            args.eval_batch_size,
            shuffle=False,
        )
        fold_frame = collect_predictions(model, fold_loader, device)
        fold_frame.insert(0, "fold", fold_idx)
        oof_frames.append(fold_frame)
        fold_summaries.append(
            {
                "fold": fold_idx,
                "train_speeches": int(len(train_speech_ids)),
                "test_speeches": int(len(test_speech_ids)),
                "fold_seconds": time.perf_counter() - fold_start,
            }
        )
        fold_histories.append({"fold": fold_idx, "history": history})

    oof_frame = pd.concat(oof_frames, ignore_index=True).sort_values("row_index")
    oof_path = output_dir / "oof_predictions.csv"
    oof_frame.drop(columns=["row_index", "text"]).to_csv(oof_path, index=False)

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
    }
    calibration_path = output_dir / "calibration.json"
    calibration_path.write_text(json.dumps(calibration, indent=2))

    final_model, final_class_weights, final_history = train_model(
        frame,
        encodings,
        tokenizer,
        args,
        device,
        args.seed + 100,
    )
    model_dir = output_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    final_model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    model_config = {
        "model_name": args.model_name,
        "context_window": args.context_window,
        "max_length": args.max_length,
        "train_batch_size": args.train_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "gradient_clip": args.gradient_clip,
        "n_folds": args.n_folds,
        "seed": args.seed,
    }
    model_config_path = output_dir / "model_config.json"
    model_config_path.write_text(json.dumps(model_config, indent=2))

    metrics = {
        "started_at": started_at,
        "ended_at": datetime.now().astimezone().isoformat(),
        "total_seconds": time.perf_counter() - run_start,
        "config": {
            "train_metadata_path": args.train_metadata_path,
            "output_dir": args.output_dir,
            "device": args.device,
            **model_config,
        },
        "num_rows": int(len(frame)),
        "num_speeches": int(frame["speech_id"].nunique()),
        "folds": fold_summaries,
        "fold_histories": fold_histories,
        "final_history": final_history,
        "class_weights": {
            "chirac": float(final_class_weights[0].item()),
            "mitterrand": float(final_class_weights[1].item()),
        },
        "calibration_threshold_range": {
            "start": CALIBRATION_THRESHOLD_START,
            "end": CALIBRATION_THRESHOLD_END,
            "step": CALIBRATION_THRESHOLD_STEP,
        },
        "oof_metrics_at_0p5": oof_metrics_0p5,
        "oof_metrics_at_best_threshold": oof_metrics_best,
        "recommended_calibration": calibration,
        "outputs": {
            "model_dir": str(model_dir.resolve()),
            "model_config": str(model_config_path.resolve()),
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
