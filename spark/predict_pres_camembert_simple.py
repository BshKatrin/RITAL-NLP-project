import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import expit, logit
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from rital_nlp_project.presidents.camembert import (
    TokenizedSentenceDataset,
    build_contextual_text_frame,
    load_metadata,
    make_collate_fn,
    tokenize_contextual_texts,
)
from rital_nlp_project.presidents.sequence import PRESIDENT_LABEL_ORDER


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
        description="Run the cleaned presidents CamemBERT classifier on the test set."
    )
    parser.add_argument(
        "--model-dir",
        default="Dataset/out/presidents_camembert_simple/model",
    )
    parser.add_argument(
        "--config-path",
        default="Dataset/out/presidents_camembert_simple/model_config.json",
    )
    parser.add_argument(
        "--calibration-path",
        default="Dataset/out/presidents_camembert_simple/calibration.json",
    )
    parser.add_argument(
        "--test-metadata-path",
        default="Dataset/clean/presidents_test_clean_bert.parquet",
    )
    parser.add_argument(
        "--output-dir",
        default="Dataset/out/presidents_camembert_simple_submission",
    )
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--decision-threshold", type=float, default=0.5)
    parser.add_argument("--device", default=default_device())
    return parser.parse_args()


def create_loader(frame, encodings, tokenizer, batch_size):
    dataset = TokenizedSentenceDataset(frame, encodings)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=make_collate_fn(tokenizer),
    )


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
            rows.append(
                pd.DataFrame(
                    {
                        "row_index": batch["row_indices"].cpu().numpy(),
                        "speech_id": batch["speech_ids"].cpu().numpy(),
                        "sentence_id": batch["sentence_ids"].cpu().numpy(),
                        "prob_mitterrand_raw": probabilities[:, NEGATIVE_INDEX].copy(),
                        "text": list(batch["texts"]),
                    }
                )
            )

    return pd.concat(rows, ignore_index=True)


def apply_score_delta(scores, delta):
    clipped = np.clip(np.asarray(scores, dtype=np.float64), 1e-9, 1.0 - 1e-9)
    return expit(logit(clipped) - float(delta))


def labels_from_scores(scores, threshold):
    return np.where(scores >= threshold, "M", "C")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    started_at = datetime.now().astimezone().isoformat()
    run_start = time.perf_counter()

    model_config = json.loads(Path(args.config_path).read_text())
    calibration = json.loads(Path(args.calibration_path).read_text())
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir).to(args.device)
    device = torch.device(args.device)
    model.to(device)

    metadata = load_metadata(args.test_metadata_path, require_labels=False)
    frame = build_contextual_text_frame(
        metadata,
        context_window=int(model_config["context_window"]),
        sep_token=tokenizer.sep_token or "</s>",
    )
    encodings = tokenize_contextual_texts(
        tokenizer,
        frame,
        int(model_config["max_length"]),
    )

    eval_batch_size = args.eval_batch_size or int(model_config["eval_batch_size"])
    dataloader = create_loader(frame, encodings, tokenizer, eval_batch_size)

    inference_start = time.perf_counter()
    predictions = collect_predictions(model, dataloader, device).sort_values("row_index")
    inference_seconds = time.perf_counter() - inference_start

    expected = metadata[["speech_id", "sentence_id"]].reset_index(drop=True)
    actual = predictions[["speech_id", "sentence_id"]].reset_index(drop=True)
    if not actual.equals(expected):
        raise ValueError("Predictions are not aligned with the test metadata")

    delta = float(calibration["equivalent_score_delta"])
    tuned_raw_threshold = float(calibration["threshold"])
    predictions["prob_mitterrand_calibrated"] = apply_score_delta(
        predictions["prob_mitterrand_raw"].to_numpy(dtype=np.float64),
        delta,
    )
    predictions["pred_label_calibrated"] = labels_from_scores(
        predictions["prob_mitterrand_calibrated"].to_numpy(dtype=np.float64),
        float(args.decision_threshold),
    )
    aligned_labels = labels_from_scores(
        predictions["prob_mitterrand_calibrated"].to_numpy(dtype=np.float64),
        0.5,
    )
    if not np.array_equal(
        aligned_labels,
        predictions["pred_label_calibrated"].to_numpy(),
    ):
        raise ValueError(
            "submission labels must be reproducible from the calibrated probability "
            "file using a 0.5 threshold"
        )

    detailed_path = output_dir / "test_predictions_detailed.csv"
    predictions.drop(columns=["row_index"]).to_csv(detailed_path, index=False)

    raw_prob_path = output_dir / "submission_prob_mitterrand_raw.csv"
    predictions[["prob_mitterrand_raw"]].to_csv(raw_prob_path, index=False, header=False)

    calibrated_prob_path = output_dir / "submission_prob_mitterrand_calibrated.csv"
    predictions[["prob_mitterrand_calibrated"]].to_csv(
        calibrated_prob_path,
        index=False,
        header=False,
    )

    submission_probability_path = output_dir / "submission_probability.csv"
    predictions[["prob_mitterrand_calibrated"]].to_csv(
        submission_probability_path,
        index=False,
        header=False,
    )

    label_path = output_dir / "submission_label_calibrated.csv"
    predictions[["pred_label_calibrated"]].to_csv(label_path, index=False, header=False)

    metrics = {
        "started_at": started_at,
        "ended_at": datetime.now().astimezone().isoformat(),
        "total_seconds": time.perf_counter() - run_start,
        "inference_seconds": inference_seconds,
        "decision_threshold": float(args.decision_threshold),
        "tuned_raw_threshold": tuned_raw_threshold,
        "equivalent_score_delta": delta,
        "label_counts": predictions["pred_label_calibrated"].value_counts().to_dict(),
        "submission_probability_semantics": (
            "Single-column probability that the row belongs to Mitterrand: "
            "0 means Chirac-like, 1 means Mitterrand-like. "
            "Thresholding submission_probability.csv at 0.5 reproduces "
            "submission_label_calibrated.csv."
        ),
        "outputs": {
            "detailed_predictions": str(detailed_path.resolve()),
            "submission_prob_mitterrand_raw": str(raw_prob_path.resolve()),
            "submission_prob_mitterrand_calibrated": str(calibrated_prob_path.resolve()),
            "submission_probability": str(submission_probability_path.resolve()),
            "submission_label_calibrated": str(label_path.resolve()),
        },
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
