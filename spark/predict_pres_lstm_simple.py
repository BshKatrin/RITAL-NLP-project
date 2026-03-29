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

from rital_nlp_project.presidents.lstm import BiLSTMSequenceTagger
from rital_nlp_project.presidents.sequence import (
    PRESIDENT_LABEL_ORDER,
    SpeechSequenceDataset,
    build_speech_sequences,
    collate_speech_sequences,
    load_embeddings_with_metadata,
)


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
        description="Run the simple presidents BiLSTM on the test set."
    )
    parser.add_argument(
        "--checkpoint-path",
        default="Dataset/out/presidents_lstm_simple/checkpoint.pt",
    )
    parser.add_argument(
        "--calibration-path",
        default="Dataset/out/presidents_lstm_simple/calibration.json",
    )
    parser.add_argument(
        "--test-embeddings-path",
        default="Dataset/embeddings/presidents_test_camembert_base_mean.npy",
    )
    parser.add_argument(
        "--test-metadata-path",
        default="Dataset/clean/presidents_test_clean_bert.parquet",
    )
    parser.add_argument(
        "--output-dir",
        default="Dataset/out/presidents_lstm_simple_submission",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--decision-threshold", type=float, default=0.5)
    parser.add_argument("--device", default=default_device())
    return parser.parse_args()


def create_loader(sequences, batch_size):
    dataset = SpeechSequenceDataset(sequences)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_speech_sequences,
    )


def load_model(checkpoint, input_dim, device):
    config = checkpoint["config"]
    model = BiLSTMSequenceTagger(
        input_dim=input_dim,
        hidden_dim=config["hidden_dim"],
        projection_dim=config["projection_dim"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
        num_labels=len(checkpoint["label_order"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def collect_predictions(model, dataloader, device):
    rows = []
    model.eval()

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch["inputs"].to(device)
            lengths = batch["lengths"]
            logits = model(inputs, lengths)
            probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
            sentence_ids = batch["sentence_ids"].cpu().numpy()
            speech_ids = batch["speech_ids"].cpu().tolist()

            for row_idx, length in enumerate(lengths.tolist()):
                rows.append(
                    pd.DataFrame(
                        {
                            "speech_id": int(speech_ids[row_idx]),
                            "sentence_id": sentence_ids[row_idx, :length].copy(),
                            "prob_mitterrand_raw": probabilities[row_idx, :length, NEGATIVE_INDEX].copy(),
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

    checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
    calibration = json.loads(Path(args.calibration_path).read_text())
    device = torch.device(args.device)
    batch_size = args.batch_size or checkpoint["config"].get("batch_size", 8)
    delta = float(calibration["equivalent_score_delta"])
    tuned_raw_threshold = float(calibration["threshold"])

    test_embeddings, test_metadata = load_embeddings_with_metadata(
        args.test_embeddings_path,
        args.test_metadata_path,
        require_labels=False,
    )
    sequences = build_speech_sequences(test_embeddings, test_metadata)
    model = load_model(checkpoint, sequences[0].embeddings.shape[1], device)
    dataloader = create_loader(sequences, batch_size)

    inference_start = time.perf_counter()
    frame = collect_predictions(model, dataloader, device)
    inference_seconds = time.perf_counter() - inference_start

    expected = test_metadata[["speech_id", "sentence_id"]].reset_index(drop=True)
    actual = frame[["speech_id", "sentence_id"]].reset_index(drop=True)
    if not actual.equals(expected):
        raise ValueError("Predictions are not aligned with the test metadata")

    frame["prob_mitterrand_calibrated"] = apply_score_delta(
        frame["prob_mitterrand_raw"].to_numpy(dtype=np.float64),
        delta,
    )
    frame["pred_label_calibrated"] = labels_from_scores(
        frame["prob_mitterrand_calibrated"].to_numpy(dtype=np.float64),
        float(args.decision_threshold),
    )
    aligned_labels = labels_from_scores(
        frame["prob_mitterrand_calibrated"].to_numpy(dtype=np.float64),
        0.5,
    )
    if not np.array_equal(aligned_labels, frame["pred_label_calibrated"].to_numpy()):
        raise ValueError(
            "submission labels must be reproducible from the calibrated probability "
            "file using a 0.5 threshold"
        )

    if "text" in test_metadata.columns:
        frame["text"] = test_metadata["text"].to_numpy()

    detailed_path = output_dir / "test_predictions_detailed.csv"
    frame.to_csv(detailed_path, index=False)

    raw_prob_path = output_dir / "submission_prob_mitterrand_raw.csv"
    frame[["prob_mitterrand_raw"]].to_csv(raw_prob_path, index=False, header=False)

    calibrated_prob_path = output_dir / "submission_prob_mitterrand_calibrated.csv"
    frame[["prob_mitterrand_calibrated"]].to_csv(
        calibrated_prob_path,
        index=False,
        header=False,
    )
    submission_probability_path = output_dir / "submission_probability.csv"
    frame[["prob_mitterrand_calibrated"]].to_csv(
        submission_probability_path,
        index=False,
        header=False,
    )

    label_path = output_dir / "submission_label_calibrated.csv"
    frame[["pred_label_calibrated"]].to_csv(label_path, index=False, header=False)

    metrics = {
        "started_at": started_at,
        "ended_at": datetime.now().astimezone().isoformat(),
        "total_seconds": time.perf_counter() - run_start,
        "inference_seconds": inference_seconds,
        "decision_threshold": float(args.decision_threshold),
        "tuned_raw_threshold": tuned_raw_threshold,
        "equivalent_score_delta": delta,
        "label_counts": frame["pred_label_calibrated"].value_counts().to_dict(),
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
