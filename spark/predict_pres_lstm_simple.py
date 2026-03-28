from __future__ import annotations

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

from rital_nlp_project.presidents import (
    BiLSTMSequenceTagger,
    PRESIDENT_LABEL_ORDER,
    SpeechSequenceDataset,
    build_speech_sequences,
    collate_speech_sequences,
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
            "Run the simple presidents BiLSTM on the unlabeled test set and "
            "export calibrated Mitterrand probabilities."
        )
    )
    parser.add_argument(
        "--checkpoint-path",
        default="Dataset/out/presidents_lstm_simple/checkpoint.pt",
        help="Path to the simple presidents checkpoint.",
    )
    parser.add_argument(
        "--calibration-path",
        default="Dataset/out/presidents_lstm_simple/calibration.json",
        help="Path to the calibration JSON learned on labeled data.",
    )
    parser.add_argument(
        "--test-embeddings-path",
        default="Dataset/embeddings/presidents_test_camembert_base_mean.npy",
        help="Path to the unlabeled president test embeddings.",
    )
    parser.add_argument(
        "--test-metadata-path",
        default="Dataset/clean/presidents_test_clean_bert.parquet",
        help="Path to the unlabeled president test metadata parquet.",
    )
    parser.add_argument(
        "--output-dir",
        default="Dataset/out/presidents_lstm_simple_submission",
        help="Directory where submission files are written.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Inference batch size. Defaults to the checkpoint batch size.",
    )
    parser.add_argument("--device", default=default_device())
    parser.add_argument(
        "--max-test-speeches",
        type=int,
        default=None,
        help="Optional cap used for smoke tests.",
    )
    return parser.parse_args()


def maybe_slice_sequences(sequences, limit: int | None):
    if limit is None:
        return list(sequences)
    return list(sequences[:limit])


def create_loader(sequences, *, batch_size: int) -> DataLoader:
    dataset = SpeechSequenceDataset(sequences)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_speech_sequences,
    )


def load_model(
    checkpoint: dict,
    *,
    input_dim: int,
    device: torch.device,
) -> BiLSTMSequenceTagger:
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
            sentence_ids = batch["sentence_ids"].cpu().numpy()
            speech_ids = batch["speech_ids"].cpu().tolist()

            for row_idx, length in enumerate(lengths.tolist()):
                predictions.append(
                    {
                        "speech_id": int(speech_ids[row_idx]),
                        "sentence_ids": sentence_ids[row_idx, :length].copy(),
                        "prob_mitterrand_raw": probabilities[row_idx, :length, NEGATIVE_INDEX].copy(),
                    }
                )

    return predictions


def apply_score_delta(scores: np.ndarray, delta: float) -> np.ndarray:
    clipped = np.clip(np.asarray(scores, dtype=np.float64), 1e-9, 1.0 - 1e-9)
    return expit(logit(clipped) - float(delta))


def predictions_to_frame(
    predictions: list[dict[str, np.ndarray | int]],
    *,
    delta: float,
) -> pd.DataFrame:
    frames = []
    for item in predictions:
        raw_scores = np.asarray(item["prob_mitterrand_raw"], dtype=np.float64)  # type: ignore[index]
        calibrated_scores = apply_score_delta(raw_scores, delta)
        labels = np.where(calibrated_scores >= 0.5, "M", "C")
        frames.append(
            pd.DataFrame(
                {
                    "speech_id": int(item["speech_id"]),
                    "sentence_id": item["sentence_ids"],  # type: ignore[index]
                    "prob_mitterrand_raw": raw_scores,
                    "prob_mitterrand_calibrated": calibrated_scores,
                    "pred_label_calibrated": labels,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    started_at = datetime.now().astimezone().isoformat()
    run_start = time.perf_counter()

    checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
    calibration = json.loads(Path(args.calibration_path).read_text())
    batch_size = args.batch_size or checkpoint["config"].get("batch_size", 8)
    delta = float(calibration["equivalent_score_delta"])
    device = torch.device(args.device)

    test_embeddings, test_metadata = load_embeddings_with_metadata(
        args.test_embeddings_path,
        args.test_metadata_path,
        require_labels=False,
    )
    test_sequences = build_speech_sequences(test_embeddings, test_metadata)
    test_sequences = maybe_slice_sequences(test_sequences, args.max_test_speeches)
    selected_speech_ids = [sequence.speech_id for sequence in test_sequences]
    selected_metadata = test_metadata[
        test_metadata["speech_id"].isin(selected_speech_ids)
    ].reset_index(drop=True)
    model = load_model(
        checkpoint,
        input_dim=int(test_sequences[0].embeddings.shape[1]),
        device=device,
    )
    test_loader = create_loader(test_sequences, batch_size=batch_size)

    inference_start = time.perf_counter()
    predictions = collect_predictions(model, test_loader, device=device)
    inference_seconds = time.perf_counter() - inference_start

    frame = predictions_to_frame(predictions, delta=delta)

    aligned_metadata = selected_metadata[["speech_id", "sentence_id"]].copy()
    if not frame[["speech_id", "sentence_id"]].reset_index(drop=True).equals(
        aligned_metadata.reset_index(drop=True)
    ):
        raise ValueError("Predictions are not aligned with the test metadata")

    if "text" in selected_metadata.columns:
        frame["text"] = selected_metadata["text"].to_numpy()

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

    calibrated_label_path = output_dir / "submission_label_calibrated.csv"
    frame[["pred_label_calibrated"]].to_csv(
        calibrated_label_path,
        index=False,
        header=False,
    )

    summary = {
        "checkpoint_path": str(Path(args.checkpoint_path).resolve()),
        "calibration_path": str(Path(args.calibration_path).resolve()),
        "device": str(device),
        "test_rows": int(len(frame)),
        "test_speeches": int(frame["speech_id"].nunique()),
        "batch_size": int(batch_size),
        "selected_score_name": calibration["selected_score_name"],
        "equivalent_score_delta": delta,
        "decision_threshold": 0.5,
        "timing": {
            "started_at": started_at,
            "ended_at": datetime.now().astimezone().isoformat(),
            "total_seconds": time.perf_counter() - run_start,
            "inference_seconds": inference_seconds,
        },
        "outputs": {
            "detailed_predictions": str(detailed_path.resolve()),
            "submission_prob_mitterrand_raw": str(raw_prob_path.resolve()),
            "submission_prob_mitterrand_calibrated": str(calibrated_prob_path.resolve()),
            "submission_label_calibrated": str(calibrated_label_path.resolve()),
        },
        "label_counts": {
            label: int(count)
            for label, count in (
                frame["pred_label_calibrated"].value_counts().sort_index().items()
            )
        },
    }

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
