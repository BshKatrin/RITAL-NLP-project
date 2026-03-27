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
    augment_sequences_with_position_features,
    augment_sequences_with_transition_features,
    build_speech_sequences,
    collate_speech_sequences,
    decode_batch,
    fit_single_span_prior,
    load_embeddings_with_metadata,
    posterior_probabilities_by_sequence,
)


INDEX_TO_LABEL = np.array(PRESIDENT_LABEL_ORDER, dtype=np.int64)
POSITIVE_INDEX = int(np.where(INDEX_TO_LABEL == 1)[0][0])
NEGATIVE_INDEX = int(np.where(INDEX_TO_LABEL == -1)[0][0])
SUBMISSION_LABELS = {1: "C", -1: "M"}


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
        description="Run a trained president BiLSTM checkpoint on the real unlabeled test set."
    )
    parser.add_argument(
        "--checkpoint-path",
        default="Dataset/out/presidents_lstm_full_mps/presidents_lstm.pt",
        help="Path to the saved BiLSTM checkpoint.",
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
        "--prior-embeddings-path",
        default="Dataset/embeddings/presidents_camembert_base_mean.npy",
        help="Path to the labeled training embeddings used to fit the single-span prior.",
    )
    parser.add_argument(
        "--prior-metadata-path",
        default="Dataset/clean/presidents_clean_bert.parquet",
        help="Path to the labeled training metadata used to fit the single-span prior.",
    )
    parser.add_argument(
        "--output-dir",
        default="Dataset/out/presidents_lstm_submission",
        help="Directory where prediction artifacts are written.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for inference. Defaults to the checkpoint config batch size.",
    )
    parser.add_argument(
        "--min-negative-span",
        type=int,
        default=None,
        help="Minimum negative span length for the single-span decoder. Defaults to the checkpoint config value.",
    )
    parser.add_argument(
        "--position-bins",
        type=int,
        default=None,
        help="Number of normalized start/end bins used by the single-span prior. Defaults to the checkpoint config value.",
    )
    parser.add_argument(
        "--position-prior-weight",
        type=float,
        default=None,
        help="Weight applied to the span location prior during constrained decoding. Defaults to the checkpoint config value.",
    )
    parser.add_argument(
        "--no-span-bias",
        type=float,
        default=0.0,
        help="Extra log-bias added to the no-span configuration during constrained decoding. Positive values make the submission more conservative.",
    )
    parser.add_argument(
        "--device",
        default=default_device(),
        help="Inference device.",
    )
    parser.add_argument(
        "--calibration-path",
        default=None,
        help=(
            "Optional JSON file produced by the CV calibration script. "
            "When provided, calibrated score and label files are exported too."
        ),
    )
    parser.add_argument(
        "--score-delta",
        type=float,
        default=None,
        help=(
            "Optional logit-space score shift applied to the selected Mitterrand "
            "probability stream before exporting calibrated scores."
        ),
    )
    parser.add_argument(
        "--decision-threshold",
        type=float,
        default=None,
        help=(
            "Optional threshold applied to the selected calibrated Mitterrand score "
            "when exporting calibrated hard labels."
        ),
    )
    return parser.parse_args()


def create_loader(sequences, *, batch_size: int) -> DataLoader:
    dataset = SpeechSequenceDataset(sequences)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_speech_sequences,
    )


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
                        "probabilities": probabilities[row_idx, :length].copy(),
                    }
                )

    return predictions


def predictions_to_frame(
    predictions: list[dict[str, np.ndarray | int]],
    *,
    prior,
    min_negative_span: int,
    position_prior_weight: float,
    no_span_bias: float,
) -> pd.DataFrame:
    sequence_probabilities = [
        item["probabilities"] for item in predictions  # type: ignore[index]
    ]
    raw_decoded = decode_batch(sequence_probabilities, decoder="argmax")
    span_decoded = decode_batch(
        sequence_probabilities,
        decoder="single_negative_span",
        prior=prior,
        min_span_length=min_negative_span,
        position_prior_weight=position_prior_weight,
        no_span_bias=no_span_bias,
    )
    span_posteriors = posterior_probabilities_by_sequence(
        sequence_probabilities,
        decoder="single_negative_span",
        prior=prior,
        min_span_length=min_negative_span,
        position_prior_weight=position_prior_weight,
        no_span_bias=no_span_bias,
    )

    frames = []
    for item, raw_pred_idx, span_pred_idx, span_probs in zip(
        predictions,
        raw_decoded,
        span_decoded,
        span_posteriors,
        strict=True,
    ):
        raw_labels = INDEX_TO_LABEL[raw_pred_idx]
        span_labels = INDEX_TO_LABEL[span_pred_idx]
        frames.append(
            pd.DataFrame(
                {
                    "speech_id": int(item["speech_id"]),
                    "sentence_id": item["sentence_ids"],  # type: ignore[index]
                    "prob_positive_raw": item["probabilities"][:, POSITIVE_INDEX],  # type: ignore[index]
                    "prob_negative_raw": item["probabilities"][:, NEGATIVE_INDEX],  # type: ignore[index]
                    "prob_positive_single_negative_span": span_probs[:, POSITIVE_INDEX],
                    "prob_negative_single_negative_span": span_probs[:, NEGATIVE_INDEX],
                    "pred_raw": raw_labels,
                    "pred_raw_label": [SUBMISSION_LABELS[int(label)] for label in raw_labels],
                    "pred_single_negative_span": span_labels,
                    "pred_single_negative_span_label": [
                        SUBMISSION_LABELS[int(label)] for label in span_labels
                    ],
                }
            )
        )

    return pd.concat(frames, ignore_index=True)


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


def apply_score_delta(scores: np.ndarray, delta: float) -> np.ndarray:
    clipped = np.clip(np.asarray(scores, dtype=np.float64), 1e-9, 1.0 - 1e-9)
    return expit(logit(clipped) - float(delta))


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    started_at = datetime.now().astimezone().isoformat()
    run_start = time.perf_counter()

    checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
    checkpoint_config = checkpoint["config"]
    batch_size = args.batch_size or checkpoint_config.get("batch_size", 8)
    min_negative_span = (
        args.min_negative_span
        or checkpoint_config.get("min_negative_span", 1)
    )
    position_bins = (
        args.position_bins
        if args.position_bins is not None
        else checkpoint_config.get("position_bins", 0)
    )
    position_prior_weight = (
        args.position_prior_weight
        if args.position_prior_weight is not None
        else checkpoint_config.get("position_prior_weight", 0.0)
    )
    position_features = checkpoint_config.get("position_features", "none")
    transition_features = bool(checkpoint_config.get("transition_features", False))
    device = torch.device(args.device)
    calibration = None
    if args.calibration_path is not None:
        calibration = json.loads(Path(args.calibration_path).read_text())

    prior_embeddings, prior_metadata = load_embeddings_with_metadata(
        args.prior_embeddings_path,
        args.prior_metadata_path,
    )
    prior_sequences = build_speech_sequences(prior_embeddings, prior_metadata)
    single_span_prior = fit_single_span_prior(
        prior_sequences,
        position_bins=position_bins,
    )

    test_embeddings, test_metadata = load_embeddings_with_metadata(
        args.test_embeddings_path,
        args.test_metadata_path,
        require_labels=False,
    )
    test_sequences = build_speech_sequences(test_embeddings, test_metadata)
    if position_features != "none":
        test_sequences = augment_sequences_with_position_features(test_sequences)
    if transition_features:
        test_sequences = augment_sequences_with_transition_features(test_sequences)
    model = load_model(
        checkpoint,
        input_dim=int(test_sequences[0].embeddings.shape[1]),
        device=device,
    )
    test_loader = create_loader(test_sequences, batch_size=batch_size)

    inference_start = time.perf_counter()
    predictions = collect_predictions(model, test_loader, device=device)
    inference_seconds = time.perf_counter() - inference_start

    frame = predictions_to_frame(
        predictions,
        prior=single_span_prior,
        min_negative_span=min_negative_span,
        position_prior_weight=position_prior_weight,
        no_span_bias=args.no_span_bias,
    )

    alignment = test_metadata[["speech_id", "sentence_id"]].reset_index(drop=True)
    predicted_alignment = frame[["speech_id", "sentence_id"]].reset_index(drop=True)
    if not predicted_alignment.equals(alignment):
        raise ValueError("Predictions are not aligned with the source test metadata")

    frame["text"] = test_metadata["text"].to_numpy()

    detailed_path = output_dir / "test_predictions_detailed.csv"
    frame.to_csv(detailed_path, index=False)

    raw_prob_path = output_dir / "submission_prob_raw_positive.csv"
    frame[["prob_positive_raw"]].to_csv(raw_prob_path, index=False, header=False)

    raw_prob_inverted_path = output_dir / "submission_prob_raw_positive_inverted.csv"
    (1.0 - frame[["prob_positive_raw"]]).to_csv(
        raw_prob_inverted_path,
        index=False,
        header=False,
    )

    span_prob_path = output_dir / "submission_prob_single_negative_span.csv"
    frame[["prob_positive_single_negative_span"]].to_csv(
        span_prob_path,
        index=False,
        header=False,
    )

    span_prob_inverted_path = output_dir / "submission_prob_single_negative_span_inverted.csv"
    (1.0 - frame[["prob_positive_single_negative_span"]]).to_csv(
        span_prob_inverted_path,
        index=False,
        header=False,
    )

    span_label_path = output_dir / "submission_label_single_negative_span.csv"
    frame[["pred_single_negative_span_label"]].to_csv(
        span_label_path,
        index=False,
        header=False,
    )

    calibration_outputs = None
    calibration_config = None
    score_delta = args.score_delta
    decision_threshold = args.decision_threshold
    selected_score_name = "prob_mitterrand_single_negative_span"

    if calibration is not None:
        recommended_calibration = calibration.get("recommended_calibration", {})
        score_delta = (
            recommended_calibration.get("equivalent_score_delta")
            if score_delta is None
            else score_delta
        )
        # The CV script stores the original best threshold and its equivalent
        # logit-space shift. Once we apply the shift to the scores, the matching
        # hard-decision threshold becomes 0.5 on the calibrated probabilities.
        decision_threshold = 0.5 if decision_threshold is None else decision_threshold
        selected_score_name = calibration.get(
            "selected_score_name",
            selected_score_name,
        )

    if score_delta is not None or decision_threshold is not None:
        if selected_score_name == "prob_mitterrand_raw":
            base_scores = frame["prob_negative_raw"].to_numpy(dtype=np.float64)
            score_output_name = "submission_prob_raw_positive_inverted_calibrated.csv"
        elif selected_score_name == "prob_mitterrand_single_negative_span":
            base_scores = frame["prob_negative_single_negative_span"].to_numpy(
                dtype=np.float64
            )
            score_output_name = (
                "submission_prob_single_negative_span_inverted_calibrated.csv"
            )
        else:
            raise ValueError(
                f"Unsupported calibrated score source: {selected_score_name}"
            )

        effective_delta = 0.0 if score_delta is None else float(score_delta)
        effective_threshold = (
            0.5 if decision_threshold is None else float(decision_threshold)
        )
        calibrated_scores = apply_score_delta(base_scores, effective_delta)
        calibrated_score_path = output_dir / score_output_name
        pd.DataFrame(calibrated_scores).to_csv(
            calibrated_score_path,
            index=False,
            header=False,
        )

        calibrated_label_path = output_dir / "submission_label_calibrated.csv"
        pd.DataFrame(
            np.where(calibrated_scores >= effective_threshold, "M", "C")
        ).to_csv(
            calibrated_label_path,
            index=False,
            header=False,
        )
        calibrated_labels = np.where(calibrated_scores >= effective_threshold, "M", "C")

        calibration_outputs = {
            "submission_prob_calibrated": str(calibrated_score_path.resolve()),
            "submission_label_calibrated": str(calibrated_label_path.resolve()),
        }
        calibration_config = {
            "calibration_path": (
                str(Path(args.calibration_path).resolve())
                if args.calibration_path is not None
                else None
            ),
            "selected_score_name": selected_score_name,
            "score_delta": float(effective_delta),
            "decision_threshold": float(effective_threshold),
            "original_threshold": (
                float(recommended_calibration["threshold"])
                if calibration is not None
                and "recommended_calibration" in calibration
                and "threshold" in calibration["recommended_calibration"]
                else None
            ),
        }
        calibration_config["calibrated_label_counts"] = {
            label: int(count)
            for label, count in (
                pd.Series(calibrated_labels).value_counts().sort_index().items()
            )
        }

    ended_at = datetime.now().astimezone().isoformat()
    total_seconds = time.perf_counter() - run_start
    summary = {
        "checkpoint_path": str(Path(args.checkpoint_path).resolve()),
        "checkpoint_output_dir": checkpoint_config.get("output_dir"),
        "device": str(device),
        "test_rows": int(len(test_metadata)),
        "test_speeches": int(test_metadata["speech_id"].nunique()),
        "batch_size": batch_size,
        "min_negative_span": int(min_negative_span),
        "position_bins": int(position_bins),
        "position_prior_weight": float(position_prior_weight),
        "no_span_bias": float(args.no_span_bias),
        "position_features": position_features,
        "transition_features": transition_features,
        "timing": {
            "started_at": started_at,
            "ended_at": ended_at,
            "total_seconds": total_seconds,
            "inference_seconds": inference_seconds,
        },
        "outputs": {
            "detailed_predictions": str(detailed_path.resolve()),
            "submission_prob_raw_positive": str(raw_prob_path.resolve()),
            "submission_prob_raw_positive_inverted": str(raw_prob_inverted_path.resolve()),
            "submission_prob_single_negative_span": str(span_prob_path.resolve()),
            "submission_prob_single_negative_span_inverted": str(
                span_prob_inverted_path.resolve()
            ),
            "submission_label_single_negative_span": str(span_label_path.resolve()),
        },
        "calibration": calibration_config,
        "calibrated_outputs": calibration_outputs,
        "label_counts": {
            label: int(count)
            for label, count in (
                frame["pred_single_negative_span_label"].value_counts().sort_index().items()
            )
        },
    }

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
