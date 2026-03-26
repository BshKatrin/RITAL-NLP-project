from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
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
        description=(
            "Sweep no-span decoder bias values for the presidents LSTM submission, "
            "compare them to a leaderboard reference file, and export shortlisted variants."
        )
    )
    parser.add_argument(
        "--checkpoint-path",
        default="Dataset/out/presidents_lstm_full_data_19ep/presidents_lstm.pt",
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
        "--prior-embeddings-path",
        default="Dataset/embeddings/presidents_camembert_base_mean.npy",
    )
    parser.add_argument(
        "--prior-metadata-path",
        default="Dataset/clean/presidents_clean_bert.parquet",
    )
    parser.add_argument(
        "--reference-path",
        default="Dataset/out/Leaderboard/submission-pres-4_88f1.csv",
        help="Leaderboard reference score file used for hard-prediction agreement.",
    )
    parser.add_argument(
        "--output-dir",
        default="Dataset/out/presidents_submission_bias_sweep",
    )
    parser.add_argument(
        "--bias-start",
        type=float,
        default=0.0,
        help="Inclusive sweep start for no-span bias.",
    )
    parser.add_argument(
        "--bias-end",
        type=float,
        default=4.0,
        help="Inclusive sweep end for no-span bias.",
    )
    parser.add_argument(
        "--bias-step",
        type=float,
        default=0.05,
        help="Sweep step size for no-span bias.",
    )
    parser.add_argument(
        "--target-counts",
        nargs="+",
        type=int,
        default=[3300, 3500, 3700],
        help="Target hard Mitterrand counts for shortlisted submissions.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Inference batch size. Defaults to the checkpoint config value.",
    )
    parser.add_argument(
        "--device",
        default=default_device(),
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
    predictions: list[dict[str, np.ndarray | int]] = []
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


def load_reference_labels(path: str | Path) -> pd.Series:
    scores = pd.read_csv(path, header=None).iloc[:, 0].astype(float)
    return pd.Series(
        np.where(scores >= 0.5, "Mitterrand", "Chirac"),
        index=scores.index,
        name="reference_label",
    )


def build_sweep_biases(start: float, end: float, step: float) -> list[float]:
    if step <= 0:
        raise ValueError("bias-step must be strictly positive")
    if end < start:
        raise ValueError("bias-end must be greater than or equal to bias-start")

    num_steps = int(math.floor((end - start) / step))
    values = [round(start + step * idx, 10) for idx in range(num_steps + 1)]
    if values[-1] < end:
        values.append(round(end, 10))
    return values


def labels_from_probability_sequence(sequence_probs: list[np.ndarray]) -> np.ndarray:
    mitterrand_probs = np.concatenate(
        [probabilities[:, NEGATIVE_INDEX] for probabilities in sequence_probs]
    )
    return np.where(mitterrand_probs >= 0.5, "Mitterrand", "Chirac")


def build_variant_frame(
    predictions: list[dict[str, np.ndarray | int]],
    *,
    sequence_posteriors: list[np.ndarray],
    decoded_indices: list[np.ndarray],
) -> pd.DataFrame:
    frames = []
    for item, posteriors, decoded in zip(
        predictions,
        sequence_posteriors,
        decoded_indices,
        strict=True,
    ):
        decoded_labels = INDEX_TO_LABEL[decoded]
        frames.append(
            pd.DataFrame(
                {
                    "speech_id": int(item["speech_id"]),
                    "sentence_id": item["sentence_ids"],  # type: ignore[index]
                    "prob_positive_single_negative_span": posteriors[:, POSITIVE_INDEX],
                    "prob_negative_single_negative_span": posteriors[:, NEGATIVE_INDEX],
                    "pred_single_negative_span": decoded_labels,
                    "pred_single_negative_span_label": [
                        SUBMISSION_LABELS[int(label)] for label in decoded_labels
                    ],
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
    checkpoint_config = checkpoint["config"]
    batch_size = args.batch_size or checkpoint_config.get("batch_size", 8)
    min_negative_span = checkpoint_config.get("min_negative_span", 1)
    position_bins = int(checkpoint_config.get("position_bins", 0))
    position_prior_weight = float(checkpoint_config.get("position_prior_weight", 0.0))
    position_features = checkpoint_config.get("position_features", "none")
    transition_features = bool(checkpoint_config.get("transition_features", False))
    device = torch.device(args.device)

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
    loader = create_loader(test_sequences, batch_size=batch_size)

    inference_start = time.perf_counter()
    predictions = collect_predictions(model, loader, device=device)
    inference_seconds = time.perf_counter() - inference_start

    sequence_probabilities = [
        item["probabilities"] for item in predictions  # type: ignore[index]
    ]
    reference_labels = load_reference_labels(args.reference_path)
    if len(reference_labels) != len(test_metadata):
        raise ValueError(
            "Reference submission length mismatch: "
            f"{len(reference_labels)} != {len(test_metadata)}"
        )

    sweep_records = []
    best_agreement_record = None

    for no_span_bias in build_sweep_biases(
        args.bias_start,
        args.bias_end,
        args.bias_step,
    ):
        decoded = decode_batch(
            sequence_probabilities,
            decoder="single_negative_span",
            prior=single_span_prior,
            min_span_length=min_negative_span,
            position_prior_weight=position_prior_weight,
            no_span_bias=no_span_bias,
        )
        predicted_labels = INDEX_TO_LABEL[np.concatenate(decoded)]
        submission_labels = np.where(predicted_labels == -1, "Mitterrand", "Chirac")
        agreement = submission_labels == reference_labels.to_numpy()
        mitterrand_count = int((submission_labels == "Mitterrand").sum())

        record = {
            "no_span_bias": float(no_span_bias),
            "mitterrand_count": mitterrand_count,
            "chirac_count": int(len(submission_labels) - mitterrand_count),
            "agreement_to_reference": float(agreement.mean()),
            "disagreement_to_reference": int((~agreement).sum()),
        }
        sweep_records.append(record)

        if (
            best_agreement_record is None
            or record["agreement_to_reference"]
            > best_agreement_record["agreement_to_reference"]
        ):
            best_agreement_record = record

    sweep_frame = pd.DataFrame(sweep_records).sort_values("no_span_bias").reset_index(
        drop=True
    )
    sweep_frame.to_csv(output_dir / "bias_sweep_summary.csv", index=False)

    selected_biases = {0.0}
    for target in args.target_counts:
        row = sweep_frame.iloc[
            (sweep_frame["mitterrand_count"] - target).abs().argmin()
        ]
        selected_biases.add(float(row["no_span_bias"]))
    if best_agreement_record is not None:
        selected_biases.add(float(best_agreement_record["no_span_bias"]))

    selected_records = []
    for no_span_bias in sorted(selected_biases):
        bias_label = f"bias_{no_span_bias:.2f}".replace(".", "p")
        variant_dir = output_dir / bias_label
        variant_dir.mkdir(parents=True, exist_ok=True)

        posteriors = posterior_probabilities_by_sequence(
            sequence_probabilities,
            decoder="single_negative_span",
            prior=single_span_prior,
            min_span_length=min_negative_span,
            position_prior_weight=position_prior_weight,
            no_span_bias=no_span_bias,
        )
        decoded = decode_batch(
            sequence_probabilities,
            decoder="single_negative_span",
            prior=single_span_prior,
            min_span_length=min_negative_span,
            position_prior_weight=position_prior_weight,
            no_span_bias=no_span_bias,
        )
        frame = build_variant_frame(
            predictions,
            sequence_posteriors=posteriors,
            decoded_indices=decoded,
        )

        alignment = test_metadata[["speech_id", "sentence_id"]].reset_index(drop=True)
        predicted_alignment = frame[["speech_id", "sentence_id"]].reset_index(drop=True)
        if not predicted_alignment.equals(alignment):
            raise ValueError(
                f"Predictions are not aligned with the test metadata for bias {no_span_bias}"
            )

        detailed_path = variant_dir / "test_predictions_detailed.csv"
        frame.assign(text=test_metadata["text"].to_numpy()).to_csv(
            detailed_path,
            index=False,
        )

        span_prob_path = variant_dir / "submission_prob_single_negative_span.csv"
        frame[["prob_positive_single_negative_span"]].to_csv(
            span_prob_path,
            index=False,
            header=False,
        )

        span_prob_inverted_path = (
            variant_dir / "submission_prob_single_negative_span_inverted.csv"
        )
        mitterrand_scores = frame[["prob_negative_single_negative_span"]]
        mitterrand_scores.to_csv(
            span_prob_inverted_path,
            index=False,
            header=False,
        )

        span_label_path = variant_dir / "submission_label_single_negative_span.csv"
        frame[["pred_single_negative_span_label"]].to_csv(
            span_label_path,
            index=False,
            header=False,
        )

        predicted_from_scores = np.where(
            frame["prob_negative_single_negative_span"].to_numpy() >= 0.5,
            "Mitterrand",
            "Chirac",
        )
        reference_array = reference_labels.to_numpy()
        agreement = predicted_from_scores == reference_array

        record = {
            "no_span_bias": float(no_span_bias),
            "mitterrand_count": int((predicted_from_scores == "Mitterrand").sum()),
            "agreement_to_reference": float(agreement.mean()),
            "disagreement_to_reference": int((~agreement).sum()),
            "variant_dir": str(variant_dir.resolve()),
            "outputs": {
                "detailed_predictions": str(detailed_path.resolve()),
                "submission_prob_single_negative_span": str(span_prob_path.resolve()),
                "submission_prob_single_negative_span_inverted": str(
                    span_prob_inverted_path.resolve()
                ),
                "submission_label_single_negative_span": str(span_label_path.resolve()),
            },
        }
        selected_records.append(record)
        (variant_dir / "metrics.json").write_text(json.dumps(record, indent=2))

    ended_at = datetime.now().astimezone().isoformat()
    total_seconds = time.perf_counter() - run_start
    summary = {
        "checkpoint_path": str(Path(args.checkpoint_path).resolve()),
        "reference_path": str(Path(args.reference_path).resolve()),
        "device": str(device),
        "test_rows": int(len(test_metadata)),
        "test_speeches": int(test_metadata["speech_id"].nunique()),
        "batch_size": int(batch_size),
        "min_negative_span": int(min_negative_span),
        "position_bins": position_bins,
        "position_prior_weight": position_prior_weight,
        "position_features": position_features,
        "transition_features": transition_features,
        "selected_target_counts": args.target_counts,
        "best_agreement_record": best_agreement_record,
        "selected_variants": selected_records,
        "timing": {
            "started_at": started_at,
            "ended_at": ended_at,
            "total_seconds": total_seconds,
            "inference_seconds": inference_seconds,
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
