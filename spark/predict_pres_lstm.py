from __future__ import annotations

import argparse
import json
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
