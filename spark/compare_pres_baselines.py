from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from rital_nlp_project.presidents import (
    PRESIDENT_LABEL_ORDER,
    build_speech_sequences,
    decode_batch,
    fit_single_span_prior,
    load_embeddings_with_metadata,
    smooth_probabilities_by_sequence,
    speech_train_test_split,
)


INDEX_TO_LABEL = np.array(PRESIDENT_LABEL_ORDER, dtype=np.int64)
POSITIVE_INDEX = int(np.where(INDEX_TO_LABEL == 1)[0][0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare president sequence baselines on the speech-aware split."
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
        default="Dataset/out/presidents_baseline_compare",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--validation-size", type=float, default=0.1)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--min-negative-span", type=int, default=3)
    parser.add_argument(
        "--selection-metric",
        choices=("f1_macro", "balanced_accuracy"),
        default="f1_macro",
    )
    parser.add_argument(
        "--sigma-grid",
        nargs="+",
        type=float,
        default=[0.0, 0.5, 0.7, 1.0, 1.5, 2.0],
    )
    return parser.parse_args()


def flatten_sequences(sequences):
    X = np.concatenate([sequence.embeddings for sequence in sequences], axis=0)
    y = np.concatenate([sequence.labels for sequence in sequences], axis=0)
    return X, y


def predict_sequence_probabilities(model, sequences):
    column_indices = [
        int(np.where(model.classes_ == label)[0][0])
        for label in PRESIDENT_LABEL_ORDER
    ]
    probabilities = []
    for sequence in sequences:
        probabilities.append(model.predict_proba(sequence.embeddings)[:, column_indices])
    return probabilities


def evaluate_predictions(
    sequence_probabilities: list[np.ndarray],
    sequences,
    *,
    decoder: str,
    prior=None,
    min_negative_span: int,
):
    decoded = decode_batch(
        sequence_probabilities,
        decoder=decoder,
        prior=prior,
        min_span_length=min_negative_span,
    )

    y_true = np.concatenate([sequence.labels for sequence in sequences], axis=0)
    y_pred_idx = np.concatenate(decoded, axis=0)
    y_pred = INDEX_TO_LABEL[y_pred_idx]
    positive_scores = np.concatenate(
        [probabilities[:, POSITIVE_INDEX] for probabilities in sequence_probabilities],
        axis=0,
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
    sequences,
    raw_probabilities: list[np.ndarray],
    *,
    single_span_prior,
    min_negative_span: int,
    gaussian_sigma_argmax: float,
    gaussian_sigma_span: float,
):
    raw_decoded = decode_batch(raw_probabilities, decoder="argmax")
    span_decoded = decode_batch(
        raw_probabilities,
        decoder="single_negative_span",
        prior=single_span_prior,
        min_span_length=min_negative_span,
    )
    gaussian_probabilities = smooth_probabilities_by_sequence(
        raw_probabilities,
        sigma=gaussian_sigma_argmax,
    )
    gaussian_span_probabilities = smooth_probabilities_by_sequence(
        raw_probabilities,
        sigma=gaussian_sigma_span,
    )
    gaussian_decoded = decode_batch(gaussian_probabilities, decoder="argmax")
    gaussian_span_decoded = decode_batch(
        gaussian_span_probabilities,
        decoder="single_negative_span",
        prior=single_span_prior,
        min_span_length=min_negative_span,
    )

    frames = []
    for sequence, raw_probs, raw_pred, span_pred, gauss_probs, gauss_pred, gauss_span_pred in zip(
        sequences,
        raw_probabilities,
        raw_decoded,
        span_decoded,
        gaussian_probabilities,
        gaussian_decoded,
        gaussian_span_decoded,
        strict=True,
    ):
        frames.append(
            pd.DataFrame(
                {
                    "speech_id": sequence.speech_id,
                    "sentence_id": sequence.sentence_ids,
                    "true_label": sequence.labels,
                    "prob_positive_raw": raw_probs[:, POSITIVE_INDEX],
                    "prob_negative_raw": raw_probs[:, 1 - POSITIVE_INDEX],
                    "prob_positive_gaussian": gauss_probs[:, POSITIVE_INDEX],
                    "prob_negative_gaussian": gauss_probs[:, 1 - POSITIVE_INDEX],
                    "sigma_gaussian_argmax": gaussian_sigma_argmax,
                    "sigma_gaussian_single_negative_span": gaussian_sigma_span,
                    "pred_raw_argmax": INDEX_TO_LABEL[raw_pred],
                    "pred_single_negative_span": INDEX_TO_LABEL[span_pred],
                    "pred_gaussian_argmax": INDEX_TO_LABEL[gauss_pred],
                    "pred_gaussian_single_negative_span": INDEX_TO_LABEL[gauss_span_pred],
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

    X_train, y_train = flatten_sequences(train_sequences)
    fit_start = time.perf_counter()
    baseline = LogisticRegression(
        max_iter=args.max_iter,
        random_state=args.random_state,
    )
    baseline.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - fit_start

    inference_start = time.perf_counter()
    val_probabilities = predict_sequence_probabilities(baseline, val_sequences)
    test_probabilities = predict_sequence_probabilities(baseline, test_sequences)
    inference_seconds = time.perf_counter() - inference_start

    single_span_prior = fit_single_span_prior(train_sequences)

    validation_results = {
        "raw_argmax": evaluate_predictions(
            val_probabilities,
            val_sequences,
            decoder="argmax",
            prior=None,
            min_negative_span=args.min_negative_span,
        ),
        "single_negative_span": evaluate_predictions(
            val_probabilities,
            val_sequences,
            decoder="single_negative_span",
            prior=single_span_prior,
            min_negative_span=args.min_negative_span,
        ),
    }

    sigma_grid_records = []
    best_gaussian_sigma = None
    best_gaussian_score = float("-inf")
    best_gaussian_span_sigma = None
    best_gaussian_span_score = float("-inf")

    for sigma in args.sigma_grid:
        smoothed_val_probabilities = smooth_probabilities_by_sequence(
            val_probabilities,
            sigma=sigma,
        )
        gaussian_metrics = evaluate_predictions(
            smoothed_val_probabilities,
            val_sequences,
            decoder="argmax",
            prior=None,
            min_negative_span=args.min_negative_span,
        )
        gaussian_span_metrics = evaluate_predictions(
            smoothed_val_probabilities,
            val_sequences,
            decoder="single_negative_span",
            prior=single_span_prior,
            min_negative_span=args.min_negative_span,
        )
        sigma_grid_records.append(
            {
                "sigma": sigma,
                "gaussian_argmax": gaussian_metrics,
                "gaussian_single_negative_span": gaussian_span_metrics,
            }
        )

        score_name = args.selection_metric
        if gaussian_metrics[score_name] > best_gaussian_score:
            best_gaussian_score = gaussian_metrics[score_name]
            best_gaussian_sigma = sigma
            validation_results["gaussian_argmax"] = gaussian_metrics
        if gaussian_span_metrics[score_name] > best_gaussian_span_score:
            best_gaussian_span_score = gaussian_span_metrics[score_name]
            best_gaussian_span_sigma = sigma
            validation_results["gaussian_single_negative_span"] = gaussian_span_metrics

    if best_gaussian_sigma is None or best_gaussian_span_sigma is None:
        raise RuntimeError("sigma selection failed")

    test_results = {
        "raw_argmax": evaluate_predictions(
            test_probabilities,
            test_sequences,
            decoder="argmax",
            prior=None,
            min_negative_span=args.min_negative_span,
        ),
        "single_negative_span": evaluate_predictions(
            test_probabilities,
            test_sequences,
            decoder="single_negative_span",
            prior=single_span_prior,
            min_negative_span=args.min_negative_span,
        ),
        "gaussian_argmax": evaluate_predictions(
            smooth_probabilities_by_sequence(test_probabilities, sigma=best_gaussian_sigma),
            test_sequences,
            decoder="argmax",
            prior=None,
            min_negative_span=args.min_negative_span,
        ),
        "gaussian_single_negative_span": evaluate_predictions(
            smooth_probabilities_by_sequence(
                test_probabilities,
                sigma=best_gaussian_span_sigma,
            ),
            test_sequences,
            decoder="single_negative_span",
            prior=single_span_prior,
            min_negative_span=args.min_negative_span,
        ),
    }

    ended_at = datetime.now().astimezone().isoformat()
    total_seconds = time.perf_counter() - run_start

    summary = {
        "config": vars(args),
        "model": "LogisticRegression",
        "split": {
            "all_speeches": len(sequences),
            "train_speeches": len(train_sequences),
            "validation_speeches": len(val_sequences),
            "test_speeches": len(test_sequences),
            "train_rows": int(sum(sequence.length for sequence in train_sequences)),
            "validation_rows": int(sum(sequence.length for sequence in val_sequences)),
            "test_rows": int(sum(sequence.length for sequence in test_sequences)),
        },
        "timing": {
            "started_at": started_at,
            "ended_at": ended_at,
            "fit_seconds": fit_seconds,
            "inference_seconds": inference_seconds,
            "total_seconds": total_seconds,
        },
        "selection": {
            "metric": args.selection_metric,
            "best_gaussian_sigma": best_gaussian_sigma,
            "best_gaussian_single_negative_span_sigma": best_gaussian_span_sigma,
        },
        "validation": validation_results,
        "test": test_results,
    }

    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    (output_dir / "validation_sigma_grid.json").write_text(
        json.dumps(sigma_grid_records, indent=2)
    )
    predictions_to_frame(
        test_sequences,
        test_probabilities,
        single_span_prior=single_span_prior,
        min_negative_span=args.min_negative_span,
        gaussian_sigma_argmax=best_gaussian_sigma,
        gaussian_sigma_span=best_gaussian_span_sigma,
    ).to_csv(output_dir / "test_predictions.csv", index=False)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
