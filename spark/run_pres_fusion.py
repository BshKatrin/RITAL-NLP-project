from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression

from rital_nlp_project.presidents import (
    SpeechSequence,
    fit_single_span_prior,
    posterior_probabilities_by_sequence,
)
from rital_nlp_project.presidents.eval_utils import (
    KNOWN_SCORE_COLUMNS,
    apply_score_delta,
    build_grouped_splits,
    compute_boundary_diagnostics,
    evaluate_proxy_predictions,
    speech_target_frame,
    sweep_thresholds,
    threshold_grid,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fuse two presidents score streams with weighted logit averaging and "
            "optional logistic stacking, then calibrate on OOF predictions."
        )
    )
    parser.add_argument("--oof-a-path", required=True)
    parser.add_argument("--oof-b-path", required=True)
    parser.add_argument("--test-a-path", required=True)
    parser.add_argument("--test-b-path", required=True)
    parser.add_argument(
        "--train-metadata-path",
        default="Dataset/clean/presidents_clean_bert.parquet",
    )
    parser.add_argument(
        "--test-metadata-path",
        default="Dataset/clean/presidents_test_clean_bert.parquet",
    )
    parser.add_argument("--oof-a-score-column", default=None)
    parser.add_argument("--oof-b-score-column", default=None)
    parser.add_argument("--test-a-score-column", default=None)
    parser.add_argument("--test-b-score-column", default=None)
    parser.add_argument(
        "--output-dir",
        default="Dataset/out/presidents_fusion",
    )
    parser.add_argument("--alpha-start", type=float, default=0.0)
    parser.add_argument("--alpha-end", type=float, default=1.0)
    parser.add_argument("--alpha-step", type=float, default=0.05)
    parser.add_argument("--threshold-start", type=float, default=0.05)
    parser.add_argument("--threshold-end", type=float, default=0.95)
    parser.add_argument("--threshold-step", type=float, default=0.005)
    parser.add_argument("--stacking-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--apply-single-span-decoder",
        action="store_true",
        help="Apply the simple single contiguous-span structural decoder to fused scores.",
    )
    parser.add_argument("--min-negative-span", type=int, default=3)
    parser.add_argument("--no-span-bias", type=float, default=0.0)
    parser.add_argument(
        "--proxy-labels-path",
        default=None,
    )
    parser.add_argument(
        "--ranking-mode",
        choices=("clean", "proxy", "hybrid"),
        default="hybrid",
    )
    return parser.parse_args()


def infer_score_column(frame: pd.DataFrame) -> str:
    for column in KNOWN_SCORE_COLUMNS:
        if column in frame.columns:
            return column
    if frame.shape[1] == 1:
        return frame.columns[0]
    raise ValueError("Could not infer a score column; pass it explicitly.")


def load_score_frame(
    path: str,
    *,
    metadata_path: str,
    score_column: str | None,
    include_labels: bool,
) -> pd.DataFrame:
    raw = pd.read_csv(path, header=None)
    metadata = pd.read_parquet(metadata_path).reset_index(drop=True)
    metadata_order = metadata[["speech_id", "sentence_id"]].copy()
    metadata_order["row_order"] = np.arange(len(metadata_order), dtype=np.int64)

    if raw.shape[1] == 1 and len(raw) == len(metadata):
        frame = metadata_order.copy()
        if include_labels:
            frame["true_label"] = metadata["label"].to_numpy()
        frame["score"] = raw.iloc[:, 0].astype(float).to_numpy()
        return frame

    frame = pd.read_csv(path)
    column = score_column or infer_score_column(frame)
    selected = frame.copy()
    rename_map = {column: "score"}
    if "label" in selected.columns and "true_label" not in selected.columns:
        rename_map["label"] = "true_label"
    selected = selected.rename(columns=rename_map)

    required = {"speech_id", "sentence_id", "score"}
    if include_labels:
        required.add("true_label")
    missing = required - set(selected.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")

    ordered_columns = ["speech_id", "sentence_id", "score"]
    if include_labels:
        ordered_columns.insert(2, "true_label")
    selected = selected[ordered_columns].copy()
    selected = selected.merge(
        metadata_order,
        on=["speech_id", "sentence_id"],
        how="left",
        validate="one_to_one",
    )
    if selected["row_order"].isna().any():
        raise ValueError(f"Could not align {path} to metadata row order")
    return selected.sort_values("row_order").reset_index(drop=True)


def align_frames(
    frame_a: pd.DataFrame,
    frame_b: pd.DataFrame,
    *,
    include_labels: bool,
) -> pd.DataFrame:
    merged = frame_a.merge(
        frame_b,
        on=["speech_id", "sentence_id"],
        suffixes=("_a", "_b"),
        how="inner",
    )
    if len(merged) != len(frame_a) or len(merged) != len(frame_b):
        raise ValueError("Fusion inputs are not perfectly aligned by speech_id/sentence_id")

    if include_labels:
        if not np.array_equal(
            merged["true_label_a"].to_numpy(dtype=np.int64),
            merged["true_label_b"].to_numpy(dtype=np.int64),
        ):
            raise ValueError("OOF inputs disagree on true labels")
        merged["true_label"] = merged["true_label_a"].to_numpy(dtype=np.int64)
        merged = merged.drop(columns=["true_label_a", "true_label_b"])

    if "row_order_a" in merged.columns and "row_order_b" in merged.columns:
        if not np.array_equal(
            merged["row_order_a"].to_numpy(dtype=np.int64),
            merged["row_order_b"].to_numpy(dtype=np.int64),
        ):
            raise ValueError("Fusion inputs disagree on metadata row order")
        merged["row_order"] = merged["row_order_a"].to_numpy(dtype=np.int64)
        merged = merged.drop(columns=["row_order_a", "row_order_b"])
    return merged.sort_values("row_order").reset_index(drop=True)


def build_alpha_grid(start: float, end: float, step: float) -> list[float]:
    if step <= 0:
        raise ValueError("alpha-step must be strictly positive")
    if end < start:
        raise ValueError("alpha-end must be >= alpha-start")
    values = np.arange(start, end + step / 2.0, step, dtype=np.float64)
    return [float(round(value, 10)) for value in values]


def weighted_logit_average(
    score_a: np.ndarray,
    score_b: np.ndarray,
    *,
    alpha: float,
) -> np.ndarray:
    clipped_a = np.clip(np.asarray(score_a, dtype=np.float64), 1e-9, 1.0 - 1e-9)
    clipped_b = np.clip(np.asarray(score_b, dtype=np.float64), 1e-9, 1.0 - 1e-9)
    fused_logits = float(alpha) * logit(clipped_a) + (1.0 - float(alpha)) * logit(
        clipped_b
    )
    return expit(fused_logits)


def build_single_span_prior_from_metadata(metadata_path: str) -> Any:
    metadata = pd.read_parquet(metadata_path).reset_index(drop=True)
    sequences = []
    for speech_id, group in metadata.groupby("speech_id", sort=False):
        labels = None
        if "label" in group.columns:
            labels = group["label"].to_numpy(dtype=np.int64, copy=True)
        sequences.append(
            SpeechSequence(
                speech_id=int(speech_id),
                sentence_ids=group["sentence_id"].to_numpy(dtype=np.int64, copy=True),
                embeddings=np.zeros((len(group), 1), dtype=np.float32),
                labels=labels,
            )
        )
    return fit_single_span_prior(sequences, position_bins=0)


def apply_single_span_decoder(
    frame: pd.DataFrame,
    *,
    prior,
    score_column: str,
    min_negative_span: int,
    no_span_bias: float,
) -> pd.Series:
    ordered = frame.sort_values(["speech_id", "sentence_id"]).reset_index()
    fused_scores = np.empty(len(frame), dtype=np.float64)
    for _, group in ordered.groupby("speech_id", sort=False):
        scores = group[score_column].to_numpy(dtype=np.float64)
        probabilities = np.column_stack([1.0 - scores, scores])
        posteriors = posterior_probabilities_by_sequence(
            [probabilities],
            decoder="single_negative_span",
            prior=prior,
            min_span_length=min_negative_span,
            position_prior_weight=0.0,
            no_span_bias=no_span_bias,
        )[0]
        original_indices = group["index"].to_numpy(dtype=np.int64)
        fused_scores[original_indices] = posteriors[:, 1]
    return pd.Series(fused_scores, index=frame.index, name=score_column)


def attach_proxy_summary(
    record: dict[str, Any],
    *,
    proxy_metrics: dict[str, Any],
) -> dict[str, Any]:
    enriched = dict(record)
    enriched["proxy_f1_macro"] = float(proxy_metrics["f1_macro"])
    enriched["proxy_accuracy"] = float(proxy_metrics["accuracy"])
    enriched["proxy_balanced_accuracy"] = float(proxy_metrics["balanced_accuracy"])
    enriched["proxy_agreement"] = float(proxy_metrics["agreement"])
    return enriched


def choose_best_record(frame: pd.DataFrame, *, ranking_mode: str) -> pd.Series:
    has_proxy = "proxy_f1_macro" in frame.columns and frame["proxy_f1_macro"].notna().any()
    if ranking_mode == "clean" or not has_proxy:
        ordered = frame.sort_values(
            ["f1_macro", "balanced_accuracy", "threshold"],
            ascending=[False, False, True],
        ).reset_index(drop=True)
        return ordered.iloc[0]
    if ranking_mode == "proxy":
        ordered = frame.sort_values(
            ["proxy_f1_macro", "proxy_balanced_accuracy", "f1_macro"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
        return ordered.iloc[0]
    ordered = frame.sort_values(
        ["proxy_f1_macro", "f1_macro", "proxy_balanced_accuracy"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    return ordered.iloc[0]


def export_submission_variant(
    frame: pd.DataFrame,
    *,
    output_dir: Path,
    score_column: str,
    threshold: float,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    delta = float(logit(float(threshold)))
    calibrated_scores = apply_score_delta(
        frame[score_column].to_numpy(dtype=np.float64),
        delta,
    )
    labels = np.where(calibrated_scores >= 0.5, "M", "C")
    detailed = frame.copy()
    detailed["prob_mitterrand_calibrated"] = calibrated_scores
    detailed["pred_label_calibrated"] = labels

    detailed_path = output_dir / "test_predictions_detailed.csv"
    raw_score_path = output_dir / "submission_prob_mitterrand_raw.csv"
    calibrated_score_path = output_dir / "submission_prob_mitterrand_calibrated.csv"
    calibrated_label_path = output_dir / "submission_label_calibrated.csv"

    detailed.to_csv(detailed_path, index=False)
    detailed[[score_column]].to_csv(raw_score_path, index=False, header=False)
    detailed[["prob_mitterrand_calibrated"]].to_csv(
        calibrated_score_path,
        index=False,
        header=False,
    )
    detailed[["pred_label_calibrated"]].to_csv(
        calibrated_label_path,
        index=False,
        header=False,
    )

    return {
        "detailed_predictions": str(detailed_path.resolve()),
        "submission_prob_mitterrand_raw": str(raw_score_path.resolve()),
        "submission_prob_mitterrand_calibrated": str(calibrated_score_path.resolve()),
        "submission_label_calibrated": str(calibrated_label_path.resolve()),
        "delta": delta,
        "label_counts": {
            label: int(count)
            for label, count in pd.Series(labels).value_counts().sort_index().items()
        },
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    oof_a = load_score_frame(
        args.oof_a_path,
        metadata_path=args.train_metadata_path,
        score_column=args.oof_a_score_column,
        include_labels=True,
    )
    oof_b = load_score_frame(
        args.oof_b_path,
        metadata_path=args.train_metadata_path,
        score_column=args.oof_b_score_column,
        include_labels=True,
    )
    test_a = load_score_frame(
        args.test_a_path,
        metadata_path=args.test_metadata_path,
        score_column=args.test_a_score_column,
        include_labels=False,
    )
    test_b = load_score_frame(
        args.test_b_path,
        metadata_path=args.test_metadata_path,
        score_column=args.test_b_score_column,
        include_labels=False,
    )

    oof_frame = align_frames(oof_a, oof_b, include_labels=True)
    test_frame = align_frames(test_a, test_b, include_labels=False)

    oof_frame["score_a_logit"] = logit(
        np.clip(oof_frame["score_a"].to_numpy(dtype=np.float64), 1e-9, 1.0 - 1e-9)
    )
    oof_frame["score_b_logit"] = logit(
        np.clip(oof_frame["score_b"].to_numpy(dtype=np.float64), 1e-9, 1.0 - 1e-9)
    )
    test_frame["score_a_logit"] = logit(
        np.clip(test_frame["score_a"].to_numpy(dtype=np.float64), 1e-9, 1.0 - 1e-9)
    )
    test_frame["score_b_logit"] = logit(
        np.clip(test_frame["score_b"].to_numpy(dtype=np.float64), 1e-9, 1.0 - 1e-9)
    )

    thresholds = threshold_grid(
        args.threshold_start,
        args.threshold_end,
        args.threshold_step,
    )
    prior = None
    if args.apply_single_span_decoder:
        prior = build_single_span_prior_from_metadata(args.train_metadata_path)

    weighted_records = []
    for alpha in build_alpha_grid(args.alpha_start, args.alpha_end, args.alpha_step):
        oof_scores = weighted_logit_average(
            oof_frame["score_a"].to_numpy(dtype=np.float64),
            oof_frame["score_b"].to_numpy(dtype=np.float64),
            alpha=alpha,
        )
        test_scores = weighted_logit_average(
            test_frame["score_a"].to_numpy(dtype=np.float64),
            test_frame["score_b"].to_numpy(dtype=np.float64),
            alpha=alpha,
        )

        variant_oof = oof_frame[["speech_id", "sentence_id", "true_label"]].copy()
        variant_test = test_frame[["speech_id", "sentence_id"]].copy()
        variant_oof["prob_mitterrand_fused"] = oof_scores
        variant_test["prob_mitterrand_fused"] = test_scores

        if args.apply_single_span_decoder:
            variant_oof["prob_mitterrand_fused"] = apply_single_span_decoder(
                variant_oof,
                prior=prior,
                score_column="prob_mitterrand_fused",
                min_negative_span=args.min_negative_span,
                no_span_bias=args.no_span_bias,
            ).to_numpy()
            variant_test["prob_mitterrand_fused"] = apply_single_span_decoder(
                variant_test,
                prior=prior,
                score_column="prob_mitterrand_fused",
                min_negative_span=args.min_negative_span,
                no_span_bias=args.no_span_bias,
            ).to_numpy()

        _, best_record = sweep_thresholds(
            variant_oof,
            score_name="prob_mitterrand_fused",
            thresholds=thresholds,
        )
        boundary_summary, _ = compute_boundary_diagnostics(
            variant_oof,
            threshold=float(best_record["threshold"]),
            score_column="prob_mitterrand_fused",
        )
        record = {
            "method": "weighted_logit_average",
            "alpha": float(alpha),
            "threshold": float(best_record["threshold"]),
            "equivalent_score_delta": float(best_record["equivalent_score_delta"]),
            "accuracy": float(best_record["accuracy"]),
            "balanced_accuracy": float(best_record["balanced_accuracy"]),
            "f1_macro": float(best_record["f1_macro"]),
            "mitterrand_count": int(best_record["mitterrand_count"]),
            "chirac_count": int(best_record["chirac_count"]),
            "multi_block_speeches": int(
                boundary_summary["speeches_with_multiple_predicted_mitterrand_blocks"]
            ),
            "start_lag_mean": boundary_summary["start_lag_mean"],
            "end_lag_mean": boundary_summary["end_lag_mean"],
        }

        if args.proxy_labels_path is not None:
            test_labels = np.where(
                apply_score_delta(
                    variant_test["prob_mitterrand_fused"].to_numpy(dtype=np.float64),
                    float(best_record["equivalent_score_delta"]),
                )
                >= 0.5,
                "M",
                "C",
            )
            proxy_metrics = evaluate_proxy_predictions(
                pd.read_csv(args.proxy_labels_path).reset_index(drop=True),
                test_labels,
                predicted_scores=apply_score_delta(
                    variant_test["prob_mitterrand_fused"].to_numpy(dtype=np.float64),
                    float(best_record["equivalent_score_delta"]),
                ),
            ).summary
            record = attach_proxy_summary(record, proxy_metrics=proxy_metrics)

        weighted_records.append(record)

    weighted_frame = pd.DataFrame(weighted_records)
    weighted_frame.to_csv(output_dir / "weighted_average_summary.csv", index=False)

    grouped_targets = speech_target_frame(
        oof_frame.rename(columns={"true_label": "label"}),
        label_column="label",
    )
    splits = build_grouped_splits(
        grouped_targets,
        n_splits=args.stacking_folds,
        seed=args.seed,
    )
    stacked_oof_scores = np.zeros(len(oof_frame), dtype=np.float64)
    oof_features = oof_frame[["score_a_logit", "score_b_logit"]].to_numpy(dtype=np.float64)
    y_binary = (oof_frame["true_label"].to_numpy(dtype=np.int64) == -1).astype(np.int64)

    for train_idx, test_idx in splits:
        train_speech_ids = set(grouped_targets.iloc[train_idx]["speech_id"].astype(int).tolist())
        test_speech_ids = set(grouped_targets.iloc[test_idx]["speech_id"].astype(int).tolist())

        train_mask = oof_frame["speech_id"].isin(train_speech_ids).to_numpy()
        test_mask = oof_frame["speech_id"].isin(test_speech_ids).to_numpy()

        estimator = LogisticRegression(
            random_state=args.seed,
            class_weight="balanced",
            max_iter=1000,
        )
        estimator.fit(oof_features[train_mask], y_binary[train_mask])
        stacked_oof_scores[test_mask] = estimator.predict_proba(oof_features[test_mask])[:, 1]

    final_estimator = LogisticRegression(
        random_state=args.seed,
        class_weight="balanced",
        max_iter=1000,
    )
    final_estimator.fit(oof_features, y_binary)
    stacked_test_scores = final_estimator.predict_proba(
        test_frame[["score_a_logit", "score_b_logit"]].to_numpy(dtype=np.float64)
    )[:, 1]

    stacked_oof = oof_frame[["speech_id", "sentence_id", "true_label"]].copy()
    stacked_test = test_frame[["speech_id", "sentence_id"]].copy()
    stacked_oof["prob_mitterrand_fused"] = stacked_oof_scores
    stacked_test["prob_mitterrand_fused"] = stacked_test_scores

    if args.apply_single_span_decoder:
        stacked_oof["prob_mitterrand_fused"] = apply_single_span_decoder(
            stacked_oof,
            prior=prior,
            score_column="prob_mitterrand_fused",
            min_negative_span=args.min_negative_span,
            no_span_bias=args.no_span_bias,
        ).to_numpy()
        stacked_test["prob_mitterrand_fused"] = apply_single_span_decoder(
            stacked_test,
            prior=prior,
            score_column="prob_mitterrand_fused",
            min_negative_span=args.min_negative_span,
            no_span_bias=args.no_span_bias,
        ).to_numpy()

    _, stacked_best = sweep_thresholds(
        stacked_oof,
        score_name="prob_mitterrand_fused",
        thresholds=thresholds,
    )
    stacked_boundary, _ = compute_boundary_diagnostics(
        stacked_oof,
        threshold=float(stacked_best["threshold"]),
        score_column="prob_mitterrand_fused",
    )
    stacking_record: dict[str, Any] = {
        "method": "logistic_stacking",
        "alpha": float("nan"),
        "threshold": float(stacked_best["threshold"]),
        "equivalent_score_delta": float(stacked_best["equivalent_score_delta"]),
        "accuracy": float(stacked_best["accuracy"]),
        "balanced_accuracy": float(stacked_best["balanced_accuracy"]),
        "f1_macro": float(stacked_best["f1_macro"]),
        "mitterrand_count": int(stacked_best["mitterrand_count"]),
        "chirac_count": int(stacked_best["chirac_count"]),
        "multi_block_speeches": int(
            stacked_boundary["speeches_with_multiple_predicted_mitterrand_blocks"]
        ),
        "start_lag_mean": stacked_boundary["start_lag_mean"],
        "end_lag_mean": stacked_boundary["end_lag_mean"],
    }
    if args.proxy_labels_path is not None:
        proxy_metrics = evaluate_proxy_predictions(
            pd.read_csv(args.proxy_labels_path).reset_index(drop=True),
            np.where(
                apply_score_delta(
                    stacked_test["prob_mitterrand_fused"].to_numpy(dtype=np.float64),
                    float(stacked_best["equivalent_score_delta"]),
                )
                >= 0.5,
                "M",
                "C",
            ),
            predicted_scores=apply_score_delta(
                stacked_test["prob_mitterrand_fused"].to_numpy(dtype=np.float64),
                float(stacked_best["equivalent_score_delta"]),
            ),
        ).summary
        stacking_record = attach_proxy_summary(
            stacking_record,
            proxy_metrics=proxy_metrics,
        )

    comparison_frame = pd.concat(
        [
            weighted_frame,
            pd.DataFrame([stacking_record]),
        ],
        ignore_index=True,
    )
    comparison_frame.to_csv(output_dir / "fusion_comparison_summary.csv", index=False)
    best_record = choose_best_record(comparison_frame, ranking_mode=args.ranking_mode)

    if str(best_record["method"]) == "logistic_stacking":
        best_test_frame = stacked_test.copy()
        best_score_column = "prob_mitterrand_fused"
    else:
        best_alpha = float(best_record["alpha"])
        best_test_frame = test_frame[["speech_id", "sentence_id"]].copy()
        best_test_frame["prob_mitterrand_fused"] = weighted_logit_average(
            test_frame["score_a"].to_numpy(dtype=np.float64),
            test_frame["score_b"].to_numpy(dtype=np.float64),
            alpha=best_alpha,
        )
        if args.apply_single_span_decoder:
            best_test_frame["prob_mitterrand_fused"] = apply_single_span_decoder(
                best_test_frame,
                prior=prior,
                score_column="prob_mitterrand_fused",
                min_negative_span=args.min_negative_span,
                no_span_bias=args.no_span_bias,
            ).to_numpy()
        best_score_column = "prob_mitterrand_fused"

    best_outputs = export_submission_variant(
        best_test_frame,
        output_dir=output_dir / "best_submission",
        score_column=best_score_column,
        threshold=float(best_record["threshold"]),
    )

    summary = {
        "config": vars(args),
        "best_record": best_record.to_dict(),
        "comparison_summary_csv": str(
            (output_dir / "fusion_comparison_summary.csv").resolve()
        ),
        "weighted_average_summary_csv": str(
            (output_dir / "weighted_average_summary.csv").resolve()
        ),
        "best_outputs": best_outputs,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
