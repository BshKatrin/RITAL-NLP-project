from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from rital_nlp_project.presidents.eval_utils import evaluate_proxy_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Repair presidents test predictions written in speech/sentence sort order "
            "by restoring the canonical hidden-test row order from metadata."
        )
    )
    parser.add_argument(
        "--detailed-path",
        required=True,
        help="Detailed prediction CSV with speech_id/sentence_id columns.",
    )
    parser.add_argument(
        "--test-metadata-path",
        default="Dataset/clean/presidents_test_clean_bert.parquet",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory where repaired outputs are written. Defaults to <detailed parent>/canonical_order.",
    )
    parser.add_argument(
        "--raw-score-column",
        default="prob_mitterrand_raw",
    )
    parser.add_argument(
        "--calibrated-score-column",
        default="prob_mitterrand_calibrated",
    )
    parser.add_argument(
        "--label-column",
        default="pred_label_calibrated",
    )
    parser.add_argument(
        "--proxy-labels-path",
        default=None,
        help="Optional accepted-proxy CSV for immediate rescoring.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    detailed_path = Path(args.detailed_path)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else detailed_path.parent / "canonical_order"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    detailed = pd.read_csv(detailed_path)
    required = {"speech_id", "sentence_id"}
    missing = required - set(detailed.columns)
    if missing:
        raise ValueError(
            f"Detailed file {detailed_path} is missing required columns: {sorted(missing)}"
        )

    metadata = pd.read_parquet(args.test_metadata_path).reset_index(drop=True)
    canonical = metadata[["speech_id", "sentence_id"]].copy()
    canonical["row_order"] = np.arange(len(canonical), dtype=np.int64)

    reordered = canonical.merge(
        detailed,
        on=["speech_id", "sentence_id"],
        how="left",
        validate="one_to_one",
        sort=False,
    )
    value_columns = [column for column in detailed.columns if column not in required]
    if reordered[value_columns].isna().all(axis=None):
        raise ValueError("Failed to align any detailed prediction rows to canonical metadata")
    if reordered[value_columns].isna().any(axis=None):
        missing_rows = int(reordered[value_columns].isna().all(axis=1).sum())
        raise ValueError(
            f"Some prediction rows could not be aligned to canonical metadata: {missing_rows}"
        )

    ordered_columns = ["speech_id", "sentence_id", *value_columns]
    repaired = reordered.sort_values("row_order").reset_index(drop=True)[ordered_columns]

    detailed_out = output_dir / "test_predictions_detailed.csv"
    repaired.to_csv(detailed_out, index=False)

    outputs: dict[str, str] = {
        "test_predictions_detailed": str(detailed_out.resolve()),
    }

    if args.raw_score_column in repaired.columns:
        raw_path = output_dir / "submission_prob_mitterrand_raw.csv"
        repaired[[args.raw_score_column]].to_csv(raw_path, index=False, header=False)
        outputs["submission_prob_mitterrand_raw"] = str(raw_path.resolve())

    if args.calibrated_score_column in repaired.columns:
        calibrated_path = output_dir / "submission_prob_mitterrand_calibrated.csv"
        repaired[[args.calibrated_score_column]].to_csv(
            calibrated_path,
            index=False,
            header=False,
        )
        outputs["submission_prob_mitterrand_calibrated"] = str(calibrated_path.resolve())

    if args.label_column in repaired.columns:
        label_path = output_dir / "submission_label_calibrated.csv"
        repaired[[args.label_column]].to_csv(label_path, index=False, header=False)
        outputs["submission_label_calibrated"] = str(label_path.resolve())

    summary: dict[str, object] = {
        "detailed_path": str(detailed_path.resolve()),
        "test_metadata_path": str(Path(args.test_metadata_path).resolve()),
        "rows": int(len(repaired)),
        "output_dir": str(output_dir.resolve()),
        "outputs": outputs,
    }

    if args.proxy_labels_path is not None and args.label_column in repaired.columns:
        proxy_path = Path(args.proxy_labels_path)
        proxy_frame = pd.read_csv(proxy_path).reset_index(drop=True)
        predicted_scores = None
        score_column = None
        if args.calibrated_score_column in repaired.columns:
            score_column = args.calibrated_score_column
        elif args.raw_score_column in repaired.columns:
            score_column = args.raw_score_column
        if score_column is not None:
            predicted_scores = repaired[score_column].to_numpy(dtype=np.float64)

        proxy_eval = evaluate_proxy_predictions(
            proxy_frame,
            repaired[args.label_column].tolist(),
            predicted_scores=predicted_scores,
        )
        proxy_output_dir = output_dir / "proxy_eval"
        proxy_output_dir.mkdir(parents=True, exist_ok=True)
        proxy_eval.accepted_rows.to_csv(
            proxy_output_dir / "accepted_rows_with_predictions.csv",
            index=False,
        )
        proxy_eval.unresolved_rows.to_csv(
            proxy_output_dir / "unresolved_rows_with_predictions.csv",
            index=False,
        )
        proxy_eval.per_method.to_csv(
            proxy_output_dir / "per_method_metrics.csv",
            index=False,
        )
        proxy_summary = {
            "prediction_path": outputs["submission_label_calibrated"],
            "proxy_labels_path": str(proxy_path.resolve()),
            "accepted_rows_path": str(
                (proxy_output_dir / "accepted_rows_with_predictions.csv").resolve()
            ),
            "unresolved_rows_path": str(
                (proxy_output_dir / "unresolved_rows_with_predictions.csv").resolve()
            ),
            "per_method_path": str((proxy_output_dir / "per_method_metrics.csv").resolve()),
            **proxy_eval.summary,
        }
        (proxy_output_dir / "metrics.json").write_text(json.dumps(proxy_summary, indent=2))
        summary["proxy_eval"] = proxy_summary

    (output_dir / "repair_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
