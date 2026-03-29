from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from rital_nlp_project.presidents.eval_utils import (
    evaluate_proxy_predictions,
    load_prediction_column,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a presidents prediction file against the accepted archive "
            "proxy labels using the exact row order of row_labels_with_evidence.csv."
        )
    )
    parser.add_argument(
        "--prediction-path",
        required=True,
        help="CSV prediction file. Can be a one-column submission file or a detailed CSV.",
    )
    parser.add_argument(
        "--proxy-labels-path",
        default="Dataset/out/presidents_archive_matching_dgx_final/row_labels_with_evidence.csv",
        help="Accepted/unresolved proxy labels CSV in hidden-test row order.",
    )
    parser.add_argument(
        "--label-column",
        default=None,
        help="Optional label column name when --prediction-path is a detailed CSV.",
    )
    parser.add_argument(
        "--score-path",
        default=None,
        help="Optional score file aligned with the prediction rows.",
    )
    parser.add_argument(
        "--score-column",
        default=None,
        help="Optional score column name when --score-path is a detailed CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory where metrics and joined accepted-row diagnostics are written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    proxy_path = Path(args.proxy_labels_path)
    prediction_path = Path(args.prediction_path)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else prediction_path.parent / "proxy_eval"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    proxy_frame = pd.read_csv(proxy_path).reset_index(drop=True)
    expected_rows = int(len(proxy_frame))

    predicted_labels = load_prediction_column(
        str(prediction_path),
        expected_rows=expected_rows,
        explicit_column=args.label_column,
        kind="label",
    )

    predicted_scores = None
    if args.score_path is not None:
        predicted_scores = load_prediction_column(
            args.score_path,
            expected_rows=expected_rows,
            explicit_column=args.score_column,
            kind="score",
        )

    result = evaluate_proxy_predictions(
        proxy_frame,
        predicted_labels,
        predicted_scores=predicted_scores,
    )

    accepted_rows = result.accepted_rows.copy()
    unresolved_rows = result.unresolved_rows.copy()

    summary = {
        "prediction_path": str(prediction_path.resolve()),
        "proxy_labels_path": str(proxy_path.resolve()),
        "accepted_rows_path": str((output_dir / "accepted_rows_with_predictions.csv").resolve()),
        "unresolved_rows_path": str((output_dir / "unresolved_rows_with_predictions.csv").resolve()),
        "per_method_path": str((output_dir / "per_method_metrics.csv").resolve()),
        **result.summary,
    }

    accepted_rows.to_csv(output_dir / "accepted_rows_with_predictions.csv", index=False)
    unresolved_rows.to_csv(output_dir / "unresolved_rows_with_predictions.csv", index=False)
    result.per_method.to_csv(output_dir / "per_method_metrics.csv", index=False)
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
