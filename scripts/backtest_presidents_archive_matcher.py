from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from rital_nlp_project.presidents.archive_labeling import (
    ArchiveMatcher,
    TRAIN_LABEL_TO_SUBMISSION,
    apply_row_acceptance_threshold,
    build_blocks,
    build_row_matches,
    candidate_matches_to_frame,
    choose_threshold,
    compute_threshold_grid,
    load_archive_docs,
    load_labeled_rows,
    score_blocks,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Backtest official-archive matching on labeled presidents training rows "
            "and calibrate a high-precision row acceptance threshold."
        )
    )
    parser.add_argument(
        "--input-path",
        default="Dataset/clean/presidents_clean_bert.parquet",
        help="Labeled presidents parquet used for backtesting.",
    )
    parser.add_argument(
        "--archive-docs-path",
        default="Dataset/out/presidents_archive_labeling/archive_docs.parquet",
        help="Local archive document table produced by the crawler.",
    )
    parser.add_argument(
        "--output-dir",
        default="Dataset/out/presidents_archive_backtest",
        help="Directory where backtest artifacts are written.",
    )
    parser.add_argument("--block-size", type=int, default=5)
    parser.add_argument("--block-step", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--block-min-score", type=float, default=0.46)
    parser.add_argument("--block-min-margin", type=float, default=0.05)
    parser.add_argument("--block-min-sentence-hits", type=int, default=2)
    parser.add_argument(
        "--min-precision",
        type=float,
        default=0.99,
        help="Minimum accepted-row precision required when choosing the row threshold.",
    )
    parser.add_argument(
        "--threshold-start",
        type=float,
        default=0.2,
        help="Start of the threshold sweep interval.",
    )
    parser.add_argument(
        "--threshold-end",
        type=float,
        default=0.95,
        help="End of the threshold sweep interval.",
    )
    parser.add_argument(
        "--threshold-step",
        type=float,
        default=0.01,
        help="Step size for the threshold sweep interval.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes used for block scoring. Use 1 to stay serial.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=None,
        help="Number of query blocks assigned to each process task. Default picks an automatic size.",
    )
    parser.add_argument(
        "--progress-every-blocks",
        type=int,
        default=250,
        help="Emit a progress log event after this many blocks have completed. Use 0 to disable.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_labeled_rows(args.input_path)
    archive_docs = load_archive_docs(args.archive_docs_path)
    blocks = build_blocks(
        rows,
        dataset_name="train",
        block_size=args.block_size,
        block_step=args.block_step,
    )
    matcher = ArchiveMatcher(archive_docs)

    candidate_matches = score_blocks(
        matcher,
        blocks,
        top_k=args.top_k,
        min_score=args.block_min_score,
        min_margin=args.block_min_margin,
        min_sentence_hits=args.block_min_sentence_hits,
        workers=args.workers,
        chunksize=args.chunksize,
        progress_every_blocks=args.progress_every_blocks,
        log_prefix="backtest",
    )

    candidate_matches_to_frame(candidate_matches).to_csv(
        output_dir / "speech_candidates.csv",
        index=False,
    )

    row_matches = build_row_matches(rows, candidate_matches)
    gold_labels = rows["label"].map(TRAIN_LABEL_TO_SUBMISSION)
    threshold_grid = compute_threshold_grid(
        row_matches,
        gold_labels=gold_labels,
        min_threshold=args.threshold_start,
        max_threshold=args.threshold_end,
        step=args.threshold_step,
    )
    threshold_grid.to_csv(output_dir / "threshold_sweep.csv", index=False)
    recommendation = choose_threshold(
        threshold_grid,
        min_precision=args.min_precision,
    )

    recommended_threshold = recommendation["recommended_threshold"]
    if recommended_threshold is None:
        recommended_threshold = float(args.threshold_end)
    calibrated_rows = apply_row_acceptance_threshold(
        row_matches,
        min_score=float(recommended_threshold),
    )
    calibrated_rows = calibrated_rows.copy()
    calibrated_rows["gold_label"] = gold_labels
    calibrated_rows["correct"] = (
        calibrated_rows["review_status"].eq("accepted")
        & calibrated_rows["matched_label"].eq(calibrated_rows["gold_label"])
    )
    calibrated_rows.to_csv(output_dir / "backtest_rows.csv", index=False)

    summary = {
        "input_rows": int(len(rows)),
        "unique_speeches": int(rows["speech_id"].nunique()),
        "blocks_scored": int(len(blocks)),
        "accepted_rows_at_recommended_threshold": int(
            calibrated_rows["review_status"].eq("accepted").sum()
        ),
        "recommended_threshold": recommendation["recommended_threshold"],
        "recommended_precision": recommendation["precision"],
        "recommended_coverage": recommendation["coverage"],
        "min_precision_constraint": float(args.min_precision),
        "block_min_score": float(args.block_min_score),
        "block_min_margin": float(args.block_min_margin),
        "block_min_sentence_hits": int(args.block_min_sentence_hits),
        "workers": int(args.workers),
        "chunksize": None if args.chunksize is None else int(args.chunksize),
        "progress_every_blocks": int(args.progress_every_blocks),
        "outputs": {
            "threshold_sweep_csv": str((output_dir / "threshold_sweep.csv").resolve()),
            "backtest_rows_csv": str((output_dir / "backtest_rows.csv").resolve()),
            "speech_candidates_csv": str((output_dir / "speech_candidates.csv").resolve()),
        },
    }
    (output_dir / "backtest_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
