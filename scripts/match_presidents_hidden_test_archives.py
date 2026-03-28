from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from rital_nlp_project.presidents.archive_labeling import (
    ArchiveMatcher,
    apply_row_acceptance_threshold,
    build_blocks,
    build_row_matches,
    candidate_matches_to_frame,
    load_archive_docs,
    load_model_triage,
    parse_hidden_test_corpus,
    recover_review_rows,
    score_blocks,
    submission_from_row_matches,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Match hidden presidents test rows against official archive documents and "
            "export evidence-backed row labels."
        )
    )
    parser.add_argument(
        "--input-path",
        default="Dataset/raw/test/presidents/corpus.tache1.test.utf8",
        help="Hidden test corpus to label.",
    )
    parser.add_argument(
        "--archive-docs-path",
        default="Dataset/out/presidents_archive_labeling/archive_docs.parquet",
        help="Local archive document table produced by the crawler.",
    )
    parser.add_argument(
        "--output-dir",
        default="Dataset/out/presidents_archive_matching",
        help="Directory where matching artifacts are written.",
    )
    parser.add_argument(
        "--model-triage-path",
        default="Dataset/out/presidents_lstm_submission_cv_calibrated_full/test_predictions_detailed.csv",
        help="Optional existing model prediction file used only for review prioritization.",
    )
    parser.add_argument(
        "--threshold-summary-path",
        default=None,
        help="Optional JSON summary from the backtest script containing a recommended row score threshold.",
    )
    parser.add_argument("--block-size", type=int, default=5)
    parser.add_argument("--block-step", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--block-min-score",
        type=float,
        default=0.46,
        help="Minimum block score required for an auto-accepted supporting block.",
    )
    parser.add_argument(
        "--block-min-margin",
        type=float,
        default=0.05,
        help="Minimum gap between the best and second-best block candidate.",
    )
    parser.add_argument(
        "--block-min-sentence-hits",
        type=int,
        default=2,
        help="Minimum number of block sentences that must align inside the winning archive window.",
    )
    parser.add_argument(
        "--row-min-score",
        type=float,
        default=0.62,
        help=(
            "Minimum row score for the final accepted label. "
            "If threshold-summary-path is given, its recommendation overrides this value."
        ),
    )
    parser.add_argument(
        "--row-conflict-margin",
        type=float,
        default=0.05,
        help="Rows with competing accepted blocks within this score gap are marked for review.",
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
    parser.add_argument(
        "--exact-sentence-min-full-tokens",
        type=int,
        default=5,
        help=(
            "Minimum number of normalized tokens required before exact-sentence fallback "
            "can auto-accept a review row."
        ),
    )
    parser.add_argument(
        "--single-source-sentence-min-score",
        type=float,
        default=0.46,
        help=(
            "Minimum sentence-level score for the second-pass recovery used only inside "
            "speeches that already have exactly one accepted source document."
        ),
    )
    return parser.parse_args()


def load_threshold(path: str | None, fallback: float) -> float:
    if path is None:
        return fallback
    summary = json.loads(Path(path).read_text(encoding="utf-8"))
    threshold = summary.get("recommended_threshold")
    if threshold is None:
        return fallback
    return float(threshold)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = parse_hidden_test_corpus(args.input_path)
    archive_docs = load_archive_docs(args.archive_docs_path)
    triage_frame = load_model_triage(args.model_triage_path)
    row_min_score = load_threshold(args.threshold_summary_path, args.row_min_score)

    blocks = build_blocks(
        rows,
        dataset_name="test",
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
        log_prefix="hidden_test_match",
    )

    speech_candidates = candidate_matches_to_frame(candidate_matches)
    speech_candidates.to_csv(output_dir / "speech_candidates.csv", index=False)

    row_matches = build_row_matches(
        rows,
        candidate_matches,
        triage_frame=triage_frame,
        row_conflict_margin=args.row_conflict_margin,
    )
    row_matches = apply_row_acceptance_threshold(
        row_matches,
        min_score=row_min_score,
    )
    row_matches = recover_review_rows(
        row_matches,
        archive_docs=archive_docs,
        matcher=matcher,
        top_k=args.top_k,
        exact_sentence_min_full_tokens=args.exact_sentence_min_full_tokens,
        single_source_sentence_min_score=args.single_source_sentence_min_score,
        single_source_sentence_workers=args.workers,
        single_source_sentence_chunksize=args.chunksize,
        progress_every_blocks=args.progress_every_blocks,
        log_prefix="hidden_test_review_recovery",
    )
    row_matches.to_csv(output_dir / "row_labels_with_evidence.csv", index=False)

    submission = submission_from_row_matches(row_matches)
    submission.to_csv(
        output_dir / "submission_label_reconstructed.csv",
        index=False,
        header=False,
    )

    summary = {
        "input_rows": int(len(rows)),
        "unique_speeches": int(rows["speech_id"].nunique()),
        "blocks_scored": int(len(blocks)),
        "candidate_rows": int(len(speech_candidates)),
        "accepted_rows": int(row_matches["review_status"].eq("accepted").sum()),
        "review_rows": int(row_matches["review_status"].ne("accepted").sum()),
        "model_disagreement_rows": int(row_matches["model_disagreement"].sum()),
        "row_min_score": float(row_min_score),
        "workers": int(args.workers),
        "chunksize": None if args.chunksize is None else int(args.chunksize),
        "progress_every_blocks": int(args.progress_every_blocks),
        "exact_sentence_min_full_tokens": int(args.exact_sentence_min_full_tokens),
        "single_source_sentence_min_score": float(args.single_source_sentence_min_score),
        "resolution_method_counts": {
            str(method): int(count)
            for method, count in row_matches["resolution_method"].fillna("unresolved").value_counts().items()
        },
        "outputs": {
            "speech_candidates_csv": str((output_dir / "speech_candidates.csv").resolve()),
            "row_labels_with_evidence_csv": str((output_dir / "row_labels_with_evidence.csv").resolve()),
            "submission_label_reconstructed_csv": str((output_dir / "submission_label_reconstructed.csv").resolve()),
        },
    }
    (output_dir / "match_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
