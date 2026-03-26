from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep a logit shift over a one-column presidents submission score file, "
            "compare it to a leaderboard reference file, and export shortlisted calibrated variants."
        )
    )
    parser.add_argument(
        "--input-path",
        default=(
            "Dataset/out/presidents_lstm_submission_full_data_19ep/"
            "submission_prob_single_negative_span_inverted.csv"
        ),
        help="One-column submission score file interpreted as P(Mitterrand).",
    )
    parser.add_argument(
        "--reference-path",
        default="Dataset/out/Leaderboard/submission-pres-4_88f1.csv",
        help="Leaderboard reference score file used for hard-prediction agreement.",
    )
    parser.add_argument(
        "--output-dir",
        default="Dataset/out/presidents_submission_score_calibration",
    )
    parser.add_argument("--delta-start", type=float, default=0.0)
    parser.add_argument("--delta-end", type=float, default=8.0)
    parser.add_argument("--delta-step", type=float, default=0.05)
    parser.add_argument(
        "--target-counts",
        nargs="+",
        type=int,
        default=[3300, 3500, 3700],
    )
    return parser.parse_args()


def load_scores(path: str | Path) -> pd.Series:
    return pd.read_csv(path, header=None).iloc[:, 0].astype(float)


def build_grid(start: float, end: float, step: float) -> list[float]:
    if step <= 0:
        raise ValueError("delta-step must be strictly positive")
    if end < start:
        raise ValueError("delta-end must be >= delta-start")
    num_steps = int(np.floor((end - start) / step))
    values = [round(start + step * idx, 10) for idx in range(num_steps + 1)]
    if values[-1] < end:
        values.append(round(end, 10))
    return values


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_scores = load_scores(args.input_path)
    reference_scores = load_scores(args.reference_path)
    if len(base_scores) != len(reference_scores):
        raise ValueError(
            f"Input/reference length mismatch: {len(base_scores)} != {len(reference_scores)}"
        )

    reference_labels = reference_scores.to_numpy() >= 0.5
    base_logits = logit(np.clip(base_scores.to_numpy(), 1e-9, 1 - 1e-9))

    sweep_records = []
    best_record = None

    for delta in build_grid(args.delta_start, args.delta_end, args.delta_step):
        shifted_scores = expit(base_logits - delta)
        shifted_labels = shifted_scores >= 0.5
        agreement = shifted_labels == reference_labels
        record = {
            "delta": float(delta),
            "mitterrand_count": int(shifted_labels.sum()),
            "chirac_count": int(len(shifted_labels) - shifted_labels.sum()),
            "agreement_to_reference": float(agreement.mean()),
            "disagreement_to_reference": int((~agreement).sum()),
        }
        sweep_records.append(record)
        if best_record is None or record["agreement_to_reference"] > best_record["agreement_to_reference"]:
            best_record = record

    sweep_frame = pd.DataFrame(sweep_records).sort_values("delta").reset_index(drop=True)
    sweep_frame.to_csv(output_dir / "score_shift_sweep_summary.csv", index=False)

    selected_deltas = set()
    for target in args.target_counts:
        row = sweep_frame.iloc[
            (sweep_frame["mitterrand_count"] - target).abs().argmin()
        ]
        selected_deltas.add(float(row["delta"]))
    if best_record is not None:
        selected_deltas.add(float(best_record["delta"]))
    selected_deltas.add(0.0)

    selected_records = []
    for delta in sorted(selected_deltas):
        shifted_scores = expit(base_logits - delta)
        shifted_labels = shifted_scores >= 0.5
        agreement = shifted_labels == reference_labels

        variant_dir = output_dir / f"delta_{delta:.2f}".replace(".", "p")
        variant_dir.mkdir(parents=True, exist_ok=True)

        score_path = variant_dir / "submission_prob_single_negative_span_inverted.csv"
        pd.DataFrame(shifted_scores).to_csv(score_path, index=False, header=False)

        label_path = variant_dir / "submission_label_thresholded.csv"
        pd.DataFrame(
            np.where(shifted_labels, "M", "C")
        ).to_csv(label_path, index=False, header=False)

        record = {
            "delta": float(delta),
            "mitterrand_count": int(shifted_labels.sum()),
            "agreement_to_reference": float(agreement.mean()),
            "disagreement_to_reference": int((~agreement).sum()),
            "variant_dir": str(variant_dir.resolve()),
            "outputs": {
                "submission_prob_single_negative_span_inverted": str(score_path.resolve()),
                "submission_label_thresholded": str(label_path.resolve()),
            },
        }
        selected_records.append(record)
        (variant_dir / "metrics.json").write_text(json.dumps(record, indent=2))

    summary = {
        "input_path": str(Path(args.input_path).resolve()),
        "reference_path": str(Path(args.reference_path).resolve()),
        "target_counts": args.target_counts,
        "best_record": best_record,
        "selected_variants": selected_records,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
