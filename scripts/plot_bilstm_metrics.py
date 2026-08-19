#!/usr/bin/env python3
"""Generate report figures from a completed presidents BiLSTM run."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("Dataset/out/presidents_lstm_simple_full"),
        help="Directory containing oof_predictions.csv and threshold_sweep.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/RITAL_NLP_Projet-2_h22/figures"),
        help="Destination for the generated PDF figures",
    )
    parser.add_argument("--speech-id", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.875)
    return parser.parse_args()


def require_columns(frame: pd.DataFrame, columns: set[str], source: Path) -> None:
    missing = columns.difference(frame.columns)
    if missing:
        raise ValueError(f"{source} is missing columns: {', '.join(sorted(missing))}")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({"font.size": 12})

    oof_source = args.input_dir / "oof_predictions.csv"
    sweep_source = args.input_dir / "threshold_sweep.csv"
    oof_df = pd.read_csv(oof_source)
    sweep_df = pd.read_csv(sweep_source)
    require_columns(
        oof_df,
        {"speech_id", "sentence_id", "true_label", "prob_mitterrand_raw", "fold"},
        oof_source,
    )
    require_columns(
        sweep_df,
        {"threshold", "f1_macro", "accuracy", "balanced_accuracy"},
        sweep_source,
    )

    speech_df = oof_df[oof_df["speech_id"] == args.speech_id].sort_values("sentence_id")
    if speech_df.empty:
        available = ", ".join(str(value) for value in sorted(oof_df["speech_id"].unique()))
        raise ValueError(f"speech_id {args.speech_id} is unavailable; choose one of: {available}")

    plt.figure(figsize=(10, 4))
    plt.plot(
        speech_df["sentence_id"],
        (speech_df["true_label"] == 1).astype(int),
        label="True Mitterrand",
        color="teal",
        drawstyle="steps-mid",
        linewidth=2,
    )
    plt.plot(
        speech_df["sentence_id"],
        speech_df["prob_mitterrand_raw"],
        label="Predicted P(Mitterrand)",
        color="orange",
        alpha=0.9,
        linewidth=2,
    )
    plt.axhline(args.threshold, color="gray", linestyle="--", label=f"Threshold ({args.threshold:g})")
    plt.xlabel("Sentence Position")
    plt.ylabel("Probability / State")
    plt.title(f"BiLSTM Speech Probability Trajectory (Speech {args.speech_id})")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(args.output_dir / "presidents_lstm_trajectory.pdf")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(sweep_df["threshold"], sweep_df["f1_macro"], label="Macro-F1", color="indigo", linewidth=2.5)
    plt.plot(sweep_df["threshold"], sweep_df["accuracy"], label="Accuracy", color="green", linewidth=1.5, alpha=0.7)
    plt.plot(sweep_df["threshold"], sweep_df["balanced_accuracy"], label="Balanced Accuracy", color="crimson", linewidth=1.5, alpha=0.7)
    plt.axvline(args.threshold, color="black", linestyle="--", label=f"Threshold ({args.threshold:g})")
    plt.xlabel("Decision Threshold (P_Mitterrand >= t)")
    plt.ylabel("Score")
    plt.title("Out-of-Fold Threshold Calibration Sweep")
    plt.legend(loc="lower center")
    plt.tight_layout()
    plt.savefig(args.output_dir / "presidents_lstm_sweep.pdf")
    plt.close()

    metrics = []
    for fold in sorted(oof_df["fold"].unique()):
        fold_df = oof_df[oof_df["fold"] == fold]
        y_true = (fold_df["true_label"] == 1).astype(int)
        y_pred = (fold_df["prob_mitterrand_raw"] >= args.threshold).astype(int)
        metrics.append(
            {
                "Fold": f"Fold {fold}",
                "Macro-F1": f1_score(y_true, y_pred, average="macro"),
                "Precision (M)": precision_score(y_true, y_pred, zero_division=0),
                "Recall (M)": recall_score(y_true, y_pred, zero_division=0),
                "Accuracy": accuracy_score(y_true, y_pred),
            }
        )

    metrics_df = pd.DataFrame(metrics).set_index("Fold")
    metrics_df.plot(kind="bar", figsize=(10, 5), colormap="viridis", edgecolor="black")
    plt.ylim(0.7, 1.0)
    plt.title(f"OOF Cross-Validation Stability (Threshold={args.threshold:g})")
    plt.ylabel("Score")
    plt.xticks(rotation=0)
    plt.legend(loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.2))
    plt.tight_layout()
    plt.savefig(args.output_dir / "presidents_lstm_folds.pdf")
    plt.close()

    print(f"Figures generated in {args.output_dir}")


if __name__ == "__main__":
    main()
