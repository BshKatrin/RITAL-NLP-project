from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare two president submission score files row-by-row using a 0.5 threshold "
            "and export agreement summaries plus visualizations."
        )
    )
    parser.add_argument(
        "--head-prob-path",
        default="Dataset/out/ekatModels/presidents_test_pred_head.csv",
        help="CSV from the head model. The prob_1 column is used.",
    )
    parser.add_argument(
        "--lstm-prob-path",
        default=(
            "Dataset/out/presidents_lstm_submission_checkpoint/"
            "submission_prob_single_negative_span_inverted.csv"
        ),
        help="One-column CSV from the LSTM model.",
    )
    parser.add_argument(
        "--metadata-path",
        default="Dataset/clean/presidents_test_clean_bert.parquet",
        help="President test metadata parquet used to recover speech and sentence ids.",
    )
    parser.add_argument(
        "--output-dir",
        default="Dataset/out/presidents_model_compare",
        help="Directory where comparison CSVs, plots and summary JSON are written.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold used to convert scores into predicted presidents.",
    )
    parser.add_argument(
        "--positive-label",
        default="Mitterrand",
        help="Label assigned when probability >= threshold.",
    )
    parser.add_argument(
        "--negative-label",
        default="Chirac",
        help="Label assigned when probability < threshold.",
    )
    parser.add_argument(
        "--top-speeches",
        type=int,
        default=6,
        help="How many most-disagreed speeches to visualize in detail.",
    )
    return parser.parse_args()


def load_scores(path: str | Path) -> pd.Series:
    path = Path(path)
    if path.suffix.lower() != ".csv":
        raise ValueError(f"Expected a CSV file, got: {path}")

    frame = pd.read_csv(path)
    if "prob_1" in frame.columns:
        return frame["prob_1"].astype(float)

    raw_frame = pd.read_csv(path, header=None)
    if raw_frame.shape[1] == 1:
        return raw_frame.iloc[:, 0].astype(float)

    raise ValueError(
        f"Could not infer score column in {path}. Expected prob_1 or a single-column CSV."
    )


def scores_to_labels(
    scores: pd.Series,
    *,
    threshold: float,
    positive_label: str,
    negative_label: str,
) -> pd.Series:
    return pd.Series(
        np.where(scores >= threshold, positive_label, negative_label),
        index=scores.index,
        name="predicted_president",
    )


def build_comparison_frame(
    metadata: pd.DataFrame,
    *,
    head_scores: pd.Series,
    lstm_scores: pd.Series,
    threshold: float,
    positive_label: str,
    negative_label: str,
) -> pd.DataFrame:
    if len(metadata) != len(head_scores) or len(metadata) != len(lstm_scores):
        raise ValueError(
            "Metadata and score files must have the same number of rows: "
            f"{len(metadata)} vs {len(head_scores)} vs {len(lstm_scores)}"
        )

    frame = metadata.copy().reset_index(drop=True)
    frame["prob_head"] = head_scores.reset_index(drop=True)
    frame["prob_lstm"] = lstm_scores.reset_index(drop=True)
    frame["pred_head"] = scores_to_labels(
        frame["prob_head"],
        threshold=threshold,
        positive_label=positive_label,
        negative_label=negative_label,
    )
    frame["pred_lstm"] = scores_to_labels(
        frame["prob_lstm"],
        threshold=threshold,
        positive_label=positive_label,
        negative_label=negative_label,
    )
    frame["head_is_positive"] = frame["pred_head"] == positive_label
    frame["lstm_is_positive"] = frame["pred_lstm"] == positive_label
    frame["agree"] = frame["pred_head"] == frame["pred_lstm"]
    frame["agreement_type"] = np.where(frame["agree"], "agree", "disagree")
    return frame


def build_speech_summary(comparison: pd.DataFrame) -> pd.DataFrame:
    summary = (
        comparison.groupby("speech_id", sort=True)
        .agg(
            sentences=("sentence_id", "size"),
            head_positive=("head_is_positive", "sum"),
            lstm_positive=("lstm_is_positive", "sum"),
            agreements=("agree", "sum"),
            disagreements=("agree", lambda s: int((~s).sum())),
            agreement_rate=("agree", "mean"),
        )
        .reset_index()
    )
    return summary.sort_values(
        by=["agreement_rate", "disagreements", "sentences"],
        ascending=[True, False, False],
    ).reset_index(drop=True)


def build_cross_table(
    comparison: pd.DataFrame,
    *,
    positive_label: str,
    negative_label: str,
) -> pd.DataFrame:
    labels = [negative_label, positive_label]
    table = pd.crosstab(
        comparison["pred_head"],
        comparison["pred_lstm"],
        rownames=["head_model"],
        colnames=["lstm_model"],
        dropna=False,
    )
    return table.reindex(index=labels, columns=labels, fill_value=0)


def plot_agreement_matrix(
    cross_table: pd.DataFrame,
    *,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    values = cross_table.to_numpy(dtype=float)
    image = ax.imshow(values, cmap="Blues")
    ax.set_xticks(range(len(cross_table.columns)))
    ax.set_xticklabels(cross_table.columns, rotation=20, ha="right")
    ax.set_yticks(range(len(cross_table.index)))
    ax.set_yticklabels(cross_table.index)
    ax.set_xlabel("LSTM model")
    ax.set_ylabel("Head model")
    ax.set_title("Prediction Agreement Matrix")

    for row_idx in range(values.shape[0]):
        for col_idx in range(values.shape[1]):
            ax.text(
                col_idx,
                row_idx,
                f"{int(values[row_idx, col_idx])}",
                ha="center",
                va="center",
                color="black",
                fontsize=10,
            )

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_speech_agreement(
    speech_summary: pd.DataFrame,
    *,
    output_path: Path,
    top_n: int,
) -> None:
    top = speech_summary.head(top_n).copy()
    labels = [str(speech_id) for speech_id in top["speech_id"]]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#b91c1c" if rate < 0.5 else "#2563eb" for rate in top["agreement_rate"]]
    ax.bar(labels, top["agreement_rate"], color=colors)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Agreement rate")
    ax.set_xlabel("Speech ID")
    ax.set_title(f"Lowest-Agreement Speeches (Top {len(top)})")
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1, alpha=0.5)

    for idx, (_, row) in enumerate(top.iterrows()):
        ax.text(
            idx,
            min(row["agreement_rate"] + 0.03, 0.99),
            f"{row['disagreements']}/{row['sentences']}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_top_speech_sequences(
    comparison: pd.DataFrame,
    speech_summary: pd.DataFrame,
    *,
    output_path: Path,
    top_n: int,
    positive_label: str,
    negative_label: str,
) -> None:
    selected = speech_summary.head(top_n)["speech_id"].tolist()
    if not selected:
        return

    label_to_value = {negative_label: 0, positive_label: 1}
    fig, axes = plt.subplots(
        len(selected),
        1,
        figsize=(12, 2.4 * len(selected)),
        sharex=False,
        sharey=True,
    )
    if len(selected) == 1:
        axes = [axes]

    for axis, speech_id in zip(axes, selected, strict=True):
        speech = comparison.loc[comparison["speech_id"] == speech_id].copy()
        x = speech["sentence_id"].to_numpy()
        head_values = speech["pred_head"].map(label_to_value).to_numpy()
        lstm_values = speech["pred_lstm"].map(label_to_value).to_numpy()
        disagreements = ~speech["agree"].to_numpy()

        axis.step(x, head_values, where="mid", label="Head model", color="#b91c1c")
        axis.step(x, lstm_values, where="mid", label="LSTM", color="#2563eb")
        axis.scatter(
            x[disagreements],
            lstm_values[disagreements],
            color="#111827",
            s=18,
            zorder=3,
            label="Disagreement" if speech_id == selected[0] else None,
        )
        axis.set_yticks([0, 1])
        axis.set_yticklabels([negative_label, positive_label])
        axis.set_title(
            f"Speech {speech_id}: {int(disagreements.sum())} disagreements / {len(speech)} sentences"
        )
        axis.grid(axis="y", linestyle="--", alpha=0.4)

    axes[-1].set_xlabel("Sentence ID")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_parquet(args.metadata_path)
    head_scores = load_scores(args.head_prob_path)
    lstm_scores = load_scores(args.lstm_prob_path)
    comparison = build_comparison_frame(
        metadata,
        head_scores=head_scores,
        lstm_scores=lstm_scores,
        threshold=args.threshold,
        positive_label=args.positive_label,
        negative_label=args.negative_label,
    )

    speech_summary = build_speech_summary(comparison)
    disagreements = comparison.loc[~comparison["agree"]].copy()
    cross_table = build_cross_table(
        comparison,
        positive_label=args.positive_label,
        negative_label=args.negative_label,
    )

    comparison_path = output_dir / "sentence_level_comparison.csv"
    comparison.to_csv(comparison_path, index=False)

    disagreements_path = output_dir / "sentence_level_disagreements.csv"
    disagreements.to_csv(disagreements_path, index=False)

    speech_summary_path = output_dir / "speech_level_summary.csv"
    speech_summary.to_csv(speech_summary_path, index=False)

    cross_table_path = output_dir / "prediction_crosstab.csv"
    cross_table.to_csv(cross_table_path)

    agreement_matrix_path = output_dir / "prediction_agreement_matrix.png"
    plot_agreement_matrix(cross_table, output_path=agreement_matrix_path)

    speech_agreement_plot_path = output_dir / "lowest_agreement_speeches.png"
    plot_speech_agreement(
        speech_summary,
        output_path=speech_agreement_plot_path,
        top_n=min(20, len(speech_summary)),
    )

    top_sequence_plot_path = output_dir / "top_disagreement_speech_sequences.png"
    plot_top_speech_sequences(
        comparison,
        speech_summary,
        output_path=top_sequence_plot_path,
        top_n=min(args.top_speeches, len(speech_summary)),
        positive_label=args.positive_label,
        negative_label=args.negative_label,
    )

    summary = {
        "rows": int(len(comparison)),
        "speeches": int(comparison["speech_id"].nunique()),
        "threshold": float(args.threshold),
        "positive_label": args.positive_label,
        "negative_label": args.negative_label,
        "agreement_rate": float(comparison["agree"].mean()),
        "disagreement_count": int((~comparison["agree"]).sum()),
        "head_positive_count": int((comparison["pred_head"] == args.positive_label).sum()),
        "lstm_positive_count": int((comparison["pred_lstm"] == args.positive_label).sum()),
        "outputs": {
            "sentence_level_comparison": str(comparison_path.resolve()),
            "sentence_level_disagreements": str(disagreements_path.resolve()),
            "speech_level_summary": str(speech_summary_path.resolve()),
            "prediction_crosstab": str(cross_table_path.resolve()),
            "prediction_agreement_matrix": str(agreement_matrix_path.resolve()),
            "lowest_agreement_speeches_plot": str(speech_agreement_plot_path.resolve()),
            "top_disagreement_speech_sequences_plot": str(top_sequence_plot_path.resolve()),
        },
    }

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
