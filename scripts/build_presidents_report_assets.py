import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.special import expit, logit
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
    roc_curve,
)


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "Dataset/out/presidents_report_assets"
FIGURES_DIR = OUT_DIR / "figures"
TABLES_DIR = OUT_DIR / "tables"
METADATA_DIR = OUT_DIR / "metadata"

SIMPLE_SWEEP_SUMMARY = ROOT / "Dataset/out/presidents_explainable_push_20260329/simple/presidents_lstm_simple_tuned_sweep/comparison_summary.json"
STRUCTURED_SWEEP_SUMMARY = ROOT / "Dataset/out/presidents_explainable_push_20260329/structured/presidents_lstm_structured_sweep/comparison_summary.json"
CLEAN_MAIN_SIMPLE_TRAIN_DIR = ROOT / "Dataset/out/presidents_lstm_simple_main_full"
RESEARCH_CAMEMBERT_DIR = ROOT / "Dataset/out/presidents_explainable_push_20260329/camembert/base_ctx2_len256_lr5e6_ep4"
FUSION_FULLCHECK_DIR = Path("/tmp/presidents_bilstm_camembert_fusion_fullcheck")
REPORT_SAFE_SIMPLE_NAME = "BiLSTM (Sweep Winner)"
REPORT_SAFE_CLEAN_NAME = "BiLSTM (Clean Main Rerun)"
REPORT_SAFE_STRUCTURED_NAME = "Structured BiLSTM"
REPORT_SAFE_CAMEMBERT_NAME = "CamemBERT Context Classifier"
REPORT_SAFE_FUSION_NAME = "Weighted BiLSTM + CamemBERT Fusion"
MODEL_COLOR_MAP = {
    REPORT_SAFE_SIMPLE_NAME: "#4c72b0",
    REPORT_SAFE_CLEAN_NAME: "#8172b2",
    REPORT_SAFE_STRUCTURED_NAME: "#c44e52",
    REPORT_SAFE_CAMEMBERT_NAME: "#dd8452",
    REPORT_SAFE_FUSION_NAME: "#55a868",
}
MODEL_PLOT_LABEL_MAP = {
    REPORT_SAFE_SIMPLE_NAME: "BiLSTM (Sweep)",
    REPORT_SAFE_CLEAN_NAME: "BiLSTM (Rerun)",
    REPORT_SAFE_STRUCTURED_NAME: "Structured BiLSTM",
    REPORT_SAFE_CAMEMBERT_NAME: "CamemBERT",
    REPORT_SAFE_FUSION_NAME: "Fusion",
}
POSITIVE_CLASS_NAME = "Mitterrand"
POSITIVE_PRECISION_LABEL = "Precision (M)"
POSITIVE_RECALL_LABEL = "Recall (M)"
POSITIVE_F1_LABEL = "F1 (M)"


def configure_plot_style(seaborn_theme="darkgrid", context="talk"):
    sns.set_theme(style=seaborn_theme, context=context)


def ensure_output_dirs():
    for directory in (
        OUT_DIR,
        FIGURES_DIR,
        TABLES_DIR,
        METADATA_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def load_json(path):
    return json.loads(Path(path).read_text())


def save_table(frame, stem, directory, float_columns=None):
    csv_path = directory / f"{stem}.csv"
    tex_path = directory / f"{stem}.tex"
    frame.to_csv(csv_path, index=False)

    latex_frame = frame.copy()
    if float_columns is not None:
        for column in float_columns:
            if column in latex_frame.columns:
                latex_frame[column] = latex_frame[column].map(
                    lambda value: f"{value:.4f}" if pd.notna(value) else ""
                )
    tex_path.write_text(latex_frame.to_latex(index=False, escape=True))
    return csv_path, tex_path


def save_figure(fig, stem, directory):
    png_path = directory / f"{stem}.png"
    pdf_path = directory / f"{stem}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def probability_to_label(probabilities, threshold):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    return np.where(probabilities >= threshold, -1, 1)


def as_binary_mitterrand(labels):
    labels = np.asarray(labels)
    if labels.dtype.kind in {"U", "S", "O"}:
        return np.where(labels == "M", 1, 0)
    return np.where(labels == -1, 1, 0)


def compute_metrics(y_true, probabilities, threshold):
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    # Some exported decoder scores can drift a hair outside [0, 1] due to
    # floating-point arithmetic, which breaks Brier scoring in sklearn.
    probabilities_for_brier = np.clip(probabilities, 0.0, 1.0)
    y_pred = probability_to_label(probabilities, threshold)
    y_true_binary = as_binary_mitterrand(y_true)
    y_pred_binary = as_binary_mitterrand(y_pred)

    metrics = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_mitterrand": float(precision_score(y_true_binary, y_pred_binary, zero_division=0)),
        "recall_mitterrand": float(recall_score(y_true_binary, y_pred_binary, zero_division=0)),
        "f1_mitterrand": float(f1_score(y_true_binary, y_pred_binary, zero_division=0)),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "mitterrand_count": int((y_pred == -1).sum()),
        "chirac_count": int((y_pred == 1).sum()),
        "brier_score": float(brier_score_loss(y_true_binary, probabilities_for_brier)),
        "roc_auc": np.nan,
        "pr_auc": np.nan,
    }

    if len(np.unique(y_true_binary)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true_binary, probabilities))
        metrics["pr_auc"] = float(average_precision_score(y_true_binary, probabilities))

    return metrics


def apply_score_delta(probabilities, delta):
    clipped = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-9, 1.0 - 1e-9)
    return expit(logit(clipped) - float(delta))


def weighted_logit_average(prob_a, prob_b, alpha):
    clipped_a = np.clip(np.asarray(prob_a, dtype=np.float64), 1e-9, 1.0 - 1.0e-9)
    clipped_b = np.clip(np.asarray(prob_b, dtype=np.float64), 1e-9, 1.0 - 1.0e-9)
    fused_logits = float(alpha) * logit(clipped_a) + (1.0 - float(alpha)) * logit(clipped_b)
    return expit(fused_logits)


def contiguous_blocks(sentence_ids, mask):
    positive_ids = np.asarray(sentence_ids, dtype=np.int64)[np.asarray(mask, dtype=bool)]
    if positive_ids.size == 0:
        return []

    blocks = []
    start = int(positive_ids[0])
    previous = int(positive_ids[0])
    for sentence_id in positive_ids[1:]:
        sentence_id = int(sentence_id)
        if sentence_id == previous + 1:
            previous = sentence_id
            continue
        blocks.append((start, previous))
        start = sentence_id
        previous = sentence_id
    blocks.append((start, previous))
    return blocks


def compute_speech_block_metrics(frame, probability_column, threshold):
    frame = frame.sort_values(["speech_id", "sentence_id"]).copy()
    records = []

    for speech_id, group in frame.groupby("speech_id", sort=False):
        sentence_ids = group["sentence_id"].to_numpy(dtype=np.int64)
        y_true_mask = group["true_label"].to_numpy(dtype=np.int64) == -1
        y_pred_mask = group[probability_column].to_numpy(dtype=np.float64) >= threshold

        true_blocks = contiguous_blocks(sentence_ids, y_true_mask)
        pred_blocks = contiguous_blocks(sentence_ids, y_pred_mask)

        true_start = true_blocks[0][0] if true_blocks else np.nan
        true_end = true_blocks[-1][1] if true_blocks else np.nan
        pred_start = pred_blocks[0][0] if pred_blocks else np.nan
        pred_end = pred_blocks[-1][1] if pred_blocks else np.nan

        records.append(
            {
                "speech_id": int(speech_id),
                "num_sentences": int(len(group)),
                "true_negative_sentence_count": int(y_true_mask.sum()),
                "predicted_negative_sentence_count": int(y_pred_mask.sum()),
                "has_true_mitterrand_block": bool(len(true_blocks) > 0),
                "has_predicted_mitterrand_block": bool(len(pred_blocks) > 0),
                "num_true_mitterrand_blocks": int(len(true_blocks)),
                "num_predicted_mitterrand_blocks": int(len(pred_blocks)),
                "true_start": true_start,
                "true_end": true_end,
                "predicted_start": pred_start,
                "predicted_end": pred_end,
                "start_lag": pred_start - true_start if true_blocks and pred_blocks else np.nan,
                "end_lag": pred_end - true_end if true_blocks and pred_blocks else np.nan,
            }
        )

    return pd.DataFrame(records)


def make_short_run_label(record):
    return (
        f"hd={int(record['hidden_dim'])}, pd={int(record['projection_dim'])}, "
        f"do={record['dropout']:.2f}, mw={record['minority_weight_scale']:.1f}, "
        f"ep={int(record['epochs'])}"
    )


def aggregate_simple_sweep_runs():
    summary = load_json(SIMPLE_SWEEP_SUMMARY)
    records = []
    for rank, run in enumerate(summary["runs"], start=1):
        run_dir = Path(run["output_dir"])
        calibration = load_json(run_dir / "calibration.json")
        oof = pd.read_csv(run_dir / "oof_predictions.csv")
        speech_metrics = pd.read_csv(run_dir / "oof_speech_metrics_at_best_threshold.csv")
        history_path = run_dir / "history.jsonl"
        history = pd.read_json(history_path, lines=True)

        tuned_metrics = compute_metrics(
            oof["true_label"].to_numpy(dtype=np.int64),
            oof["prob_mitterrand_raw"].to_numpy(dtype=np.float64),
            float(calibration["threshold"]),
        )
        default_metrics = compute_metrics(
            oof["true_label"].to_numpy(dtype=np.int64),
            oof["prob_mitterrand_raw"].to_numpy(dtype=np.float64),
            0.5,
        )

        record = {
            "run_name": run["run_name"],
            "short_label": make_short_run_label(run),
            "hidden_dim": int(run["hidden_dim"]),
            "projection_dim": int(run["projection_dim"]),
            "dropout": float(run["dropout"]),
            "minority_weight_scale": float(run["minority_weight_scale"]),
            "epochs": int(run["epochs"]),
            "best_epoch": int(run["best_epoch"]),
            "validation_best_f1": float(run["validation_best_f1"]),
            "tuned_threshold": float(calibration["threshold"]),
            "train_time_seconds": float(history["epoch_seconds"].sum()),
            "oof_accuracy_tuned": float(tuned_metrics["accuracy"]),
            "oof_precision_mitterrand_tuned": float(tuned_metrics["precision_mitterrand"]),
            "oof_recall_mitterrand_tuned": float(tuned_metrics["recall_mitterrand"]),
            "oof_f1_mitterrand_tuned": float(tuned_metrics["f1_mitterrand"]),
            "oof_balanced_accuracy_tuned": float(tuned_metrics["balanced_accuracy"]),
            "oof_precision_macro_tuned": float(tuned_metrics["precision_macro"]),
            "oof_recall_macro_tuned": float(tuned_metrics["recall_macro"]),
            "oof_f1_macro_tuned": float(tuned_metrics["f1_macro"]),
            "oof_roc_auc": float(tuned_metrics["roc_auc"]),
            "oof_pr_auc": float(tuned_metrics["pr_auc"]),
            "oof_accuracy_0p5": float(default_metrics["accuracy"]),
            "oof_f1_mitterrand_0p5": float(default_metrics["f1_mitterrand"]),
            "oof_f1_macro_0p5": float(default_metrics["f1_macro"]),
            "start_lag_mean": float(speech_metrics["start_lag"].mean()),
            "end_lag_mean": float(speech_metrics["end_lag"].mean()),
            "multi_block_speeches": int((speech_metrics["num_predicted_mitterrand_blocks"] > 1).sum()),
        }
        records.append(record)

    frame = pd.DataFrame(records).sort_values(
        ["oof_f1_mitterrand_tuned", "oof_roc_auc"],
        ascending=[False, False],
    ).reset_index(drop=True)
    frame.insert(0, "rank_oof_tuned_f1", np.arange(1, len(frame) + 1))
    frame.insert(0, "plot_id", [f"R{i+1}" for i in range(len(frame))])
    return frame


def aggregate_clean_main_simple():
    calibration = load_json(CLEAN_MAIN_SIMPLE_TRAIN_DIR / "calibration.json")
    oof = pd.read_csv(CLEAN_MAIN_SIMPLE_TRAIN_DIR / "oof_predictions.csv")

    tuned = compute_metrics(
        oof["true_label"].to_numpy(dtype=np.int64),
        oof["prob_mitterrand_raw"].to_numpy(dtype=np.float64),
        float(calibration["threshold"]),
    )

    return {
        "model": REPORT_SAFE_CLEAN_NAME,
        "oof_accuracy": float(tuned["accuracy"]),
        "oof_precision_mitterrand": float(tuned["precision_mitterrand"]),
        "oof_recall_mitterrand": float(tuned["recall_mitterrand"]),
        "oof_f1_mitterrand": float(tuned["f1_mitterrand"]),
        "oof_balanced_accuracy": float(tuned["balanced_accuracy"]),
        "oof_precision_macro": float(tuned["precision_macro"]),
        "oof_recall_macro": float(tuned["recall_macro"]),
        "oof_f1_macro": float(tuned["f1_macro"]),
        "oof_roc_auc": float(tuned["roc_auc"]),
        "oof_pr_auc": float(tuned["pr_auc"]),
        "threshold": float(calibration["threshold"]),
    }


def locate_camembert_train_dir():
    clean_train = ROOT / "Dataset/out/presidents_camembert_simple_main_full"
    if (clean_train / "metrics.json").exists():
        return clean_train
    return RESEARCH_CAMEMBERT_DIR


def aggregate_camembert(train_dir):
    calibration = load_json(train_dir / "calibration.json")
    oof = pd.read_csv(train_dir / "oof_predictions.csv")

    tuned = compute_metrics(
        oof["true_label"].to_numpy(dtype=np.int64),
        oof["prob_mitterrand_raw"].to_numpy(dtype=np.float64),
        float(calibration["threshold"]),
    )
    return {
        "model": REPORT_SAFE_CAMEMBERT_NAME,
        "oof_accuracy": float(tuned["accuracy"]),
        "oof_precision_mitterrand": float(tuned["precision_mitterrand"]),
        "oof_recall_mitterrand": float(tuned["recall_mitterrand"]),
        "oof_f1_mitterrand": float(tuned["f1_mitterrand"]),
        "oof_balanced_accuracy": float(tuned["balanced_accuracy"]),
        "oof_precision_macro": float(tuned["precision_macro"]),
        "oof_recall_macro": float(tuned["recall_macro"]),
        "oof_f1_macro": float(tuned["f1_macro"]),
        "oof_roc_auc": float(tuned["roc_auc"]),
        "oof_pr_auc": float(tuned["pr_auc"]),
        "threshold": float(calibration["threshold"]),
    }


def aggregate_simple_camembert_fusion(camembert_train_dir):
    if (ROOT / "Dataset/out/presidents_bilstm_camembert_fusion_main_full/metrics.json").exists():
        fusion_dir = ROOT / "Dataset/out/presidents_bilstm_camembert_fusion_main_full"
    else:
        fusion_dir = FUSION_FULLCHECK_DIR

    fusion_metrics = load_json(fusion_dir / "metrics.json")
    alpha = float(fusion_metrics["calibration"]["alpha"])
    threshold = float(fusion_metrics["calibration"]["threshold"])

    simple_oof = pd.read_csv(CLEAN_MAIN_SIMPLE_TRAIN_DIR / "oof_predictions.csv").sort_values(
        ["speech_id", "sentence_id"]
    )
    camembert_oof = pd.read_csv(camembert_train_dir / "oof_predictions.csv").sort_values(
        ["speech_id", "sentence_id"]
    )
    if not simple_oof[["speech_id", "sentence_id"]].reset_index(drop=True).equals(
        camembert_oof[["speech_id", "sentence_id"]].reset_index(drop=True)
    ):
        raise ValueError("BiLSTM and CamemBERT OOF predictions are not aligned")

    fused_oof = weighted_logit_average(
        simple_oof["prob_mitterrand_raw"].to_numpy(dtype=np.float64),
        camembert_oof["prob_mitterrand_raw"].to_numpy(dtype=np.float64),
        alpha,
    )
    tuned = compute_metrics(
        simple_oof["true_label"].to_numpy(dtype=np.int64),
        fused_oof,
        threshold,
    )
    return {
        "model": REPORT_SAFE_FUSION_NAME,
        "oof_accuracy": float(tuned["accuracy"]),
        "oof_precision_mitterrand": float(tuned["precision_mitterrand"]),
        "oof_recall_mitterrand": float(tuned["recall_mitterrand"]),
        "oof_f1_mitterrand": float(tuned["f1_mitterrand"]),
        "oof_balanced_accuracy": float(tuned["balanced_accuracy"]),
        "oof_precision_macro": float(tuned["precision_macro"]),
        "oof_recall_macro": float(tuned["recall_macro"]),
        "oof_f1_macro": float(tuned["f1_macro"]),
        "oof_roc_auc": float(tuned["roc_auc"]),
        "oof_pr_auc": float(tuned["pr_auc"]),
        "threshold": threshold,
        "alpha": alpha,
    }


def aggregate_report_safe_model_comparison(simple_sweep_frame, camembert_train_dir):
    best_simple = simple_sweep_frame.iloc[0]
    clean_main = aggregate_clean_main_simple()
    camembert = aggregate_camembert(camembert_train_dir)
    fusion = aggregate_simple_camembert_fusion(camembert_train_dir)

    frame = pd.DataFrame(
        [
            {
                "model": REPORT_SAFE_SIMPLE_NAME,
                "oof_accuracy": best_simple["oof_accuracy_tuned"],
                "oof_precision_mitterrand": best_simple["oof_precision_mitterrand_tuned"],
                "oof_recall_mitterrand": best_simple["oof_recall_mitterrand_tuned"],
                "oof_f1_mitterrand": best_simple["oof_f1_mitterrand_tuned"],
                "oof_balanced_accuracy": best_simple["oof_balanced_accuracy_tuned"],
                "oof_precision_macro": best_simple["oof_precision_macro_tuned"],
                "oof_recall_macro": best_simple["oof_recall_macro_tuned"],
                "oof_f1_macro": best_simple["oof_f1_macro_tuned"],
                "oof_roc_auc": best_simple["oof_roc_auc"],
                "oof_pr_auc": best_simple["oof_pr_auc"],
                "threshold": best_simple["tuned_threshold"],
            },
            clean_main,
            camembert,
            fusion,
        ]
    )
    return frame


def make_curve_source(model, y_true, scores, threshold):
    return {
        "model": model,
        "plot_label": MODEL_PLOT_LABEL_MAP[model],
        "y_true": np.asarray(y_true, dtype=np.int64),
        "scores": np.asarray(scores, dtype=np.float64),
        "threshold": float(threshold),
    }


def locate_structured_best_run():
    summary = load_json(STRUCTURED_SWEEP_SUMMARY)
    best = summary["best_ranked_run"]
    calibration = load_json(Path(best["calibration_dir"]) / "calibration.json")
    recommended = calibration.get("recommended_calibration", {})
    score_name = recommended.get("score_name", best.get("selected_score_name", "prob_mitterrand_raw"))
    threshold = float(recommended.get("threshold", best["tuned_threshold"]))
    oof = pd.read_csv(Path(best["calibration_dir"]) / "oof_predictions.csv")
    tuned = compute_metrics(
        oof["true_label"].to_numpy(dtype=np.int64),
        oof[score_name].to_numpy(dtype=np.float64),
        threshold,
    )
    return {
        "model": REPORT_SAFE_STRUCTURED_NAME,
        "run_name": best["run_name"],
        "run_dir": Path(best["run_dir"]),
        "calibration_dir": Path(best["calibration_dir"]),
        "score_name": score_name,
        "threshold": threshold,
        "selected_score_name": best.get("selected_score_name", score_name),
        "short_label": (
            f"hd={int(best['hidden_dim'])}, pd={int(best['projection_dim'])}, "
            f"do={float(best['dropout']):.2f}, mns={int(best['min_negative_span'])}, "
            f"lr={float(best['learning_rate']):.4f}, ep={int(best['epochs'])}"
        ),
        "oof": oof,
        "metrics": {
            "model": REPORT_SAFE_STRUCTURED_NAME,
            "oof_accuracy": float(tuned["accuracy"]),
            "oof_precision_mitterrand": float(tuned["precision_mitterrand"]),
            "oof_recall_mitterrand": float(tuned["recall_mitterrand"]),
            "oof_f1_mitterrand": float(tuned["f1_mitterrand"]),
            "oof_balanced_accuracy": float(tuned["balanced_accuracy"]),
            "oof_precision_macro": float(tuned["precision_macro"]),
            "oof_recall_macro": float(tuned["recall_macro"]),
            "oof_f1_macro": float(tuned["f1_macro"]),
            "oof_roc_auc": float(tuned["roc_auc"]),
            "oof_pr_auc": float(tuned["pr_auc"]),
            "threshold": threshold,
        },
    }


def build_report_safe_curve_sources(simple_sweep_frame, camembert_train_dir):
    best_run_dir = Path(load_json(SIMPLE_SWEEP_SUMMARY)["best_ranked_run"]["output_dir"])
    best_simple_oof = pd.read_csv(best_run_dir / "oof_predictions.csv")
    best_simple_threshold = float(load_json(best_run_dir / "calibration.json")["threshold"])

    clean_main_oof = pd.read_csv(CLEAN_MAIN_SIMPLE_TRAIN_DIR / "oof_predictions.csv")
    clean_main_threshold = float(load_json(CLEAN_MAIN_SIMPLE_TRAIN_DIR / "calibration.json")["threshold"])

    camembert_oof = pd.read_csv(camembert_train_dir / "oof_predictions.csv")
    camembert_threshold = float(load_json(camembert_train_dir / "calibration.json")["threshold"])

    fusion_dir = (
        ROOT / "Dataset/out/presidents_bilstm_camembert_fusion_main_full"
        if (ROOT / "Dataset/out/presidents_bilstm_camembert_fusion_main_full/metrics.json").exists()
        else FUSION_FULLCHECK_DIR
    )
    fusion_metrics = load_json(fusion_dir / "metrics.json")
    fusion_alpha = float(fusion_metrics["calibration"]["alpha"])
    fusion_threshold = float(fusion_metrics["calibration"]["threshold"])

    clean_main_aligned = clean_main_oof.sort_values(["speech_id", "sentence_id"]).reset_index(drop=True)
    camembert_aligned = camembert_oof.sort_values(["speech_id", "sentence_id"]).reset_index(drop=True)
    if not clean_main_aligned[["speech_id", "sentence_id"]].equals(
        camembert_aligned[["speech_id", "sentence_id"]]
    ):
        raise ValueError("BiLSTM and CamemBERT OOF predictions are not aligned")

    fusion_scores = weighted_logit_average(
        clean_main_aligned["prob_mitterrand_raw"].to_numpy(dtype=np.float64),
        camembert_aligned["prob_mitterrand_raw"].to_numpy(dtype=np.float64),
        fusion_alpha,
    )

    return [
        make_curve_source(
            REPORT_SAFE_SIMPLE_NAME,
            best_simple_oof["true_label"],
            best_simple_oof["prob_mitterrand_raw"],
            best_simple_threshold,
        ),
        make_curve_source(
            REPORT_SAFE_CLEAN_NAME,
            clean_main_oof["true_label"],
            clean_main_oof["prob_mitterrand_raw"],
            clean_main_threshold,
        ),
        make_curve_source(
            REPORT_SAFE_CAMEMBERT_NAME,
            camembert_oof["true_label"],
            camembert_oof["prob_mitterrand_raw"],
            camembert_threshold,
        ),
        make_curve_source(
            REPORT_SAFE_FUSION_NAME,
            clean_main_aligned["true_label"],
            fusion_scores,
            fusion_threshold,
        ),
    ]


def build_structured_curve_sources(simple_sweep_frame):
    structured_model = locate_structured_best_run()
    best_run_dir = Path(load_json(SIMPLE_SWEEP_SUMMARY)["best_ranked_run"]["output_dir"])
    best_simple_oof = pd.read_csv(best_run_dir / "oof_predictions.csv")
    best_simple_threshold = float(load_json(best_run_dir / "calibration.json")["threshold"])
    clean_main_oof = pd.read_csv(CLEAN_MAIN_SIMPLE_TRAIN_DIR / "oof_predictions.csv")
    clean_main_threshold = float(load_json(CLEAN_MAIN_SIMPLE_TRAIN_DIR / "calibration.json")["threshold"])
    return [
        make_curve_source(
            REPORT_SAFE_SIMPLE_NAME,
            best_simple_oof["true_label"],
            best_simple_oof["prob_mitterrand_raw"],
            best_simple_threshold,
        ),
        make_curve_source(
            REPORT_SAFE_CLEAN_NAME,
            clean_main_oof["true_label"],
            clean_main_oof["prob_mitterrand_raw"],
            clean_main_threshold,
        ),
        make_curve_source(
            REPORT_SAFE_STRUCTURED_NAME,
            structured_model["oof"]["true_label"],
            structured_model["oof"][structured_model["score_name"]],
            structured_model["threshold"],
        ),
    ]


def build_paper_model_table(model_comparison_frame, structured_model=None):
    return model_comparison_frame[
        [
            "model",
            "oof_accuracy",
            "oof_precision_mitterrand",
            "oof_recall_mitterrand",
            "oof_f1_mitterrand",
            "oof_roc_auc",
            "oof_pr_auc",
        ]
    ].rename(
        columns={
            "model": "Model",
            "oof_accuracy": "Accuracy",
            "oof_precision_mitterrand": POSITIVE_PRECISION_LABEL,
            "oof_recall_mitterrand": POSITIVE_RECALL_LABEL,
            "oof_f1_mitterrand": POSITIVE_F1_LABEL,
            "oof_roc_auc": "ROC AUC",
            "oof_pr_auc": "PR AUC",
        }
    )


def build_camera_ready_model_table(model_comparison_frame, structured_model):
    rows = [
        {
            "Model": "BiLSTM",
            "Variant": "Sweep winner",
            "Accuracy": model_comparison_frame.loc[
                model_comparison_frame["model"] == REPORT_SAFE_SIMPLE_NAME, "oof_accuracy"
            ].iloc[0],
            POSITIVE_PRECISION_LABEL: model_comparison_frame.loc[
                model_comparison_frame["model"] == REPORT_SAFE_SIMPLE_NAME, "oof_precision_mitterrand"
            ].iloc[0],
            POSITIVE_RECALL_LABEL: model_comparison_frame.loc[
                model_comparison_frame["model"] == REPORT_SAFE_SIMPLE_NAME, "oof_recall_mitterrand"
            ].iloc[0],
            POSITIVE_F1_LABEL: model_comparison_frame.loc[
                model_comparison_frame["model"] == REPORT_SAFE_SIMPLE_NAME, "oof_f1_mitterrand"
            ].iloc[0],
            "ROC AUC": model_comparison_frame.loc[
                model_comparison_frame["model"] == REPORT_SAFE_SIMPLE_NAME, "oof_roc_auc"
            ].iloc[0],
            "PR AUC": model_comparison_frame.loc[
                model_comparison_frame["model"] == REPORT_SAFE_SIMPLE_NAME, "oof_pr_auc"
            ].iloc[0],
        },
        {
            "Model": "BiLSTM",
            "Variant": "Clean rerun",
            "Accuracy": model_comparison_frame.loc[
                model_comparison_frame["model"] == REPORT_SAFE_CLEAN_NAME, "oof_accuracy"
            ].iloc[0],
            POSITIVE_PRECISION_LABEL: model_comparison_frame.loc[
                model_comparison_frame["model"] == REPORT_SAFE_CLEAN_NAME, "oof_precision_mitterrand"
            ].iloc[0],
            POSITIVE_RECALL_LABEL: model_comparison_frame.loc[
                model_comparison_frame["model"] == REPORT_SAFE_CLEAN_NAME, "oof_recall_mitterrand"
            ].iloc[0],
            POSITIVE_F1_LABEL: model_comparison_frame.loc[
                model_comparison_frame["model"] == REPORT_SAFE_CLEAN_NAME, "oof_f1_mitterrand"
            ].iloc[0],
            "ROC AUC": model_comparison_frame.loc[
                model_comparison_frame["model"] == REPORT_SAFE_CLEAN_NAME, "oof_roc_auc"
            ].iloc[0],
            "PR AUC": model_comparison_frame.loc[
                model_comparison_frame["model"] == REPORT_SAFE_CLEAN_NAME, "oof_pr_auc"
            ].iloc[0],
        },
        {
            "Model": "Structured BiLSTM",
            "Variant": "Single-span decoder",
            "Accuracy": structured_model["metrics"]["oof_accuracy"],
            POSITIVE_PRECISION_LABEL: structured_model["metrics"]["oof_precision_mitterrand"],
            POSITIVE_RECALL_LABEL: structured_model["metrics"]["oof_recall_mitterrand"],
            POSITIVE_F1_LABEL: structured_model["metrics"]["oof_f1_mitterrand"],
            "ROC AUC": structured_model["metrics"]["oof_roc_auc"],
            "PR AUC": structured_model["metrics"]["oof_pr_auc"],
        },
        {
            "Model": "CamemBERT",
            "Variant": "Context classifier",
            "Accuracy": model_comparison_frame.loc[
                model_comparison_frame["model"] == REPORT_SAFE_CAMEMBERT_NAME, "oof_accuracy"
            ].iloc[0],
            POSITIVE_PRECISION_LABEL: model_comparison_frame.loc[
                model_comparison_frame["model"] == REPORT_SAFE_CAMEMBERT_NAME, "oof_precision_mitterrand"
            ].iloc[0],
            POSITIVE_RECALL_LABEL: model_comparison_frame.loc[
                model_comparison_frame["model"] == REPORT_SAFE_CAMEMBERT_NAME, "oof_recall_mitterrand"
            ].iloc[0],
            POSITIVE_F1_LABEL: model_comparison_frame.loc[
                model_comparison_frame["model"] == REPORT_SAFE_CAMEMBERT_NAME, "oof_f1_mitterrand"
            ].iloc[0],
            "ROC AUC": model_comparison_frame.loc[
                model_comparison_frame["model"] == REPORT_SAFE_CAMEMBERT_NAME, "oof_roc_auc"
            ].iloc[0],
            "PR AUC": model_comparison_frame.loc[
                model_comparison_frame["model"] == REPORT_SAFE_CAMEMBERT_NAME, "oof_pr_auc"
            ].iloc[0],
        },
        {
            "Model": "Fusion",
            "Variant": (
                "Weighted logit, "
                f"alpha={model_comparison_frame.loc[model_comparison_frame['model'] == REPORT_SAFE_FUSION_NAME, 'alpha'].iloc[0]:.2f}"
            ),
            "Accuracy": model_comparison_frame.loc[
                model_comparison_frame["model"] == REPORT_SAFE_FUSION_NAME, "oof_accuracy"
            ].iloc[0],
            POSITIVE_PRECISION_LABEL: model_comparison_frame.loc[
                model_comparison_frame["model"] == REPORT_SAFE_FUSION_NAME, "oof_precision_mitterrand"
            ].iloc[0],
            POSITIVE_RECALL_LABEL: model_comparison_frame.loc[
                model_comparison_frame["model"] == REPORT_SAFE_FUSION_NAME, "oof_recall_mitterrand"
            ].iloc[0],
            POSITIVE_F1_LABEL: model_comparison_frame.loc[
                model_comparison_frame["model"] == REPORT_SAFE_FUSION_NAME, "oof_f1_mitterrand"
            ].iloc[0],
            "ROC AUC": model_comparison_frame.loc[
                model_comparison_frame["model"] == REPORT_SAFE_FUSION_NAME, "oof_roc_auc"
            ].iloc[0],
            "PR AUC": model_comparison_frame.loc[
                model_comparison_frame["model"] == REPORT_SAFE_FUSION_NAME, "oof_pr_auc"
            ].iloc[0],
        },
    ]
    return pd.DataFrame(rows)


def latex_escape(value):
    return (
        str(value)
        .replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("_", "\\_")
        .replace("#", "\\#")
    )


def render_camera_ready_latex_table(frame, caption, label):
    columns = list(frame.columns)
    alignment = "ll" + "r" * (len(columns) - 2)
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "{\\scriptsize",
        "\\setlength{\\tabcolsep}{4pt}",
        f"\\begin{{tabular}}{{{alignment}}}",
        "\\hline",
        " & ".join(f"\\textbf{{{latex_escape(column)}}}" for column in columns) + " \\\\",
        "\\hline",
    ]
    for row in frame.itertuples(index=False):
        row_values = []
        for index, value in enumerate(row):
            if isinstance(value, (float, np.floating)):
                row_values.append(f"{float(value):.3f}")
            else:
                escaped = latex_escape(value)
                row_values.append(f"\\textbf{{{escaped}}}" if index == 0 else escaped)
        lines.append(" & ".join(row_values) + " \\\\")
    lines.extend(
        [
            "\\hline",
            "\\end{tabular}",
            "}",
            f"\\caption{{{latex_escape(caption)}}}",
            f"\\label{{{label}}}",
            "\\end{table}",
        ]
    )
    return "\n".join(lines) + "\n"


def save_styled_latex_table(frame, stem, directory, caption, label):
    tex_path = directory / f"{stem}.tex"
    tex_path.write_text(render_camera_ready_latex_table(frame, caption, label))
    return tex_path


def compute_boundary_lag_summary(speech_metrics):
    start_lag = speech_metrics["start_lag"].dropna().astype(float)
    end_lag = speech_metrics["end_lag"].dropna().astype(float)
    return {
        "mean_abs_start_lag": float(start_lag.abs().mean()) if not start_lag.empty else np.nan,
        "mean_abs_end_lag": float(end_lag.abs().mean()) if not end_lag.empty else np.nan,
        "multi_block_speeches": int((speech_metrics["num_predicted_mitterrand_blocks"] > 1).sum()),
    }


def select_boundary_comparison_run():
    summary = load_json(SIMPLE_SWEEP_SUMMARY)
    best_run_name = summary["best_ranked_run"]["run_name"]
    candidates = []

    for run in summary["runs"]:
        if run["run_name"] == best_run_name:
            continue

        run_dir = Path(run["output_dir"])
        speech_metrics = pd.read_csv(run_dir / "oof_speech_metrics_at_best_threshold.csv")
        lag_summary = compute_boundary_lag_summary(speech_metrics)
        candidates.append(
            {
                "run_name": run["run_name"],
                "run_dir": run_dir,
                "short_label": make_short_run_label(run),
                "speech_metrics": speech_metrics,
                **lag_summary,
                "lag_score": lag_summary["mean_abs_start_lag"] + lag_summary["mean_abs_end_lag"],
            }
        )

    if not candidates:
        raise ValueError("No boundary comparison BiLSTM candidate was found")

    candidates.sort(
        key=lambda item: (item["lag_score"], item["multi_block_speeches"]),
        reverse=True,
    )
    return candidates[0]


def compute_fold_metrics(oof_frame, threshold):
    records = []
    for fold, group in oof_frame.groupby("fold", sort=True):
        metrics = compute_metrics(
            group["true_label"].to_numpy(dtype=np.int64),
            group["prob_mitterrand_raw"].to_numpy(dtype=np.float64),
            threshold,
        )
        records.append({"fold": int(fold), **metrics})
    return pd.DataFrame(records)


def build_threshold_sweep_frame(run_dir):
    run_dir = Path(run_dir)
    oof_frame = pd.read_csv(run_dir / "oof_predictions.csv")
    stored_sweep = pd.read_csv(run_dir / "threshold_sweep.csv")
    score_name = (
        stored_sweep["selected_score_name"].iloc[0]
        if "selected_score_name" in stored_sweep.columns
        else "prob_mitterrand_raw"
    )
    records = []
    for threshold in stored_sweep["threshold"].to_numpy(dtype=np.float64):
        metrics = compute_metrics(
            oof_frame["true_label"].to_numpy(dtype=np.int64),
            oof_frame[score_name].to_numpy(dtype=np.float64),
            threshold,
        )
        records.append(metrics)
    return pd.DataFrame(records).sort_values("threshold").reset_index(drop=True)


def build_fusion_alpha_sweep_frame(fusion_dir, camembert_train_dir):
    fusion_dir = Path(fusion_dir)
    stored_alpha_sweep = pd.read_csv(fusion_dir / "alpha_sweep.csv").sort_values("alpha").reset_index(drop=True)

    simple_oof = pd.read_csv(CLEAN_MAIN_SIMPLE_TRAIN_DIR / "oof_predictions.csv").sort_values(
        ["speech_id", "sentence_id"]
    ).reset_index(drop=True)
    camembert_oof = pd.read_csv(Path(camembert_train_dir) / "oof_predictions.csv").sort_values(
        ["speech_id", "sentence_id"]
    ).reset_index(drop=True)
    if not simple_oof[["speech_id", "sentence_id"]].equals(camembert_oof[["speech_id", "sentence_id"]]):
        raise ValueError("BiLSTM and CamemBERT OOF predictions are not aligned")

    records = []
    for row in stored_alpha_sweep.itertuples(index=False):
        fused_scores = weighted_logit_average(
            simple_oof["prob_mitterrand_raw"].to_numpy(dtype=np.float64),
            camembert_oof["prob_mitterrand_raw"].to_numpy(dtype=np.float64),
            float(row.alpha),
        )
        metrics = compute_metrics(
            simple_oof["true_label"].to_numpy(dtype=np.int64),
            fused_scores,
            float(row.threshold),
        )
        records.append(
            {
                "method": row.method,
                "alpha": float(row.alpha),
                **metrics,
                "equivalent_score_delta": float(row.equivalent_score_delta),
            }
        )
    return pd.DataFrame(records).sort_values("alpha").reset_index(drop=True)


def choose_example_speeches(speech_metrics):
    clean_candidates = speech_metrics[
        (speech_metrics["has_true_mitterrand_block"])
        & (speech_metrics["has_predicted_mitterrand_block"])
        & (speech_metrics["num_predicted_mitterrand_blocks"] == 1)
        & (speech_metrics["num_sentences"] >= 60)
    ].copy()
    clean_candidates["total_abs_lag"] = (
        clean_candidates["start_lag"].abs() + clean_candidates["end_lag"].abs()
    )
    clean_example = (
        clean_candidates.sort_values(["total_abs_lag", "num_sentences"], ascending=[True, False])
        .iloc[0]["speech_id"]
    )

    noisy_candidates = speech_metrics[
        (speech_metrics["has_true_mitterrand_block"])
        & (speech_metrics["num_predicted_mitterrand_blocks"] > 1)
    ].copy()
    if noisy_candidates.empty:
        noisy_candidates = speech_metrics[
            speech_metrics["has_true_mitterrand_block"]
        ].copy()
        noisy_candidates["total_abs_lag"] = (
            noisy_candidates["start_lag"].abs() + noisy_candidates["end_lag"].abs()
        )
        noisy_example = (
            noisy_candidates.sort_values(["total_abs_lag", "num_sentences"], ascending=[False, False])
            .iloc[0]["speech_id"]
        )
    else:
        noisy_example = (
            noisy_candidates.sort_values(
                ["num_predicted_mitterrand_blocks", "num_sentences"],
                ascending=[False, False],
            )
            .iloc[0]["speech_id"]
        )

    return int(clean_example), int(noisy_example)


def plot_simple_sweep_ranking(simple_sweep_frame):
    plot_frame = simple_sweep_frame.sort_values("oof_f1_mitterrand_tuned", ascending=True)
    fig, ax = plt.subplots(figsize=(11, 8))
    palette = ["#b8c6db"] * len(plot_frame)
    highlight_index = int(plot_frame.index[plot_frame["rank_oof_tuned_f1"] == 1][0])
    palette[list(plot_frame.index).index(highlight_index)] = "#dd8452"
    ax.barh(plot_frame["plot_id"], plot_frame["oof_f1_mitterrand_tuned"], color=palette, edgecolor="#2f2f2f")
    ax.set_xlabel("OOF F1 (Mitterrand positive) at Tuned Threshold")
    ax.set_ylabel("Run ID")
    ax.set_title("BiLSTM Hyperparameter Sweep (12 Runs)")
    ax.set_xlim(
        plot_frame["oof_f1_mitterrand_tuned"].min() - 0.003,
        plot_frame["oof_f1_mitterrand_tuned"].max() + 0.003,
    )
    for _, row in plot_frame.iterrows():
        ax.text(
            row["oof_f1_mitterrand_tuned"] + 0.0003,
            row["plot_id"],
            row["short_label"],
            va="center",
            fontsize=9,
        )
    return fig


def plot_threshold_sweep(best_run_dir):
    sweep = build_threshold_sweep_frame(best_run_dir)
    selected_threshold = float(load_json(Path(best_run_dir) / "calibration.json")["threshold"])
    selected_row = sweep.loc[np.isclose(sweep["threshold"], selected_threshold)].iloc[0]
    fig, ax = plt.subplots(figsize=(9, 5))
    for column, label, color in (
        ("f1_mitterrand", "F1 (Mitterrand positive)", "#dd8452"),
        ("accuracy", "Accuracy", "#4c72b0"),
    ):
        ax.plot(sweep["threshold"], sweep[column], label=label, linewidth=2.2, color=color)
    ax.axvline(selected_threshold, color="#2f2f2f", linestyle="--", linewidth=1.5)
    ax.text(
        selected_threshold + 0.003,
        float(selected_row["f1_mitterrand"]) - 0.003,
        f"selected={selected_threshold:.3f}",
        fontsize=10,
    )
    ax.set_xlabel("Decision Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Threshold Calibration on Out-of-Fold Predictions")
    ax.legend(frameon=True)
    return fig


def plot_epoch_selection(best_run_dir):
    history = pd.read_json(best_run_dir / "history.jsonl", lines=True)
    history["validation_f1_tuned"] = history["validation_selection"].map(lambda value: value["f1_macro"])
    history["validation_f1_0p5"] = history["validation_metrics_at_0p5"].map(lambda value: value["f1_macro"])
    best_epoch = int(history.loc[history["validation_f1_tuned"].idxmax(), "epoch"])

    fig, ax1 = plt.subplots(figsize=(10.5, 6.2))
    ax1.plot(
        history["epoch"],
        history["validation_f1_tuned"],
        marker="o",
        linewidth=2.5,
        markersize=9,
        label="Validation Macro-F1 (Tuned)",
        color="#dd8452",
    )
    ax1.plot(
        history["epoch"],
        history["validation_f1_0p5"],
        marker="s",
        linewidth=2.5,
        markersize=9,
        label="Validation Macro-F1 (0.5)",
        color="#4c72b0",
    )
    ax1.axvline(best_epoch, color="#2f2f2f", linestyle="--", linewidth=1.4)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Validation Macro-F1")
    ax1.set_title("Epoch Selection for the Chosen BiLSTM")
    ax1.set_xlim(history["epoch"].min() - 0.4, history["epoch"].max() + 0.4)
    ax1.set_ylim(
        history[["validation_f1_tuned", "validation_f1_0p5"]].min().min() - 0.01,
        history[["validation_f1_tuned", "validation_f1_0p5"]].max().max() + 0.01,
    )

    ax2 = ax1.twinx()
    ax2.plot(
        history["epoch"],
        history["train_loss"],
        marker="^",
        linewidth=2.3,
        markersize=8,
        alpha=0.6,
        label="Train Loss",
        color="#55a868",
    )
    ax2.set_ylabel("Train Loss")
    ax2.set_ylim(history["train_loss"].min() * 0.9, history["train_loss"].max() * 1.05)

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines_1 + lines_2,
        labels_1 + labels_2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=3,
        frameon=True,
        columnspacing=1.4,
        handlelength=2.8,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return fig


def plot_fold_heatmap(fold_metrics):
    heatmap_frame = fold_metrics.set_index("fold")[
        [
            "accuracy",
            "precision_mitterrand",
            "recall_mitterrand",
            "f1_mitterrand",
            "roc_auc",
            "pr_auc",
        ]
    ].rename(
        columns={
            "accuracy": "Accuracy",
            "precision_mitterrand": POSITIVE_PRECISION_LABEL,
            "recall_mitterrand": POSITIVE_RECALL_LABEL,
            "f1_mitterrand": POSITIVE_F1_LABEL,
            "roc_auc": "ROC AUC",
            "pr_auc": "PR AUC",
        }
    )
    fig, ax = plt.subplots(figsize=(10.4, 5.0))
    sns.heatmap(
        heatmap_frame,
        annot=True,
        fmt=".3f",
        cmap="viridis",
        cbar=True,
        cbar_kws={"label": "Metric value", "shrink": 0.92, "aspect": 16, "pad": 0.05},
        linewidths=1.0,
        linecolor="white",
        annot_kws={"size": 8.5, "weight": "semibold"},
        ax=ax,
    )
    color_mesh = ax.collections[0]
    for text, value in zip(ax.texts, heatmap_frame.to_numpy().ravel()):
        text.set_color("white" if color_mesh.norm(value) < 0.55 else "#1f1f1f")
    cbar = color_mesh.colorbar
    cbar.ax.tick_params(labelsize=8.5)
    cbar.set_label("Metric value", size=9.5, labelpad=8)
    ax.set_title("Selected BiLSTM: Metrics by OOF Fold")
    ax.set_xlabel("")
    ax.set_ylabel("Fold")
    ax.tick_params(axis="x", labelsize=8.5)
    ax.tick_params(axis="y", labelsize=8.5)
    fig.tight_layout()
    return fig


def plot_roc_pr_curves(oof_frame, threshold):
    y_true_binary = as_binary_mitterrand(oof_frame["true_label"].to_numpy(dtype=np.int64))
    scores = oof_frame["prob_mitterrand_raw"].to_numpy(dtype=np.float64)
    fpr, tpr, _ = roc_curve(y_true_binary, scores)
    precision, recall, _ = precision_recall_curve(y_true_binary, scores)
    auc_roc = roc_auc_score(y_true_binary, scores)
    auc_pr = average_precision_score(y_true_binary, scores)
    operating_point = compute_operating_point(y_true_binary, scores, threshold)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(fpr, tpr, color="#dd8452", linewidth=2.5, label=f"ROC AUC = {auc_roc:.3f}")
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="#777777")
    axes[0].scatter(
        operating_point["fpr"],
        operating_point["tpr"],
        s=58,
        color="#dd8452",
        edgecolors="white",
        linewidths=1.1,
        zorder=5,
    )
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("OOF ROC Curve (Selected BiLSTM)")
    axes[0].legend(loc="lower right", title="marker = tuned threshold", title_fontsize=9)

    axes[1].plot(recall, precision, color="#4c72b0", linewidth=2.5, label=f"PR AUC = {auc_pr:.3f}")
    axes[1].scatter(
        operating_point["recall"],
        operating_point["precision"],
        s=58,
        color="#4c72b0",
        edgecolors="white",
        linewidths=1.1,
        zorder=5,
    )
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("OOF Precision-Recall Curve (Selected BiLSTM)")
    axes[1].legend(loc="lower left", title="marker = tuned threshold", title_fontsize=9)
    return fig


def compute_operating_point(y_true_binary, scores, threshold):
    y_true_binary = np.asarray(y_true_binary, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    predictions = (scores >= float(threshold)).astype(np.int64)
    true_positive = int(((predictions == 1) & (y_true_binary == 1)).sum())
    false_positive = int(((predictions == 1) & (y_true_binary == 0)).sum())
    true_negative = int(((predictions == 0) & (y_true_binary == 0)).sum())
    false_negative = int(((predictions == 0) & (y_true_binary == 1)).sum())
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
    false_positive_rate = (
        false_positive / (false_positive + true_negative) if (false_positive + true_negative) else 0.0
    )
    return {
        "fpr": float(false_positive_rate),
        "tpr": float(recall),
        "recall": float(recall),
        "precision": float(precision),
    }


def plot_curve_comparison(
    curve_sources,
    roc_title,
    pr_title,
    figsize=(13.2, 5.4),
    pr_legend_loc="lower left",
):
    positive_rate = as_binary_mitterrand(curve_sources[0]["y_true"]).mean()
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    for source in curve_sources:
        y_true_binary = as_binary_mitterrand(source["y_true"])
        scores = np.asarray(source["scores"], dtype=np.float64)
        color = MODEL_COLOR_MAP[source["model"]]

        fpr, tpr, _ = roc_curve(y_true_binary, scores)
        precision, recall, _ = precision_recall_curve(y_true_binary, scores)
        auc_roc = roc_auc_score(y_true_binary, scores)
        auc_pr = average_precision_score(y_true_binary, scores)
        operating_point = compute_operating_point(y_true_binary, scores, source["threshold"])

        axes[0].plot(
            fpr,
            tpr,
            color=color,
            linewidth=2.3,
            label=f"{source['plot_label']} ({auc_roc:.3f})",
        )
        axes[0].scatter(
            operating_point["fpr"],
            operating_point["tpr"],
            s=62,
            color=color,
            edgecolors="white",
            linewidths=1.1,
            zorder=5,
        )
        axes[1].plot(
            recall,
            precision,
            color=color,
            linewidth=2.3,
            label=f"{source['plot_label']} ({auc_pr:.3f})",
        )
        axes[1].scatter(
            operating_point["recall"],
            operating_point["precision"],
            s=62,
            color=color,
            edgecolors="white",
            linewidths=1.1,
            zorder=5,
        )

    axes[0].plot([0, 1], [0, 1], linestyle="--", color="#777777", linewidth=1.3, label="Chance")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title(roc_title)
    axes[0].legend(
        loc="lower right",
        frameon=True,
        fontsize=9,
        title="AUC; marker = tuned threshold",
        title_fontsize=9,
    )

    axes[1].axhline(
        positive_rate,
        linestyle="--",
        color="#777777",
        linewidth=1.3,
        label=f"Positive prevalence ({positive_rate:.3f})",
    )
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title(pr_title)
    axes[1].legend(
        loc=pr_legend_loc,
        frameon=True,
        fontsize=9,
        title="AUC; marker = tuned threshold",
        title_fontsize=9,
    )
    fig.tight_layout()
    return fig


def plot_model_roc_pr_comparison(curve_sources):
    return plot_curve_comparison(
        curve_sources,
        roc_title="OOF ROC Comparison Across Models",
        pr_title="OOF Precision-Recall Comparison Across Models",
    )


def plot_structured_roc_pr_comparison(curve_sources):
    return plot_curve_comparison(
        curve_sources,
        roc_title="OOF ROC Comparison Across BiLSTM Variants",
        pr_title="OOF Precision-Recall Comparison Across BiLSTM Variants",
        figsize=(13.4, 5.4),
    )


def plot_lag_distributions(
    speech_metrics,
    comparison_speech_metrics=None,
    comparison_label="Worse-Boundary BiLSTM",
    focus_range=(-7, 7),
    bin_width=1.0,
):
    fig, ax = plt.subplots(figsize=(10.4, 6.4))
    series = [
        {
            "values": speech_metrics["start_lag"].dropna().astype(float).to_numpy(),
            "label": "Selected - Start Lag",
            "color": "#4c72b0",
            "linestyle": "-",
        },
        {
            "values": speech_metrics["end_lag"].dropna().astype(float).to_numpy(),
            "label": "Selected - End Lag",
            "color": "#dd8452",
            "linestyle": "-",
        },
    ]

    if comparison_speech_metrics is not None:
        series.extend(
            [
                {
                    "values": comparison_speech_metrics["start_lag"].dropna().astype(float).to_numpy(),
                    "label": f"{comparison_label} - Start Lag",
                    "color": "#8c2d4f",
                    "linestyle": "--",
                },
                {
                    "values": comparison_speech_metrics["end_lag"].dropna().astype(float).to_numpy(),
                    "label": f"{comparison_label} - End Lag",
                    "color": "#b24a2f",
                    "linestyle": "--",
                },
            ]
        )

    low, high = focus_range
    bins = np.arange(low - 0.5, high + 1.5, bin_width)
    hidden_points = 0

    for item in series:
        if item["values"].size == 0:
            continue
        mean_abs_lag = np.mean(np.abs(item["values"]))
        visible_values = item["values"][(item["values"] >= low) & (item["values"] <= high)]
        hidden_points += int(item["values"].size - visible_values.size)
        if visible_values.size == 0:
            continue
        ax.hist(
            visible_values,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=2.2,
            color=item["color"],
            linestyle=item["linestyle"],
            alpha=0.95,
            label=f"{item['label']} | mean abs={mean_abs_lag:.2f}",
        )

    ax.axvline(0, color="#2f2f2f", linestyle="--", linewidth=1.2)
    ax.set_ylabel("Density")
    ax.set_xlabel("Lag in Sentence Indices")
    ax.set_xlim(low - 0.5, high + 0.5)
    ax.set_xticks(np.arange(low, high + 1, 1))
    ax.set_title(
        "Boundary Error Distribution for the Selected BiLSTM"
        if comparison_speech_metrics is None
        else "Boundary Error Distribution Around the True Span",
        pad=14,
    )
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        frameon=True,
        fontsize=9,
        ncol=2,
        columnspacing=1.3,
        handlelength=2.8,
    )
    if comparison_speech_metrics is not None and hidden_points > 0:
        ax.text(
            0.5,
            1.005,
            f"Central window [{low}, {high}] sentence indices; {hidden_points} extreme lag values omitted.",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=8.8,
            color="#3a3a3a",
        )
        fig.tight_layout(rect=(0, 0.12, 1, 0.95))
    else:
        fig.tight_layout(rect=(0, 0.12, 1, 1))
    return fig


def build_trajectory_frame(clean_main_threshold, cam_threshold, fusion_alpha, fusion_threshold, camembert_train_dir):
    simple_oof = pd.read_csv(CLEAN_MAIN_SIMPLE_TRAIN_DIR / "oof_predictions.csv")
    simple_calibration = load_json(CLEAN_MAIN_SIMPLE_TRAIN_DIR / "calibration.json")
    camembert_oof = pd.read_csv(camembert_train_dir / "oof_predictions.csv")
    camembert_calibration = load_json(camembert_train_dir / "calibration.json")

    merged = simple_oof.merge(
        camembert_oof[["speech_id", "sentence_id", "prob_mitterrand_raw"]],
        on=["speech_id", "sentence_id"],
        suffixes=("_simple", "_camembert"),
    )
    merged["prob_simple_calibrated"] = apply_score_delta(
        merged["prob_mitterrand_raw_simple"].to_numpy(dtype=np.float64),
        simple_calibration["equivalent_score_delta"],
    )
    merged["prob_camembert_calibrated"] = apply_score_delta(
        merged["prob_mitterrand_raw_camembert"].to_numpy(dtype=np.float64),
        camembert_calibration["equivalent_score_delta"],
    )
    merged["prob_fusion_raw"] = weighted_logit_average(
        merged["prob_mitterrand_raw_simple"].to_numpy(dtype=np.float64),
        merged["prob_mitterrand_raw_camembert"].to_numpy(dtype=np.float64),
        fusion_alpha,
    )
    merged["prob_fusion_calibrated"] = apply_score_delta(
        merged["prob_fusion_raw"].to_numpy(dtype=np.float64),
        logit(fusion_threshold),
    )
    return merged


def plot_probability_trajectories(trajectory_frame, speech_ids):
    labels = {
        speech_ids[0]: "Clean Boundary Example",
        speech_ids[1]: "Noisy Boundary Example",
    }
    fig, axes = plt.subplots(2, 1, figsize=(12.8, 8.4), sharey=True)

    for ax, speech_id in zip(axes, speech_ids):
        group = trajectory_frame[trajectory_frame["speech_id"] == speech_id].sort_values("sentence_id")
        x = group["sentence_id"].to_numpy(dtype=np.int64)
        true_mask = group["true_label"].to_numpy(dtype=np.int64) == -1

        for block_index, (start, end) in enumerate(contiguous_blocks(x, true_mask)):
            ax.axvspan(
                start - 0.5,
                end + 0.5,
                color="#bdbdbd",
                alpha=0.32,
                zorder=0,
                label="True Mitterrand Span" if block_index == 0 else None,
            )
            ax.axvline(start - 0.5, color="#8c8c8c", linestyle=":", linewidth=1.1, alpha=0.9, zorder=1)
            ax.axvline(end + 0.5, color="#8c8c8c", linestyle=":", linewidth=1.1, alpha=0.9, zorder=1)

        ax.plot(x, group["prob_simple_calibrated"], color="#4c72b0", linewidth=2.0, label="BiLSTM", zorder=3)
        ax.plot(
            x,
            group["prob_camembert_calibrated"],
            color="#dd8452",
            linewidth=2.0,
            label="CamemBERT",
            zorder=3,
        )
        ax.plot(
            x,
            group["prob_fusion_calibrated"],
            color="#55a868",
            linewidth=2.2,
            label="Weighted Fusion",
            zorder=3,
        )
        ax.axhline(0.5, color="#2f2f2f", linestyle="--", linewidth=1.0)
        ax.set_title(f"{labels[speech_id]} (speech_id={speech_id})")
        ax.set_ylim(-0.02, 1.02)

    axes[-1].set_xlabel("Sentence Index Inside Speech")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.93), ncol=4, frameon=True)
    fig.supylabel("Calibrated Mitterrand Probability", x=0.02)
    fig.suptitle("Sentence-Level Probability Trajectories", y=0.99)
    fig.tight_layout(rect=(0.04, 0.04, 1, 0.86))
    return fig


def plot_model_comparison_oof(model_comparison_frame):
    plot_frame = model_comparison_frame.sort_values("oof_f1_mitterrand", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(
        plot_frame["model"],
        plot_frame["oof_f1_mitterrand"],
        color=[MODEL_COLOR_MAP[model] for model in plot_frame["model"]],
    )
    ax.set_xlabel("OOF F1 (Mitterrand positive) at Tuned Threshold")
    ax.set_title("Model Comparison on Train OOF Predictions")
    for _, row in plot_frame.iterrows():
        ax.text(row["oof_f1_mitterrand"] + 0.002, row["model"], f"{row['oof_f1_mitterrand']:.3f}", va="center")
    return fig


def plot_fusion_alpha_sweep(fusion_dir, camembert_train_dir):
    alpha_sweep = build_fusion_alpha_sweep_frame(fusion_dir, camembert_train_dir)
    selected_alpha = float(load_json(Path(fusion_dir) / "metrics.json")["calibration"]["alpha"])
    selected_row = alpha_sweep.loc[np.isclose(alpha_sweep["alpha"], selected_alpha)].iloc[0]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(alpha_sweep["alpha"], alpha_sweep["f1_mitterrand"], marker="o", linewidth=2.2, color="#55a868")
    ax.axvline(selected_alpha, color="#2f2f2f", linestyle="--", linewidth=1.2)
    ax.set_xlabel("BiLSTM Weight Alpha")
    ax.set_ylabel("OOF F1 (Mitterrand positive)")
    ax.set_title("Weighted Fusion Alpha Sweep")
    ax.text(selected_alpha + 0.01, float(selected_row["f1_mitterrand"]) - 0.003, f"selected={selected_alpha:.2f}")
    return fig


def build_readme(asset_paths):
    lines = [
        "# Presidents Report Assets",
        "",
        "This folder was generated by `scripts/build_presidents_report_assets.py`.",
        "",
        "## Main Report-Safe Figures",
    ]
    for path in asset_paths["report_figures"]:
        lines.append(f"- `{path.relative_to(OUT_DIR)}`")
    lines.extend(["", "## Main Tables"])
    for path in asset_paths["report_tables"]:
        lines.append(f"- `{path.relative_to(OUT_DIR)}`")
    (OUT_DIR / "README.md").write_text("\n".join(lines) + "\n")


def main():
    configure_plot_style()
    ensure_output_dirs()

    asset_paths = {
        "report_figures": [],
        "report_tables": [],
    }

    simple_sweep_frame = aggregate_simple_sweep_runs()
    camembert_train_dir = locate_camembert_train_dir()
    model_comparison_frame = aggregate_report_safe_model_comparison(
        simple_sweep_frame,
        camembert_train_dir,
    )
    curve_sources = build_report_safe_curve_sources(simple_sweep_frame, camembert_train_dir)
    structured_model = locate_structured_best_run()
    structured_curve_sources = build_structured_curve_sources(simple_sweep_frame)
    paper_model_table = build_paper_model_table(model_comparison_frame)
    camera_ready_table = build_camera_ready_model_table(model_comparison_frame, structured_model)
    boundary_comparison_run = select_boundary_comparison_run()

    float_columns = [
        column for column in simple_sweep_frame.columns
        if simple_sweep_frame[column].dtype.kind == "f"
    ]
    for path in save_table(
        simple_sweep_frame[
            [
                "plot_id",
                "short_label",
                "best_epoch",
                "tuned_threshold",
                "oof_accuracy_tuned",
                "oof_precision_mitterrand_tuned",
                "oof_recall_mitterrand_tuned",
                "oof_f1_mitterrand_tuned",
                "oof_roc_auc",
                "oof_pr_auc",
                "start_lag_mean",
                "end_lag_mean",
                "multi_block_speeches",
            ]
        ],
        "simple_sweep_runs_report_safe",
        TABLES_DIR,
        float_columns=float_columns,
    ):
        asset_paths["report_tables"].append(path)

    best_run_dir = Path(load_json(SIMPLE_SWEEP_SUMMARY)["best_ranked_run"]["output_dir"])
    best_run_oof = pd.read_csv(best_run_dir / "oof_predictions.csv")
    best_run_threshold = float(load_json(best_run_dir / "calibration.json")["threshold"])
    best_fold_metrics = compute_fold_metrics(best_run_oof, best_run_threshold)

    for path in save_table(
        best_fold_metrics[
            [
                "fold",
                "accuracy",
                "balanced_accuracy",
                "precision_mitterrand",
                "recall_mitterrand",
                "f1_mitterrand",
                "roc_auc",
                "pr_auc",
            ]
        ],
        "simple_best_fold_metrics",
        TABLES_DIR,
        float_columns=[column for column in best_fold_metrics.columns if column != "fold"],
    ):
        asset_paths["report_tables"].append(path)

    for path in save_table(
        model_comparison_frame[
            [
                "model",
                "oof_accuracy",
                "oof_precision_mitterrand",
                "oof_recall_mitterrand",
                "oof_f1_mitterrand",
                "oof_roc_auc",
                "oof_pr_auc",
                "threshold",
            ]
        ],
        "model_comparison_report_safe",
        TABLES_DIR,
        float_columns=[column for column in model_comparison_frame.columns if column != "model"],
    ):
        asset_paths["report_tables"].append(path)

    for path in save_table(
        paper_model_table,
        "model_comparison_paper",
        TABLES_DIR,
        float_columns=[column for column in paper_model_table.columns if column != "Model"],
    ):
        asset_paths["report_tables"].append(path)

    for path in save_table(
        camera_ready_table,
        "model_comparison_camera_ready",
        TABLES_DIR,
        float_columns=[column for column in camera_ready_table.columns if column not in {"Model", "Variant"}],
    ):
        asset_paths["report_tables"].append(path)
    asset_paths["report_tables"].append(
        save_styled_latex_table(
            camera_ready_table,
            "model_comparison_camera_ready_styled",
            TABLES_DIR,
            caption="Grouped out-of-fold comparison across the main report-safe models.",
            label="tab:presidents-model-comparison-oof",
        )
    )

    speech_metrics = compute_speech_block_metrics(
        pd.read_csv(CLEAN_MAIN_SIMPLE_TRAIN_DIR / "oof_predictions.csv"),
        "prob_mitterrand_raw",
        float(load_json(CLEAN_MAIN_SIMPLE_TRAIN_DIR / "calibration.json")["threshold"]),
    )
    selected_boundary_summary = compute_boundary_lag_summary(speech_metrics)
    clean_example_speech, noisy_example_speech = choose_example_speeches(speech_metrics)
    selected_speeches = {
        "clean_example_speech_id": clean_example_speech,
        "noisy_example_speech_id": noisy_example_speech,
    }
    (METADATA_DIR / "selected_example_speeches.json").write_text(json.dumps(selected_speeches, indent=2))

    for path in save_figure(plot_simple_sweep_ranking(simple_sweep_frame), "simple_sweep_ranking_oof", FIGURES_DIR):
        asset_paths["report_figures"].append(path)
    for path in save_figure(plot_threshold_sweep(best_run_dir), "simple_threshold_sweep", FIGURES_DIR):
        asset_paths["report_figures"].append(path)
    for path in save_figure(plot_epoch_selection(best_run_dir), "simple_epoch_selection", FIGURES_DIR):
        asset_paths["report_figures"].append(path)
    for path in save_figure(plot_fold_heatmap(best_fold_metrics), "simple_fold_metrics_heatmap", FIGURES_DIR):
        asset_paths["report_figures"].append(path)
    for path in save_figure(plot_roc_pr_curves(best_run_oof, best_run_threshold), "simple_roc_pr_curves", FIGURES_DIR):
        asset_paths["report_figures"].append(path)
    for path in save_figure(plot_model_roc_pr_comparison(curve_sources), "model_comparison_roc_pr_oof", FIGURES_DIR):
        asset_paths["report_figures"].append(path)
    for path in save_figure(
        plot_structured_roc_pr_comparison(structured_curve_sources),
        "structured_model_roc_pr_oof",
        FIGURES_DIR,
    ):
        asset_paths["report_figures"].append(path)
    for path in save_figure(
        plot_lag_distributions(
            speech_metrics,
            comparison_speech_metrics=boundary_comparison_run["speech_metrics"],
        ),
        "simple_boundary_lags",
        FIGURES_DIR,
    ):
        asset_paths["report_figures"].append(path)

    fusion_source_dir = (
        ROOT / "Dataset/out/presidents_bilstm_camembert_fusion_main_full"
        if (ROOT / "Dataset/out/presidents_bilstm_camembert_fusion_main_full/metrics.json").exists()
        else FUSION_FULLCHECK_DIR
    )
    fusion_metrics = load_json(fusion_source_dir / "metrics.json")
    trajectory_frame = build_trajectory_frame(
        clean_main_threshold=float(load_json(CLEAN_MAIN_SIMPLE_TRAIN_DIR / "calibration.json")["threshold"]),
        cam_threshold=float(load_json(camembert_train_dir / "calibration.json")["threshold"]),
        fusion_alpha=float(fusion_metrics["calibration"]["alpha"]),
        fusion_threshold=float(fusion_metrics["calibration"]["threshold"]),
        camembert_train_dir=camembert_train_dir,
    )
    for path in save_figure(
        plot_probability_trajectories(
            trajectory_frame,
            (clean_example_speech, noisy_example_speech),
        ),
        "probability_trajectories_examples",
        FIGURES_DIR,
    ):
        asset_paths["report_figures"].append(path)

    for path in save_figure(plot_model_comparison_oof(model_comparison_frame), "model_comparison_oof", FIGURES_DIR):
        asset_paths["report_figures"].append(path)
    for path in save_figure(
        plot_fusion_alpha_sweep(fusion_source_dir, camembert_train_dir),
        "fusion_alpha_sweep",
        FIGURES_DIR,
    ):
        asset_paths["report_figures"].append(path)

    metadata = {
        "simple_sweep_summary": str(SIMPLE_SWEEP_SUMMARY.resolve()),
        "clean_main_simple_train_dir": str(CLEAN_MAIN_SIMPLE_TRAIN_DIR.resolve()),
        "camembert_train_dir": str(camembert_train_dir.resolve()),
        "fusion_source_dir": str(fusion_source_dir.resolve()),
        "structured_best_run": {
            "run_name": structured_model["run_name"],
            "run_dir": str(structured_model["run_dir"].resolve()),
            "calibration_dir": str(structured_model["calibration_dir"].resolve()),
            "score_name": structured_model["score_name"],
            "threshold": structured_model["threshold"],
            "short_label": structured_model["short_label"],
        },
        "selected_example_speeches": selected_speeches,
        "selected_boundary_lag_summary": selected_boundary_summary,
        "boundary_comparison_run": {
            "run_name": boundary_comparison_run["run_name"],
            "short_label": boundary_comparison_run["short_label"],
            "run_dir": str(boundary_comparison_run["run_dir"].resolve()),
            "mean_abs_start_lag": boundary_comparison_run["mean_abs_start_lag"],
            "mean_abs_end_lag": boundary_comparison_run["mean_abs_end_lag"],
            "multi_block_speeches": boundary_comparison_run["multi_block_speeches"],
        },
    }
    (METADATA_DIR / "report_bundle_metadata.json").write_text(json.dumps(metadata, indent=2))
    build_readme(asset_paths)

    print(json.dumps({"output_dir": str(OUT_DIR.resolve()), **metadata}, indent=2))


if __name__ == "__main__":
    main()
