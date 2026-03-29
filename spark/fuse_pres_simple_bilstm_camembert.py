import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


CALIBRATION_THRESHOLD_START = 0.05
CALIBRATION_THRESHOLD_END = 0.95
CALIBRATION_THRESHOLD_STEP = 0.005


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fuse the cleaned presidents BiLSTM and CamemBERT with a weighted logit average."
    )
    parser.add_argument(
        "--bilstm-train-dir",
        default="Dataset/out/presidents_lstm_simple",
    )
    parser.add_argument(
        "--bilstm-submission-dir",
        default="Dataset/out/presidents_lstm_simple_submission",
    )
    parser.add_argument(
        "--camembert-train-dir",
        default="Dataset/out/presidents_camembert_simple",
    )
    parser.add_argument(
        "--camembert-submission-dir",
        default="Dataset/out/presidents_camembert_simple_submission",
    )
    parser.add_argument(
        "--output-dir",
        default="Dataset/out/presidents_bilstm_camembert_fusion",
    )
    parser.add_argument(
        "--alpha-values",
        default="0.25,0.35,0.45,0.55,0.65,0.75",
    )
    parser.add_argument("--decision-threshold", type=float, default=0.5)
    return parser.parse_args()


def parse_alpha_values(raw_values):
    values = [float(value.strip()) for value in raw_values.split(",") if value.strip()]
    if not values:
        raise ValueError("alpha-values must contain at least one numeric value")
    clipped = np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)
    return np.unique(clipped)


def threshold_grid(start, end, step):
    values = np.arange(start, end + step / 2.0, step, dtype=np.float64)
    return np.clip(values, 1e-6, 1.0 - 1e-6)


def evaluate_threshold(y_true, scores, threshold):
    y_pred = np.where(scores >= threshold, -1, 1)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "mitterrand_count": int((y_pred == -1).sum()),
        "chirac_count": int((y_pred == 1).sum()),
    }


def selection_key(record):
    return (
        float(record["f1_macro"]),
        float(record["balanced_accuracy"]),
        -float(record["threshold"]),
    )


def weighted_logit_average(prob_a, prob_b, alpha):
    clipped_a = np.clip(np.asarray(prob_a, dtype=np.float64), 1e-9, 1.0 - 1e-9)
    clipped_b = np.clip(np.asarray(prob_b, dtype=np.float64), 1e-9, 1.0 - 1e-9)
    fused_logits = float(alpha) * logit(clipped_a) + (1.0 - float(alpha)) * logit(clipped_b)
    return expit(fused_logits)


def sweep_thresholds(y_true, scores, thresholds):
    records = []
    for threshold in thresholds:
        record = evaluate_threshold(y_true, scores, float(threshold))
        record["threshold"] = float(threshold)
        record["equivalent_score_delta"] = float(logit(float(threshold)))
        records.append(record)

    sweep_frame = pd.DataFrame(records).sort_values(
        ["f1_macro", "balanced_accuracy", "threshold"],
        ascending=[False, False, True],
    )
    return pd.DataFrame(records), sweep_frame.iloc[0].to_dict()


def load_aligned_oof(bilstm_path, camembert_path):
    bilstm = pd.read_csv(bilstm_path).sort_values(["speech_id", "sentence_id"]).reset_index(drop=True)
    camembert = pd.read_csv(camembert_path).sort_values(["speech_id", "sentence_id"]).reset_index(drop=True)

    bilstm_keys = bilstm[["speech_id", "sentence_id"]]
    camembert_keys = camembert[["speech_id", "sentence_id"]]
    if not bilstm_keys.equals(camembert_keys):
        raise ValueError("BiLSTM and CamemBERT OOF predictions are not aligned")
    if not np.array_equal(
        bilstm["true_label"].to_numpy(dtype=np.int64),
        camembert["true_label"].to_numpy(dtype=np.int64),
    ):
        raise ValueError("BiLSTM and CamemBERT OOF labels do not match")

    return pd.DataFrame(
        {
            "speech_id": bilstm["speech_id"].to_numpy(dtype=np.int64),
            "sentence_id": bilstm["sentence_id"].to_numpy(dtype=np.int64),
            "true_label": bilstm["true_label"].to_numpy(dtype=np.int64),
            "prob_bilstm_raw": bilstm["prob_mitterrand_raw"].to_numpy(dtype=np.float64),
            "prob_camembert_raw": camembert["prob_mitterrand_raw"].to_numpy(dtype=np.float64),
        }
    )


def load_aligned_test(bilstm_path, camembert_path):
    bilstm = pd.read_csv(bilstm_path)
    camembert = pd.read_csv(camembert_path)

    bilstm_keys = bilstm[["speech_id", "sentence_id"]].reset_index(drop=True)
    camembert_keys = camembert[["speech_id", "sentence_id"]].reset_index(drop=True)
    if not bilstm_keys.equals(camembert_keys):
        raise ValueError("BiLSTM and CamemBERT test predictions are not aligned")

    frame = pd.DataFrame(
        {
            "speech_id": bilstm["speech_id"].to_numpy(dtype=np.int64),
            "sentence_id": bilstm["sentence_id"].to_numpy(dtype=np.int64),
            "prob_bilstm_raw": bilstm["prob_mitterrand_raw"].to_numpy(dtype=np.float64),
            "prob_camembert_raw": camembert["prob_mitterrand_raw"].to_numpy(dtype=np.float64),
        }
    )
    if "text" in bilstm.columns:
        frame["text"] = bilstm["text"].astype(str).to_numpy()
    return frame


def labels_from_scores(scores, threshold):
    return np.where(scores >= threshold, "M", "C")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    started_at = datetime.now().astimezone().isoformat()
    run_start = time.perf_counter()

    oof_frame = load_aligned_oof(
        Path(args.bilstm_train_dir) / "oof_predictions.csv",
        Path(args.camembert_train_dir) / "oof_predictions.csv",
    )
    alpha_values = parse_alpha_values(args.alpha_values)
    thresholds = threshold_grid(
        CALIBRATION_THRESHOLD_START,
        CALIBRATION_THRESHOLD_END,
        CALIBRATION_THRESHOLD_STEP,
    )

    y_true = oof_frame["true_label"].to_numpy(dtype=np.int64)
    alpha_records = []
    best_record = None

    for alpha in alpha_values:
        fused_scores = weighted_logit_average(
            oof_frame["prob_bilstm_raw"].to_numpy(dtype=np.float64),
            oof_frame["prob_camembert_raw"].to_numpy(dtype=np.float64),
            alpha,
        )
        _, threshold_record = sweep_thresholds(y_true, fused_scores, thresholds)
        record = {
            "method": "weighted_logit_average",
            "alpha": float(alpha),
            **threshold_record,
        }
        alpha_records.append(record)
        if best_record is None or selection_key(record) > selection_key(best_record):
            best_record = record

    alpha_sweep = pd.DataFrame(alpha_records).sort_values(
        ["f1_macro", "balanced_accuracy", "alpha"],
        ascending=[False, False, True],
    )
    alpha_sweep_path = output_dir / "alpha_sweep.csv"
    alpha_sweep.to_csv(alpha_sweep_path, index=False)

    best_alpha = float(best_record["alpha"])
    best_threshold = float(best_record["threshold"])
    best_delta = float(best_record["equivalent_score_delta"])
    fused_oof_scores = weighted_logit_average(
        oof_frame["prob_bilstm_raw"].to_numpy(dtype=np.float64),
        oof_frame["prob_camembert_raw"].to_numpy(dtype=np.float64),
        best_alpha,
    )

    calibration = {
        "alpha": best_alpha,
        "threshold": best_threshold,
        "equivalent_score_delta": best_delta,
    }
    calibration_path = output_dir / "calibration.json"
    calibration_path.write_text(json.dumps(calibration, indent=2))

    test_frame = load_aligned_test(
        Path(args.bilstm_submission_dir) / "test_predictions_detailed.csv",
        Path(args.camembert_submission_dir) / "test_predictions_detailed.csv",
    )
    test_frame["prob_mitterrand_raw"] = weighted_logit_average(
        test_frame["prob_bilstm_raw"].to_numpy(dtype=np.float64),
        test_frame["prob_camembert_raw"].to_numpy(dtype=np.float64),
        best_alpha,
    )
    test_frame["prob_mitterrand_calibrated"] = expit(
        logit(np.clip(test_frame["prob_mitterrand_raw"].to_numpy(dtype=np.float64), 1e-9, 1.0 - 1e-9))
        - best_delta
    )
    test_frame["pred_label_calibrated"] = labels_from_scores(
        test_frame["prob_mitterrand_calibrated"].to_numpy(dtype=np.float64),
        float(args.decision_threshold),
    )
    aligned_labels = labels_from_scores(
        test_frame["prob_mitterrand_calibrated"].to_numpy(dtype=np.float64),
        0.5,
    )
    if not np.array_equal(aligned_labels, test_frame["pred_label_calibrated"].to_numpy()):
        raise ValueError(
            "submission labels must be reproducible from the calibrated probability "
            "file using a 0.5 threshold"
        )

    detailed_path = output_dir / "test_predictions_detailed.csv"
    test_frame.to_csv(detailed_path, index=False)

    raw_prob_path = output_dir / "submission_prob_mitterrand_raw.csv"
    test_frame[["prob_mitterrand_raw"]].to_csv(raw_prob_path, index=False, header=False)

    calibrated_prob_path = output_dir / "submission_prob_mitterrand_calibrated.csv"
    test_frame[["prob_mitterrand_calibrated"]].to_csv(
        calibrated_prob_path,
        index=False,
        header=False,
    )

    submission_probability_path = output_dir / "submission_probability.csv"
    test_frame[["prob_mitterrand_calibrated"]].to_csv(
        submission_probability_path,
        index=False,
        header=False,
    )

    label_path = output_dir / "submission_label_calibrated.csv"
    test_frame[["pred_label_calibrated"]].to_csv(label_path, index=False, header=False)

    oof_metrics_0p5 = evaluate_threshold(y_true, fused_oof_scores, 0.5)
    oof_metrics_best = evaluate_threshold(y_true, fused_oof_scores, best_threshold)

    metrics = {
        "started_at": started_at,
        "ended_at": datetime.now().astimezone().isoformat(),
        "total_seconds": time.perf_counter() - run_start,
        "alpha_values": alpha_values.tolist(),
        "selected_alpha": best_alpha,
        "decision_threshold": float(args.decision_threshold),
        "oof_metrics_at_0p5": oof_metrics_0p5,
        "oof_metrics_at_best_threshold": oof_metrics_best,
        "calibration": calibration,
        "label_counts": test_frame["pred_label_calibrated"].value_counts().to_dict(),
        "outputs": {
            "alpha_sweep": str(alpha_sweep_path.resolve()),
            "calibration": str(calibration_path.resolve()),
            "detailed_predictions": str(detailed_path.resolve()),
            "submission_prob_mitterrand_raw": str(raw_prob_path.resolve()),
            "submission_prob_mitterrand_calibrated": str(calibrated_prob_path.resolve()),
            "submission_probability": str(submission_probability_path.resolve()),
            "submission_label_calibrated": str(label_path.resolve()),
        },
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
