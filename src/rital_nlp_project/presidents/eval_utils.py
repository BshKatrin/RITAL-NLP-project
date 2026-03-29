from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence
import warnings

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedShuffleSplit


PRESIDENT_SUBMISSION_LABELS = ("C", "M")
KNOWN_LABEL_COLUMNS = (
    "pred_label_calibrated",
    "pred_single_negative_span_label",
    "pred_raw_label",
    "label",
    "matched_label",
)
KNOWN_SCORE_COLUMNS = (
    "prob_mitterrand_calibrated",
    "prob_mitterrand_raw",
    "prob_negative_single_negative_span",
    "prob_negative_raw",
)


@dataclass(frozen=True)
class ProxyEvaluation:
    summary: dict[str, object]
    per_method: pd.DataFrame
    accepted_rows: pd.DataFrame
    unresolved_rows: pd.DataFrame


def threshold_grid(start: float, end: float, step: float) -> np.ndarray:
    if not 0.0 < start < 1.0 or not 0.0 < end < 1.0:
        raise ValueError("threshold range must be strictly between 0 and 1")
    if end <= start:
        raise ValueError("threshold-end must be greater than threshold-start")
    if step <= 0:
        raise ValueError("threshold-step must be strictly positive")
    values = np.arange(start, end + step / 2.0, step, dtype=np.float64)
    return np.clip(values, 1e-6, 1.0 - 1e-6)


def apply_score_delta(scores: np.ndarray, delta: float) -> np.ndarray:
    clipped = np.clip(np.asarray(scores, dtype=np.float64), 1e-9, 1.0 - 1e-9)
    return expit(logit(clipped) - float(delta))


def labels_from_threshold(
    scores: np.ndarray,
    *,
    threshold: float,
    positive_label: int = -1,
    negative_label: int = 1,
) -> np.ndarray:
    return np.where(scores >= threshold, positive_label, negative_label)


def threshold_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    threshold: float,
    positive_label: int = -1,
    negative_label: int = 1,
) -> dict[str, float]:
    y_pred = labels_from_threshold(
        scores,
        threshold=threshold,
        positive_label=positive_label,
        negative_label=negative_label,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def build_threshold_record(
    scores: np.ndarray,
    y_true: np.ndarray,
    *,
    threshold: float,
    score_name: str,
    positive_label: int = -1,
    negative_label: int = 1,
) -> dict[str, float]:
    metrics = threshold_metrics(
        y_true,
        scores,
        threshold=float(threshold),
        positive_label=positive_label,
        negative_label=negative_label,
    )
    return {
        "selected_score_name": score_name,
        "score_name": score_name,
        "threshold": float(threshold),
        "best_threshold": float(threshold),
        "equivalent_score_delta": float(logit(float(threshold))),
        **metrics,
        "mitterrand_count": int((scores >= threshold).sum()),
        "chirac_count": int((scores < threshold).sum()),
    }


def sweep_thresholds(
    frame: pd.DataFrame,
    *,
    score_name: str,
    thresholds: np.ndarray,
    true_label_column: str = "true_label",
    positive_label: int = -1,
    negative_label: int = 1,
) -> tuple[pd.DataFrame, dict[str, float]]:
    y_true = frame[true_label_column].to_numpy(dtype=np.int64)
    scores = frame[score_name].to_numpy(dtype=np.float64)
    records = [
        build_threshold_record(
            scores,
            y_true,
            threshold=float(threshold),
            score_name=score_name,
            positive_label=positive_label,
            negative_label=negative_label,
        )
        for threshold in thresholds
    ]
    sweep_frame = pd.DataFrame(records).sort_values(
        ["f1_macro", "balanced_accuracy", "threshold"],
        ascending=[False, False, True],
    )
    return pd.DataFrame(records), sweep_frame.iloc[0].to_dict()


def contiguous_blocks(mask: np.ndarray) -> list[tuple[int, int]]:
    mask = np.asarray(mask, dtype=np.bool_)
    if mask.ndim != 1:
        raise ValueError("mask must be 1-dimensional")
    changes = np.diff(np.r_[0, mask.astype(np.int8), 0])
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0] - 1
    return [(int(start), int(end)) for start, end in zip(starts, ends, strict=True)]


def maybe_float(value: float) -> float | None:
    if np.isnan(value):
        return None
    return float(value)


def compute_boundary_diagnostics(
    frame: pd.DataFrame,
    *,
    threshold: float,
    score_column: str = "prob_mitterrand_raw",
    true_label_column: str = "true_label",
    speech_id_column: str = "speech_id",
    sentence_id_column: str = "sentence_id",
    negative_label: int = -1,
    positive_label: int = 1,
) -> tuple[dict[str, object], pd.DataFrame]:
    required_columns = {
        speech_id_column,
        sentence_id_column,
        true_label_column,
        score_column,
    }
    missing_columns = required_columns - set(frame.columns)
    if missing_columns:
        raise ValueError(
            f"Missing required columns for diagnostics: {sorted(missing_columns)}"
        )

    rows = []
    ordered = frame.sort_values([speech_id_column, sentence_id_column]).reset_index(
        drop=True
    )
    for speech_id, group in ordered.groupby(speech_id_column, sort=False):
        true_labels = group[true_label_column].to_numpy(dtype=np.int64)
        scores = group[score_column].to_numpy(dtype=np.float64)
        predicted_labels = labels_from_threshold(
            scores,
            threshold=threshold,
            positive_label=negative_label,
            negative_label=positive_label,
        )

        true_blocks = contiguous_blocks(true_labels == negative_label)
        predicted_blocks = contiguous_blocks(predicted_labels == negative_label)

        has_true_block = bool(true_blocks)
        has_predicted_block = bool(predicted_blocks)
        true_start = true_blocks[0][0] if has_true_block else np.nan
        true_end = true_blocks[-1][1] if has_true_block else np.nan
        pred_start = predicted_blocks[0][0] if has_predicted_block else np.nan
        pred_end = predicted_blocks[-1][1] if has_predicted_block else np.nan

        rows.append(
            {
                "speech_id": int(speech_id),
                "num_sentences": int(len(group)),
                "true_negative_sentence_count": int((true_labels == negative_label).sum()),
                "predicted_negative_sentence_count": int(
                    (predicted_labels == negative_label).sum()
                ),
                "has_true_mitterrand_block": has_true_block,
                "has_predicted_mitterrand_block": has_predicted_block,
                "num_true_mitterrand_blocks": int(len(true_blocks)),
                "num_predicted_mitterrand_blocks": int(len(predicted_blocks)),
                "true_start": maybe_float(true_start),
                "true_end": maybe_float(true_end),
                "predicted_start": maybe_float(pred_start),
                "predicted_end": maybe_float(pred_end),
                "start_lag": maybe_float(pred_start - true_start)
                if has_true_block and has_predicted_block
                else None,
                "end_lag": maybe_float(pred_end - true_end)
                if has_true_block and has_predicted_block
                else None,
            }
        )

    per_speech = pd.DataFrame(rows).sort_values("speech_id").reset_index(drop=True)
    start_lags = pd.to_numeric(per_speech["start_lag"], errors="coerce")
    end_lags = pd.to_numeric(per_speech["end_lag"], errors="coerce")

    summary: dict[str, object] = {
        "threshold": float(threshold),
        "score_column": score_column,
        "speeches_total": int(len(per_speech)),
        "speeches_with_true_mitterrand_block": int(
            per_speech["has_true_mitterrand_block"].sum()
        ),
        "speeches_with_predicted_mitterrand_block": int(
            per_speech["has_predicted_mitterrand_block"].sum()
        ),
        "speeches_with_true_and_predicted_mitterrand_block": int(
            (
                per_speech["has_true_mitterrand_block"]
                & per_speech["has_predicted_mitterrand_block"]
            ).sum()
        ),
        "speeches_with_multiple_predicted_mitterrand_blocks": int(
            (per_speech["num_predicted_mitterrand_blocks"] > 1).sum()
        ),
        "start_lag_mean": maybe_float(float(start_lags.mean()))
        if start_lags.notna().any()
        else None,
        "start_lag_median": maybe_float(float(start_lags.median()))
        if start_lags.notna().any()
        else None,
        "end_lag_mean": maybe_float(float(end_lags.mean()))
        if end_lags.notna().any()
        else None,
        "end_lag_median": maybe_float(float(end_lags.median()))
        if end_lags.notna().any()
        else None,
    }
    return summary, per_speech


def speech_target_frame(
    frame: pd.DataFrame,
    *,
    speech_id_column: str = "speech_id",
    label_column: str = "label",
    negative_label: int = -1,
) -> pd.DataFrame:
    required_columns = {speech_id_column, label_column}
    missing_columns = required_columns - set(frame.columns)
    if missing_columns:
        raise ValueError(
            f"Missing required columns for grouped targets: {sorted(missing_columns)}"
        )
    grouped = (
        frame.groupby(speech_id_column, sort=False)[label_column]
        .apply(lambda values: int(np.any(values.to_numpy(dtype=np.int64) == negative_label)))
        .reset_index(name="target")
    )
    return grouped


def build_grouped_splits(
    group_targets: pd.DataFrame,
    *,
    n_splits: int,
    seed: int,
    speech_id_column: str = "speech_id",
    target_column: str = "target",
) -> list[tuple[np.ndarray, np.ndarray]]:
    targets = group_targets[target_column].to_numpy(dtype=np.int64)
    indices = np.arange(len(group_targets))
    min_count = int(np.bincount(targets).min()) if len(targets) else 0

    if min_count >= n_splits:
        splitter = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=seed,
        )
        return list(splitter.split(indices, targets))

    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(splitter.split(indices))


def split_grouped_train_validation(
    group_targets: pd.DataFrame,
    *,
    validation_size: float,
    seed: int,
    target_column: str = "target",
) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 < validation_size < 1.0:
        raise ValueError("validation_size must be strictly between 0 and 1")
    if len(group_targets) < 2:
        raise ValueError("Need at least two groups for a validation split")

    targets = group_targets[target_column].to_numpy(dtype=np.int64)
    indices = np.arange(len(group_targets))
    min_count = int(np.bincount(targets).min()) if len(targets) else 0

    if min_count >= 2:
        splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=validation_size,
            random_state=seed,
        )
        train_idx, val_idx = next(splitter.split(indices, targets))
        return np.asarray(train_idx), np.asarray(val_idx)

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(indices)
    val_size = max(1, int(round(len(group_targets) * validation_size)))
    val_idx = np.sort(shuffled[:val_size])
    train_idx = np.sort(shuffled[val_size:])
    return np.asarray(train_idx), np.asarray(val_idx)


def normalize_submission_labels(
    values: Iterable[object],
    *,
    allow_missing: bool = False,
) -> np.ndarray:
    normalized = []
    for value in values:
        if pd.isna(value):
            if allow_missing:
                normalized.append(None)
                continue
            raise ValueError(f"Unsupported submission label value: {value!r}")
        text = str(value).strip().lower()
        if text in {"c", "chirac", "1", "+1"}:
            normalized.append("C")
        elif text in {"m", "mitterrand", "francois mitterrand", "-1"}:
            normalized.append("M")
        else:
            raise ValueError(f"Unsupported submission label value: {value!r}")
    return np.asarray(normalized, dtype=object)


def infer_label_column(frame: pd.DataFrame) -> str:
    for column in KNOWN_LABEL_COLUMNS:
        if column in frame.columns:
            return column
    if frame.shape[1] == 1:
        return frame.columns[0]
    raise ValueError(
        "Could not infer a label column. Pass --label-column explicitly."
    )


def infer_score_column(frame: pd.DataFrame) -> str:
    for column in KNOWN_SCORE_COLUMNS:
        if column in frame.columns:
            return column
    if frame.shape[1] == 1:
        return frame.columns[0]
    raise ValueError(
        "Could not infer a score column. Pass --score-column explicitly."
    )


def load_prediction_column(
    path: str,
    *,
    expected_rows: int,
    explicit_column: str | None = None,
    kind: str = "label",
) -> pd.Series:
    if explicit_column is None:
        raw = pd.read_csv(path, header=None)
        if raw.shape[1] == 1 and len(raw) == expected_rows:
            return raw.iloc[:, 0]

    frame = pd.read_csv(path)
    if explicit_column is not None:
        if explicit_column not in frame.columns:
            raise ValueError(
                f"Column {explicit_column!r} not found in prediction file {path}"
            )
        series = frame[explicit_column]
    else:
        column = infer_label_column(frame) if kind == "label" else infer_score_column(frame)
        series = frame[column]

    if len(series) != expected_rows:
        raise ValueError(
            f"Prediction length mismatch for {path}: {len(series)} != {expected_rows}"
        )
    return series.reset_index(drop=True)


def classification_summary(
    y_true: Sequence[object],
    y_pred: Sequence[object],
    *,
    labels: Sequence[str] = PRESIDENT_SUBMISSION_LABELS,
) -> dict[str, object]:
    y_true_arr = normalize_submission_labels(y_true)
    y_pred_arr = normalize_submission_labels(y_pred)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        matrix = confusion_matrix(y_true_arr, y_pred_arr, labels=list(labels))
        return {
            "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true_arr, y_pred_arr)),
            "precision_macro": float(
                precision_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)
            ),
            "recall_macro": float(
                recall_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)
            ),
            "f1_macro": float(
                f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)
            ),
            "confusion_matrix": matrix.tolist(),
            "labels": list(labels),
            "predicted_counts": {
                label: int(count)
                for label, count in pd.Series(y_pred_arr).value_counts().sort_index().items()
            },
            "true_counts": {
                label: int(count)
                for label, count in pd.Series(y_true_arr).value_counts().sort_index().items()
            },
            "agreement": float(np.mean(y_true_arr == y_pred_arr)),
            "disagreement_count": int(np.sum(y_true_arr != y_pred_arr)),
        }


def evaluate_proxy_predictions(
    proxy_frame: pd.DataFrame,
    predicted_labels: Sequence[object],
    *,
    predicted_scores: Sequence[float] | None = None,
    review_status_column: str = "review_status",
    accepted_value: str = "accepted",
    label_column: str = "matched_label",
    resolution_method_column: str = "resolution_method",
) -> ProxyEvaluation:
    frame = proxy_frame.copy().reset_index(drop=True)
    frame["predicted_label"] = normalize_submission_labels(
        predicted_labels,
        allow_missing=True,
    )

    if predicted_scores is not None:
        scores = np.asarray(predicted_scores, dtype=np.float64)
        if len(scores) != len(frame):
            raise ValueError(
                "predicted_scores must match the number of proxy rows: "
                f"{len(scores)} != {len(frame)}"
            )
        frame["predicted_score"] = scores

    accepted_mask = frame[review_status_column].eq(accepted_value)
    accepted_rows = frame.loc[accepted_mask].copy().reset_index(drop=True)
    unresolved_rows = frame.loc[~accepted_mask].copy().reset_index(drop=True)
    if accepted_rows["predicted_label"].isna().any():
        raise ValueError("Accepted proxy rows must have predicted labels")

    summary = classification_summary(
        accepted_rows[label_column],
        accepted_rows["predicted_label"],
    )
    summary.update(
        {
            "rows_total": int(len(frame)),
            "accepted_rows": int(accepted_mask.sum()),
            "unresolved_rows": int((~accepted_mask).sum()),
        }
    )

    per_method_records = []
    for method, group in accepted_rows.groupby(resolution_method_column, dropna=False):
        record = classification_summary(
            group[label_column],
            group["predicted_label"],
        )
        per_method_records.append(
            {
                "resolution_method": method,
                "rows": int(len(group)),
                **record,
            }
        )
    per_method = pd.DataFrame(per_method_records).sort_values(
        ["f1_macro", "rows"],
        ascending=[False, False],
    )

    if "predicted_score" in unresolved_rows.columns:
        score_series = unresolved_rows["predicted_score"].astype(float)
        summary["unresolved_score_summary"] = {
            "mean": float(score_series.mean()) if len(score_series) else None,
            "median": float(score_series.median()) if len(score_series) else None,
            "min": float(score_series.min()) if len(score_series) else None,
            "max": float(score_series.max()) if len(score_series) else None,
        }

    return ProxyEvaluation(
        summary=summary,
        per_method=per_method.reset_index(drop=True),
        accepted_rows=accepted_rows,
        unresolved_rows=unresolved_rows,
    )
