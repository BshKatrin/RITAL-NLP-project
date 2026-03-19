from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


MetricFunc = Callable[[Any, Any, Any], float]


@dataclass(frozen=True)
class BinaryTaskSpec:
    name: str
    negative_label: int
    positive_label: int
    cv_n_splits: int = 5
    test_size: float = 0.2
    random_state: int = 42
    sequential: bool = False


def infer_binary_task_spec(name: str, y: np.ndarray | list[int]) -> BinaryTaskSpec:
    labels = tuple(sorted(np.unique(y).tolist()))
    if len(labels) != 2:
        raise ValueError(f"{name}: expected exactly 2 labels, got {labels!r}")
    return BinaryTaskSpec(name=name, negative_label=labels[0], positive_label=labels[1])


def get_positive_class_scores(
    estimator: Any,
    X: Any,
    positive_label: int,
) -> np.ndarray | None:
    classes = getattr(estimator, "classes_", None)

    if hasattr(estimator, "predict_proba"):
        proba = np.asarray(estimator.predict_proba(X))
        if proba.ndim == 1:
            return proba
        if proba.ndim == 2:
            if classes is None:
                return proba[:, -1]
            idx = int(np.where(np.asarray(classes) == positive_label)[0][0])
            return proba[:, idx]

    if hasattr(estimator, "decision_function"):
        scores = np.asarray(estimator.decision_function(X))
        if scores.ndim == 1:
            return scores
        if scores.ndim == 2:
            if classes is None:
                return scores[:, -1]
            idx = int(np.where(np.asarray(classes) == positive_label)[0][0])
            return scores[:, idx]

    return None


def classification_report_for_task(
    y_true: np.ndarray | list[int],
    y_pred: np.ndarray | list[int],
    task: BinaryTaskSpec,
) -> str:
    labels = [task.negative_label, task.positive_label]
    return classification_report(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )


def compute_binary_metrics(
    y_true: np.ndarray | list[int],
    y_pred: np.ndarray | list[int],
    *,
    task: BinaryTaskSpec,
    y_score: np.ndarray | None = None,
) -> dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_pos": float(
            precision_score(
                y_true,
                y_pred,
                pos_label=task.positive_label,
                average="binary",
                zero_division=0,
            )
        ),
        "recall_pos": float(
            recall_score(
                y_true,
                y_pred,
                pos_label=task.positive_label,
                average="binary",
                zero_division=0,
            )
        ),
        "f1_pos": float(
            f1_score(
                y_true,
                y_pred,
                pos_label=task.positive_label,
                average="binary",
                zero_division=0,
            )
        ),
        "precision_neg": float(
            precision_score(
                y_true,
                y_pred,
                pos_label=task.negative_label,
                average="binary",
                zero_division=0,
            )
        ),
        "recall_neg": float(
            recall_score(
                y_true,
                y_pred,
                pos_label=task.negative_label,
                average="binary",
                zero_division=0,
            )
        ),
        "f1_neg": float(
            f1_score(
                y_true,
                y_pred,
                pos_label=task.negative_label,
                average="binary",
                zero_division=0,
            )
        ),
    }

    if y_score is not None and len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
        metrics["average_precision"] = float(average_precision_score(y_true == task.positive_label, y_score))

    return metrics


def make_binary_cv_scorers(task: BinaryTaskSpec) -> dict[str, MetricFunc]:
    def _metric_from_predictions(
        estimator: Any,
        X: Any,
        y: Any,
        key: str,
    ) -> float:
        y_pred = estimator.predict(X)
        return compute_binary_metrics(y, y_pred, task=task)[key]

    def _metric_from_scores(
        estimator: Any,
        X: Any,
        y: Any,
        key: str,
    ) -> float:
        y_pred = estimator.predict(X)
        y_score = get_positive_class_scores(estimator, X, task.positive_label)
        return compute_binary_metrics(y, y_pred, task=task, y_score=y_score)[key]

    return {
        "accuracy": lambda estimator, X, y: _metric_from_predictions(estimator, X, y, "accuracy"),
        "balanced_accuracy": lambda estimator, X, y: _metric_from_predictions(estimator, X, y, "balanced_accuracy"),
        "precision_macro": lambda estimator, X, y: _metric_from_predictions(estimator, X, y, "precision_macro"),
        "recall_macro": lambda estimator, X, y: _metric_from_predictions(estimator, X, y, "recall_macro"),
        "f1_macro": lambda estimator, X, y: _metric_from_predictions(estimator, X, y, "f1_macro"),
        "precision_pos": lambda estimator, X, y: _metric_from_predictions(estimator, X, y, "precision_pos"),
        "recall_pos": lambda estimator, X, y: _metric_from_predictions(estimator, X, y, "recall_pos"),
        "f1_pos": lambda estimator, X, y: _metric_from_predictions(estimator, X, y, "f1_pos"),
        "precision_neg": lambda estimator, X, y: _metric_from_predictions(estimator, X, y, "precision_neg"),
        "recall_neg": lambda estimator, X, y: _metric_from_predictions(estimator, X, y, "recall_neg"),
        "f1_neg": lambda estimator, X, y: _metric_from_predictions(estimator, X, y, "f1_neg"),
        "roc_auc": lambda estimator, X, y: _metric_from_scores(estimator, X, y, "roc_auc"),
        "average_precision": lambda estimator, X, y: _metric_from_scores(estimator, X, y, "average_precision"),
    }


DEFAULT_SCORING_NAMES = tuple(make_binary_cv_scorers(BinaryTaskSpec("default", 0, 1)).keys())
