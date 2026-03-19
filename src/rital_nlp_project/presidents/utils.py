from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import PredefinedSplit


def concat_seq_by_class(texts, classes, n=None):
    """Concatenate consecutive texts with the same class label."""

    if not texts:
        return [], []
    if len(texts) != len(classes):
        raise ValueError("texts and classes must have the same length")
    if n is not None and n < 1:
        raise ValueError("n must be >= 1")

    max_block = n
    new_texts = []
    new_classes = []
    i = 0
    length = len(texts)

    while i < length:
        current_class = classes[i]
        j = i + 1
        while j < length and classes[j] == current_class:
            j += 1

        start = i
        while start < j:
            end = min(start + max_block, j) if max_block else j
            new_texts.append(" ".join(texts[start:end]))
            new_classes.append(current_class)
            start = end
        i = j

    return new_texts, new_classes


def smooth_predictions(proba, sigma=1.0, groups=None):
    """Gaussian smoothing for ordered predictions."""

    proba = np.asarray(proba, dtype=float)
    if groups is None:
        smoothed = gaussian_filter1d(proba, sigma=sigma, axis=0)
    else:
        groups = np.asarray(groups)
        if len(groups) != len(proba):
            raise ValueError("groups and predictions must have the same length")

        smoothed = np.empty_like(proba, dtype=float)
        start = 0
        while start < len(proba):
            end = start + 1
            while end < len(proba) and groups[end] == groups[start]:
                end += 1
            smoothed[start:end] = gaussian_filter1d(proba[start:end], sigma=sigma, axis=0)
            start = end

    smoothed = np.clip(smoothed, 0.0, None)
    row_sums = smoothed.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return smoothed / row_sums


def smooth_positive_scores(scores, sigma=1.0, groups=None):
    scores = np.asarray(scores, dtype=float)
    scores = np.clip(scores, 0.0, 1.0)
    proba = np.column_stack([1.0 - scores, scores])
    return smooth_predictions(proba, sigma=sigma, groups=groups)[:, 1]


def _contiguous_fold_ids(n_samples: int, n_splits: int) -> np.ndarray:
    fold_sizes = np.full(n_splits, n_samples // n_splits, dtype=int)
    fold_sizes[: n_samples % n_splits] += 1
    fold_ids = np.empty(n_samples, dtype=int)

    start = 0
    for fold_id, fold_size in enumerate(fold_sizes):
        stop = start + fold_size
        fold_ids[start:stop] = fold_id
        start = stop

    return fold_ids


def _is_valid_binary_fold_assignment(classes: np.ndarray, fold_ids: np.ndarray) -> bool:
    labels = np.unique(classes)
    if labels.size != 2:
        return False

    for fold_id in np.unique(fold_ids):
        test_mask = fold_ids == fold_id
        train_mask = ~test_mask
        if np.unique(classes[test_mask]).size != 2:
            return False
        if np.unique(classes[train_mask]).size != 2:
            return False

    return True


def split_for_cv(classes, n_splits):
    """Return a blocked CV splitter with contiguous test folds."""

    classes = np.asarray(classes)
    for candidate_splits in range(n_splits, 1, -1):
        fold_ids = _contiguous_fold_ids(len(classes), candidate_splits)
        if _is_valid_binary_fold_assignment(classes, fold_ids):
            return PredefinedSplit(test_fold=fold_ids)

    raise ValueError(
        "Could not build a valid blocked cross-validation split with at least 2 folds."
    )


def sequential_holdout_split(X, y, test_size):
    """Use the last contiguous slice as holdout while keeping both classes in train/test."""

    y = np.asarray(y)
    split_idx = int(len(y) * (1 - test_size))
    split_idx = min(max(split_idx, 1), len(y) - 1)

    while split_idx > 1:
        y_train = y[:split_idx]
        y_test = y[split_idx:]
        if np.unique(y_train).size == 2 and np.unique(y_test).size == 2:
            return X[:split_idx], X[split_idx:], y_train, y_test
        split_idx -= 1

    raise ValueError("Could not find a valid sequential holdout split containing both classes.")


class SmoothedProbaClassifier(BaseEstimator, ClassifierMixin):
    _estimator_type = "classifier"

    def __init__(self, base_estimator, sigma=1.0):
        self.base_estimator = base_estimator
        self.sigma = sigma

    def fit(self, X, y):
        self.estimator_ = clone(self.base_estimator)
        self.estimator_.fit(X, y)
        self.classes_ = self.estimator_.classes_
        return self

    def predict_proba(self, X):
        proba = self.estimator_.predict_proba(X)
        return smooth_predictions(proba, sigma=self.sigma)

    def decision_function(self, X):
        proba = self.predict_proba(X)
        if proba.ndim == 1 or proba.shape[1] == 1:
            return proba.ravel()
        if proba.shape[1] == 2:
            return proba[:, 1]
        return proba

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]
