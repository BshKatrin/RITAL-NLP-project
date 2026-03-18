import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import PredefinedSplit

from rital_nlp_project.presidents.models_config import *


def cv_fn(X, y):
    unique_speakers = np.unique(y)
    fold_ids = np.zeros(len(y), dtype=int)

    for s in unique_speakers:
        idx = np.where(y == s)[0]
        blocks = np.array_split(idx, cv_n_splits)
        for k, b in enumerate(blocks):
            fold_ids[b] = k

    # to iterate (cv) : for train_idx, test_idx in .split()
    # for train/test : train_idx, test = next(.split())
    return PredefinedSplit(test_fold=fold_ids)


def split_fn(X, y):
    split_idx = int(len(X) * (1-test_size))

    X_train = X[:split_idx]
    X_test = X[split_idx:]
    y_train = y[:split_idx]
    y_test = y[split_idx:]
    return X_train, X_test, y_train, y_test


def smooth_predictions(proba, sigma=1.0):
    """
    Gaussian smoothing for sequential data.
    """
    smoothed = gaussian_filter1d(proba, sigma=sigma, axis=0)
    return np.clip(smoothed, 0, 1)


class SmoothedProbaClassifier(BaseEstimator, ClassifierMixin):
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
        smoothed = smooth_predictions(proba, sigma=self.sigma)
        row_sums = smoothed.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return smoothed / row_sums

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def decision_function(self, X):
        if not hasattr(self.estimator_, "decision_function"):
            raise AttributeError("base_estimator has no decision_function")
        return self.estimator_.decision_function(X)
