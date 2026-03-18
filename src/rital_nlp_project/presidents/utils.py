import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import PredefinedSplit


def concat_seq_by_class(texts, classes, n=None):
    """
    Concatenate at most n consecutive texts of the same class into one.

    Args:
        texts (list of str): List of text strings.
        classes (list of int): List of class labels (same length as texts).
        n (int): Maximum number of consecutive phrases to combine (must be >= 1).
        If None, the limit is not set.

    Returns:
        tuple: (new_texts, new_classes)
            new_texts (list of str): Concatenated texts.
            new_classes (list of int): Corresponding class labels.
    """

    # new_texts = [texts[0]]
    # new_classes = [classes[0]]
    # for t, c in zip(texts[1:], classes[1:]):
    #     if c == new_classes[-1]:
    #         new_texts[-1] += " " + t
    #     else:
    #         new_texts.append(t)
    #         new_classes.append(c)
    # return new_texts, new_classes
    if not texts:
        return [], []
    if len(texts) != len(classes):
        raise ValueError("texts and classes must have the same length")
    if n and n < 1:
        raise ValueError("n must be >= 1")

    max_block = n
    new_texts = []
    new_classes = []
    i = 0
    L = len(texts)
    while i < L:
        curr_class = classes[i]
        # find end of run of the same class
        j = i + 1
        while j < L and classes[j] == curr_class:
            j += 1
        # split the run [i, j) into blocks of size <= max_block
        start = i
        while start < j:
            end = min(start + max_block, j) if max_block else j
            new_texts.append(" ".join(texts[start:end]))
            new_classes.append(curr_class)
            start = end
        i = j
    return new_texts, new_classes


def smooth_predictions(proba, sigma=1.0):
    """
    Gaussian smoothing for sequential data.
    """
    smoothed = gaussian_filter1d(proba, sigma=sigma, axis=0)
    return np.clip(smoothed, 0, 1)


def split_for_cv(classes, n_splits):
    unique_speakers = np.unique(classes)
    fold_ids = np.zeros(len(classes), dtype=int)

    for s in unique_speakers:
        idx = np.where(classes == s)[0]
        blocks = np.array_split(idx, n_splits)
        for k, b in enumerate(blocks):
            fold_ids[b] = k

    # to iterate (cv) : for train_idx, test_idx in .split()
    # for train/test : train_idx, test = next(.split())
    return PredefinedSplit(test_fold=fold_ids)


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
