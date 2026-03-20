import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.base import BaseEstimator, ClassifierMixin, clone


def smooth_gauss(proba, sigma=1.0):
    """
    Gaussian smoothing for sequential data.
    """
    smoothed = gaussian_filter1d(proba, sigma=sigma, axis=0)
    return np.clip(smoothed, 0, 1)


class SmoothedProbaClassifier(ClassifierMixin, BaseEstimator):
    _estimator_type = "classifier"

    def __init__(self, base_estimator, sigma=1.0):
        self.base_estimator = base_estimator
        self.sigma = sigma

    def fit(self, X, y):
        self._inner = clone(self.base_estimator)
        self._inner.fit(X, y)
        self.classes_ = self._inner.classes_
        return self

    def predict_proba(self, X):
        proba = self._inner.predict_proba(X)
        smoothed = smooth_gauss(proba, sigma=self.sigma)
        row_sums = smoothed.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return smoothed / row_sums

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]
