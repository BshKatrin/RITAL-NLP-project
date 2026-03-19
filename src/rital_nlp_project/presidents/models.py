from rital_nlp_project.presidents.models_config import cv_n_splits, test_size
from rital_nlp_project.presidents.utils import (
    SmoothedProbaClassifier,
    sequential_holdout_split,
    split_for_cv,
)


def cv_fn(X, y):
    return split_for_cv(y, n_splits=cv_n_splits)


def split_fn(X, y):
    return sequential_holdout_split(X, y, test_size=test_size)


__all__ = ["SmoothedProbaClassifier", "cv_fn", "split_fn"]
