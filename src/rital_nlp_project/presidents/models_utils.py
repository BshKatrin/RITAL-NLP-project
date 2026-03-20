import numpy as np
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
    # Temporal cutoff
    split_idx = int(len(X) * (1-test_size))

    X_train = X[:split_idx]
    X_test = X[split_idx:]
    y_train = y[:split_idx]
    y_test = y[split_idx:]
    return X_train, X_test, y_train, y_test
