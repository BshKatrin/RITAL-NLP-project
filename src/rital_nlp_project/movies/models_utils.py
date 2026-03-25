from sklearn.model_selection import StratifiedKFold, train_test_split
from rital_nlp_project.movies.models_config import *


def cv_fn(X, y):
    return StratifiedKFold(n_splits=cv_n_splits, shuffle=True, random_state=random_state)


def split_fn(X, y):
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state, shuffle=True)
