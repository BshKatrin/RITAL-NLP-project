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


def concat_neighbours(X, n):
    n_rows, n_cols = len(X) - 2*n, X.shape[1] * (2 * n + 1)
    X_exp = np.zeros(shape=(n_rows, n_cols))

    # print(len(X), n_rows, len(X)-n-2)
    for i in range(n, len(X)-n-2):
        tmp = np.array([X[i-j] for j in range(-n, n+1)])
        X_exp[i] = np.concatenate(tmp)

    return X_exp


def add_extra_features(X):
    n_rows, n_cols = X.shape
    extra = np.zeros(shape=(n_rows-2, n_cols * 2 + 2))
    for i in range(1, len(X)-2):
        past = X[i-1]
        futur = X[i+1]
        pres = X[i]

        # Cosine distance between pres and past phrase
        cos_dist_past = np.dot(past, pres) / np.linalg.norm(past) / np.linalg.norm(pres)

        # Cosine distance between pres and futur phrase
        cos_dist_futur = np.dot(futur, pres) / np.linalg.norm(futur) / np.linalg.norm(pres)

        diff_past = pres - pres
        diff_futur = futur - pres

        extra[i] = np.concatenate([[cos_dist_past], [cos_dist_futur], diff_past, diff_futur])

    return extra


def expand_train_test(X_train, X_test, y_train, y_test=None, n=1, add_extra=True):
    X_train_concat = concat_neighbours(X_train, n)
    X_test_concat = concat_neighbours(X_test, n)
    y_train_concat = y_train[n:-n]

    if add_extra:
        X_train_extra = add_extra_features(X_train)
        X_test_extra = add_extra_features(X_test)
        X_train_concat = np.concatenate([X_train_concat, X_train_extra], axis=1)
        X_test_concat = np.concatenate([X_test_concat, X_test_extra], axis=1)

    if y_test is not None:
        y_test_concat = y_test[n:-n]
        return X_train_concat, X_test_concat, y_train_concat, y_test_concat

    return X_train_concat, X_test_concat, y_train_concat


def smooth_pred(pred, passes=1, n=1):
    pred_smooth = pred.copy()
    for _ in range(passes):
        for i in range(n, len(pred_smooth)-n-1):
            past = pred_smooth[i-n]
            futur = pred_smooth[i+n]
            if past == futur:
                for j in range(i-n, i+n):
                    if j < len(pred_smooth):
                        pred_smooth[j] = past
    return pred_smooth


def get_scores_smooth_duration(log_probs, D_max: list, kde: list):
    """Compute duration-aware scores used for probability smoothing.

    The idea is to calculate a score of how well the last d observations
    support y, given how long each president's speech lasts,
    by summing over all possible d to take different possible durations into account.

    For each time 't' and class 'y', the score S[t, y] aggregates segment scores
    over candidate durations 'd in [1, D_max[y]]':

    - segment score: mean(log_probs[t-d+1:t+1, y])
    - duration weight: P(d|y), obtained from kde

    Args:
    - log_probs[i, y] = log P(y | x_i)
    - D_max[y] = maximum duration for class y
    - kde[y] = duration density estimator for class y
    """

    T, Y = log_probs.shape

    D = [np.arange(1, d+1) for d in D_max]
    # kde = [kde_0, kde_1]

    S = np.zeros((T, Y))
    for t in range(T):
        print(t)
        for y in range(Y):
            score = 0.0
            for d in D[y]:
                if t - d + 1 < 0:
                    continue

                window = log_probs[t-d+1:t+1, y]
                seg_score = window.mean()

                log_pd = kde[y].score_samples([[d]])[0]
                weight = np.exp(log_pd)

                score += weight * seg_score

            S[t, y] = score
    return S


def smooth_prob_duration(probs, alpha, S):
    """Apply (weighted) smoothing to probabilities"""
    # alpha = 0.5
    S_final = alpha * probs + (1-alpha) * S

    # Trick to avoid overflow
    P_smoothed = np.exp(S_final - S_final.max(axis=1, keepdims=True))
    P_smoothed /= P_smoothed.sum(axis=1, keepdims=True)
    return P_smoothed


def predict_y_ext(probs, X_ext, n):
    """Predict edge cases (when phrase has no previous or next phrase)"""
    pred = np.argmax(probs,)
    return np.pad(pred, pad_width=n, mode='edge')


def expand_train_test(X_train, X_test, y_train, y_test=None, n=1, add_extra=True):
    X_train_concat = concat_neighbours(X_train, n)
    X_test_concat = concat_neighbours(X_test, n)
    y_train_concat = y_train[n:-n]

    if add_extra:
        X_train_extra = add_extra_features(X_train)
        X_test_extra = add_extra_features(X_test)
        X_train_concat = np.concatenate([X_train_concat, X_train_extra], axis=1)
        X_test_concat = np.concatenate([X_test_concat, X_test_extra], axis=1)

    if y_test is not None:
        y_test_concat = y_test[n:-n]
        return X_train_concat, X_test_concat, y_train_concat, y_test_concat

    return X_train_concat, X_test_concat, y_train_concat


def get_seq_len(classes):
    unique_class = classes[np.r_[0, np.where(np.diff(classes) != 0)[0] + 1]]
    lengths = np.diff(np.r_[0, np.where(np.diff(classes) != 0)[0] + 1, len(classes)])
    return unique_class, lengths
