from scipy.ndimage import gaussian_filter1d
import numpy as np


def concat_seq_by_class(texts, classes, n=20):
    """
    Concatenate at most n consecutive texts of the same class into one.
    If n > 20, runs will still be split every 20 items (i.e. effective max = min(n, 20)).

    Args:
        texts (list of str): List of text strings.
        classes (list of int): List of class labels (same length as texts).
        n (int): Maximum number of consecutive phrases to combine (must be >= 1).

    Returns:
        tuple: (new_texts, new_classes)
            new_texts (list of str): Concatenated texts.
            new_classes (list of int): Corresponding class labels.
    """
    if not texts:
        return [], []
    if len(texts) != len(classes):
        raise ValueError("texts and classes must have the same length")
    if n < 1:
        raise ValueError("n must be >= 1")

    max_block = min(n, 20)
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
            end = min(start + max_block, j)
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
