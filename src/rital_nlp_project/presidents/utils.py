from scipy.ndimage import gaussian_filter1d
import numpy as np


def concat_seq_by_class(texts, classes):
    """
    Concatenate texts if the same class is sequential in the list of classes.

    Args:
        texts (list of str): List of text strings.
        classes (list of int): List of class labels (same length as texts).

    Returns:
        tuple: (new_texts, new_classes)
            new_texts (list of str): Concatenated texts.
            new_classes (list of int): Corresponding class labels.
    """

    new_texts = [texts[0]]
    new_classes = [classes[0]]
    for t, c in zip(texts[1:], classes[1:]):
        if c == new_classes[-1]:
            new_texts[-1] += " " + t
        else:
            new_texts.append(t)
            new_classes.append(c)
    return new_texts, new_classes


def smooth_predictions(proba, sigma=1.0):
    """
    Gaussian smoothing for sequential data.
    """
    smoothed = gaussian_filter1d(proba, sigma=sigma, axis=0)
    return np.clip(smoothed, 0, 1)
