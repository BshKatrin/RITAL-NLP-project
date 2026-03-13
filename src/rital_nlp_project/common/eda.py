import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
import pandas as pd
from scipy.stats import contingency


def plot_class_wordclouds(classes, matrix, vocab):
    """Plot word clouds for each class in the dataset.

    Args:
        classes : Array of class labels for each document.
        matrix (scipy.sparse matrix or np.ndarray): Document-term matrix (shape: n_samples x n_features).
        vocab (np.ndarray or list): List of vocabulary terms corresponding to columns in the matrix.

    Returns:
        matplotlib.figure.Figure: The matplotlib Figure object containing the wordclouds.
    """
    fig, axes = plt.subplots(1, 2, figsize=(8, 5))
    axes = axes.flatten()

    cloud = WordCloud(background_color='white', stopwords=[], max_words=100,
                      colormap="viridis", relative_scaling=0.2, random_state=42)
    for i, target in enumerate(np.unique(classes)):
        # Extract data related to target
        mask = (classes == target)
        matrix_mask = matrix[mask, :]

        # Extract frequencies
        freqs = {word: np.log1p(int(freq)) for word, freq in zip(vocab, matrix_mask.sum(axis=0).A1)}

        # Generate wordcloud
        cloud = cloud.generate_from_frequencies(freqs)

        axes[i].imshow(cloud)
        axes[i].axis("off")
        axes[i].set_title(target)

    plt.tight_layout()
    return fig


def get_emp_zipf(matrix, vocab):
    """Compute the empirical word frequency distribution (Zipf's law) for a corpus.

    Args:
        matrix (scipy.sparse matrix or np.ndarray): Document-term matrix (shape: n_samples x n_features).
        vocab (np.ndarray or list): List of vocabulary terms corresponding to columns in the matrix.

    Returns:
        tuple: (freqs, ranks)
            freqs (np.ndarray): Normalized word frequencies sorted in descending order.
            ranks (np.ndarray): Ranks corresponding to the sorted frequencies (1-based).
    """
    freqs = np.asarray(matrix.sum(axis=0)).flatten()
    freqs = sorted(freqs, reverse=True)
    freqs = freqs / np.sum(freqs)
    ranks = np.arange(1, len(vocab) + 1)
    return freqs, ranks


def get_theo_zipf(matrix, vocab):
    """Compute the theoretical Zipf distribution for a given vocabulary size.

    Args:
        matrix (scipy.sparse matrix or np.ndarray): Document-term matrix (not used, included for API consistency).
        vocab (np.ndarray or list): List of vocabulary terms (used to determine vocabulary size).

    Returns:
        tuple: (zipf_theory, ranks)
            zipf_theory (np.ndarray): Theoretical Zipf probabilities for each rank.
            ranks (np.ndarray): Ranks (1-based) for the vocabulary size.
    """
    ranks = np.arange(1, len(vocab) + 1)
    zipf_theory = 1 / ranks
    zipf_theory = zipf_theory / np.sum(zipf_theory)
    return zipf_theory, ranks


def get_log_odds_ratio(matrix, vocab, classes, classes_values, alpha=1, percentile=5):
    """Compute smoothed log-odds ratios of terms between two classes.    

    Args:
        matrix (array-like or sparse matrix): Document-term matrix (n_docs x n_terms).
        vocab (Sequence[str]): Terms corresponding to columns of matrix.
        classes (array-like): Class label per document.
        classes_values (Sequence): Two class labels to compare (first / second).
        alpha (float): Laplace smoothing parameter (default 1).
        percentile (int): Percentile for trimming extremes (default 5).

    Note:
        log_odds (log_or) tends to +inf if a term is more associated with classes_values[0],
        and tends to -inf if a term is more associated with classes_values[1].

    Returns:
        pandas.DataFrame: Columns = ["word", "or", "log_or"], trimmed by percentile.
    """
    if len(classes_values) != 2:
        raise ValueError("classes_values must contain exactly two elements")

    v0, v1 = classes_values
    X0 = matrix[np.asarray(classes) == v0, :]
    X1 = matrix[np.asarray(classes) == v1, :]

    # Laplace-like smoothing: add alpha to term counts and alpha*V to denominators
    V = len(vocab)
    n0 = X0.shape[0] + alpha * V
    n1 = X1.shape[0] + alpha * V

    docs_0 = np.asarray((X0 > 0).sum(axis=0)).ravel() + alpha
    docs_1 = np.asarray((X1 > 0).sum(axis=0)).ravel() + alpha

    p0 = docs_0 / n0
    p1 = docs_1 / n1

    odds_ratio = (p0 / (1 - p0)) * ((1 - p1) / p1)
    log_or = np.log(odds_ratio)

    df = pd.DataFrame({"word": vocab, "or": odds_ratio, "log_or": log_or})

    low, high = np.percentile(df["log_or"], [percentile, 100 - percentile])
    return df[(df["log_or"] >= low) & (df["log_or"] <= high)]
