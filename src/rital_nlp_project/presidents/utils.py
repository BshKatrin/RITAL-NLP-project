import pandas as pd
import re
import numpy as np


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


def get_seq_len(classes):
    mask = np.where(np.diff(classes) != 0)[0]
    unique_class = classes[np.r_[0, mask + 1]]
    lengths = np.diff(np.r_[0, mask + 1, len(classes)])
    return unique_class, lengths


def load_with_numbers(input_path, output_path=None):
    data = []

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            match = re.match(r"<(\d+):\d+(?::[^>]+)?>\s*(.*)", line)

            if match:
                doc_id = int(match.group(1))   # first number
                text = match.group(2)          # the text after >

                data.append((doc_id, text))

    df = pd.DataFrame(data, columns=["label", "text"])
    if output_path is None:
        return df

    df.to_parquet(output_path, index=False)


if __name__ == "__main__":
    load_with_numbers("Dataset/test/corpus.tache1.test.utf8", "Dataset/clean/presidents_test.parquet")
