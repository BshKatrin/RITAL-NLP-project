import os
import re
import codecs
from pathlib import Path
from typing import Callable

import pandas as pd

PRESIDENTS_TRAIN_PATTERN = re.compile(r"<([0-9]+):([0-9]+):([MC])>(.*)")
PRESIDENTS_TEST_PATTERN = re.compile(r"<([0-9]+):([0-9]+)>\s?(.*)")


def load_pres_dataframe(fname: str, *, labeled: bool = True) -> pd.DataFrame:
    rows = []
    pattern = PRESIDENTS_TRAIN_PATTERN if labeled else PRESIDENTS_TEST_PATTERN
    with codecs.open(fname, "r", "utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if len(line) < 3:
                continue
            text = line.rstrip("\n")
            match = pattern.fullmatch(text)
            if match is None:
                raise ValueError(f"Could not parse presidents line {line_number}: {text[:120]!r}")

            if labeled:
                speech_id, sentence_id, label, content = match.groups()
                rows.append(
                    {
                        "speech_id": int(speech_id),
                        "sentence_id": int(sentence_id),
                        "text": content,
                        "label": -1 if label == "M" else 1,
                    }
                )
            else:
                speech_id, sentence_id, content = match.groups()
                rows.append(
                    {
                        "speech_id": int(speech_id),
                        "sentence_id": int(sentence_id),
                        "text": content,
                    }
                )

    return pd.DataFrame(rows)


def load_movies_dataframe(path2data: str, *, labeled: bool = True) -> pd.DataFrame:
    rows = []
    if labeled:
        path = Path(path2data)
        class_dirs = sorted(child for child in path.iterdir() if child.is_dir())
        review_id = 0
        for label, class_dir in enumerate(class_dirs):
            for file_path in sorted(class_dir.iterdir()):
                if not file_path.is_file() or file_path.name.startswith("."):
                    continue
                rows.append(
                    {
                        "review_id": review_id,
                        "text": file_path.read_text(encoding="utf-8"),
                        "label": label,
                    }
                )
                review_id += 1
        return pd.DataFrame(rows)

    path = Path(path2data)
    lines = path.read_text(encoding="utf-8").splitlines()
    for review_id, text in enumerate(lines):
        rows.append({"review_id": review_id, "text": text})
    return pd.DataFrame(rows)


def load_pres(fname):
    """Load data (on presidents)"""
    df = load_pres_dataframe(fname, labeled=True)
    return df["text"].tolist(), df["label"].tolist()


def load_movies(path2data):  # 1 classe par répertoire
    df = load_movies_dataframe(path2data, labeled=True)
    return df["text"].tolist(), df["label"].tolist()


def clean_dataset(load_func: Callable, data_path: str, out_path: str, preprocessor):
    texts, classes = load_func(data_path)
    text_preprocessor = preprocessor.fit(texts)
    texts_cleaned = text_preprocessor.transform(texts)
    df = pd.DataFrame({"text": texts_cleaned, "label": classes})
    output_path = Path(out_path).with_suffix(".parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)


def clean_dataframe(df: pd.DataFrame, out_path: str, preprocessor, *, text_col: str = "text"):
    texts = df[text_col].tolist()
    text_preprocessor = preprocessor.fit(texts)
    cleaned = df.copy()
    cleaned[text_col] = text_preprocessor.transform(texts)
    output_path = Path(out_path).with_suffix(".parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_parquet(output_path, index=False)


def load_dataframe(path):
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported dataframe format: {path.suffix}")


def load_clean_data(path):
    df = load_dataframe(path)
    return df["text"].tolist(), df["label"].to_numpy()


def load_texts(path, text_col: str = "text"):
    path = Path(path)
    if path.suffix == ".txt":
        return [line.rstrip("\n") for line in path.read_text().splitlines()]
    df = load_dataframe(path)
    return df[text_col].tolist()


def save_predictions(path, predictions, *, ids=None, label_col: str = "label"):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({label_col: predictions})
    if ids is not None:
        df.insert(0, "id", ids)
    save_dataframe(output_path, df)


def save_prediction_lines(path, values, *, formatter=str):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(f"{formatter(value)}\n")


def save_dataframe(path, df: pd.DataFrame):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix == ".parquet":
        df.to_parquet(output_path, index=False)
    else:
        df.to_csv(output_path, index=False)
