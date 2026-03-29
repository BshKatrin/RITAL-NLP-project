from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

from rital_nlp_project.common.preprocess.stopwords import STOPWORDS
from rital_nlp_project.presidents.transformer import TextPreprocessor


TRAIN_INPUT_PATH = "Dataset/raw/presidents/presidents.learn.utf8.txt"
TEST_INPUT_PATH = "Dataset/raw/test/presidents/corpus.tache1.test.utf8"
TRAIN_METADATA_OUTPUT_PATH = "Dataset/clean/presidents_clean_bert.parquet"
TEST_METADATA_OUTPUT_PATH = "Dataset/clean/presidents_test_clean_bert.parquet"
TRAIN_EMBEDDINGS_OUTPUT_PATH = "Dataset/embeddings/presidents_camembert_base_mean.npy"
TEST_EMBEDDINGS_OUTPUT_PATH = "Dataset/embeddings/presidents_test_camembert_base_mean.npy"
SUMMARY_OUTPUT_PATH = "Dataset/out/presidents_artifacts_build_summary.json"
DEFAULT_MODEL_NAME = "camembert-base"
DEFAULT_BATCH_SIZE = 32
DEFAULT_MAX_LENGTH = 256
PRESIDENTS_LANGUAGE = "french"
PREPROCESSING_MODE = "bert"

TRAIN_LINE_RE = re.compile(r"^<(\d+):(\d+):([CM])>\s*(.*)$")
TEST_LINE_RE = re.compile(r"^<(\d+):(\d+)>\s*(.*)$")
LABEL_MAP = {"C": 1, "M": -1}


@dataclass(frozen=True)
class CorpusConfig:
    name: str
    input_path: str | Path
    metadata_output_path: str | Path
    embeddings_output_path: str | Path
    has_labels: bool


def default_device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build presidents sentence-level BERT metadata parquet files and "
            "CamemBERT mean-pooled embedding matrices for the sequence models."
        )
    )
    parser.add_argument(
        "--train-input-path",
        default=TRAIN_INPUT_PATH,
    )
    parser.add_argument(
        "--test-input-path",
        default=TEST_INPUT_PATH,
    )
    parser.add_argument(
        "--train-metadata-output-path",
        default=TRAIN_METADATA_OUTPUT_PATH,
    )
    parser.add_argument(
        "--test-metadata-output-path",
        default=TEST_METADATA_OUTPUT_PATH,
    )
    parser.add_argument(
        "--train-embeddings-output-path",
        default=TRAIN_EMBEDDINGS_OUTPUT_PATH,
    )
    parser.add_argument(
        "--test-embeddings-output-path",
        default=TEST_EMBEDDINGS_OUTPUT_PATH,
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="Hugging Face model used for sentence embeddings.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size used for transformer inference.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=DEFAULT_MAX_LENGTH,
        help="Tokenizer truncation length.",
    )
    parser.add_argument(
        "--device",
        default=default_device(),
        help="Device used for embedding inference.",
    )
    parser.add_argument(
        "--summary-output-path",
        default=SUMMARY_OUTPUT_PATH,
    )
    return parser.parse_args()


def parse_corpus_lines(path: str | Path, has_labels: bool) -> pd.DataFrame:
    path = Path(path)
    rows: list[dict[str, object]] = []
    line_re = TRAIN_LINE_RE if has_labels else TEST_LINE_RE

    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue

            match = line_re.match(line)
            if match is None:
                raise ValueError(f"Could not parse line {line_number} in {path}")

            if has_labels:
                speech_id_str, sentence_id_str, label_char, text = match.groups()
                rows.append(
                    {
                        "speech_id": int(speech_id_str),
                        "sentence_id": int(sentence_id_str),
                        "text_raw": text,
                        "label": LABEL_MAP[label_char],
                    }
                )
            else:
                speech_id_str, sentence_id_str, text = match.groups()
                rows.append(
                    {
                        "speech_id": int(speech_id_str),
                        "sentence_id": int(sentence_id_str),
                        "text_raw": text,
                    }
                )

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError(f"No rows were parsed from {path}")

    block_ids = (frame["speech_id"] != frame["speech_id"].shift()).cumsum()
    blocks_per_speech = frame.assign(block_id=block_ids).groupby("speech_id", sort=False)[
        "block_id"
    ].nunique()
    problematic = blocks_per_speech[blocks_per_speech > 1]
    if not problematic.empty:
        raise ValueError(
            f"speech_id must appear in one contiguous block in {path}, "
            f"got {problematic.to_dict()}"
        )

    per_speech = frame.groupby("speech_id", sort=False)["sentence_id"].apply(
        lambda values: values.is_monotonic_increasing
    )
    if not bool(per_speech.all()):
        raise ValueError(f"sentence_id is not monotonic within speech_id in {path}")

    return frame.reset_index(drop=True)


def clean_for_bert(frame: pd.DataFrame) -> pd.DataFrame:
    preprocessor = TextPreprocessor(
        stem=False,
        lemmatize=True,
        stopwords=STOPWORDS[PRESIDENTS_LANGUAGE],
        pipeline_mode=PREPROCESSING_MODE,
    )
    cleaned_texts = preprocessor.fit(frame["text_raw"].tolist()).transform(
        frame["text_raw"].tolist()
    )

    cleaned = frame.copy()
    cleaned["text"] = cleaned_texts

    columns = ["speech_id", "sentence_id", "text"]
    if "label" in cleaned.columns:
        columns.append("label")
    columns.append("text_raw")
    return cleaned[columns].copy()


def mean_pool_hidden_states(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    # The BiLSTM consumes one fixed vector per sentence, so we average token
    # states over the non-padding positions to get one sentence embedding.
    mask = attention_mask.unsqueeze(-1).to(dtype=hidden_states.dtype)
    pooled = (hidden_states * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1.0)
    return pooled / counts


def build_corpus_configs(args: argparse.Namespace) -> list[CorpusConfig]:
    return [
        CorpusConfig(
            name="train",
            input_path=args.train_input_path,
            metadata_output_path=args.train_metadata_output_path,
            embeddings_output_path=args.train_embeddings_output_path,
            has_labels=True,
        ),
        CorpusConfig(
            name="test",
            input_path=args.test_input_path,
            metadata_output_path=args.test_metadata_output_path,
            embeddings_output_path=args.test_embeddings_output_path,
            has_labels=False,
        ),
    ]


def encode_texts(
    texts: list[str],
    tokenizer,
    model,
    batch_size: int,
    max_length: int,
    device: torch.device,
) -> np.ndarray:
    all_embeddings: list[np.ndarray] = []
    model.eval()

    with torch.no_grad():
        for batch_start in range(0, len(texts), batch_size):
            batch_end = min(batch_start + batch_size, len(texts))
            batch_texts = texts[batch_start:batch_end]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {
                key: value.to(device)
                for key, value in encoded.items()
            }
            outputs = model(**encoded)
            pooled = mean_pool_hidden_states(
                outputs.last_hidden_state,
                encoded["attention_mask"],
            )
            all_embeddings.append(pooled.cpu().numpy().astype(np.float32, copy=False))

    return np.concatenate(all_embeddings, axis=0)


def save_corpus_artifacts(
    corpus_config: CorpusConfig,
    tokenizer,
    model,
    *,
    batch_size: int,
    max_length: int,
    device: torch.device,
) -> dict[str, object]:
    started = time.perf_counter()
    parsed = parse_corpus_lines(corpus_config.input_path, corpus_config.has_labels)
    metadata = clean_for_bert(parsed)
    embeddings = encode_texts(
        metadata["text"].tolist(),
        tokenizer,
        model,
        batch_size,
        max_length,
        device,
    )

    metadata_output_path = Path(corpus_config.metadata_output_path)
    embeddings_output_path = Path(corpus_config.embeddings_output_path)
    metadata_output_path.parent.mkdir(parents=True, exist_ok=True)
    embeddings_output_path.parent.mkdir(parents=True, exist_ok=True)

    metadata.to_parquet(metadata_output_path, index=False)
    np.save(embeddings_output_path, embeddings)

    summary: dict[str, object] = {
        "name": corpus_config.name,
        "input_path": str(Path(corpus_config.input_path).resolve()),
        "metadata_output_path": str(metadata_output_path.resolve()),
        "embeddings_output_path": str(embeddings_output_path.resolve()),
        "rows": int(len(metadata)),
        "speeches": int(metadata["speech_id"].nunique()),
        "embedding_dim": int(embeddings.shape[1]),
        "seconds": time.perf_counter() - started,
    }
    if "label" in metadata.columns:
        summary["label_counts"] = (
            metadata["label"].value_counts().sort_index().to_dict()
        )
    return summary


def main():
    args = parse_args()
    started_at = datetime.now().astimezone().isoformat()
    run_start = time.perf_counter()

    model_name = str(args.model_name)
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)

    summaries = [
        save_corpus_artifacts(
            corpus_config,
            tokenizer,
            model,
            batch_size=int(args.batch_size),
            max_length=int(args.max_length),
            device=device,
        )
        for corpus_config in build_corpus_configs(args)
    ]

    summary_output_path = Path(args.summary_output_path)
    summary_output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "started_at": started_at,
        "ended_at": datetime.now().astimezone().isoformat(),
        "total_seconds": time.perf_counter() - run_start,
        "model_name": model_name,
        "config": vars(args),
        "corpora": summaries,
    }
    summary_output_path.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
