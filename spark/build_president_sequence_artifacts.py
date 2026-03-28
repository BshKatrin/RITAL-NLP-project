from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

from rital_nlp_project.common.preprocess.stopwords import STOPWORDS
from rital_nlp_project.presidents.transformer import TextPreprocessor


TRAIN_LINE_RE = re.compile(r"^<(\d+):(\d+):([CM])>\s*(.*)$")
TEST_LINE_RE = re.compile(r"^<(\d+):(\d+)>\s*(.*)$")
LABEL_MAP = {"C": 1, "M": -1}


@dataclass(frozen=True)
class CorpusConfig:
    name: str
    input_path: str
    metadata_output_path: str
    embeddings_output_path: str
    has_labels: bool


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if (
        getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available()
    ):
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
        default="Dataset/raw/presidents/presidents.learn.utf8.txt",
    )
    parser.add_argument(
        "--test-input-path",
        default="Dataset/raw/test/presidents/corpus.tache1.test.utf8",
    )
    parser.add_argument(
        "--train-metadata-output-path",
        default="Dataset/clean/presidents_clean_bert.parquet",
    )
    parser.add_argument(
        "--test-metadata-output-path",
        default="Dataset/clean/presidents_test_clean_bert.parquet",
    )
    parser.add_argument(
        "--train-embeddings-output-path",
        default="Dataset/embeddings/presidents_camembert_base_mean.npy",
    )
    parser.add_argument(
        "--test-embeddings-output-path",
        default="Dataset/embeddings/presidents_test_camembert_base_mean.npy",
    )
    parser.add_argument(
        "--model-name",
        default="camembert-base",
        help="Hugging Face model used for sentence embeddings.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size used for transformer inference.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Tokenizer truncation length.",
    )
    parser.add_argument(
        "--device",
        default=default_device(),
        help="Device used for embedding inference.",
    )
    parser.add_argument(
        "--summary-output-path",
        default="Dataset/out/presidents_artifacts_build_summary.json",
    )
    return parser.parse_args()


def parse_corpus_lines(path: str | Path, *, has_labels: bool) -> pd.DataFrame:
    records = []
    line_re = TRAIN_LINE_RE if has_labels else TEST_LINE_RE

    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue

            match = line_re.match(line)
            if match is None:
                raise ValueError(
                    f"Could not parse line {line_number} in {path}: {raw_line.rstrip()}"
                )

            if has_labels:
                speech_id_str, sentence_id_str, label_char, text = match.groups()
                label = LABEL_MAP[label_char]
            else:
                speech_id_str, sentence_id_str, text = match.groups()
                label = None

            records.append(
                {
                    "speech_id": int(speech_id_str),
                    "sentence_id": int(sentence_id_str),
                    "text_raw": text,
                    "label": label,
                }
            )

    frame = pd.DataFrame(records)
    if frame.empty:
        raise ValueError(f"No rows were parsed from {path}")

    frame = frame.reset_index(drop=True)
    block_ids = frame["speech_id"].ne(frame["speech_id"].shift()).cumsum()
    blocks_per_speech = frame.groupby("speech_id", sort=False).apply(
        lambda group: int(block_ids.loc[group.index].nunique()),
        include_groups=False,
    )
    if not bool((blocks_per_speech == 1).all()):
        problematic = blocks_per_speech[blocks_per_speech > 1].to_dict()
        raise ValueError(
            f"speech_id must appear in one contiguous block in {path}, got {problematic}"
        )

    per_speech = frame.groupby("speech_id", sort=False)["sentence_id"]
    is_monotonic = per_speech.apply(lambda values: values.is_monotonic_increasing)
    if not bool(is_monotonic.all()):
        raise ValueError(f"sentence_id is not monotonic within speech_id in {path}")

    if not has_labels:
        frame = frame.drop(columns=["label"])

    return frame


def clean_for_bert(frame: pd.DataFrame) -> pd.DataFrame:
    preprocessor = TextPreprocessor(
        stem=False,
        lemmatize=True,
        stopwords=STOPWORDS["french"],
        pipeline_mode="bert",
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
    return cleaned[columns]


def mean_pool_hidden_states(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
    pooled = (hidden_states * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1.0)
    return pooled / counts


def encode_texts(
    texts: list[str],
    *,
    tokenizer: AutoTokenizer,
    model: AutoModel,
    batch_size: int,
    max_length: int,
    device: torch.device,
) -> np.ndarray:
    all_embeddings: list[np.ndarray] = []
    model.eval()

    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            end = min(start + batch_size, len(texts))
            batch_texts = texts[start:end]
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
            print(
                json.dumps(
                    {
                        "batch_start": start,
                        "batch_end": end,
                        "num_rows": len(batch_texts),
                    }
                ),
                flush=True,
            )

    return np.concatenate(all_embeddings, axis=0)


def save_corpus_artifacts(
    corpus_config: CorpusConfig,
    *,
    tokenizer: AutoTokenizer,
    model: AutoModel,
    batch_size: int,
    max_length: int,
    device: torch.device,
) -> dict[str, object]:
    start = time.perf_counter()
    parsed = parse_corpus_lines(
        corpus_config.input_path,
        has_labels=corpus_config.has_labels,
    )
    cleaned = clean_for_bert(parsed)
    embeddings = encode_texts(
        cleaned["text"].tolist(),
        tokenizer=tokenizer,
        model=model,
        batch_size=batch_size,
        max_length=max_length,
        device=device,
    )

    metadata_output_path = Path(corpus_config.metadata_output_path)
    embeddings_output_path = Path(corpus_config.embeddings_output_path)
    metadata_output_path.parent.mkdir(parents=True, exist_ok=True)
    embeddings_output_path.parent.mkdir(parents=True, exist_ok=True)

    cleaned.to_parquet(metadata_output_path, index=False)
    np.save(embeddings_output_path, embeddings)

    summary: dict[str, object] = {
        "name": corpus_config.name,
        "input_path": str(Path(corpus_config.input_path).resolve()),
        "metadata_output_path": str(metadata_output_path.resolve()),
        "embeddings_output_path": str(embeddings_output_path.resolve()),
        "rows": int(len(cleaned)),
        "speeches": int(cleaned["speech_id"].nunique()),
        "embedding_dim": int(embeddings.shape[1]),
        "seconds": time.perf_counter() - start,
    }
    if corpus_config.has_labels:
        summary["label_counts"] = {
            str(int(label)): int(count)
            for label, count in cleaned["label"].value_counts().sort_index().items()
        }
    return summary


def main() -> None:
    args = parse_args()
    started_at = datetime.now().astimezone().isoformat()
    run_start = time.perf_counter()

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name).to(device)

    corpora = [
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

    summaries = [
        save_corpus_artifacts(
            corpus_config,
            tokenizer=tokenizer,
            model=model,
            batch_size=args.batch_size,
            max_length=args.max_length,
            device=device,
        )
        for corpus_config in corpora
    ]

    payload = {
        "started_at": started_at,
        "ended_at": datetime.now().astimezone().isoformat(),
        "total_seconds": time.perf_counter() - run_start,
        "device": str(device),
        "model_name": args.model_name,
        "batch_size": int(args.batch_size),
        "max_length": int(args.max_length),
        "corpora": summaries,
        "config": {
            key: value
            for key, value in vars(args).items()
        },
    }

    summary_output_path = Path(args.summary_output_path)
    summary_output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_output_path.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
