from __future__ import annotations

import html
import json
import math
import re
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from unidecode import unidecode


PRESIDENT_NAME_TO_LABEL = {
    "Jacques Chirac": "C",
    "Francois Mitterrand": "M",
}
PRESIDENT_SLUG_TO_NAME = {
    "jacques-chirac": "Jacques Chirac",
    "francois-mitterrand": "Francois Mitterrand",
}
PRESIDENT_LABEL_TO_NAME = {
    "C": "Jacques Chirac",
    "M": "Francois Mitterrand",
}
TRAIN_LABEL_TO_SUBMISSION = {
    1: "C",
    -1: "M",
}
STOPWORDS = {
    "a",
    "ai",
    "aie",
    "aient",
    "ainsi",
    "alors",
    "au",
    "aucun",
    "aucune",
    "aussi",
    "autre",
    "autres",
    "aux",
    "avec",
    "avoir",
    "bien",
    "car",
    "ce",
    "cela",
    "celle",
    "celles",
    "celui",
    "cependant",
    "ces",
    "cet",
    "cette",
    "ceux",
    "chaque",
    "chez",
    "comme",
    "comment",
    "d",
    "dans",
    "de",
    "des",
    "du",
    "elle",
    "elles",
    "en",
    "encore",
    "entre",
    "est",
    "et",
    "etre",
    "eux",
    "fait",
    "font",
    "grace",
    "ici",
    "il",
    "ils",
    "j",
    "je",
    "l",
    "la",
    "le",
    "les",
    "leur",
    "leurs",
    "lors",
    "lui",
    "m",
    "ma",
    "mais",
    "me",
    "mes",
    "moi",
    "mon",
    "monsieur",
    "mme",
    "madame",
    "mesdames",
    "messieurs",
    "n",
    "ne",
    "ni",
    "non",
    "nos",
    "notre",
    "nous",
    "on",
    "ont",
    "ou",
    "par",
    "pas",
    "plus",
    "pour",
    "qu",
    "que",
    "quel",
    "quelle",
    "quelles",
    "quels",
    "qui",
    "s",
    "sa",
    "sans",
    "se",
    "ses",
    "si",
    "son",
    "sont",
    "sur",
    "t",
    "te",
    "tes",
    "toi",
    "ton",
    "tous",
    "tout",
    "toute",
    "toutes",
    "un",
    "une",
    "vos",
    "votre",
    "vous",
    "y",
}
PLACEHOLDER_RE = re.compile(r"<\s*(?:nom|date)\s*>", flags=re.IGNORECASE)
TAG_RE = re.compile(r"<[^>]+>")
TOKEN_RE = re.compile(r"[a-z0-9]+")
ROW_RE = re.compile(r"^<(?P<speech_id>\d+):(?P<sentence_id>\d+)(?::(?P<label>[A-Z]))?>\s*(?P<text>.*)$")
TITLE_RE = re.compile(r"<h1[^>]*>(.*?)</h1>", flags=re.IGNORECASE | re.DOTALL)
META_RE_TEMPLATE = r'<meta\s+property="{property_name}"\s+content="(.*?)"'
ELYSEE_BODY_RE = re.compile(
    r'<div\s+dir="auto"\s+class="card__body ck-styled">(.*?)</div>',
    flags=re.IGNORECASE | re.DOTALL,
)
VP_BODY_RE = re.compile(
    r'<div\s+class="field field--name-field-texte-integral .*? field__item">(.*?)</div>',
    flags=re.IGNORECASE | re.DOTALL,
)
VP_AUTHOR_RE = re.compile(
    r'<a href="/auteur/[^"]+"[^>]*>(.*?)</a>',
    flags=re.IGNORECASE | re.DOTALL,
)
ELYSEE_ID_RE = re.compile(
    r"EA_datalayer\.push\('content_id','(\d+)'\);",
    flags=re.IGNORECASE,
)
BR_RE = re.compile(r"(?i)<br\s*/?>")
BLOCK_BREAK_RE = re.compile(r"(?i)</p>|</div>")
LINE_BREAK_RE = re.compile(r"\n+")
SPLIT_RE = re.compile(r"(?:\n+|(?<=[.!?])\s+)")
WHITESPACE_RE = re.compile(r"\s+")


class ArchiveLabelingError(RuntimeError):
    pass


@dataclass(frozen=True)
class ArchiveDoc:
    doc_id: str
    president: str
    source_name: str
    source_url: str
    title: str
    date: str
    raw_text: str
    normalized_text: str
    sentences: tuple[str, ...]
    normalized_sentences: tuple[str, ...]


@dataclass(frozen=True)
class QueryBlock:
    block_id: str
    dataset_name: str
    speech_id: int
    start_row_idx: int
    end_row_idx: int
    start_sentence_id: int
    end_sentence_id: int
    row_indices: tuple[int, ...]
    query_text: str
    normalized_query_text: str
    query_retrieval_text: str
    query_tokens: tuple[str, ...]
    raw_sentences: tuple[str, ...]
    normalized_sentences: tuple[str, ...]
    sentence_tokens: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class CandidateMatch:
    block_id: str
    speech_id: int
    start_row_idx: int
    end_row_idx: int
    start_sentence_id: int
    end_sentence_id: int
    rank: int
    doc_id: str
    president: str
    matched_label: str
    source_name: str
    source_url: str
    source_title: str
    source_date: str
    retrieval_score: float
    alignment_score: float
    match_score: float
    score_margin: float
    sentence_hits: int
    sentence_hit_fraction: float
    evidence_snippet: str
    doc_window_start: int
    doc_window_end: int
    accepted: bool


@dataclass(frozen=True)
class RowMatch:
    row_idx: int
    speech_id: int
    sentence_id: int
    text: str
    matched_president: str | None
    matched_label: str | None
    source_url: str | None
    source_title: str | None
    source_date: str | None
    source_name: str | None
    match_score: float
    score_margin: float
    review_status: str
    evidence_snippet: str | None
    block_id: str | None
    model_disagreement: bool
    triage_priority: float


def html_to_text(fragment: str) -> str:
    text = BR_RE.sub("\n", fragment)
    text = BLOCK_BREAK_RE.sub("\n", text)
    text = TAG_RE.sub(" ", text)
    text = html.unescape(text)
    text = text.replace("\xa0", " ")
    return text


def clean_visible_text(text: str) -> str:
    text = html.unescape(text)
    text = text.replace("\xa0", " ")
    text = text.replace("’", "'").replace("`", "'").replace("´", "'")
    text = re.sub(r"[‐‑‒–—−]", "-", text)
    text = PLACEHOLDER_RE.sub(" ", text)
    text = re.sub(r"(?m)^\s*-\s*", "", text)
    text = LINE_BREAK_RE.sub("\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


@lru_cache(maxsize=400_000)
def normalize_text(text: str) -> str:
    text = clean_visible_text(text)
    text = unidecode(text.lower())
    text = text.replace("'", " ")
    text = re.sub(r"[^a-z0-9\n ]+", " ", text)
    text = LINE_BREAK_RE.sub("\n", text)
    text = WHITESPACE_RE.sub(" ", text)
    return text.strip()


def split_text_into_sentences(text: str) -> list[str]:
    text = clean_visible_text(text)
    parts = [part.strip(" -") for part in SPLIT_RE.split(text) if part.strip(" -")]
    return parts


@lru_cache(maxsize=400_000)
def full_tokens_from_normalized(text: str) -> tuple[str, ...]:
    return tuple(TOKEN_RE.findall(text))


@lru_cache(maxsize=400_000)
def informative_tokens_from_normalized(text: str) -> tuple[str, ...]:
    tokens = full_tokens_from_normalized(text)
    return tuple(token for token in tokens if len(token) >= 3 and token not in STOPWORDS)


def tokenize(text: str, *, informative_only: bool = False) -> list[str]:
    normalized = normalize_text(text)
    tokens = (
        informative_tokens_from_normalized(normalized)
        if informative_only
        else full_tokens_from_normalized(normalized)
    )
    return list(tokens)


def parse_hidden_test_corpus(path: str | Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle):
            stripped = line.rstrip("\n")
            if not stripped:
                continue
            match = ROW_RE.match(stripped)
            if match is None:
                raise ArchiveLabelingError(f"Cannot parse line {line_number + 1} from {path}")
            rows.append(
                {
                    "row_idx": len(rows),
                    "speech_id": int(match.group("speech_id")),
                    "sentence_id": int(match.group("sentence_id")),
                    "text": clean_visible_text(match.group("text")),
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ArchiveLabelingError(f"No rows found in {path}")
    return frame


def load_labeled_rows(path: str | Path) -> pd.DataFrame:
    frame = pd.read_parquet(path).copy()
    required = {"speech_id", "sentence_id", "text", "label"}
    missing = required - set(frame.columns)
    if missing:
        raise ArchiveLabelingError(
            f"Missing required columns in {path}: {sorted(missing)}"
        )
    frame = frame.sort_values(["speech_id", "sentence_id"], kind="stable").reset_index(drop=True)
    frame.insert(0, "row_idx", np.arange(len(frame), dtype=np.int64))
    frame["text"] = frame["text"].map(clean_visible_text)
    return frame


def load_row_frame(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".parquet":
        return load_labeled_rows(path)
    return parse_hidden_test_corpus(path)


def ensure_parent_dir(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def get_elysee_archive_urls(session: requests.Session) -> list[tuple[str, str]]:
    response = session.get("https://www.elysee.fr/sitemap.publication.xml", timeout=60)
    response.raise_for_status()
    root = ET.fromstring(response.text)
    namespace = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    urls: list[tuple[str, str]] = []
    for loc in root.findall("sm:url/sm:loc", namespace):
        url = (loc.text or "").strip()
        if not url:
            continue
        for slug, president_name in PRESIDENT_SLUG_TO_NAME.items():
            if f"/{slug}/" in url:
                urls.append((url, president_name))
                break
    if not urls:
        raise ArchiveLabelingError("No Jacques Chirac or François Mitterrand URLs found in Élysée sitemap")
    return urls


def _extract_first(pattern: re.Pattern[str], text: str) -> str | None:
    match = pattern.search(text)
    if match is None:
        return None
    return html.unescape(match.group(1)).strip()


def _extract_meta_property(html_text: str, property_name: str) -> str | None:
    pattern = re.compile(
        META_RE_TEMPLATE.format(property_name=re.escape(property_name)),
        flags=re.IGNORECASE,
    )
    return _extract_first(pattern, html_text)


def _extract_elysee_doc_id(html_text: str, source_url: str) -> str:
    match = ELYSEE_ID_RE.search(html_text)
    if match is not None:
        return f"elysee:{match.group(1)}"
    return f"elysee:{Path(urlparse(source_url).path).name}"


def parse_elysee_document(source_url: str, html_text: str, president: str) -> ArchiveDoc:
    title = _extract_first(TITLE_RE, html_text) or _extract_meta_property(html_text, "og:title")
    if title is None:
        raise ArchiveLabelingError(f"Missing title in {source_url}")
    body_html_match = ELYSEE_BODY_RE.search(html_text)
    if body_html_match is None:
        raise ArchiveLabelingError(f"Missing transcript body in {source_url}")
    body_text = clean_visible_text(html_to_text(body_html_match.group(1)))
    date_value = _extract_meta_property(html_text, "og:article:published_time") or ""
    if date_value:
        date_value = datetime.fromisoformat(date_value).date().isoformat()
    sentences = tuple(split_text_into_sentences(body_text))
    normalized_sentences = tuple(
        normalized
        for normalized in (normalize_text(sentence) for sentence in sentences)
        if normalized
    )
    return ArchiveDoc(
        doc_id=_extract_elysee_doc_id(html_text, source_url),
        president=president,
        source_name="elysee",
        source_url=source_url,
        title=clean_visible_text(title),
        date=date_value,
        raw_text=body_text,
        normalized_text=normalize_text(body_text),
        sentences=sentences,
        normalized_sentences=normalized_sentences,
    )


def parse_vie_publique_document(source_url: str, html_text: str) -> ArchiveDoc:
    title = _extract_first(TITLE_RE, html_text)
    if title is None:
        raise ArchiveLabelingError(f"Missing title in {source_url}")
    body_html_match = VP_BODY_RE.search(html_text)
    if body_html_match is None:
        raise ArchiveLabelingError(f"Missing transcript body in {source_url}")
    body_text = clean_visible_text(html_to_text(body_html_match.group(1)))
    time_match = re.search(
        r'<time[^>]*datetime="([^"]+)"[^>]*>(.*?)</time>',
        html_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    date_value = ""
    if time_match is not None:
        date_value = time_match.group(1)
        try:
            date_value = datetime.fromisoformat(date_value.replace("Z", "+00:00")).date().isoformat()
        except ValueError:
            date_value = clean_visible_text(time_match.group(2))
    author = _extract_first(VP_AUTHOR_RE, html_text)
    if author is None:
        raise ArchiveLabelingError(f"Missing intervenant in {source_url}")
    normalized_author = unidecode(clean_visible_text(author))
    if "jacques chirac" in normalized_author.lower():
        president = "Jacques Chirac"
    elif "francois mitterrand" in normalized_author.lower():
        president = "Francois Mitterrand"
    else:
        raise ArchiveLabelingError(f"Unsupported intervenant '{author}' in {source_url}")
    discourse_id_match = re.search(r"/discours/(\d+)", source_url)
    if discourse_id_match is None:
        raise ArchiveLabelingError(f"Missing discourse id in {source_url}")
    sentences = tuple(split_text_into_sentences(body_text))
    normalized_sentences = tuple(
        normalized
        for normalized in (normalize_text(sentence) for sentence in sentences)
        if normalized
    )
    return ArchiveDoc(
        doc_id=f"vie-publique:{discourse_id_match.group(1)}",
        president=president,
        source_name="vie-publique",
        source_url=source_url,
        title=clean_visible_text(title),
        date=date_value,
        raw_text=body_text,
        normalized_text=normalize_text(body_text),
        sentences=sentences,
        normalized_sentences=normalized_sentences,
    )


def archive_doc_to_record(doc: ArchiveDoc) -> dict[str, object]:
    return {
        "doc_id": doc.doc_id,
        "president": doc.president,
        "matched_label": PRESIDENT_NAME_TO_LABEL[doc.president],
        "source_name": doc.source_name,
        "source_url": doc.source_url,
        "title": doc.title,
        "date": doc.date,
        "raw_text": doc.raw_text,
        "normalized_text": doc.normalized_text,
        "sentences_json": json.dumps(doc.sentences, ensure_ascii=True),
        "normalized_sentences_json": json.dumps(doc.normalized_sentences, ensure_ascii=True),
    }


def load_archive_docs(path: str | Path) -> list[ArchiveDoc]:
    path = Path(path)
    if path.suffix == ".parquet":
        frame = pd.read_parquet(path)
    else:
        frame = pd.read_csv(path)
    docs: list[ArchiveDoc] = []
    for row in frame.to_dict(orient="records"):
        docs.append(
            ArchiveDoc(
                doc_id=str(row["doc_id"]),
                president=str(row["president"]),
                source_name=str(row["source_name"]),
                source_url=str(row["source_url"]),
                title=str(row["title"]),
                date=str(row["date"]),
                raw_text=str(row["raw_text"]),
                normalized_text=str(row["normalized_text"]),
                sentences=tuple(json.loads(row.get("sentences_json", "[]"))),
                normalized_sentences=tuple(json.loads(row.get("normalized_sentences_json", "[]"))),
            )
        )
    if not docs:
        raise ArchiveLabelingError(f"No archive docs found in {path}")
    return docs


def read_url_list(path: str | Path | None) -> list[str]:
    if path is None:
        return []
    urls: list[str] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                urls.append(stripped)
    return urls


def build_blocks(
    rows: pd.DataFrame,
    *,
    dataset_name: str,
    block_size: int = 5,
    block_step: int = 3,
    min_sentence_tokens: int = 3,
    query_sentence_limit: int = 5,
) -> list[QueryBlock]:
    blocks: list[QueryBlock] = []
    grouped = rows.groupby("speech_id", sort=False)
    for speech_id, group in grouped:
        group = group.sort_values("sentence_id", kind="stable").reset_index(drop=True)
        total = len(group)
        if total == 0:
            continue
        starts = list(range(0, total, block_step))
        if starts[-1] != max(total - block_size, 0):
            starts.append(max(total - block_size, 0))
        starts = sorted(set(starts))
        for local_start in starts:
            local_end = min(total, local_start + block_size)
            block_rows = group.iloc[local_start:local_end]
            raw_sentences = tuple(block_rows["text"].tolist())
            normalized_sentences = tuple(normalize_text(text) for text in raw_sentences)
            sentence_tokens = tuple(
                informative_tokens_from_normalized(sentence)
                for sentence in normalized_sentences
            )
            query_candidates = [
                sentence
                for sentence, tokens in zip(raw_sentences, sentence_tokens, strict=False)
                if len(tokens) >= min_sentence_tokens
            ]
            if not query_candidates:
                query_candidates = list(raw_sentences)
            query_text = " ".join(query_candidates[:query_sentence_limit]).strip()
            normalized_query_text = normalize_text(query_text)
            if not normalized_query_text:
                continue
            query_retrieval_text = build_retrieval_text_from_normalized(normalized_query_text)
            query_tokens = informative_tokens_from_normalized(normalized_query_text)
            if not query_tokens:
                query_tokens = full_tokens_from_normalized(normalized_query_text)
            start_row_idx = int(block_rows["row_idx"].iloc[0])
            end_row_idx = int(block_rows["row_idx"].iloc[-1])
            start_sentence_id = int(block_rows["sentence_id"].iloc[0])
            end_sentence_id = int(block_rows["sentence_id"].iloc[-1])
            blocks.append(
                QueryBlock(
                    block_id=f"{dataset_name}:{int(speech_id)}:{start_sentence_id}-{end_sentence_id}",
                    dataset_name=dataset_name,
                    speech_id=int(speech_id),
                    start_row_idx=start_row_idx,
                    end_row_idx=end_row_idx,
                    start_sentence_id=start_sentence_id,
                    end_sentence_id=end_sentence_id,
                    row_indices=tuple(int(v) for v in block_rows["row_idx"].tolist()),
                    query_text=query_text,
                    normalized_query_text=normalized_query_text,
                    query_retrieval_text=query_retrieval_text,
                    query_tokens=query_tokens,
                    raw_sentences=raw_sentences,
                    normalized_sentences=normalized_sentences,
                    sentence_tokens=sentence_tokens,
                )
            )
    if not blocks:
        raise ArchiveLabelingError("No query blocks generated")
    return blocks


def build_retrieval_text_from_normalized(text: str) -> str:
    informative_tokens = informative_tokens_from_normalized(text)
    if not informative_tokens:
        informative_tokens = full_tokens_from_normalized(text)
    return " ".join(informative_tokens)


def build_retrieval_text(text: str) -> str:
    return build_retrieval_text_from_normalized(normalize_text(text))


class ArchiveMatcher:
    def __init__(self, docs: Iterable[ArchiveDoc]):
        self.docs = list(docs)
        if not self.docs:
            raise ArchiveLabelingError("Archive matcher requires at least one document")
        retrieval_texts = [
            build_retrieval_text_from_normalized(doc.normalized_text) for doc in self.docs
        ]
        self.doc_sentence_tokens = [
            tuple(
                informative_tokens_from_normalized(sentence)
                for sentence in doc.normalized_sentences
            )
            for doc in self.docs
        ]
        self.doc_sentence_token_sets = [
            tuple(frozenset(sentence_tokens) for sentence_tokens in doc_sentence_tokens)
            for doc_sentence_tokens in self.doc_sentence_tokens
        ]
        self.vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
            lowercase=False,
        )
        self.doc_matrix = self.vectorizer.fit_transform(retrieval_texts)

    def retrieve_candidate_indices(self, block: QueryBlock, *, top_k: int) -> list[tuple[int, float]]:
        query_vector = self.vectorizer.transform([block.query_retrieval_text])
        similarities = linear_kernel(query_vector, self.doc_matrix).ravel()
        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [
            (int(idx), float(similarities[idx]))
            for idx in top_indices
            if float(similarities[idx]) > 0.0
        ]


def counter_overlap(left: Iterable[str], right: Iterable[str]) -> int:
    left_tokens = tuple(left)
    right_tokens = tuple(right)
    if not left_tokens or not right_tokens:
        return 0
    left_counter = Counter(left_tokens)
    right_counter = Counter(right_tokens)
    overlap = left_counter & right_counter
    return int(sum(overlap.values()))


def counter_overlap_from_counters(
    left_counter: Counter[str],
    right_counter: Counter[str],
) -> int:
    if not left_counter or not right_counter:
        return 0
    if len(left_counter) > len(right_counter):
        left_counter, right_counter = right_counter, left_counter
    overlap = 0
    for token, left_count in left_counter.items():
        right_count = right_counter.get(token, 0)
        if right_count:
            overlap += min(left_count, right_count)
    return int(overlap)


def score_token_pair(
    query_tokens: tuple[str, ...],
    candidate_tokens: tuple[str, ...],
    *,
    query_counter: Counter[str] | None = None,
) -> tuple[float, float, float]:
    if not query_tokens or not candidate_tokens:
        return 0.0, 0.0, 0.0
    candidate_counter = Counter(candidate_tokens)
    overlap = counter_overlap_from_counters(
        query_counter or Counter(query_tokens),
        candidate_counter,
    )
    precision = overlap / len(candidate_tokens)
    recall = overlap / len(query_tokens)
    if precision + recall == 0.0:
        token_f1 = 0.0
    else:
        token_f1 = 2.0 * precision * recall / (precision + recall)
    containment = overlap / min(len(query_tokens), len(candidate_tokens))
    return token_f1, recall, containment


def score_sentence_against_window(
    query_tokens: tuple[str, ...],
    window_sentence_tokens: tuple[tuple[str, ...], ...],
) -> float:
    best_score = 0.0
    if not query_tokens:
        return 0.0
    query_counter = Counter(query_tokens)
    for start in range(len(window_sentence_tokens)):
        for width in (1, 2):
            segment_tokens = tuple(
                token
                for sentence_tokens in window_sentence_tokens[start : start + width]
                for token in sentence_tokens
            )
            if not segment_tokens:
                continue
            overlap = counter_overlap_from_counters(query_counter, Counter(segment_tokens))
            containment = overlap / min(len(query_tokens), len(segment_tokens))
            best_score = max(best_score, containment)
    return best_score


def _top_anchor_indices(
    query_tokens: tuple[str, ...],
    doc_sentence_token_sets: tuple[frozenset[str], ...],
    *,
    limit: int = 4,
) -> list[int]:
    if not query_tokens:
        return []
    query_token_set = frozenset(query_tokens)
    scores = [
        (len(query_token_set & sentence_token_set), idx)
        for idx, sentence_token_set in enumerate(doc_sentence_token_sets)
    ]
    scores = [item for item in scores if item[0] > 0]
    scores.sort(reverse=True)
    return [idx for _, idx in scores[:limit]]


def _candidate_window_ranges(
    block: QueryBlock,
    doc: ArchiveDoc,
    doc_sentence_token_sets: tuple[frozenset[str], ...],
) -> list[tuple[int, int]]:
    block_sentence_count = max(1, len(block.normalized_sentences))
    min_window = max(1, block_sentence_count - 2)
    max_window = min(len(doc.normalized_sentences), block_sentence_count + 2)
    if max_window <= 0:
        return []

    query_sentence_tokens = [tokens for tokens in block.sentence_tokens if tokens]
    anchor_indices: set[int] = set()
    if query_sentence_tokens:
        anchor_indices.update(_top_anchor_indices(query_sentence_tokens[0], doc_sentence_token_sets))
        anchor_indices.update(_top_anchor_indices(query_sentence_tokens[-1], doc_sentence_token_sets))
        middle_idx = len(query_sentence_tokens) // 2
        anchor_indices.update(_top_anchor_indices(query_sentence_tokens[middle_idx], doc_sentence_token_sets))

    windows: set[tuple[int, int]] = set()
    if not anchor_indices:
        step = max(1, len(doc.normalized_sentences) // 10)
        for start in range(0, len(doc.normalized_sentences), step):
            for window_size in range(min_window, max_window + 1):
                end = min(len(doc.normalized_sentences), start + window_size)
                if end > start:
                    windows.add((start, end))
    else:
        for anchor_idx in anchor_indices:
            for window_size in range(min_window, max_window + 1):
                for offset in (-2, -1, 0, 1, 2):
                    start = max(0, anchor_idx + offset)
                    end = start + window_size
                    if end <= len(doc.normalized_sentences):
                        windows.add((start, end))
                    anchored_end = anchor_idx + 1 + offset
                    anchored_start = anchored_end - window_size
                    if 0 <= anchored_start < anchored_end <= len(doc.normalized_sentences):
                        windows.add((anchored_start, anchored_end))
    return sorted(windows)


def align_block_to_doc(
    block: QueryBlock,
    doc: ArchiveDoc,
    doc_sentence_tokens: tuple[tuple[str, ...], ...],
    doc_sentence_token_sets: tuple[frozenset[str], ...],
) -> dict[str, object]:
    if not doc.normalized_sentences:
        return {
            "alignment_score": 0.0,
            "sentence_hits": 0,
            "sentence_hit_fraction": 0.0,
            "doc_window_start": -1,
            "doc_window_end": -1,
            "evidence_snippet": "",
        }
    block_query_counter = Counter(block.query_tokens)
    best: dict[str, object] = {
        "alignment_score": 0.0,
        "sentence_hits": 0,
        "sentence_hit_fraction": 0.0,
        "doc_window_start": -1,
        "doc_window_end": -1,
        "evidence_snippet": "",
    }
    for start, end in _candidate_window_ranges(block, doc, doc_sentence_token_sets):
        window_sentence_tokens = doc_sentence_tokens[start:end]
        window_tokens = tuple(
            token
            for sentence_tokens in window_sentence_tokens
            for token in sentence_tokens
        )
        if not window_tokens:
            continue
        token_f1, recall, window_containment = score_token_pair(
            block.query_tokens,
            window_tokens,
            query_counter=block_query_counter,
        )
        local_scores = [
            score_sentence_against_window(sentence_tokens, window_sentence_tokens)
            for sentence_tokens in block.sentence_tokens
            if sentence_tokens
        ]
        sentence_hits = sum(score >= 0.65 for score in local_scores)
        sentence_hit_fraction = (
            sentence_hits / len(local_scores) if local_scores else 0.0
        )
        alignment_score = (
            0.45 * token_f1
            + 0.25 * recall
            + 0.15 * window_containment
            + 0.15 * sentence_hit_fraction
        )
        if alignment_score > float(best["alignment_score"]):
            raw_evidence = " ".join(doc.sentences[start:end]).strip()
            best = {
                "alignment_score": float(alignment_score),
                "sentence_hits": int(sentence_hits),
                "sentence_hit_fraction": float(sentence_hit_fraction),
                "doc_window_start": int(start),
                "doc_window_end": int(end - 1),
                "evidence_snippet": raw_evidence[:1200],
            }
    return best


def score_block_candidates(
    matcher: ArchiveMatcher,
    block: QueryBlock,
    *,
    top_k: int,
    min_score: float,
    min_margin: float,
    min_sentence_hits: int,
) -> list[CandidateMatch]:
    retrieved = matcher.retrieve_candidate_indices(block, top_k=top_k)
    scored: list[CandidateMatch] = []
    interim: list[dict[str, object]] = []
    for rank, (doc_idx, retrieval_score) in enumerate(retrieved, start=1):
        doc = matcher.docs[doc_idx]
        alignment = align_block_to_doc(
            block,
            doc,
            matcher.doc_sentence_tokens[doc_idx],
            matcher.doc_sentence_token_sets[doc_idx],
        )
        match_score = 0.35 * float(retrieval_score) + 0.65 * float(alignment["alignment_score"])
        interim.append(
            {
                "rank": rank,
                "doc": doc,
                "retrieval_score": float(retrieval_score),
                "alignment_score": float(alignment["alignment_score"]),
                "match_score": float(match_score),
                "sentence_hits": int(alignment["sentence_hits"]),
                "sentence_hit_fraction": float(alignment["sentence_hit_fraction"]),
                "evidence_snippet": str(alignment["evidence_snippet"]),
                "doc_window_start": int(alignment["doc_window_start"]),
                "doc_window_end": int(alignment["doc_window_end"]),
            }
        )
    if not interim:
        return []
    interim.sort(key=lambda item: float(item["match_score"]), reverse=True)
    second_score = float(interim[1]["match_score"]) if len(interim) > 1 else 0.0
    for item in interim:
        doc = item["doc"]
        assert isinstance(doc, ArchiveDoc)
        score_margin = float(item["match_score"]) - second_score if item is interim[0] else 0.0
        accepted = bool(
            item is interim[0]
            and float(item["match_score"]) >= min_score
            and score_margin >= min_margin
            and int(item["sentence_hits"]) >= min_sentence_hits
        )
        scored.append(
            CandidateMatch(
                block_id=block.block_id,
                speech_id=block.speech_id,
                start_row_idx=block.start_row_idx,
                end_row_idx=block.end_row_idx,
                start_sentence_id=block.start_sentence_id,
                end_sentence_id=block.end_sentence_id,
                rank=int(item["rank"]),
                doc_id=doc.doc_id,
                president=doc.president,
                matched_label=PRESIDENT_NAME_TO_LABEL[doc.president],
                source_name=doc.source_name,
                source_url=doc.source_url,
                source_title=doc.title,
                source_date=doc.date,
                retrieval_score=float(item["retrieval_score"]),
                alignment_score=float(item["alignment_score"]),
                match_score=float(item["match_score"]),
                score_margin=float(score_margin),
                sentence_hits=int(item["sentence_hits"]),
                sentence_hit_fraction=float(item["sentence_hit_fraction"]),
                evidence_snippet=str(item["evidence_snippet"]),
                doc_window_start=int(item["doc_window_start"]),
                doc_window_end=int(item["doc_window_end"]),
                accepted=accepted,
            )
        )
    return scored


def candidate_matches_to_frame(matches: Iterable[CandidateMatch]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "block_id": match.block_id,
                "speech_id": match.speech_id,
                "start_row_idx": match.start_row_idx,
                "end_row_idx": match.end_row_idx,
                "start_sentence_id": match.start_sentence_id,
                "end_sentence_id": match.end_sentence_id,
                "rank": match.rank,
                "doc_id": match.doc_id,
                "president": match.president,
                "matched_label": match.matched_label,
                "source_name": match.source_name,
                "source_url": match.source_url,
                "source_title": match.source_title,
                "source_date": match.source_date,
                "retrieval_score": match.retrieval_score,
                "alignment_score": match.alignment_score,
                "match_score": match.match_score,
                "score_margin": match.score_margin,
                "sentence_hits": match.sentence_hits,
                "sentence_hit_fraction": match.sentence_hit_fraction,
                "evidence_snippet": match.evidence_snippet,
                "doc_window_start": match.doc_window_start,
                "doc_window_end": match.doc_window_end,
                "accepted": match.accepted,
            }
            for match in matches
        ]
    )


def load_model_triage(path: str | Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    frame = pd.read_csv(path)
    if {"speech_id", "sentence_id", "pred_raw_label", "pred_single_negative_span_label"} - set(frame.columns):
        return None
    frame = frame.copy()
    frame["model_disagreement"] = frame["pred_raw_label"] != frame["pred_single_negative_span_label"]
    return frame[
        [
            "speech_id",
            "sentence_id",
            "pred_raw_label",
            "pred_single_negative_span_label",
            "model_disagreement",
        ]
    ]


def build_row_matches(
    rows: pd.DataFrame,
    candidate_matches: Iterable[CandidateMatch],
    *,
    triage_frame: pd.DataFrame | None = None,
    row_conflict_margin: float = 0.05,
) -> pd.DataFrame:
    by_row: dict[int, list[CandidateMatch]] = {}
    for candidate in candidate_matches:
        if not candidate.accepted:
            continue
        for row_idx in range(candidate.start_row_idx, candidate.end_row_idx + 1):
            by_row.setdefault(row_idx, []).append(candidate)
    triage_lookup: dict[tuple[int, int], bool] = {}
    if triage_frame is not None:
        triage_lookup = {
            (int(row["speech_id"]), int(row["sentence_id"])): bool(row["model_disagreement"])
            for row in triage_frame.to_dict(orient="records")
        }
    results: list[RowMatch] = []
    for row in rows.to_dict(orient="records"):
        row_idx = int(row["row_idx"])
        speech_id = int(row["speech_id"])
        sentence_id = int(row["sentence_id"])
        text = str(row["text"])
        model_disagreement = triage_lookup.get((speech_id, sentence_id), False)
        candidates = sorted(
            by_row.get(row_idx, []),
            key=lambda candidate: candidate.match_score,
            reverse=True,
        )
        if not candidates:
            results.append(
                RowMatch(
                    row_idx=row_idx,
                    speech_id=speech_id,
                    sentence_id=sentence_id,
                    text=text,
                    matched_president=None,
                    matched_label=None,
                    source_url=None,
                    source_title=None,
                    source_date=None,
                    source_name=None,
                    match_score=0.0,
                    score_margin=0.0,
                    review_status="review_unmatched",
                    evidence_snippet=None,
                    block_id=None,
                    model_disagreement=model_disagreement,
                    triage_priority=2.0 + float(model_disagreement),
                )
            )
            continue
        best = candidates[0]
        conflict = any(
            candidate.source_url != best.source_url
            and best.match_score - candidate.match_score <= row_conflict_margin
            for candidate in candidates[1:]
        )
        review_status = "review_conflict" if conflict else "accepted"
        triage_priority = (
            (3.0 if review_status != "accepted" else 0.0)
            + (1.0 if model_disagreement else 0.0)
            + (1.0 - best.match_score)
        )
        results.append(
            RowMatch(
                row_idx=row_idx,
                speech_id=speech_id,
                sentence_id=sentence_id,
                text=text,
                matched_president=best.president if not conflict else None,
                matched_label=best.matched_label if not conflict else None,
                source_url=best.source_url if not conflict else None,
                source_title=best.source_title if not conflict else None,
                source_date=best.source_date if not conflict else None,
                source_name=best.source_name if not conflict else None,
                match_score=best.match_score,
                score_margin=best.score_margin,
                review_status=review_status,
                evidence_snippet=best.evidence_snippet,
                block_id=best.block_id,
                model_disagreement=model_disagreement,
                triage_priority=triage_priority,
            )
        )
    frame = pd.DataFrame(
        [
            {
                "row_idx": result.row_idx,
                "speech_id": result.speech_id,
                "sentence_id": result.sentence_id,
                "text": result.text,
                "matched_president": result.matched_president,
                "matched_label": result.matched_label,
                "source_url": result.source_url,
                "source_title": result.source_title,
                "source_date": result.source_date,
                "source_name": result.source_name,
                "match_score": result.match_score,
                "score_margin": result.score_margin,
                "review_status": result.review_status,
                "evidence_snippet": result.evidence_snippet,
                "block_id": result.block_id,
                "model_disagreement": result.model_disagreement,
                "triage_priority": result.triage_priority,
            }
            for result in results
        ]
    )
    return frame.sort_values(["row_idx"], kind="stable").reset_index(drop=True)


def apply_row_acceptance_threshold(
    frame: pd.DataFrame,
    *,
    min_score: float,
) -> pd.DataFrame:
    adjusted = frame.copy()
    low_confidence_mask = (
        adjusted["review_status"].eq("accepted") & adjusted["match_score"].lt(min_score)
    )
    adjusted.loc[low_confidence_mask, "review_status"] = "review_low_confidence"
    for column in [
        "matched_president",
        "matched_label",
        "source_url",
        "source_title",
        "source_date",
        "source_name",
        "evidence_snippet",
        "block_id",
    ]:
        adjusted.loc[low_confidence_mask, column] = None
    adjusted["triage_priority"] = (
        adjusted["triage_priority"]
        + adjusted["review_status"].ne("accepted").astype(float)
    )
    return adjusted


def submission_from_row_matches(frame: pd.DataFrame) -> pd.DataFrame:
    labels = frame["matched_label"].where(frame["review_status"].eq("accepted"), "")
    return pd.DataFrame({"label": labels.fillna("")})


def compute_threshold_grid(
    row_matches: pd.DataFrame,
    *,
    gold_labels: pd.Series,
    min_threshold: float = 0.2,
    max_threshold: float = 0.95,
    step: float = 0.01,
) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for threshold in np.arange(min_threshold, max_threshold + step / 2.0, step):
        accepted = (
            row_matches["review_status"].eq("accepted")
            & row_matches["match_score"].ge(float(threshold))
        )
        accepted_count = int(accepted.sum())
        if accepted_count == 0:
            precision = math.nan
        else:
            predicted = row_matches.loc[accepted, "matched_label"].reset_index(drop=True)
            truth = gold_labels.loc[accepted].reset_index(drop=True)
            precision = float((predicted == truth).mean())
        coverage = accepted_count / len(row_matches)
        rows.append(
            {
                "threshold": float(round(threshold, 4)),
                "accepted_rows": accepted_count,
                "coverage": float(coverage),
                "precision": precision,
            }
        )
    return pd.DataFrame(rows)


def choose_threshold(
    threshold_grid: pd.DataFrame,
    *,
    min_precision: float,
) -> dict[str, float | int | None]:
    eligible = threshold_grid.loc[threshold_grid["precision"].ge(min_precision)].copy()
    if eligible.empty:
        return {
            "recommended_threshold": None,
            "accepted_rows": 0,
            "coverage": 0.0,
            "precision": None,
        }
    eligible = eligible.sort_values(
        ["coverage", "threshold"],
        ascending=[False, False],
        kind="stable",
    ).reset_index(drop=True)
    best = eligible.iloc[0]
    return {
        "recommended_threshold": float(best["threshold"]),
        "accepted_rows": int(best["accepted_rows"]),
        "coverage": float(best["coverage"]),
        "precision": float(best["precision"]),
    }
