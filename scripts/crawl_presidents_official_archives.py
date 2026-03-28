from __future__ import annotations

import argparse
import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

from rital_nlp_project.presidents.archive_labeling import (
    ArchiveDoc,
    ArchiveLabelingError,
    archive_doc_to_record,
    get_elysee_archive_urls,
    parse_elysee_document,
    parse_vie_publique_document,
    read_url_list,
)


RETRYABLE_STATUS_CODES = {403, 429, 500, 502, 503, 504}
THREAD_LOCAL = threading.local()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Crawl official Jacques Chirac and François Mitterrand archive pages "
            "from the Élysée publication sitemap and write a normalized local corpus."
        )
    )
    parser.add_argument(
        "--output-dir",
        default="Dataset/out/presidents_archive_labeling",
        help="Directory where archive docs and crawl metadata are written.",
    )
    parser.add_argument(
        "--max-elysee-docs",
        type=int,
        default=None,
        help="Optional cap for Élysée documents, useful for smoke tests.",
    )
    parser.add_argument(
        "--vie-publique-urls-path",
        default=None,
        help=(
            "Optional text file containing one official vie-publique discourse URL per line. "
            "These are appended as fallback documents."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Maximum number of concurrent HTTP fetches.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=45,
        help="Per-request timeout in seconds.",
    )
    parser.add_argument(
        "--min-request-interval-seconds",
        type=float,
        default=1.0,
        help="Minimum delay between two outbound requests across all workers.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=6,
        help="Maximum number of fetch attempts for retryable responses such as 403/429.",
    )
    parser.add_argument(
        "--retry-backoff-seconds",
        type=float,
        default=20.0,
        help="Base exponential backoff in seconds for retryable responses.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing archive_docs file in the output directory and skip URLs already fetched.",
    )
    return parser.parse_args()


def make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/123.0.0.0 Safari/537.36"
            ),
            "Accept": (
                "text/html,application/xhtml+xml,application/xml;q=0.9,"
                "image/avif,image/webp,*/*;q=0.8"
            ),
            "Accept-Language": "fr-FR,fr;q=0.9,en-US;q=0.8,en;q=0.7",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
            "Referer": "https://www.elysee.fr/",
        }
    )
    return session


class RequestLimiter:
    def __init__(self, min_interval_seconds: float):
        self.min_interval_seconds = max(float(min_interval_seconds), 0.0)
        self._lock = threading.Lock()
        self._next_allowed_at = 0.0

    def wait(self) -> None:
        if self.min_interval_seconds <= 0.0:
            return
        while True:
            with self._lock:
                now = time.monotonic()
                wait_seconds = self._next_allowed_at - now
                if wait_seconds <= 0.0:
                    self._next_allowed_at = now + self.min_interval_seconds
                    return
            time.sleep(min(wait_seconds, 0.25))


def get_thread_session() -> requests.Session:
    session = getattr(THREAD_LOCAL, "session", None)
    if session is None:
        session = make_session()
        THREAD_LOCAL.session = session
    return session


def reset_thread_session() -> None:
    session = getattr(THREAD_LOCAL, "session", None)
    if session is not None:
        session.close()
    THREAD_LOCAL.session = make_session()


def fetch_html_with_retries(
    source_url: str,
    *,
    timeout_seconds: int,
    limiter: RequestLimiter,
    max_retries: int,
    retry_backoff_seconds: float,
) -> str:
    attempts = max(int(max_retries), 1)
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        limiter.wait()
        session = get_thread_session()
        try:
            response = session.get(source_url, timeout=timeout_seconds)
            if response.status_code in RETRYABLE_STATUS_CODES:
                raise requests.HTTPError(
                    f"{response.status_code} Client Error: {response.reason} for url: {source_url}",
                    response=response,
                )
            response.raise_for_status()
            return response.text
        except Exception as exc:  # pragma: no cover - network failures vary
            last_error = exc
            if attempt >= attempts:
                break
            reset_thread_session()
            response = getattr(exc, "response", None)
            retry_after = None if response is None else response.headers.get("Retry-After")
            if retry_after is not None:
                try:
                    sleep_seconds = float(retry_after)
                except ValueError:
                    sleep_seconds = retry_backoff_seconds
            else:
                sleep_seconds = retry_backoff_seconds * (2 ** (attempt - 1))
            sleep_seconds += random.uniform(0.0, 1.5)
            time.sleep(sleep_seconds)
    assert last_error is not None
    raise last_error


def fetch_elysee_doc(
    source_url: str,
    president: str,
    *,
    timeout_seconds: int,
    limiter: RequestLimiter,
    max_retries: int,
    retry_backoff_seconds: float,
) -> ArchiveDoc:
    html_text = fetch_html_with_retries(
        source_url,
        timeout_seconds=timeout_seconds,
        limiter=limiter,
        max_retries=max_retries,
        retry_backoff_seconds=retry_backoff_seconds,
    )
    return parse_elysee_document(source_url, html_text, president)


def fetch_vie_publique_doc(
    source_url: str,
    *,
    timeout_seconds: int,
    limiter: RequestLimiter,
    max_retries: int,
    retry_backoff_seconds: float,
) -> ArchiveDoc:
    html_text = fetch_html_with_retries(
        source_url,
        timeout_seconds=timeout_seconds,
        limiter=limiter,
        max_retries=max_retries,
        retry_backoff_seconds=retry_backoff_seconds,
    )
    return parse_vie_publique_document(source_url, html_text)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    limiter = RequestLimiter(args.min_request_interval_seconds)

    session = make_session()
    elysee_urls = get_elysee_archive_urls(session)
    if args.max_elysee_docs is not None:
        elysee_urls = elysee_urls[: args.max_elysee_docs]
    vie_publique_urls = read_url_list(args.vie_publique_urls_path)

    parquet_path = output_dir / "archive_docs.parquet"
    csv_path = output_dir / "archive_docs.csv"
    failures_path = output_dir / "crawl_failures.csv"
    summary_path = output_dir / "crawl_summary.json"

    existing_docs: list[ArchiveDoc] = []
    already_fetched_urls: set[str] = set()
    if args.resume and parquet_path.exists():
        existing_frame = pd.read_parquet(parquet_path)
        if not existing_frame.empty:
            already_fetched_urls = set(existing_frame["source_url"].astype(str))
            existing_docs = [
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
                for row in existing_frame.to_dict(orient="records")
            ]

    elysee_urls = [
        (source_url, president)
        for source_url, president in elysee_urls
        if source_url not in already_fetched_urls
    ]
    vie_publique_urls = [
        source_url
        for source_url in vie_publique_urls
        if source_url not in already_fetched_urls
    ]

    docs: list[ArchiveDoc] = list(existing_docs)
    failures: list[dict[str, str]] = []
    submitted: dict[object, tuple[str, str]] = {}
    total_to_fetch = len(elysee_urls) + len(vie_publique_urls)

    print(
        json.dumps(
            {
                "event": "crawl_start",
                "elysee_urls_requested": len(elysee_urls),
                "vie_publique_urls_requested": len(vie_publique_urls),
                "docs_reused_from_resume": len(existing_docs),
                "workers": int(args.workers),
                "min_request_interval_seconds": float(args.min_request_interval_seconds),
                "max_retries": int(args.max_retries),
                "retry_backoff_seconds": float(args.retry_backoff_seconds),
            }
        ),
        flush=True,
    )

    with ThreadPoolExecutor(max_workers=max(args.workers, 1)) as executor:
        for source_url, president in elysee_urls:
            future = executor.submit(
                fetch_elysee_doc,
                source_url,
                president,
                timeout_seconds=args.timeout_seconds,
                limiter=limiter,
                max_retries=args.max_retries,
                retry_backoff_seconds=args.retry_backoff_seconds,
            )
            submitted[future] = ("elysee", source_url)
        for source_url in vie_publique_urls:
            future = executor.submit(
                fetch_vie_publique_doc,
                source_url,
                timeout_seconds=args.timeout_seconds,
                limiter=limiter,
                max_retries=args.max_retries,
                retry_backoff_seconds=args.retry_backoff_seconds,
            )
            submitted[future] = ("vie-publique", source_url)

        for completed_count, future in enumerate(as_completed(submitted), start=1):
            source_name, source_url = submitted[future]
            try:
                docs.append(future.result())
            except Exception as exc:  # pragma: no cover - network failures vary
                failures.append(
                    {
                        "source_name": source_name,
                        "source_url": source_url,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
            if completed_count % 25 == 0 or completed_count == total_to_fetch:
                print(
                    json.dumps(
                        {
                            "event": "crawl_progress",
                            "completed": completed_count,
                            "total_to_fetch": total_to_fetch,
                            "succeeded_this_run": len(docs) - len(existing_docs),
                            "failed_this_run": len(failures),
                        }
                    ),
                    flush=True,
                )

    if not docs:
        raise ArchiveLabelingError("Crawler did not produce any archive document")

    docs.sort(key=lambda doc: (doc.president, doc.date, doc.doc_id))
    frame = pd.DataFrame([archive_doc_to_record(doc) for doc in docs])
    frame = frame.drop_duplicates(subset=["source_url"], keep="first").reset_index(drop=True)

    frame.to_parquet(parquet_path, index=False)
    frame.to_csv(csv_path, index=False)
    pd.DataFrame(failures).to_csv(failures_path, index=False)

    summary = {
        "elysee_urls_requested": len(elysee_urls),
        "vie_publique_urls_requested": len(vie_publique_urls),
        "docs_reused_from_resume": len(existing_docs),
        "docs_written": int(len(frame)),
        "elysee_docs_written": int(frame["source_name"].eq("elysee").sum()),
        "vie_publique_docs_written": int(frame["source_name"].eq("vie-publique").sum()),
        "failures": len(failures),
        "workers": int(args.workers),
        "min_request_interval_seconds": float(args.min_request_interval_seconds),
        "max_retries": int(args.max_retries),
        "retry_backoff_seconds": float(args.retry_backoff_seconds),
        "outputs": {
            "archive_docs_parquet": str(parquet_path.resolve()),
            "archive_docs_csv": str(csv_path.resolve()),
            "crawl_failures_csv": str(failures_path.resolve()),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"event": "crawl_complete", **summary}), flush=True)


if __name__ == "__main__":
    main()
