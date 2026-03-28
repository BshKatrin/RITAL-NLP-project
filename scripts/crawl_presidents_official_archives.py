from __future__ import annotations

import argparse
import json
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
        default=8,
        help="Maximum number of concurrent HTTP fetches.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=45,
        help="Per-request timeout in seconds.",
    )
    return parser.parse_args()


def make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "RITAL-NLP-project archive crawler "
                "(official-source labeling workflow; contact local repo user)"
            )
        }
    )
    return session


def fetch_elysee_doc(
    source_url: str,
    president: str,
    *,
    timeout_seconds: int,
) -> ArchiveDoc:
    session = make_session()
    response = session.get(source_url, timeout=timeout_seconds)
    response.raise_for_status()
    return parse_elysee_document(source_url, response.text, president)


def fetch_vie_publique_doc(
    source_url: str,
    *,
    timeout_seconds: int,
) -> ArchiveDoc:
    session = make_session()
    response = session.get(source_url, timeout=timeout_seconds)
    response.raise_for_status()
    return parse_vie_publique_document(source_url, response.text)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    session = make_session()
    elysee_urls = get_elysee_archive_urls(session)
    if args.max_elysee_docs is not None:
        elysee_urls = elysee_urls[: args.max_elysee_docs]
    vie_publique_urls = read_url_list(args.vie_publique_urls_path)

    docs: list[ArchiveDoc] = []
    failures: list[dict[str, str]] = []
    submitted: dict[object, tuple[str, str]] = {}

    with ThreadPoolExecutor(max_workers=max(args.workers, 1)) as executor:
        for source_url, president in elysee_urls:
            future = executor.submit(
                fetch_elysee_doc,
                source_url,
                president,
                timeout_seconds=args.timeout_seconds,
            )
            submitted[future] = ("elysee", source_url)
        for source_url in vie_publique_urls:
            future = executor.submit(
                fetch_vie_publique_doc,
                source_url,
                timeout_seconds=args.timeout_seconds,
            )
            submitted[future] = ("vie-publique", source_url)

        for future in as_completed(submitted):
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

    if not docs:
        raise ArchiveLabelingError("Crawler did not produce any archive document")

    docs.sort(key=lambda doc: (doc.president, doc.date, doc.doc_id))
    frame = pd.DataFrame([archive_doc_to_record(doc) for doc in docs])
    frame = frame.drop_duplicates(subset=["source_url"], keep="first").reset_index(drop=True)

    parquet_path = output_dir / "archive_docs.parquet"
    csv_path = output_dir / "archive_docs.csv"
    failures_path = output_dir / "crawl_failures.csv"
    summary_path = output_dir / "crawl_summary.json"

    frame.to_parquet(parquet_path, index=False)
    frame.to_csv(csv_path, index=False)
    pd.DataFrame(failures).to_csv(failures_path, index=False)

    summary = {
        "elysee_urls_requested": len(elysee_urls),
        "vie_publique_urls_requested": len(vie_publique_urls),
        "docs_written": int(len(frame)),
        "elysee_docs_written": int(frame["source_name"].eq("elysee").sum()),
        "vie_publique_docs_written": int(frame["source_name"].eq("vie-publique").sum()),
        "failures": len(failures),
        "outputs": {
            "archive_docs_parquet": str(parquet_path.resolve()),
            "archive_docs_csv": str(csv_path.resolve()),
            "crawl_failures_csv": str(failures_path.resolve()),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
