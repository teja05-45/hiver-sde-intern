"""
Ingestion layer for the raw `twcs.csv` Customer Support on Twitter dataset.

Design notes (see docs/decision-log.md for the full rationale):

- The full dataset (~2.81M usable rows once you account for CSV fields that
  contain embedded newlines, which is why `wc -l` overcounts vs the real
  row count -- see `docs/decision-log.md`, decision "wc -l vs parsed rows")
  fits a *lightweight 4-column index* (tweet_id, author_id, inbound,
  in_response_to_tweet_id) comfortably in memory (~120MB measured). We
  exploit that: a single indexed pass replaces the graph-reconstruction
  problem with dict lookups, which is both simpler and faster than
  re-scanning the file per brand.
- The *full text* column is NOT loaded for all 2.81M rows by default --
  only for the rows belonging to the brand(s) actually being processed.
  This keeps peak memory well under the container's ~3GB budget even
  though the raw CSV is ~500MB on disk.
- Everything here is deterministic given the same input file (no random
  sampling at this layer -- sampling happens downstream, with an explicit
  seed, in `scripts/build_dataset.py`).
"""
from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

logger = logging.getLogger(__name__)

EXPECTED_COLUMNS = [
    "tweet_id",
    "author_id",
    "inbound",
    "created_at",
    "text",
    "response_tweet_id",
    "in_response_to_tweet_id",
]


@dataclass(frozen=True)
class TweetIndexEntry:
    """Lightweight per-tweet record used for graph traversal (no text)."""

    tweet_id: int
    author_id: str
    inbound: bool
    in_response_to_tweet_id: Optional[int]


@dataclass
class IngestionStats:
    """Data-quality counters accumulated during a streaming pass."""

    rows_seen: int = 0
    rows_valid: int = 0
    rows_missing_text: int = 0
    rows_missing_tweet_id: int = 0
    rows_malformed_reference: int = 0
    rows_duplicate_tweet_id: int = 0

    def as_dict(self) -> dict:
        return {
            "rows_seen": self.rows_seen,
            "rows_valid": self.rows_valid,
            "rows_missing_text": self.rows_missing_text,
            "rows_missing_tweet_id": self.rows_missing_tweet_id,
            "rows_malformed_reference": self.rows_malformed_reference,
            "rows_duplicate_tweet_id": self.rows_duplicate_tweet_id,
            "valid_rate": round(self.rows_valid / self.rows_seen, 5) if self.rows_seen else 0.0,
        }


def _parse_optional_int(raw: str) -> Optional[int]:
    """Parse a possibly-empty, possibly comma-joined reference id field.

    `response_tweet_id` can contain multiple comma-separated ids (a tweet
    can receive several replies). For graph-traversal purposes here we only
    need `in_response_to_tweet_id`, which the source dataset guarantees is
    single-valued, but we defensively take the first token in case of
    malformed rows rather than raising.
    """
    if raw is None:
        return None
    raw = raw.strip()
    if raw == "" or raw.lower() == "nan":
        return None
    first = raw.split(",")[0].strip()
    try:
        return int(first)
    except ValueError:
        return None


def validate_file_header(path: Path) -> None:
    """Raise a clear, actionable error if the CSV doesn't match the expected schema."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Download the 'Customer Support on Twitter' "
            f"dataset (Kaggle: thoughtvector/customer-support-on-twitter), extract "
            f"twcs.csv, and place it at {path}. See README.md 'Dataset setup'."
        )
    with path.open("r", encoding="utf-8", newline="") as f:
        header = next(csv.reader(f))
    missing = [c for c in EXPECTED_COLUMNS if c not in header]
    if missing:
        raise ValueError(
            f"Dataset at {path} is missing expected columns {missing}. "
            f"Found columns: {header}. This ingestion pipeline expects the "
            f"standard 'Customer Support on Twitter' schema: {EXPECTED_COLUMNS}."
        )


def build_lightweight_index(path: Path) -> tuple[dict[int, TweetIndexEntry], IngestionStats]:
    """Stream the full CSV once and build an in-memory tweet_id -> entry index.

    Only the four columns needed for graph traversal are retained per row
    (no `text`, no `created_at`), which is what keeps this feasible in a
    constrained-memory environment for a ~2.8M-row file. Returns both the
    index and data-quality statistics, since "log data-quality statistics"
    is an explicit pipeline requirement.
    """
    validate_file_header(path)
    index: dict[int, TweetIndexEntry] = {}
    stats = IngestionStats()

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            stats.rows_seen += 1
            tweet_id_raw = row.get("tweet_id")
            if not tweet_id_raw:
                stats.rows_missing_tweet_id += 1
                continue
            try:
                tweet_id = int(tweet_id_raw)
            except ValueError:
                stats.rows_missing_tweet_id += 1
                continue

            if tweet_id in index:
                stats.rows_duplicate_tweet_id += 1
                continue

            author_id = (row.get("author_id") or "").strip()
            inbound_raw = (row.get("inbound") or "").strip().lower()
            inbound = inbound_raw == "true"
            in_response_to = _parse_optional_int(row.get("in_response_to_tweet_id"))

            index[tweet_id] = TweetIndexEntry(
                tweet_id=tweet_id,
                author_id=author_id,
                inbound=inbound,
                in_response_to_tweet_id=in_response_to,
            )
            stats.rows_valid += 1

    logger.info("Ingestion index built: %s", stats.as_dict())
    return index, stats


def iter_rows_for_authors(path: Path, author_ids: set[str], batch_size: int = 20000) -> Iterator[list[dict]]:
    """Stream the full CSV once, yielding batches of raw rows whose author_id is in `author_ids`.

    Used to fetch full text (which we deliberately don't hold in memory for
    the whole dataset) only for the brand(s) selected for downstream
    processing.
    """
    validate_file_header(path)
    batch: list[dict] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("author_id") in author_ids:
                batch.append(row)
                if len(batch) >= batch_size:
                    yield batch
                    batch = []
    if batch:
        yield batch


def iter_rows_for_tweet_ids(path: Path, tweet_ids: set[int], batch_size: int = 20000) -> Iterator[list[dict]]:
    """Stream the full CSV once, yielding batches of raw rows whose tweet_id is in `tweet_ids`.

    Unlike `iter_rows_for_authors` (which filters by who *sent* the tweet),
    this filters by a specific set of tweet ids -- used to fetch full text
    for every tweet in a reconstructed conversation thread, which includes
    both the brand's tweets and the customer's tweets.
    """
    validate_file_header(path)
    batch: list[dict] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tid_raw = row.get("tweet_id")
            if not tid_raw:
                continue
            try:
                tid = int(tid_raw)
            except ValueError:
                continue
            if tid in tweet_ids:
                batch.append(row)
                if len(batch) >= batch_size:
                    yield batch
                    batch = []
    if batch:
        yield batch
