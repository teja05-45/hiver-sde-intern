"""
Display helpers for showing REAL dataset content in a product UI.

The support corpus is real Twitter data: customer tweets are full of
@mentions, t.co shortener links, and raw agent signatures. Retrieval and
classification want that text; a support workspace should not paste
other customers' handles into a demo inbox. These helpers produce the
DISPLAY projection only -- raw text is never altered in the pipeline.

Every value here is derived deterministically from real data. Nothing in
this module invents content: it masks, truncates, or formats it.
"""
from __future__ import annotations

import re
from datetime import datetime

_MENTION_RE = re.compile(r"@\w+")
_URL_RE = re.compile(r"https?://\S+")
_WS_RE = re.compile(r"\s+")

# Twitter's created_at format, e.g. "Tue Oct 31 22:19:34 +0000 2017"
_TWITTER_TIME_FMT = "%a %b %d %H:%M:%S %z %Y"


def sanitize_display_text(text: str | None, max_chars: int = 160) -> str:
    """Mask @mentions and links, collapse whitespace, truncate.

    Truncation is display-only and always marked with an ellipsis so a
    reader can tell a preview from the full message.
    """
    if not text:
        return ""
    cleaned = _WS_RE.sub(" ", _URL_RE.sub("", _MENTION_RE.sub("", str(text)))).strip()
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 1].rstrip() + "…"


def customer_ref(conversation_id: str) -> str:
    """Stable, PII-free customer label derived from the real conversation id.

    The dataset has no customer names -- only tweet/author ids embedded in
    conversation ids like "AmazonHelp_620". A stable hash gives every real
    conversation a consistent workspace label ("Customer #1842") without
    inventing identities: the same conversation always maps to the same
    reference, different conversations map to different references.
    """
    digest = abs(hash(conversation_id)) % 10000
    return f"Customer #{digest:04d}"


def to_iso_utc(raw: str | None) -> str | None:
    """Twitter created_at -> ISO-8601 UTC. Returns None for unparseable
    input instead of guessing a timestamp."""
    if not raw:
        return None
    try:
        return datetime.strptime(raw.strip(), _TWITTER_TIME_FMT).isoformat()
    except ValueError:
        return None
