"""Lightweight text cleaning for customer support tweets.

Deliberately minimal: this is normalization for embedding/clustering and
display, not a full NLP pipeline. Aggressive cleaning (e.g. stemming,
stopword removal beyond what TF-IDF's own vectorizer handles) is avoided
because it would make it harder for a human reviewer to recognize the
original message when inspecting clusters or golden-set examples.
"""
from __future__ import annotations

import re

_URL_RE = re.compile(r"https?://\S+")
_MENTION_RE = re.compile(r"@\w+")
_WHITESPACE_RE = re.compile(r"\s+")


def clean_for_display(text: str) -> str:
    """Minimal cleanup: collapse whitespace only. Keeps mentions/URLs."""
    return _WHITESPACE_RE.sub(" ", text or "").strip()


def clean_for_modeling(text: str) -> str:
    """Cleanup for TF-IDF/embedding input: strip @mentions and URLs (both
    are near-constant boilerplate in this dataset -- every customer tweet
    starts with '@AmazonHelp' or similar, and URLs carry no topical
    signal for TF-IDF), collapse whitespace, lowercase.
    """
    text = text or ""
    text = _URL_RE.sub(" ", text)
    text = _MENTION_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip().lower()
    return text


def non_ascii_ratio(text: str) -> float:
    if not text:
        return 0.0
    non_ascii = sum(1 for c in text if ord(c) > 127)
    return non_ascii / max(len(text), 1)


def is_likely_non_english(text: str, threshold: float = 0.15) -> bool:
    """Heuristic language filter.

    This is intentionally a cheap heuristic (non-ASCII character ratio),
    not a real language-ID model (langdetect/fasttext are not installable
    in this environment -- no network access for pip). It will misclassify
    some edge cases (e.g. an English tweet quoting a non-Latin product
    name, or non-English text that happens to be mostly Latin-script like
    French/Spanish). Documented as a known limitation in
    docs/decision-log.md rather than presented as precise language ID.
    """
    return non_ascii_ratio(text) > threshold
