"""
Leakage-aware dataset splitting.

Why temporal, not random (see docs/decision-log.md for the full writeup):
this is a support dataset with recurring issue types, templated agent
replies, and repeated customer complaints about the same events (e.g. a
shipping delay affecting many customers in the same week). A random split
lets near-identical complaints from the same week land in both train and
test, which inflates classical accuracy without reflecting deployment
reality -- in production, the model only ever sees *future* messages it
was not evaluated against. A temporal split (train on earlier
conversations, evaluate on strictly later ones) is a closer proxy to
actually deploying this system.

Duplicate/near-duplicate detection is a second, complementary leakage
check: even within a temporal split, if the *same* templated complaint
text appears in both train and test (customers copy-pasting a viral
complaint, or the brand's own canned replies appearing near-verbatim),
that's still leakage. We flag it and report it, but do NOT silently drop
cross-split duplicates from the test set, because that's what a random
split would look like *if it had been leakage-checked* -- the honest move
is to measure and report the overlap, not to launder it away.
"""
from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime

_NORMALIZE_RE = re.compile(r"[^a-z0-9 ]+")
_DIGIT_RE = re.compile(r"\d+")


def parse_twitter_datetime(raw: str) -> datetime:
    return datetime.strptime(raw, "%a %b %d %H:%M:%S %z %Y")


def normalize_for_dedup(text: str) -> str:
    """Aggressive normalization used only for duplicate detection (not for
    modeling/display): lowercase, strip @mentions/URLs, collapse numbers
    (so 'Order #12345' and 'Order #98765' are treated as the same
    template), strip punctuation, collapse whitespace.
    """
    text = text.lower()
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = _DIGIT_RE.sub("0", text)
    text = _NORMALIZE_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def content_hash(normalized_text: str) -> str:
    return hashlib.sha1(normalized_text.encode("utf-8")).hexdigest()


@dataclass
class SplitAssignment:
    train_ids: list[str]
    dev_ids: list[str]
    test_ids: list[str]
    cutoff1: str
    cutoff2: str


def temporal_split(
    conversation_dates: dict[str, datetime],
    cutoff1: datetime,
    cutoff2: datetime,
) -> SplitAssignment:
    train, dev, test = [], [], []
    for conv_id, dt in conversation_dates.items():
        if dt < cutoff1:
            train.append(conv_id)
        elif dt < cutoff2:
            dev.append(conv_id)
        else:
            test.append(conv_id)
    return SplitAssignment(
        train_ids=sorted(train), dev_ids=sorted(dev), test_ids=sorted(test),
        cutoff1=cutoff1.isoformat(), cutoff2=cutoff2.isoformat(),
    )


def analyze_duplicates(texts_by_id: dict[str, str]) -> dict:
    """Return exact-duplicate groups and per-text normalized hash groups."""
    hash_to_ids: dict[str, list[str]] = defaultdict(list)
    for conv_id, text in texts_by_id.items():
        h = content_hash(normalize_for_dedup(text))
        hash_to_ids[h].append(conv_id)

    dup_groups = {h: ids for h, ids in hash_to_ids.items() if len(ids) > 1}
    total_in_dup_groups = sum(len(ids) for ids in dup_groups.values())
    return {
        "n_documents": len(texts_by_id),
        "n_unique_normalized_texts": len(hash_to_ids),
        "n_duplicate_groups": len(dup_groups),
        "n_documents_in_duplicate_groups": total_in_dup_groups,
        "duplicate_rate": round(total_in_dup_groups / max(len(texts_by_id), 1), 4),
        "dup_groups_sample": {h: ids[:5] for h, ids in list(dup_groups.items())[:10]},
    }


def cross_split_duplicate_overlap(
    hash_to_ids: dict[str, list[str]], split_of_id: dict[str, str]
) -> dict:
    """Among duplicate groups, how many contain ids from more than one split?

    This is the direct leakage measurement: a duplicate group that spans
    train and test means the model could have memorized that exact
    (templated) complaint during training and gets credit for
    "recognizing" it at test time rather than generalizing.
    """
    crossing_groups = 0
    crossing_pairs = 0
    for h, ids in hash_to_ids.items():
        if len(ids) < 2:
            continue
        splits_present = {split_of_id.get(i) for i in ids}
        splits_present.discard(None)
        if len(splits_present) > 1:
            crossing_groups += 1
            crossing_pairs += len(ids)
    return {
        "duplicate_groups_spanning_multiple_splits": crossing_groups,
        "documents_in_cross_split_duplicate_groups": crossing_pairs,
    }
