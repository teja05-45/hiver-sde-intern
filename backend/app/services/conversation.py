"""
Conversation reconstruction for a single brand.

Given the lightweight graph index (see `ingestion.py`) and a target brand,
this module:

1. Finds "engagement" edges: brand tweets that are a direct reply to a
   genuine inbound customer tweet.
2. Deterministically samples a reproducible subset of those engagements
   (full-dataset processing is explicitly not required by the assignment;
   a documented, seeded subsample is).
3. Walks each sampled engagement backward through `in_response_to_tweet_id`
   to recover the full thread (capped at a max depth to bound cost/noise
   from occasional very long threads), then re-fetches full text/timestamps
   for exactly the tweet ids involved (and only those) in one streaming
   pass over the raw file.
4. Produces a normalized `Conversation` object matching the schema
   requested in the assignment: conversation_id, brand, messages,
   customer_messages, agent_messages, timestamps, resolution.

Limitation, documented rather than hidden: a customer thread can in
principle branch (a customer tweet can receive multiple distinct brand
replies, e.g. from different agents). We reconstruct each engagement as its
own linear chain rooted at that specific agent reply, which means a
customer message involved in multiple engagements appears in multiple
Conversation objects. This over-counts total conversations, but does not
create duplicate agent replies, and preserves the "customer message x
successful historical resolution" pairing that retrieval/evidence-scoring
downstream actually depends on. See docs/decision-log.md.
"""
from __future__ import annotations

import json
import logging
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from app.services.ingestion import TweetIndexEntry, iter_rows_for_tweet_ids

logger = logging.getLogger(__name__)

MAX_THREAD_DEPTH = 12


@dataclass
class Message:
    tweet_id: int
    role: str  # "customer" | "agent"
    text: str
    created_at: str


@dataclass
class Conversation:
    conversation_id: str
    brand: str
    messages: list[Message] = field(default_factory=list)

    @property
    def customer_messages(self) -> list[str]:
        return [m.text for m in self.messages if m.role == "customer"]

    @property
    def agent_messages(self) -> list[str]:
        return [m.text for m in self.messages if m.role == "agent"]

    @property
    def timestamps(self) -> list[str]:
        return [m.created_at for m in self.messages]

    @property
    def resolution(self) -> Optional[str]:
        agent_msgs = self.agent_messages
        return agent_msgs[-1] if agent_msgs else None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["customer_messages"] = self.customer_messages
        d["agent_messages"] = self.agent_messages
        d["timestamps"] = self.timestamps
        d["resolution"] = self.resolution
        return d


def find_brand_engagement_tweet_ids(index: dict[int, TweetIndexEntry], brand: str) -> list[int]:
    """Return agent-reply tweet_ids that directly reply to a genuine inbound customer tweet."""
    engagement_ids = []
    for entry in index.values():
        if entry.inbound or entry.author_id != brand:
            continue
        parent_id = entry.in_response_to_tweet_id
        if parent_id is None:
            continue
        parent = index.get(parent_id)
        if parent is not None and parent.inbound:
            engagement_ids.append(entry.tweet_id)
    engagement_ids.sort()  # deterministic ordering before sampling
    return engagement_ids


def sample_engagement_ids(engagement_ids: list[int], sample_size: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    if sample_size >= len(engagement_ids):
        return list(engagement_ids)
    return sorted(rng.sample(engagement_ids, sample_size))


def build_thread_tweet_id_chain(index: dict[int, TweetIndexEntry], agent_tweet_id: int) -> list[int]:
    """Walk backward from an agent reply through `in_response_to_tweet_id`, return ids oldest->newest."""
    chain: list[int] = [agent_tweet_id]
    cursor = index.get(agent_tweet_id)
    depth = 1
    while cursor is not None and cursor.in_response_to_tweet_id is not None and depth < MAX_THREAD_DEPTH:
        parent_id = cursor.in_response_to_tweet_id
        parent = index.get(parent_id)
        if parent is None:
            break
        chain.append(parent_id)
        cursor = parent
        depth += 1
    chain.reverse()  # oldest first
    return chain


def reconstruct_conversations(
    index: dict[int, TweetIndexEntry],
    raw_path: Path,
    brand: str,
    sample_size: int,
    seed: int,
) -> list[Conversation]:
    all_engagements = find_brand_engagement_tweet_ids(index, brand)
    logger.info("Found %d total engagements for brand=%s", len(all_engagements), brand)

    sampled = sample_engagement_ids(all_engagements, sample_size, seed)
    logger.info("Sampled %d engagements (seed=%d)", len(sampled), seed)

    # Build the tweet_id chain for every sampled engagement, and the union
    # of all tweet ids we'll need full text for.
    chains: dict[int, list[int]] = {}
    needed_ids: set[int] = set()
    for agent_tweet_id in sampled:
        chain = build_thread_tweet_id_chain(index, agent_tweet_id)
        chains[agent_tweet_id] = chain
        needed_ids.update(chain)

    # Single streaming pass to fetch text/timestamps for exactly the tweet
    # ids we need (not the whole file, and not held in memory for the
    # 2.8M-row dataset as a whole).
    text_by_id: dict[int, dict] = {}
    for batch in iter_rows_for_tweet_ids(raw_path, needed_ids):
        for row in batch:
            tid = int(row["tweet_id"])
            text_by_id[tid] = {"text": row["text"], "created_at": row["created_at"]}

    conversations: list[Conversation] = []
    for agent_tweet_id, chain in chains.items():
        messages: list[Message] = []
        for tid in chain:
            entry = index.get(tid)
            row_text = text_by_id.get(tid)
            if entry is None or row_text is None:
                continue
            role = "agent" if (not entry.inbound and entry.author_id == brand) else "customer"
            messages.append(Message(tweet_id=tid, role=role, text=row_text["text"], created_at=row_text["created_at"]))
        if not messages:
            continue
        conversations.append(
            Conversation(conversation_id=f"{brand}_{agent_tweet_id}", brand=brand, messages=messages)
        )

    logger.info("Reconstructed %d conversations for brand=%s", len(conversations), brand)
    return conversations


def save_conversations_jsonl(conversations: list[Conversation], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for c in conversations:
            f.write(json.dumps(c.to_dict()) + "\n")


def load_conversations_jsonl(path: Path) -> list[dict]:
    conversations = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                conversations.append(json.loads(line))
    return conversations
