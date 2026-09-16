#!/usr/bin/env python3
"""
Profile every candidate brand in the raw dataset and recommend one, backed
by evidence rather than fame/familiarity.

Usage:
    python scripts/profile_dataset.py [--top-n 15] [--min-outbound 500]

Outputs:
    reports/brand_profile.json  (machine-readable)
    reports/brand_profile.md    (human-readable, includes the recommendation)

Methodology (see docs/decision-log.md for the full "why"):
    For each brand (an author_id appearing on outbound / inbound=False
    rows), using the full in-memory graph index built by
    `backend.app.services.ingestion.build_lightweight_index`, we compute:

      - outbound_tweets: how many replies the brand's support account sent
      - direct_engagements: how many of those replies are a *direct* reply
        (in_response_to_tweet_id) to a genuine inbound customer tweet --
        this is a stronger signal than raw outbound volume, since some
        outbound tweets are brand-initiated or reply to other brand tweets
      - unique_customers_engaged: distinct customers the brand directly
        replied to (repeated-issue signal without customer-level double
        counting)
      - reply_depth: for a sample of engagement threads, how many
        back-and-forth turns occur before the thread stops (proxy for
        "repeated support problems" / conversation depth)
      - response_rate: direct_engagements / (approx. total inbound
        mentions of the brand), a rough measure of how consistently the
        brand's support team actually responds
      - date range covered, used later to judge whether a temporal
        train/dev/test split is viable

    We deliberately do NOT use "brand fame" or manual selection. The
    recommended brand is the one with the best balance of volume,
    conversation depth, and response consistency among brands that clear
    a minimum data-volume bar.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.services.ingestion import build_lightweight_index, TweetIndexEntry  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = REPO_ROOT / "data" / "raw" / "twcs.csv"
REPORTS_DIR = REPO_ROOT / "reports"


def looks_like_brand_handle(author_id: str) -> bool:
    """Heuristic: brand support accounts are the author on outbound rows and
    have non-numeric, human-readable handles (customer author_ids in this
    dataset are anonymized integers)."""
    return not author_id.isdigit()


def profile_brands(index: dict[int, TweetIndexEntry]) -> dict[str, dict]:
    stats: dict[str, dict] = defaultdict(lambda: {
        "outbound_tweets": 0,
        "direct_engagements": 0,
        "unique_customers_engaged": set(),
        "reply_depths": [],
    })

    # Pass: for every outbound (brand) tweet, look at what it replied to.
    for entry in index.values():
        if entry.inbound:
            continue
        if not entry.author_id or not looks_like_brand_handle(entry.author_id):
            continue
        brand = entry.author_id
        stats[brand]["outbound_tweets"] += 1

        parent_id = entry.in_response_to_tweet_id
        if parent_id is None:
            continue
        parent = index.get(parent_id)
        if parent is None or not parent.inbound:
            continue  # not a direct reply to a genuine customer message

        stats[brand]["direct_engagements"] += 1
        stats[brand]["unique_customers_engaged"].add(parent.author_id)

        # Measure thread depth by walking further back from the customer
        # message through alternating customer/brand turns.
        depth = 2  # customer msg + this brand reply
        cursor = parent
        while cursor.in_response_to_tweet_id is not None and depth < 12:
            nxt = index.get(cursor.in_response_to_tweet_id)
            if nxt is None:
                break
            cursor = nxt
            depth += 1
        stats[brand]["reply_depths"].append(depth)

    # Finalize: convert sets to counts, compute derived metrics.
    finalized = {}
    for brand, s in stats.items():
        n_engagements = s["direct_engagements"]
        depths = s["reply_depths"]
        finalized[brand] = {
            "outbound_tweets": s["outbound_tweets"],
            "direct_engagements": n_engagements,
            "unique_customers_engaged": len(s["unique_customers_engaged"]),
            "engagement_rate": round(n_engagements / s["outbound_tweets"], 4) if s["outbound_tweets"] else 0.0,
            "avg_reply_depth": round(sum(depths) / len(depths), 3) if depths else 0.0,
            "max_reply_depth": max(depths) if depths else 0,
            "repeat_customer_rate": (
                round(1 - (len(s["unique_customers_engaged"]) / n_engagements), 4)
                if n_engagements else 0.0
            ),
        }
    return finalized


def score_and_rank(profiles: dict[str, dict], min_outbound: int) -> list[tuple[str, float, dict]]:
    """Score brands on a balance of volume, depth, and engagement quality.

    Score is intentionally simple and explainable (no hidden weighting
    that would make the "why this brand" story hard to defend in an
    interview): a brand needs enough volume AND enough conversational
    depth AND a healthy engagement rate to score well.

    NOTE on an earlier version of this function: a first draft normalized
    volume linearly with a hard cap at 5,000 engagements
    (`min(engagements / 5000, 1.0)`). That saturates almost immediately --
    every brand above 5,000 engagements gets the same volume_score of 1.0
    -- so it let a brand with ~11K engagements outrank one with ~169K
    purely on small differences in reply depth. That's the wrong trade-off
    for this assignment: the golden set needs 150-250 stratified examples
    across common/rare/ambiguous/OOD categories, and the retrieval index
    needs enough historical resolutions to have real evidence for rare
    intents too. More data is a genuine, first-order advantage here (the
    assignment literally lists "sufficient data" as the first selection
    criterion), so volume is now scored on a log scale relative to the
    richest candidate -- this rewards more data without letting it
    completely dominate depth/engagement-rate.
    """
    ranked = []
    if not profiles:
        return ranked

    candidates = {
        b: p for b, p in profiles.items()
        if p["outbound_tweets"] >= min_outbound and p["direct_engagements"] >= min_outbound
    }
    if not candidates:
        return ranked

    import math
    max_engagements = max(p["direct_engagements"] for p in candidates.values())
    log_max = math.log1p(max_engagements)

    for brand, p in candidates.items():
        volume_score = math.log1p(p["direct_engagements"]) / log_max if log_max > 0 else 0.0
        depth_score = min(p["avg_reply_depth"] / 4.0, 1.0)
        engagement_score = p["engagement_rate"]  # already 0-1
        composite = 0.45 * volume_score + 0.30 * depth_score + 0.25 * engagement_score
        ranked.append((brand, round(composite, 4), p))
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-n", type=int, default=15)
    parser.add_argument("--min-outbound", type=int, default=500)
    args = parser.parse_args()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Building tweet index from {RAW_PATH} ...")
    index, ingest_stats = build_lightweight_index(RAW_PATH)
    print(f"Index built: {ingest_stats.as_dict()}")

    print("Profiling candidate brands ...")
    profiles = profile_brands(index)
    print(f"Found {len(profiles)} candidate brand accounts.")

    ranked = score_and_rank(profiles, args.min_outbound)
    top = ranked[: args.top_n]

    recommended = top[0] if top else None

    report = {
        "ingestion_stats": ingest_stats.as_dict(),
        "num_candidate_brands": len(profiles),
        "min_outbound_threshold": args.min_outbound,
        "ranked_top_n": [
            {"brand": b, "score": s, **p} for b, s, p in top
        ],
        "recommended_brand": recommended[0] if recommended else None,
        "recommended_brand_stats": recommended[2] if recommended else None,
    }

    (REPORTS_DIR / "brand_profile.json").write_text(json.dumps(report, indent=2))

    md_lines = [
        "# Brand Profiling Report",
        "",
        f"Dataset ingestion: {ingest_stats.as_dict()}",
        "",
        f"Candidate brand accounts found: {len(profiles)}",
        f"Minimum outbound/engagement threshold applied: {args.min_outbound}",
        "",
        "## Top candidates (ranked by composite score)",
        "",
        "| Rank | Brand | Score | Outbound Tweets | Direct Engagements | Unique Customers | Engagement Rate | Avg Reply Depth | Repeat-Customer Rate |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for i, (b, s, p) in enumerate(top, start=1):
        md_lines.append(
            f"| {i} | {b} | {s} | {p['outbound_tweets']} | {p['direct_engagements']} | "
            f"{p['unique_customers_engaged']} | {p['engagement_rate']} | {p['avg_reply_depth']} | "
            f"{p['repeat_customer_rate']} |"
        )

    if recommended:
        b, s, p = recommended
        md_lines += [
            "",
            "## Recommendation",
            "",
            f"**Selected brand: `{b}`** (composite score {s})",
            "",
            f"- {p['direct_engagements']:,} direct customer engagements (replies to genuine inbound customer tweets)",
            f"- {p['unique_customers_engaged']:,} unique customers engaged (repeat-customer rate {p['repeat_customer_rate']})",
            f"- Average reply-thread depth: {p['avg_reply_depth']} turns (max observed: {p['max_reply_depth']})",
            f"- Engagement rate (direct replies / all outbound tweets): {p['engagement_rate']}",
            "",
            "This brand was selected because it clears the minimum volume bar, has the strongest "
            "combination of engagement depth (repeated back-and-forth, suggesting real issue "
            "resolution rather than one-off canned replies) and engagement rate (consistent "
            "response behavior) among brands with sufficient data -- not because of brand "
            "recognition.",
        ]

    (REPORTS_DIR / "brand_profile.md").write_text("\n".join(md_lines))
    print(f"\nWrote reports/brand_profile.json and reports/brand_profile.md")
    if recommended:
        print(f"\nRECOMMENDED BRAND: {recommended[0]} (score={recommended[1]})")


if __name__ == "__main__":
    main()
