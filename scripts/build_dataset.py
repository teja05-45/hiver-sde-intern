#!/usr/bin/env python3
"""
Build the reproducible, brand-filtered, reconstructed-conversation dataset.

Usage:
    python scripts/build_dataset.py --brand AmazonHelp --sample-size 60000 --seed 42

Reads the full raw twcs.csv (streamed, not fully loaded), reconstructs
conversation threads for a deterministic sample of the selected brand's
customer engagements, and writes:

    data/processed/conversations_<brand>.jsonl
    reports/dataset_build_report.json

This is the "reproducible subsample" step explicitly permitted (and
expected) by the assignment: "Full dataset processing is NOT required.
Use a reproducible subsample."
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.services.ingestion import build_lightweight_index  # noqa: E402
from app.services.conversation import reconstruct_conversations, save_conversations_jsonl  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("build_dataset")

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = REPO_ROOT / "data" / "raw" / "twcs.csv"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
REPORTS_DIR = REPO_ROOT / "reports"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--brand", type=str, required=True)
    parser.add_argument("--sample-size", type=int, default=60000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    t0 = time.time()
    logger.info("Building tweet index from %s", RAW_PATH)
    index, ingest_stats = build_lightweight_index(RAW_PATH)
    logger.info("Index ready in %.1fs: %s", time.time() - t0, ingest_stats.as_dict())

    t1 = time.time()
    conversations = reconstruct_conversations(
        index=index,
        raw_path=RAW_PATH,
        brand=args.brand,
        sample_size=args.sample_size,
        seed=args.seed,
    )
    logger.info("Reconstructed %d conversations in %.1fs", len(conversations), time.time() - t1)

    out_path = PROCESSED_DIR / f"conversations_{args.brand}.jsonl"
    save_conversations_jsonl(conversations, out_path)
    logger.info("Wrote %s", out_path)

    lengths = [len(c.messages) for c in conversations]
    n_with_agent_reply = sum(1 for c in conversations if c.resolution)
    report = {
        "brand": args.brand,
        "sample_size_requested": args.sample_size,
        "seed": args.seed,
        "conversations_written": len(conversations),
        "avg_messages_per_conversation": round(sum(lengths) / len(lengths), 3) if lengths else 0,
        "min_messages": min(lengths) if lengths else 0,
        "max_messages": max(lengths) if lengths else 0,
        "conversations_with_resolution": n_with_agent_reply,
        "ingestion_stats": ingest_stats.as_dict(),
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "dataset_build_report.json").write_text(json.dumps(report, indent=2))
    logger.info("Report: %s", report)


if __name__ == "__main__":
    main()
