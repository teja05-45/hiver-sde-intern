#!/usr/bin/env python3
"""
Evaluate the retrieval system: Recall@1/3/5 and MRR.

Usage:
    python scripts/evaluate_retrieval.py --brand AmazonHelp

Relevance ground truth, stated plainly: we do not have hand-labeled
"is this specific retrieved historical case actually a good match"
judgments (that would require its own large annotation effort). As a
distant-supervision proxy, a retrieved case is treated as "relevant" if it
shares the query's (silver) intent label. This measures "does retrieval
surface same-topic historical cases," which is necessary but not
sufficient for "is this good evidence for generation" -- evidence quality
scoring (separate module) adds resolution-consistency and result-count
signals on top of raw retrieval relevance. This distinction is kept
explicit in reports/retrieval_metrics.json rather than presented as a
single number.

Temporal correctness: evaluating against the test split, the retrieval
index is built ONLY from train+dev (both strictly earlier in time) --
never from test itself. See app/retrieval/retriever.py docstring.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.retrieval.embeddings import TfidfEmbeddingProvider  # noqa: E402
from app.retrieval.retriever import HistoricalRetriever  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
REPORTS_DIR = REPO_ROOT / "reports"


def load_split(brand: str) -> dict[str, list[dict]]:
    path = PROCESSED_DIR / f"labeled_{brand}.jsonl"
    by_split: dict[str, list[dict]] = {"train": [], "dev": [], "test": []}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            split = r.get("split")
            if split in by_split:
                by_split[split].append(r)
    return by_split


def evaluate_retrieval(retriever: HistoricalRetriever, queries: list[dict], k_values: list[int]) -> dict:
    max_k = max(k_values)
    hits_at_k = {k: 0 for k in k_values}
    reciprocal_ranks = []
    evidence_counts = []
    top_similarities = []

    for q in queries:
        result = retriever.retrieve(q["root_message"], k=max_k)
        evidence_counts.append(result.num_cases)
        top_similarities.append(result.top_similarity)

        true_intent = q["intent"]
        rank_of_first_relevant = None
        for rank, case in enumerate(result.cases, start=1):
            if case.intent == true_intent:
                rank_of_first_relevant = rank
                break

        for k in k_values:
            if rank_of_first_relevant is not None and rank_of_first_relevant <= k:
                hits_at_k[k] += 1

        reciprocal_ranks.append(1.0 / rank_of_first_relevant if rank_of_first_relevant else 0.0)

    n = len(queries)
    return {
        "n_queries": n,
        "recall_at_k": {f"recall@{k}": round(hits_at_k[k] / n, 4) for k in k_values},
        "mrr": round(sum(reciprocal_ranks) / n, 4) if n else 0.0,
        "avg_evidence_count": round(sum(evidence_counts) / n, 2) if n else 0.0,
        "avg_top_similarity": round(sum(top_similarities) / n, 4) if n else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--brand", type=str, required=True)
    parser.add_argument("--k-values", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--max-eval-queries", type=int, default=3000,
                         help="cap test-set queries evaluated, for runtime; sampled deterministically")
    args = parser.parse_args()

    data = load_split(args.brand)

    print("Building retrieval index from train+dev (data strictly before the test period) ...")
    t0 = time.time()
    index_records = data["train"] + data["dev"]
    retriever = HistoricalRetriever(TfidfEmbeddingProvider()).build_index(index_records)
    print(f"Index built with {len(index_records)} historical cases in {time.time()-t0:.1f}s")

    test_queries = data["test"]
    if len(test_queries) > args.max_eval_queries:
        import random
        rng = random.Random(42)
        test_queries = rng.sample(test_queries, args.max_eval_queries)
    print(f"Evaluating on {len(test_queries)} test queries ...")

    t1 = time.time()
    metrics = evaluate_retrieval(retriever, test_queries, args.k_values)
    print(f"Retrieval evaluation done in {time.time()-t1:.1f}s")
    print(json.dumps(metrics, indent=2))

    report = {
        "brand": args.brand,
        "index_source_splits": ["train", "dev"],
        "index_size": len(index_records),
        "eval_split": "test",
        "relevance_proxy": "retrieved case shares query's silver intent label (see script docstring)",
        **metrics,
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "retrieval_metrics.json").write_text(json.dumps(report, indent=2))
    print("\nWrote reports/retrieval_metrics.json")


if __name__ == "__main__":
    main()
