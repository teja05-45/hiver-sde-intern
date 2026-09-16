#!/usr/bin/env python3
"""
Sweep hybrid retrieval weights against the intent-proxy relevance metrics.

Why this script exists: the hybrid score's weights are engineering choices
with stated rationale, not God-given constants. The first full evaluation
(w_intent=0.30) measured +7.1pp Recall@1 but -4.8pp Recall@5 / -7.8pp
Recall@10 vs the baseline cosine ranking: the intent channel over-
homogenizes the window when the classifier's argmax is wrong, suppressing
true-intent cases deeper in the pool. This script measures that tradeoff
across a small weight grid on the dev split (dev is strictly earlier than
test, so tuning here does not leak into the reported test numbers), and the
chosen operating point is written to models/retrieval_tuning.json with the
full measured grid for auditability.

Usage:
    python scripts/sweep_retrieval_weights.py --brand AmazonHelp
        [--grid 0.05 0.1 0.15 0.2 0.3] [--max-eval-queries 1500]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.retrieval.embeddings import TfidfEmbeddingProvider  # noqa: E402
from app.retrieval.lexical import BM25Index  # noqa: E402
from app.retrieval.quality import (  # noqa: E402
    HybridRetrievalConfig,
    resolution_quality,
    score_candidates,
)
from app.retrieval.retriever import HistoricalRetriever  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
MODELS_DIR = REPO_ROOT / "models"


def load_split(brand: str) -> dict[str, list[dict]]:
    by_split: dict[str, list[dict]] = {"train": [], "dev": [], "test": []}
    with (PROCESSED_DIR / f"labeled_{brand}.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("split") in by_split:
                by_split[r["split"]].append(r)
    return by_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--brand", type=str, required=True)
    parser.add_argument("--grid", type=float, nargs="+",
                        default=[0.05, 0.10, 0.15, 0.20, 0.30],
                        help="w_intent values to sweep; lexical/quality/penalty stay at defaults")
    parser.add_argument("--max-eval-queries", type=int, default=1500)
    parser.add_argument("--k", type=int, default=5, help="evidence window the summary is computed over")
    args = parser.parse_args()

    data = load_split(args.brand)
    index_records = data["train"]  # tune on train-built index, evaluate on dev: no split leakage
    dev = data["dev"]
    if len(dev) > args.max_eval_queries:
        import random
        rng = random.Random(42)
        dev = rng.sample(dev, args.max_eval_queries)

    print(f"Building tuning index from train ({len(index_records)} cases) ...")
    retriever = HistoricalRetriever(TfidfEmbeddingProvider()).build_index(index_records)
    texts = [ (r["root_message"] or "").lower() for r in index_records ]  # noqa: E501  (cleaning parity with retriever)
    from app.services.text_cleaning import clean_for_modeling
    texts = [clean_for_modeling(r["root_message"]) for r in index_records]
    bm25 = BM25Index().fit(texts)
    quality_by_id = {r["conversation_id"]: resolution_quality(r.get("resolution")) for r in index_records}

    classifier = None
    try:
        import joblib
        clf_path = MODELS_DIR / f"classifier_{args.brand}.joblib"
        if clf_path.exists():
            classifier = joblib.load(clf_path)
            print("Loaded classifier for intent-compatibility channel.")
    except Exception as e:  # noqa: BLE001
        print(f"Classifier unavailable ({e}); intent channel will be 0 for all cases.")

    pool_k = max(20, min(4 * args.k, 100))
    print(f"Sweeping w_intent over {args.grid} on {len(dev)} dev queries "
          f"(window k={args.k}, pool={pool_k}) ...")

    # Precompute everything that does NOT depend on w_intent, once per
    # query: the candidate pool (vector search), the classifier intent
    # distribution, and BM25 raw scores. Each grid point then only runs
    # the cheap scoring pass. (First version re-ran retrieval + classify
    # per grid point and timed out at 10 minutes.)
    from app.services.text_cleaning import clean_for_modeling as _cfm
    per_query: list[tuple[list, dict | None, str]] = []
    t_pre = time.time()
    for q in dev:
        res = retriever.retrieve(q["root_message"], k=args.k)
        pool = res.baseline_cases[:pool_k]
        probs = None
        if classifier is not None:
            probs = classifier.predict([_cfm(q["root_message"])])[0].all_scores
        per_query.append((pool, probs, q["intent"]))
    print(f"Precomputed pools + intent probs in {time.time()-t_pre:.1f}s.")

    results = []
    for w_intent in args.grid:
        cfg = HybridRetrievalConfig(w_intent=w_intent)
        t0 = time.time()
        hits = {1: 0, 5: 0}
        rr_sum = 0.0
        res_supported = 0
        quality_sum = 0.0
        for (pool, probs, true_intent), q in zip(per_query, dev):
            scored = score_candidates(pool, q["root_message"], probs, bm25,
                                      quality_by_id, cfg)
            window = scored[:args.k]
            rank = next((r for r, s in enumerate(window, 1) if s.case.intent == true_intent), None)
            if rank == 1:
                hits[1] += 1
            if rank is not None and rank <= 5:
                hits[5] += 1
            rr_sum += 1.0 / rank if rank else 0.0
            if any(s.components["resolution_quality"] >= 0.5 for s in window):
                res_supported += 1
            quality_sum += sum(s.components["resolution_quality"] for s in window) / max(len(window), 1)
        n = len(dev)
        row = {
            "w_intent": w_intent,
            "recall_at_1": round(hits[1] / n, 4),
            "recall_at_5": round(hits[5] / n, 4),
            "mrr": round(rr_sum / n, 4),
            "resolution_supported_rate": round(res_supported / n, 4),
            "mean_resolution_quality": round(quality_sum / n, 4),
            "seconds": round(time.time() - t0, 1),
        }
        results.append(row)
        print(json.dumps(row))

    # Selection rule, fixed BEFORE looking at the results and stated here:
    # maximize Recall@1 subject to Recall@5 not dropping more than 1pp below
    # the best Recall@5 in the grid (top-1 relevance is the primary goal --
    # the evidence window leads with the strongest case -- but a window that
    # loses deep recall starves generation of corroborating history).
    best_r5 = max(r["recall_at_5"] for r in results)
    eligible = [r for r in results if r["recall_at_5"] >= best_r5 - 0.01]
    chosen = max(eligible, key=lambda r: (r["recall_at_1"], r["mrr"]))
    print(f"\nChosen operating point: w_intent={chosen['w_intent']} "
          f"(Recall@1 {chosen['recall_at_1']}, Recall@5 {chosen['recall_at_5']}, "
          f"MRR {chosen['mrr']}) by the selection rule stated in the script docstring.")

    out = {
        "brand": args.brand,
        "tuned_on": "dev (index built from train only -- no test leakage)",
        "n_queries": len(dev),
        "k": args.k,
        "selection_rule": ("maximize recall@1 subject to recall@5 within 1pp "
                           "of the grid best"),
        "grid": results,
        "chosen": chosen,
        "applied_defaults": {
            "RETRIEVAL_W_INTENT": chosen["w_intent"],
            "note": "set as env var or HybridRetrievalConfig default at build time",
        },
    }
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (MODELS_DIR / "retrieval_tuning.json").write_text(json.dumps(out, indent=2))
    print(f"Wrote {MODELS_DIR / 'retrieval_tuning.json'}")


if __name__ == "__main__":
    main()
