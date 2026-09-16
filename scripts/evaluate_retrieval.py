#!/usr/bin/env python3
"""
Evaluate retrieval quality: baseline (TF-IDF cosine) vs hybrid reranking.

Usage:
    python scripts/evaluate_retrieval.py --brand AmazonHelp
    python scripts/evaluate_retrieval.py --brand AmazonHelp --max-eval-queries 300

Relevance ground truth, stated plainly: we do not have hand-labeled
"is this specific retrieved historical case actually a good match"
judgments (that would require its own large annotation effort). As a
distant-supervision proxy, a retrieved case is treated as "relevant" if it
shares the query's (silver) intent label. Every metric below is therefore a
PROXY metric: it measures "does retrieval surface same-topic historical
cases," which is necessary but not sufficient for "is this good evidence for
generation." Nothing here is human-verified, and the report says so.

Temporal correctness: evaluating against the test split, the retrieval
index is built ONLY from train+dev (both strictly earlier in time) --
never from test itself. See app/retrieval/retriever.py docstring.

What is measured (per ranking, baseline and hybrid):
  - Recall@1/5/10 and MRR under the intent-proxy relevance definition
  - intent-consistent retrieval rate: fraction of queries whose TOP case
    shares the query intent (the cleanest single relevance number)
  - resolution-supported retrieval rate: fraction of queries whose top-k
    window contains >= 1 case with a usable resolution (quality >= 0.5) --
    retrieval is only useful if the window can actually support a draft
  - mean evidence quality: mean resolution_quality over the window
  - window size and top similarity

The two rankings are computed for the SAME queries from the SAME index
(the retriever returns both orders per query), so the delta is a paired
comparison, not two separate runs.
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
from app.services.text_cleaning import clean_for_modeling  # noqa: E402

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


def evaluate_ranking(cases_by_query: list[list], true_intents: list[str], k_values: list[int]) -> dict:
    """Compute the metric block for one ranking order.

    `cases_by_query[i]` is the ordered case list for query i (any object with
    .intent and .similarity, plus .resolution when available).
    """
    max_k = max(k_values)
    n = len(cases_by_query)
    hits_at_k = {k: 0 for k in k_values}
    reciprocal_ranks: list[float] = []
    top_intent_match = 0
    resolution_supported = 0
    quality_sums: list[float] = []
    top_sims: list[float] = []

    for cases, true_intent in zip(cases_by_query, true_intents):
        window = cases[:max_k]
        rank_of_first_relevant = None
        for rank, case in enumerate(window, start=1):
            if case.intent == true_intent:
                rank_of_first_relevant = rank
                break
        for k in k_values:
            if rank_of_first_relevant is not None and rank_of_first_relevant <= k:
                hits_at_k[k] += 1
        reciprocal_ranks.append(1.0 / rank_of_first_relevant if rank_of_first_relevant else 0.0)

        if window and window[0].intent == true_intent:
            top_intent_match += 1
        if any(resolution_quality(getattr(c, "resolution", None)) >= 0.5 for c in window):
            resolution_supported += 1
        if window:
            quality_sums.append(sum(resolution_quality(getattr(c, "resolution", None)) for c in window) / len(window))
            top_sims.append(window[0].similarity)
        else:
            quality_sums.append(0.0)
            top_sims.append(0.0)

    return {
        "n_queries": n,
        "recall_at_k": {f"recall@{k}": round(hits_at_k[k] / n, 4) for k in k_values},
        "mrr": round(sum(reciprocal_ranks) / n, 4) if n else 0.0,
        "intent_consistent_rate": round(top_intent_match / n, 4) if n else 0.0,
        "resolution_supported_rate": round(resolution_supported / n, 4) if n else 0.0,
        "mean_resolution_quality": round(sum(quality_sums) / n, 4) if n else 0.0,
        "mean_top_similarity": round(sum(top_sims) / n, 4) if n else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--brand", type=str, required=True)
    parser.add_argument("--k-values", type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument("--max-eval-queries", type=int, default=3000,
                        help="cap test-set queries evaluated, for runtime; sampled deterministically")
    args = parser.parse_args()

    data = load_split(args.brand)

    print("Building retrieval index from train+dev (data strictly before the test period) ...")
    t0 = time.time()
    index_records = data["train"] + data["dev"]
    retriever = HistoricalRetriever(TfidfEmbeddingProvider()).build_index(index_records)
    print(f"Index built with {len(index_records)} historical cases in {time.time()-t0:.1f}s")

    # The intent-compatibility channel needs the same classifier the deployed
    # agent uses. (The first version of this A/B forgot to pass intent probs,
    # which silently measured 'hybrid without its intent channel' and
    # mislabeled the result -- the report now records whether the channel
    # was actually active.)
    classifier = None
    clf_path = REPO_ROOT / "models" / f"classifier_{args.brand}.joblib"
    if clf_path.exists():
        import joblib
        classifier = joblib.load(clf_path)
        print(f"Loaded classifier for intent-compatibility channel: {clf_path.name}")
    else:
        print("WARNING: classifier artifact not found; intent channel will be 0 "
              "(this measures lexical+quality reranking only).")

    test_queries = data["test"]
    if len(test_queries) > args.max_eval_queries:
        import random
        rng = random.Random(42)
        test_queries = rng.sample(test_queries, args.max_eval_queries)
    n_queries = len(test_queries)
    print(f"Evaluating on {n_queries} test queries (intent-proxy relevance; see docstring) ...")

    max_k = max(args.k_values)
    baseline_windows: list[list] = []
    hybrid_windows: list[list] = []
    # Ablation: hybrid scoring WITHOUT the intent channel. This separates the
    # two effects the full hybrid mixes: lexical+quality reranking (pure
    # retrieval-side signals) vs classifier-steered reranking. It matters
    # because the relevance proxy is the silver intent label and the
    # classifier was trained on silver labels -- with the intent channel
    # active, intent-consistency partially measures classifier/silver
    # agreement (93.6% silver accuracy), so the headline Recall@1 delta is
    # NOT a clean measure of retrieval relevance. The ablation is the
    # circularity-free view of what reranking alone contributes.
    ablation_windows: list[list] = []
    true_intents = [q["intent"] for q in test_queries]

    texts_index = [clean_for_modeling(r["root_message"]) for r in index_records]
    bm25 = BM25Index().fit(texts_index)
    quality_by_id = {r["conversation_id"]: resolution_quality(r.get("resolution"))
                     for r in index_records}

    t1 = time.time()
    for i, q in enumerate(test_queries):
        probs = None
        if classifier is not None:
            probs = classifier.predict([clean_for_modeling(q["root_message"])])[0].all_scores
        result = retriever.retrieve(q["root_message"], k=max_k, intent_probs=probs)
        # baseline_cases = pre-rerank cosine order (the previous system's
        # ranking); cases = hybrid reranked order. Same index, same query.
        baseline_windows.append(result.baseline_cases[:max_k])
        hybrid_windows.append(result.cases if result.hybrid_enabled else result.baseline_cases[:max_k])
        # Ablation window: same pool, intent channel zeroed.
        pool = result.baseline_cases[:len(result.baseline_cases)]
        scored_abl = score_candidates(pool, q["root_message"], None, bm25,
                                      quality_by_id, HybridRetrievalConfig())
        ablation_windows.append([s.case for s in scored_abl[:max_k]])
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{n_queries} queries scored in {time.time()-t1:.1f}s")

    print(f"Retrieval evaluation done in {time.time()-t1:.1f}s")

    baseline_metrics = evaluate_ranking(baseline_windows, true_intents, args.k_values)
    hybrid_metrics = evaluate_ranking(hybrid_windows, true_intents, args.k_values)
    ablation_metrics = evaluate_ranking(ablation_windows, true_intents, args.k_values)

    print("\nBASELINE (TF-IDF cosine):")
    print(json.dumps(baseline_metrics, indent=2))
    print("\nHYBRID (reranked):")
    print(json.dumps(hybrid_metrics, indent=2))

    print("\nABLATION (hybrid WITHOUT intent channel):")
    print(json.dumps(ablation_metrics, indent=2))

    report = {
        "brand": args.brand,
        "index_source_splits": ["train", "dev"],
        "index_size": len(index_records),
        "eval_split": "test",
        "relevance_proxy": (
            "retrieved case shares query's silver intent label. PROXY/SILVER "
            "evaluation derived from cluster labels -- NOT human-verified "
            "relevance (see script docstring)"
        ),
        "baseline": {"description": "plain TF-IDF cosine ranking (pre-hybrid system)", **baseline_metrics},
        "hybrid": {"description": (
            "reranked: semantic + lexical(BM25) + intent-compatibility "
            "(classifier p) + resolution-quality - contradiction penalty"
        ), "intent_channel_active": classifier is not None, **hybrid_metrics},
        "ablations": {
            "hybrid_without_intent_channel": {
                "description": (
                    "same hybrid scoring with intent compatibility forced to 0: "
                    "isolates the lexical+quality contribution from classifier steering"
                ),
                **ablation_metrics,
            },
        },
        "caveats": [
            (
                "PROXY CIRCULARITY: relevance is defined as sharing the query's silver "
                "intent label; the classifier was trained on those same silver labels; "
                "the hybrid's intent channel steers retrieval toward the classifier's "
                "argmax. The baseline->hybrid Recall@1 delta therefore overstates "
                "independent retrieval relevance -- read it as 'intent-aligned reranking "
                "works as designed against the silver proxy,' not as a human-verified "
                "improvement. The ablation row is the circularity-free view."
            ),
            (
                "Resolution-supported rate saturates at 1.0 because the index only "
                "contains conversations with recorded resolutions; the meaningful "
                "number is mean_resolution_quality (usable vs placeholder resolutions)."
            ),
        ],
        "delta": {
            f"recall@{k}": round(hybrid_metrics["recall_at_k"][f"recall@{k}"]
                                  - baseline_metrics["recall_at_k"][f"recall@{k}"], 4)
            for k in args.k_values
        } | {
            "mrr": round(hybrid_metrics["mrr"] - baseline_metrics["mrr"], 4),
            "intent_consistent_rate": round(hybrid_metrics["intent_consistent_rate"]
                                             - baseline_metrics["intent_consistent_rate"], 4),
            "resolution_supported_rate": round(hybrid_metrics["resolution_supported_rate"]
                                                - baseline_metrics["resolution_supported_rate"], 4),
            "mean_resolution_quality": round(hybrid_metrics["mean_resolution_quality"]
                                              - baseline_metrics["mean_resolution_quality"], 4),
        },
        "recall_at_k": hybrid_metrics["recall_at_k"],  # backward-compat for UI readers
        "mrr": hybrid_metrics["mrr"],
        "avg_evidence_count": max_k,
        "avg_top_similarity": hybrid_metrics["mean_top_similarity"],
        "n_queries": n_queries,
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "retrieval_metrics.json").write_text(json.dumps(report, indent=2))
    print("\nWrote reports/retrieval_metrics.json")


if __name__ == "__main__":
    main()
