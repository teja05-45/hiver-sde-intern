#!/usr/bin/env python3
"""
Discover candidate intents from the selected brand's real customer messages.

Usage:
    python scripts/discover_intents.py --brand AmazonHelp --k-min 8 --k-max 16

Pipeline (see docs/decision-log.md for "why TF-IDF instead of neural
embeddings"): this sandbox has no network access to `pip install
sentence-transformers`, so clustering uses TF-IDF + TruncatedSVD (LSA)
instead of dense sentence embeddings. The `EmbeddingProvider` interface in
`backend/app/classification/embeddings.py` isolates this choice so
swapping in a real embedding model later is a one-line change, not a
rewrite.

Steps:
    1. Load reconstructed conversations, filter to (heuristically) English
       root customer messages.
    2. Clean text, vectorize with TF-IDF.
    3. Reduce dimensionality with TruncatedSVD (LSA).
    4. Cluster with MiniBatchKMeans across a range of k; pick k by
       silhouette score.
    5. Print top terms and sample messages per cluster to
       reports/intent_clusters.md for human inspection/naming.

This script does NOT write the final configs/intents.yaml -- that step
requires a human (me, acting as the labeler) to read the cluster
inspection output and assign meaningful names/descriptions, which is done
as a separate, explicit step per the assignment's "human naming" and
"merge/split" requirements.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.services.conversation import load_conversations_jsonl  # noqa: E402
from app.services.text_cleaning import clean_for_modeling  # noqa: E402
from app.services.lang_id import is_english  # noqa: E402

import numpy as np  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.decomposition import TruncatedSVD  # noqa: E402
from sklearn.cluster import MiniBatchKMeans  # noqa: E402
from sklearn.metrics import silhouette_score  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
REPORTS_DIR = REPO_ROOT / "reports"


def load_root_customer_messages(brand: str) -> tuple[list[str], list[str], list[int], list[str]]:
    """Returns (raw_root_messages, cleaned_root_messages, conversation_indices, conversation_ids)."""
    path = PROCESSED_DIR / f"conversations_{brand}.jsonl"
    conversations = load_conversations_jsonl(path)

    raw, cleaned, idx, conv_ids = [], [], [], []
    n_filtered_non_english = 0
    n_filtered_too_short = 0
    for i, c in enumerate(conversations):
        cust_msgs = [m["text"] for m in c["messages"] if m["role"] == "customer"]
        if not cust_msgs:
            continue
        root = cust_msgs[0]
        if not is_english(root):
            n_filtered_non_english += 1
            continue
        cleaned_text = clean_for_modeling(root)
        if len(cleaned_text.split()) < 3:
            n_filtered_too_short += 1
            continue
        raw.append(root)
        cleaned.append(cleaned_text)
        idx.append(i)
        conv_ids.append(c["conversation_id"])

    print(f"Loaded {len(conversations)} conversations.")
    print(f"Filtered {n_filtered_non_english} likely non-English root messages.")
    print(f"Filtered {n_filtered_too_short} too-short (<3 token) root messages.")
    print(f"Remaining for clustering: {len(cleaned)}")
    return raw, cleaned, idx, conv_ids


def pick_k_by_silhouette(X, k_min: int, k_max: int, sample_size: int, seed: int) -> tuple[int, dict]:
    scores = {}
    rng = np.random.RandomState(seed)
    n = X.shape[0]
    sample_idx = rng.choice(n, size=min(sample_size, n), replace=False)
    X_sample = X[sample_idx]

    for k in range(k_min, k_max + 1):
        km = MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=5, batch_size=1024)
        labels_full = km.fit_predict(X)
        labels_sample = labels_full[sample_idx]
        # silhouette needs >1 cluster present in the sample
        if len(set(labels_sample)) < 2:
            scores[k] = -1.0
            continue
        score = silhouette_score(X_sample, labels_sample, metric="euclidean")
        scores[k] = round(float(score), 4)
        print(f"  k={k}: silhouette={scores[k]}")

    best_k = max(scores, key=scores.get)
    return best_k, scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--brand", type=str, required=True)
    parser.add_argument("--k-min", type=int, default=8)
    parser.add_argument("--k-max", type=int, default=16)
    parser.add_argument("--svd-components", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--silhouette-sample-size", type=int, default=8000)
    parser.add_argument("--top-terms", type=int, default=12)
    parser.add_argument("--samples-per-cluster", type=int, default=8)
    args = parser.parse_args()

    raw, cleaned, _idx, conv_ids = load_root_customer_messages(args.brand)

    print("\nVectorizing with TF-IDF ...")
    vectorizer = TfidfVectorizer(
        max_features=20000,
        min_df=5,
        max_df=0.4,
        ngram_range=(1, 2),
        stop_words="english",
    )
    X_tfidf = vectorizer.fit_transform(cleaned)
    print(f"TF-IDF matrix: {X_tfidf.shape}")

    print(f"\nReducing to {args.svd_components} dims with TruncatedSVD ...")
    svd = TruncatedSVD(n_components=args.svd_components, random_state=args.seed)
    X_reduced = svd.fit_transform(X_tfidf)
    print(f"Explained variance ratio (sum): {svd.explained_variance_ratio_.sum():.4f}")

    print(f"\nSearching k in [{args.k_min}, {args.k_max}] by silhouette score ...")
    best_k, scores = pick_k_by_silhouette(
        X_reduced, args.k_min, args.k_max, args.silhouette_sample_size, args.seed
    )
    print(f"\nBest k by silhouette: {best_k}")

    print(f"\nFitting final MiniBatchKMeans with k={best_k} ...")
    km = MiniBatchKMeans(n_clusters=best_k, random_state=args.seed, n_init=10, batch_size=1024)
    labels = km.fit_predict(X_reduced)

    # Top TF-IDF terms per cluster: use the mean TF-IDF vector of members
    # (not the SVD centroid, so terms are human-readable words, not
    # latent dimensions).
    terms = np.array(vectorizer.get_feature_names_out())
    cluster_reports = []
    for c in range(best_k):
        member_idx = np.where(labels == c)[0]
        if len(member_idx) == 0:
            continue
        mean_tfidf = np.asarray(X_tfidf[member_idx].mean(axis=0)).ravel()
        top_term_idx = mean_tfidf.argsort()[::-1][: args.top_terms]
        top_terms = terms[top_term_idx].tolist()

        rng = np.random.RandomState(args.seed)
        sample_local_idx = rng.choice(member_idx, size=min(args.samples_per_cluster, len(member_idx)), replace=False)
        sample_msgs = [raw[i] for i in sample_local_idx]

        cluster_reports.append({
            "cluster": int(c),
            "size": int(len(member_idx)),
            "pct_of_total": round(100 * len(member_idx) / len(labels), 2),
            "top_terms": top_terms,
            "sample_messages": sample_msgs,
        })

    cluster_reports.sort(key=lambda r: r["size"], reverse=True)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "intent_clusters.json").write_text(json.dumps({
        "brand": args.brand,
        "n_documents": len(cleaned),
        "k_search_scores": scores,
        "best_k": best_k,
        "clusters": cluster_reports,
    }, indent=2), encoding="utf-8")

    md_lines = [f"# Intent Cluster Inspection Report -- {args.brand}", "",
                f"Documents clustered: {len(cleaned)}",
                f"k search range: [{args.k_min}, {args.k_max}], scores: {scores}",
                f"Selected k: {best_k}", ""]
    for r in cluster_reports:
        md_lines.append(f"## Cluster {r['cluster']} -- {r['size']} messages ({r['pct_of_total']}%)")
        md_lines.append(f"**Top terms:** {', '.join(r['top_terms'])}")
        md_lines.append("")
        md_lines.append("**Sample messages:**")
        for m in r["sample_messages"]:
            md_lines.append(f"- {m}")
        md_lines.append("")
    (REPORTS_DIR / "intent_clusters.md").write_text("\n".join(md_lines), encoding="utf-8")

    print(f"\nWrote reports/intent_clusters.json and reports/intent_clusters.md")
    print(f"{len(cluster_reports)} clusters produced. Inspect reports/intent_clusters.md to assign names.")

    # Persist per-conversation cluster assignments so downstream labeling
    # (mapping cluster -> final human-named intent via configs/intents.yaml)
    # doesn't need to re-run clustering.
    assignments = {conv_ids[i]: int(labels[i]) for i in range(len(conv_ids))}
    (PROCESSED_DIR / f"cluster_assignments_{args.brand}.json").write_text(json.dumps(assignments), encoding="utf-8")
    print(f"Wrote data/processed/cluster_assignments_{args.brand}.json ({len(assignments)} conversations)")


if __name__ == "__main__":
    main()
