#!/usr/bin/env python3
"""
Empirically calibrate evidence-score weights (rather than hand-picking them).

Usage:
    python scripts/calibrate_evidence_score.py --brand AmazonHelp

Methodology: for each dev-split example, compute the five evidence
features (retrieval_score, intent_confidence, intent_agreement_rate,
resolution_consistency, evidence_count_score) using an index built from
train only (dev is strictly later than train, preserving temporal
validity). Label each example 1 if the TF-IDF+LogReg classifier's
predicted intent matches the (silver) true intent, 0 otherwise. Fit a
logistic regression: features -> P(classifier correct). Negative
coefficients are clipped to zero (a feature that's negatively associated
with correctness has no principled place in a "more evidence = more
trustworthy" score) and the remaining coefficients are normalized to sum
to 1 and saved as configs/evidence_score_weights.json.

This is a calibration of the *scoring formula*, not of the escalation
threshold (that's a separate, later step using the golden set + explicit
business rationale, per the assignment's requirement not to "simply
choose the threshold that gives the prettiest result").

Note on intent_confidence: an earlier version of this script included the
classifier's own intent_confidence as a fifth feature. It captured ~88% of
the fitted weight, making evidence_score nearly degenerate into "restate
the classifier's self-reported confidence" -- which is exactly the
"answer because the model is confident" pattern this project's core
design principle rejects. intent_confidence is excluded from the
evidence-quality calibration for that reason and used as a separate,
explicitly named input to the escalation policy instead. See
docs/decision-log.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

import numpy as np  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from app.classification.baselines import TfidfLogisticRegressionClassifier  # noqa: E402
from app.retrieval.embeddings import TfidfEmbeddingProvider  # noqa: E402
from app.retrieval.retriever import HistoricalRetriever  # noqa: E402
from app.services.evidence_scoring import compute_evidence_features, DEFAULT_WEIGHTS_PATH  # noqa: E402
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--brand", type=str, required=True)
    parser.add_argument("--max-examples", type=int, default=2000)
    args = parser.parse_args()

    data = load_split(args.brand)
    train_texts = [clean_for_modeling(r["root_message"]) for r in data["train"]]
    train_labels = [r["intent"] for r in data["train"]]

    print("Training intent classifier on train split ...")
    clf = TfidfLogisticRegressionClassifier().fit(train_texts, train_labels)

    print("Building retrieval index on train split only (dev is later in time) ...")
    retriever = HistoricalRetriever(TfidfEmbeddingProvider()).build_index(data["train"])
    resolution_embedder = TfidfEmbeddingProvider().fit(
        [r["resolution"] for r in data["train"] if r.get("resolution")]
    )

    dev_examples = data["dev"]
    if len(dev_examples) > args.max_examples:
        import random
        dev_examples = random.Random(42).sample(dev_examples, args.max_examples)
    print(f"Computing features on {len(dev_examples)} dev examples ...")

    X, y = [], []
    for r in dev_examples:
        query_text = r["root_message"]
        pred = clf.predict([clean_for_modeling(query_text)])[0]
        evidence = retriever.retrieve(query_text, k=5)
        features = compute_evidence_features(evidence, resolution_embedder)
        X.append([
            features.retrieval_score,
            features.intent_agreement_rate, features.resolution_consistency,
            features.evidence_count_score,
        ])
        y.append(1 if pred.intent == r["intent"] else 0)

    X = np.array(X)
    y = np.array(y)
    feature_names = ["retrieval_score", "intent_agreement_rate",
                      "resolution_consistency", "evidence_count_score"]

    print(f"\nBase rate (classifier accuracy on this dev sample): {y.mean():.4f}")

    lr = LogisticRegression(max_iter=1000)
    lr.fit(X, y)
    probs = lr.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, probs) if len(set(y)) > 1 else float("nan")
    print(f"Calibration logistic regression AUC (in-sample, dev set): {auc:.4f}")

    coefs = lr.coef_[0]
    print("\nRaw coefficients:")
    for name, c in zip(feature_names, coefs):
        print(f"  {name}: {c:.4f}")

    clipped = np.clip(coefs, 0, None)
    if clipped.sum() == 0:
        print("WARNING: all coefficients non-positive; falling back to equal weights.")
        weights = {name: 1.0 / len(feature_names) for name in feature_names}
    else:
        normalized = clipped / clipped.sum()
        weights = {name: round(float(w), 4) for name, w in zip(feature_names, normalized)}

    print("\nFinal normalized weights (negative coefficients clipped to 0):")
    for name, w in weights.items():
        print(f"  {name}: {w}")

    DEFAULT_WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_WEIGHTS_PATH.write_text(json.dumps(weights, indent=2))
    print(f"\nWrote {DEFAULT_WEIGHTS_PATH}")

    report = {
        "brand": args.brand,
        "n_dev_examples_used": len(dev_examples),
        "classifier_accuracy_on_sample": round(float(y.mean()), 4),
        "calibration_auc": round(float(auc), 4) if auc == auc else None,  # NaN check
        "raw_coefficients": {name: round(float(c), 4) for name, c in zip(feature_names, coefs)},
        "final_weights": weights,
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "evidence_score_calibration.json").write_text(json.dumps(report, indent=2))
    print("Wrote reports/evidence_score_calibration.json")


if __name__ == "__main__":
    main()
