#!/usr/bin/env python3
"""
Evaluate the intent classifier against the golden set (human-verified
labels), as distinct from the silver-label test-set numbers in
reports/baseline_results.json.

Usage:
    python scripts/evaluate_against_golden.py --brand AmazonHelp

This is the number that should be trusted over the silver-label test
accuracy: the golden set was read and labeled example-by-example by the
primary labeler (see data/golden/README.md), not derived from the same
clustering/classifier pipeline being evaluated. It is also deliberately
stratified toward hard/ambiguous/rare/OOD cases (data/golden/golden_set_summary.json
shows only 21.5% "easy" examples), so accuracy here is expected to be
LOWER than the silver test-set accuracy -- that gap is itself the main
content of reports/misleading_headline_number.md, not a bug in either
number.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.classification.baselines import MajorityClassifier, TfidfLogisticRegressionClassifier  # noqa: E402
from app.evaluation.metrics import compute_classification_metrics, rare_intent_f1  # noqa: E402
from app.services.text_cleaning import clean_for_modeling  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
GOLDEN_DIR = REPO_ROOT / "data" / "golden"
REPORTS_DIR = REPO_ROOT / "reports"


def load_train(brand: str) -> list[dict]:
    train = []
    with (PROCESSED_DIR / f"labeled_{brand}.jsonl").open() as f:
        for line in f:
            r = json.loads(line)
            if r.get("split") == "train":
                train.append(r)
    return train


def load_golden() -> list[dict]:
    with (GOLDEN_DIR / "golden_set.csv").open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--brand", type=str, required=True)
    args = parser.parse_args()

    train = load_train(args.brand)
    golden = load_golden()
    print(f"Train size: {len(train)}, Golden set size: {len(golden)}")

    train_texts = [clean_for_modeling(r["root_message"]) for r in train]
    train_labels = [r["intent"] for r in train]
    golden_texts = [clean_for_modeling(r["message"]) for r in golden]
    golden_labels = [r["intent"] for r in golden]

    results = {}

    maj = MajorityClassifier().fit(train_texts, train_labels)
    maj_preds = [r.intent for r in maj.predict(golden_texts)]
    maj_metrics = compute_classification_metrics(golden_labels, maj_preds)
    results["majority"] = maj_metrics.as_dict()

    lr = TfidfLogisticRegressionClassifier().fit(train_texts, train_labels)
    lr_preds = [r.intent for r in lr.predict(golden_texts)]
    lr_metrics = compute_classification_metrics(golden_labels, lr_preds)
    lr_rare = rare_intent_f1(lr_metrics, rarity_threshold=500)
    results["tfidf_logreg"] = lr_metrics.as_dict()

    print("\n=== Golden-set (human-verified) results ===")
    for name, m in results.items():
        print(f"{name}: accuracy={m['accuracy']:.4f} macro_f1={m['macro_f1']:.4f}")

    # Direct comparison against the silver test-set numbers, if available.
    comparison = None
    silver_path = REPORTS_DIR / "baseline_results.json"
    if silver_path.exists():
        silver = json.loads(silver_path.read_text())
        comparison = {
            "tfidf_logreg_silver_test_accuracy": silver["tfidf_logreg"]["metrics"]["accuracy"],
            "tfidf_logreg_golden_accuracy": lr_metrics.accuracy,
            "gap": round(silver["tfidf_logreg"]["metrics"]["accuracy"] - lr_metrics.accuracy, 4),
        }
        print(f"\nSilver test accuracy: {comparison['tfidf_logreg_silver_test_accuracy']}")
        print(f"Golden (human-verified) accuracy: {comparison['tfidf_logreg_golden_accuracy']}")
        print(f"Gap: {comparison['gap']}")

    report = {
        "brand": args.brand,
        "golden_set_size": len(golden),
        "results": results,
        "tfidf_logreg_rare_intent_f1": lr_rare,
        "silver_vs_golden_comparison": comparison,
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "golden_set_evaluation.json").write_text(json.dumps(report, indent=2))
    print("\nWrote reports/golden_set_evaluation.json")


if __name__ == "__main__":
    main()
