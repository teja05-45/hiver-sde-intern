#!/usr/bin/env python3
"""
Identify and document the top failure modes using REAL misclassifications
on the golden set (human-verified labels) -- not invented examples.

Usage:
    python scripts/analyze_failures.py --brand AmazonHelp

Writes reports/failure_analysis.json and reports/failure_analysis.md.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.classification.baselines import TfidfLogisticRegressionClassifier  # noqa: E402
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


def categorize_failure(row: dict, true_intent: str, pred_intent: str) -> str:
    """Rule-based failure-mode categorization from real signals already
    present on each golden-set row (categories/difficulty columns written
    by build_golden_set.py) -- not a free-form judgment call per example."""
    cats = set(row["categories"].split(",")) if row["categories"] else set()
    if "ambiguous" in cats:
        return "ambiguous_intent"
    if "multi_intent" in cats:
        return "multi_intent_message"
    if "ood" in cats or pred_intent == "general_other":
        return "out_of_distribution"
    if "noisy" in cats:
        return "noisy_short_message"
    if "rare" in cats:
        return "rare_intent"
    return "other_classifier_error"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--brand", type=str, required=True)
    args = parser.parse_args()

    train = load_train(args.brand)
    train_texts = [clean_for_modeling(r["root_message"]) for r in train]
    train_labels = [r["intent"] for r in train]
    clf = TfidfLogisticRegressionClassifier().fit(train_texts, train_labels)

    with (GOLDEN_DIR / "golden_set.csv").open(encoding="utf-8") as f:
        golden = list(csv.DictReader(f))

    failures = []
    for row in golden:
        text = row["message"]
        pred = clf.predict([clean_for_modeling(text)])[0]
        if pred.intent != row["intent"]:
            failures.append({
                "id": row["id"], "message": text, "true_intent": row["intent"],
                "predicted_intent": pred.intent, "confidence": round(pred.confidence, 3),
                "difficulty": row["difficulty"], "categories": row["categories"],
                "failure_category": categorize_failure(row, row["intent"], pred.intent),
            })

    print(f"Total golden examples: {len(golden)}, failures: {len(failures)} ({len(failures)/len(golden):.1%})")

    by_category: dict[str, list[dict]] = defaultdict(list)
    for f in failures:
        by_category[f["failure_category"]].append(f)

    top5 = sorted(by_category.items(), key=lambda x: -len(x[1]))[:5]

    hypotheses = {
        "ambiguous_intent": "Two or more intents genuinely fit the message about equally well "
            "(classifier's own top-2 margin was <0.15); TF-IDF surface features can't resolve "
            "this the way semantic context could. Fix: real sentence embeddings, or a "
            "second-pass LLM classification specifically for low-margin cases.",
        "multi_intent_message": "Message raises multiple issues at once (e.g. a late delivery AND "
            "a billing complaint); the taxonomy assumes one intent per message. Fix: multi-label "
            "classification, or explicit 'primary vs secondary intent' extraction.",
        "out_of_distribution": "Message doesn't cleanly match any of the 12 taxonomy intents "
            "(promotional content, meta-questions about the support account itself, vague venting). "
            "This is partly by design (general_other exists to catch these and route to escalation) "
            "but the classifier sometimes assigns high confidence to a specific wrong intent instead "
            "of recognizing OOD. Fix: dedicated OOD/novelty detection layer, not just relying on "
            "general_other cluster membership.",
        "noisy_short_message": "Very short or heavily-punctuated messages lack enough token signal "
            "for TF-IDF to disambiguate. Fix: use conversation context (later customer turns), not "
            "just the root message, for classification.",
        "rare_intent": "Intents with <500 training examples (e.g. payment_or_billing_issue, "
            "content_availability_inquiry) don't give the classifier enough examples of their "
            "distinguishing vocabulary. Fix: oversampling, class-weighted loss (already applied, "
            "insufficient alone), or few-shot LLM classification for rare intents specifically.",
        "other_classifier_error": "Errors not explained by any of the above structural categories "
            "-- likely genuine TF-IDF feature-overlap between semantically related intents.",
    }

    report = {
        "brand": args.brand, "golden_set_size": len(golden), "n_failures": len(failures),
        "failure_rate": round(len(failures) / len(golden), 4),
        "top_failure_modes": [
            {"category": cat, "count": len(items), "pct_of_failures": round(len(items) / len(failures), 3),
             "hypothesis": hypotheses.get(cat, ""), "examples": items[:3]}
            for cat, items in top5
        ],
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "failure_analysis.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    md = ["# Failure Analysis (from real golden-set misclassifications)", "",
          f"Golden set size: {len(golden)} | Failures: {len(failures)} ({len(failures)/len(golden):.1%})", ""]
    for cat, items in top5:
        md.append(f"## {cat} -- {len(items)} cases ({len(items)/len(failures):.1%} of failures)")
        md.append(f"**Hypothesis:** {hypotheses.get(cat, '')}")
        md.append("")
        for ex in items[:3]:
            md.append(f"- **[{ex['id']}]** \"{ex['message'][:150]}\"")
            md.append(f"  - Expected: `{ex['true_intent']}` | Predicted: `{ex['predicted_intent']}` "
                       f"(confidence {ex['confidence']})")
        md.append("")
    (REPORTS_DIR / "failure_analysis.md").write_text("\n".join(md), encoding="utf-8")
    print("Wrote reports/failure_analysis.json and reports/failure_analysis.md")


if __name__ == "__main__":
    main()
