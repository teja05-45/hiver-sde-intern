#!/usr/bin/env python3
"""
Train and evaluate both baselines on the temporally-split labeled dataset.

Usage:
    python scripts/evaluate_baselines.py --brand AmazonHelp

Writes reports/baseline_results.json and reports/baseline_results.md.

Reminder (see build_labeled_dataset.py docstring): these labels are
cluster-derived silver labels, not hand-verified. These baseline numbers
are directly comparable to the main agent's numbers on the SAME silver
test set (apples to apples), but neither should be quoted as "true"
accuracy without the golden-set cross-check done later.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.services.conversation import load_conversations_jsonl  # noqa: E402 (not used but keeps import style consistent)
from app.services.text_cleaning import clean_for_modeling  # noqa: E402
from app.classification.baselines import MajorityClassifier, TfidfLogisticRegressionClassifier  # noqa: E402
from app.evaluation.metrics import compute_classification_metrics, rare_intent_f1  # noqa: E402

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
    parser.add_argument("--rare-intent-threshold", type=int, default=500,
                         help="training support below which an intent counts as 'rare'")
    args = parser.parse_args()

    data = load_split(args.brand)
    print(f"train={len(data['train'])} dev={len(data['dev'])} test={len(data['test'])}")

    train_texts = [clean_for_modeling(r["root_message"]) for r in data["train"]]
    train_labels = [r["intent"] for r in data["train"]]
    test_texts = [clean_for_modeling(r["root_message"]) for r in data["test"]]
    test_labels = [r["intent"] for r in data["test"]]

    results = {}

    print("\nTraining MajorityClassifier ...")
    maj = MajorityClassifier().fit(train_texts, train_labels)
    maj_preds = [r.intent for r in maj.predict(test_texts)]
    maj_metrics = compute_classification_metrics(test_labels, maj_preds)
    maj_rare = rare_intent_f1(maj_metrics, args.rare_intent_threshold)
    results["majority"] = {"metrics": maj_metrics.as_dict(), "rare_intent": maj_rare}
    print(f"  accuracy={maj_metrics.accuracy:.4f} macro_f1={maj_metrics.macro_f1:.4f}")

    print("\nTraining TF-IDF + LogisticRegression ...")
    lr = TfidfLogisticRegressionClassifier().fit(train_texts, train_labels)
    lr_preds = [r.intent for r in lr.predict(test_texts)]
    lr_metrics = compute_classification_metrics(test_labels, lr_preds)
    lr_rare = rare_intent_f1(lr_metrics, args.rare_intent_threshold)
    results["tfidf_logreg"] = {"metrics": lr_metrics.as_dict(), "rare_intent": lr_rare}
    print(f"  accuracy={lr_metrics.accuracy:.4f} macro_f1={lr_metrics.macro_f1:.4f}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "baseline_results.json").write_text(json.dumps(results, indent=2))

    md = [
        "# Baseline Results",
        "",
        f"Brand: {args.brand} | Test set size: {len(test_labels)} (silver/cluster-derived labels, temporal test split)",
        "",
        "| Model | Accuracy | Macro P | Macro R | Macro F1 | Weighted F1 | Rare-intent avg F1 (support<{}) |".format(args.rare_intent_threshold),
        "|---|---|---|---|---|---|---|",
    ]
    for name, r in results.items():
        m = r["metrics"]
        rare_f1 = r["rare_intent"]["avg_f1"]
        md.append(
            f"| {name} | {m['accuracy']} | {m['macro_precision']} | {m['macro_recall']} | "
            f"{m['macro_f1']} | {m['weighted_f1']} | {rare_f1} |"
        )
    md.append("")
    md.append("## Per-intent F1 (TF-IDF + LogisticRegression)")
    md.append("")
    md.append("| Intent | Precision | Recall | F1 | Support |")
    md.append("|---|---|---|---|---|")
    for intent, d in sorted(results["tfidf_logreg"]["metrics"]["per_intent"].items(), key=lambda x: -x[1]["support"]):
        md.append(f"| {intent} | {d['precision']} | {d['recall']} | {d['f1']} | {d['support']} |")

    (REPORTS_DIR / "baseline_results.md").write_text("\n".join(md))
    print("\nWrote reports/baseline_results.json and reports/baseline_results.md")


if __name__ == "__main__":
    main()
