#!/usr/bin/env python3
"""
Export the golden set joined with the model's own predictions.

Produces data/golden/golden_set_with_predictions.json, consumed by
GET /api/v1/golden-set/examples for the Golden Set page. For every golden
example we record:

  - the customer message and the HUMAN-VERIFIED gold label
  - the model's predicted intent + confidence (from the trained classifier)
  - correct / incorrect against the gold label
  - the example's stratification categories (ambiguous/multi_intent/noisy/
    ood/high_risk/common/rare/medium), for filtering

Usage:
    python scripts/export_golden_with_predictions.py --brand AmazonHelp
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

import joblib  # noqa: E402

from app.services.text_cleaning import clean_for_modeling  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_DIR = REPO_ROOT / "data" / "golden"
MODELS_DIR = REPO_ROOT / "models"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--brand", type=str, default="AmazonHelp")
    args = parser.parse_args()

    clf = joblib.load(MODELS_DIR / f"classifier_{args.brand}.joblib")

    examples = []
    with (GOLDEN_DIR / "golden_set.csv").open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            msg = row["message"]
            pred = clf.predict([clean_for_modeling(msg)])[0]
            examples.append({
                "id": row["id"],
                "customer_message": msg,
                "gold_intent": row["intent"],
                "predicted_intent": pred.intent,
                "confidence": round(pred.confidence, 4),
                "correct": pred.intent == row["intent"],
                "categories": [c for c in row.get("categories", "").split(",") if c],
                "difficulty": row.get("difficulty", ""),
                "label_source": row.get("label_source", ""),
            })

    out = {
        "brand": args.brand,
        "note": "Gold labels are human-verified (see data/golden/README.md). "
                "Predictions are from the trained classifier artifact.",
        "total": len(examples),
        "examples": examples,
    }
    out_path = GOLDEN_DIR / "golden_set_with_predictions.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Wrote {out_path} ({len(examples)} examples)")


if __name__ == "__main__":
    main()
