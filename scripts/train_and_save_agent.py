#!/usr/bin/env python3
"""
Train the intent classifier and build the retrieval index once, and
persist both to disk (models/) so the API doesn't retrain on every
startup.

Usage:
    python scripts/train_and_save_agent.py --brand AmazonHelp
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

import joblib  # noqa: E402
import yaml  # noqa: E402
from app.classification.baselines import TfidfLogisticRegressionClassifier  # noqa: E402
from app.retrieval.embeddings import TfidfEmbeddingProvider  # noqa: E402
from app.retrieval.retriever import HistoricalRetriever  # noqa: E402
from app.services.evidence_scoring import load_weights  # noqa: E402
from app.services.text_cleaning import clean_for_modeling  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
MODELS_DIR = REPO_ROOT / "models"
CONFIGS_DIR = REPO_ROOT / "configs"


def load_split(brand: str) -> dict[str, list[dict]]:
    by_split: dict[str, list[dict]] = {"train": [], "dev": [], "test": []}
    with (PROCESSED_DIR / f"labeled_{brand}.jsonl").open() as f:
        for line in f:
            r = json.loads(line)
            if r.get("split") in by_split:
                by_split[r["split"]].append(r)
    return by_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--brand", type=str, required=True)
    args = parser.parse_args()

    data = load_split(args.brand)
    train_texts = [clean_for_modeling(r["root_message"]) for r in data["train"]]
    train_labels = [r["intent"] for r in data["train"]]

    print("Training classifier ...")
    clf = TfidfLogisticRegressionClassifier().fit(train_texts, train_labels)

    print("Building retrieval index (train+dev) ...")
    retriever = HistoricalRetriever(TfidfEmbeddingProvider()).build_index(data["train"] + data["dev"])

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, MODELS_DIR / f"classifier_{args.brand}.joblib")
    joblib.dump(retriever, MODELS_DIR / f"retriever_{args.brand}.joblib")

    weights = load_weights()
    intents_cfg = {i["name"]: i for i in yaml.safe_load((CONFIGS_DIR / "intents.yaml").read_text())["intents"]}
    (MODELS_DIR / "evidence_weights.json").write_text(json.dumps(weights, indent=2))
    (MODELS_DIR / "intents_cfg.json").write_text(json.dumps(intents_cfg, indent=2))

    print(f"Saved artifacts to {MODELS_DIR}")


if __name__ == "__main__":
    main()
