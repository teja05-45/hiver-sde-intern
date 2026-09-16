#!/usr/bin/env python3
"""
Automation precision/coverage analysis: sweep escalation thresholds and
report the tradeoff, rather than picking whichever threshold looks best.

Usage:
    python scripts/evaluate_automation.py --brand AmazonHelp

IMPORTANT SCOPE NOTE: this sweep uses classification + retrieval +
evidence-scoring signals only. It does NOT yet include a grounding_score
from response generation, because that requires a live LLM call, which
this sandbox cannot make (no network access -- see README "LLM execution
status"). "Auto precision" here means: of the cases the policy would mark
AUTO, what fraction had a CORRECT intent prediction (against the silver
label). This is a necessary-but-not-sufficient proxy for "was the
auto-sent reply actually correct and safe" -- a correct intent with a
badly grounded generated reply would still be a real failure that this
number cannot see. The full pipeline (including the grounding gate) is
implemented in backend/app/escalation/policy.py and exercised end-to-end
in backend/tests/ against a mock LLM provider; re-running this exact
sweep with a real LLM_PROVIDER configured would tighten (not loosen) the
reported precision, since the grounding check can only turn AUTO cases
into ESCALATE, never the reverse. This is called out again in
reports/misleading_headline_number.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

import yaml  # noqa: E402
from app.classification.baselines import TfidfLogisticRegressionClassifier  # noqa: E402
from app.retrieval.embeddings import TfidfEmbeddingProvider  # noqa: E402
from app.retrieval.retriever import HistoricalRetriever  # noqa: E402
from app.services.evidence_scoring import compute_evidence_features, compute_evidence_score, load_weights  # noqa: E402
from app.services.text_cleaning import clean_for_modeling  # noqa: E402
from app.escalation.policy import EscalationSignals, EscalationThresholds, decide  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
REPORTS_DIR = REPO_ROOT / "reports"
CONFIGS_DIR = REPO_ROOT / "configs"


def load_split(brand: str) -> dict[str, list[dict]]:
    path = PROCESSED_DIR / f"labeled_{brand}.jsonl"
    by_split: dict[str, list[dict]] = {"train": [], "dev": [], "test": []}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("split") in by_split:
                by_split[r["split"]].append(r)
    return by_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--brand", type=str, required=True)
    parser.add_argument("--max-examples", type=int, default=2500)
    parser.add_argument("--target-auto-precision", type=float, default=0.99)
    args = parser.parse_args()

    data = load_split(args.brand)
    intents_cfg = {i["name"]: i for i in yaml.safe_load((CONFIGS_DIR / "intents.yaml").read_text())["intents"]}
    weights = load_weights()

    train_texts = [clean_for_modeling(r["root_message"]) for r in data["train"]]
    train_labels = [r["intent"] for r in data["train"]]
    print("Training classifier ...")
    clf = TfidfLogisticRegressionClassifier().fit(train_texts, train_labels)

    print("Building retrieval index (train+dev, evaluating on test) ...")
    retriever = HistoricalRetriever(TfidfEmbeddingProvider()).build_index(data["train"] + data["dev"])
    resolution_embedder = TfidfEmbeddingProvider().fit(
        [r["resolution"] for r in (data["train"] + data["dev"]) if r.get("resolution")]
    )

    test_examples = data["test"]
    if len(test_examples) > args.max_examples:
        import random
        test_examples = random.Random(42).sample(test_examples, args.max_examples)
    print(f"Scoring {len(test_examples)} test examples ...")

    scored = []
    for r in test_examples:
        text = r["root_message"]
        pred = clf.predict([clean_for_modeling(text)])[0]
        evidence = retriever.retrieve(text, k=5)
        features = compute_evidence_features(evidence, resolution_embedder)
        evidence_score = compute_evidence_score(features, weights)
        is_correct = pred.intent == r["intent"]
        escalation_tendency = intents_cfg.get(pred.intent, {}).get("escalation_tendency", "high")
        scored.append({
            "intent_confidence": pred.confidence,
            "evidence_score": evidence_score,
            "num_supporting_cases": evidence.num_cases,
            "intent_agreement_rate": evidence.intent_agreement_rate,
            "escalation_tendency": escalation_tendency,
            "predicted_intent": pred.intent,
            "true_intent": r["intent"],
            "is_correct": is_correct,
        })

    # Sweep (evidence_score_threshold, intent_confidence_threshold) pairs.
    threshold_grid = [0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
    curve = []
    for ev_t in threshold_grid:
        for ic_t in threshold_grid:
            thresholds = EscalationThresholds(
                min_evidence_score_for_auto=ev_t,
                min_intent_confidence_for_auto=ic_t,
                min_supporting_cases_for_auto=2,
                min_grounding_score_for_auto=0.0,  # not evaluated in this sweep, see docstring
            )
            n_auto, n_auto_correct = 0, 0
            for s in scored:
                signals = EscalationSignals(
                    intent=s["predicted_intent"],
                    intent_confidence=s["intent_confidence"],
                    intent_escalation_tendency=s["escalation_tendency"],
                    evidence_score=s["evidence_score"],
                    num_supporting_cases=s["num_supporting_cases"],
                    intent_agreement_rate=s["intent_agreement_rate"],
                )
                decision = decide(signals, thresholds)
                if decision.public_decision == "AUTO":
                    n_auto += 1
                    if s["is_correct"]:
                        n_auto_correct += 1
            coverage = n_auto / len(scored) if scored else 0.0
            precision = n_auto_correct / n_auto if n_auto else None
            curve.append({
                "evidence_score_threshold": ev_t,
                "intent_confidence_threshold": ic_t,
                "coverage": round(coverage, 4),
                "auto_precision": round(precision, 4) if precision is not None else None,
                "n_auto": n_auto,
            })

    # Pick the operating point: among thresholds achieving the target
    # precision, choose the one with maximum coverage. If none achieve
    # target precision, report the max precision actually achievable
    # (do NOT silently lower the bar and call it the target).
    feasible = [c for c in curve if c["auto_precision"] is not None and c["auto_precision"] >= args.target_auto_precision and c["n_auto"] >= 20]
    if feasible:
        chosen = max(feasible, key=lambda c: c["coverage"])
        target_achieved = True
    else:
        candidates = [c for c in curve if c["n_auto"] >= 20]
        chosen = max(candidates, key=lambda c: (c["auto_precision"] or 0, c["coverage"])) if candidates else None
        target_achieved = False

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "automation_curve.json").write_text(json.dumps({
        "brand": args.brand,
        "n_scored": len(scored),
        "target_auto_precision": args.target_auto_precision,
        "target_achieved": target_achieved,
        "chosen_operating_point": chosen,
        "full_curve": curve,
        "scope_note": "Intent-correctness proxy only; does not yet include grounding_score (no live LLM in this sandbox). See script docstring.",
    }, indent=2))

    # CSV as explicitly requested by the assignment.
    import csv
    with (REPORTS_DIR / "automation_curve.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["evidence_score_threshold", "intent_confidence_threshold", "coverage", "auto_precision", "n_auto"])
        writer.writeheader()
        for row in curve:
            writer.writerow(row)

    print("\nChosen operating point:")
    print(json.dumps(chosen, indent=2))
    print(f"\nTarget precision ({args.target_auto_precision}) achieved: {target_achieved}")
    print("\nWrote reports/automation_curve.json and reports/automation_curve.csv")


if __name__ == "__main__":
    main()
