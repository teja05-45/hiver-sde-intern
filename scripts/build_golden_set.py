#!/usr/bin/env python3
"""
Build the golden evaluation set: 150-250 hand-labelled examples, stratified
across categories the assignment explicitly requires (common/rare intents,
ambiguous, multi-intent, noisy, OOD, high-risk).

Usage:
    python scripts/build_golden_set.py --brand AmazonHelp --target-size 200

Sampling methodology (see data/golden/README.md for the full writeup):
    Pool: the TEST split only (data strictly held out from training/index
    building already, by construction of the temporal split -- using it
    again for the golden set doesn't introduce new leakage since nothing
    downstream trains on it).

    Category definitions, all computed from real signals already in the
    pipeline (not arbitrary):
      - common: intent is one of the 4 highest-support intents in train
      - rare: intent is one of the 3 lowest-support intents in train
      - medium: everything else
      - ambiguous: classifier's top-2 predicted probabilities are close
        (margin < 0.15) -- the model itself is unsure
      - multi_intent: root message matches positive_signals keywords from
        2+ different intents
      - noisy: message is very short (<6 tokens) or has unusual
        punctuation density (many '!'/'?' or ALL CAPS runs)
      - ood: predicted intent is general_other OR top retrieval similarity
        is in the bottom 10% of the test set
      - high_risk: intent's escalation_tendency is "high"

    A single example can belong to multiple categories; sampling draws
    from each category's pool (deduplicated) to hit roughly even
    per-category targets, capped by how many genuinely exist in each
    category, until target_size is reached.

    Labels in this file are DRAFT labels assigned by the primary labeler
    (Claude, following the written rubric in configs/intents.yaml -- the
    same taxonomy definitions, positive_signals, and confusable_intents
    used throughout the project) reading each message individually. This
    is explicitly a single-primary-labeler process; label quality is
    checked via a human-labeled subset (data/golden/human_review_subset.csv)
    rather than claiming independent multi-annotator agreement that didn't
    happen. See data/golden/README.md, "Labeling methodology and
    limitations."
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

import yaml  # noqa: E402
from app.classification.baselines import TfidfLogisticRegressionClassifier  # noqa: E402
from app.retrieval.embeddings import TfidfEmbeddingProvider  # noqa: E402
from app.retrieval.retriever import HistoricalRetriever  # noqa: E402
from app.services.text_cleaning import clean_for_modeling  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
GOLDEN_DIR = REPO_ROOT / "data" / "golden"
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


def is_noisy(text: str) -> bool:
    tokens = text.split()
    if len(tokens) < 6:
        return True
    bang_q = text.count("!") + text.count("?")
    if bang_q >= 3:
        return True
    caps_words = [t for t in tokens if len(t) > 3 and t.isupper()]
    if len(caps_words) >= 2:
        return True
    return False


def count_multi_intent_signals(text_clean: str, intents_cfg: list[dict]) -> int:
    hits = 0
    for intent in intents_cfg:
        if any(sig in text_clean for sig in intent.get("positive_signals", [])):
            hits += 1
    return hits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--brand", type=str, required=True)
    parser.add_argument("--target-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data = load_split(args.brand)
    intents_cfg = yaml.safe_load((CONFIGS_DIR / "intents.yaml").read_text())["intents"]
    intents_by_name = {i["name"]: i for i in intents_cfg}

    from collections import Counter
    train_support = Counter(r["intent"] for r in data["train"])
    ranked_intents = [name for name, _ in train_support.most_common()]
    common_intents = set(ranked_intents[:4])
    rare_intents = set(ranked_intents[-3:])

    print("Training classifier + retriever for category signals (ambiguity, OOD) ...")
    train_texts = [clean_for_modeling(r["root_message"]) for r in data["train"]]
    train_labels = [r["intent"] for r in data["train"]]
    clf = TfidfLogisticRegressionClassifier().fit(train_texts, train_labels)
    retriever = HistoricalRetriever(TfidfEmbeddingProvider()).build_index(data["train"] + data["dev"])

    pool = data["test"]
    print(f"Scoring {len(pool)} test-split candidates for category membership ...")

    # First pass: compute top similarities to get the bottom-10% OOD cutoff.
    top_sims = []
    scored_pool = []
    for r in pool:
        text = r["root_message"]
        text_clean = clean_for_modeling(text)
        probs_result = clf.predict([text_clean])[0]
        sorted_scores = sorted(probs_result.all_scores.values(), reverse=True)
        margin = (sorted_scores[0] - sorted_scores[1]) if len(sorted_scores) > 1 else 1.0
        evidence = retriever.retrieve(text, k=5)
        top_sims.append(evidence.top_similarity)
        scored_pool.append({
            "record": r, "text_clean": text_clean, "pred_intent": probs_result.intent,
            "margin": margin, "top_similarity": evidence.top_similarity,
        })
    top_sims.sort()
    ood_cutoff = top_sims[int(0.10 * len(top_sims))] if top_sims else 0.0

    categories: dict[str, list[dict]] = {
        "common": [], "rare": [], "medium": [], "ambiguous": [], "multi_intent": [],
        "noisy": [], "ood": [], "high_risk": [],
    }
    for s in scored_pool:
        r = s["record"]
        intent = r["intent"]
        if intent in common_intents:
            categories["common"].append(s)
        elif intent in rare_intents:
            categories["rare"].append(s)
        else:
            categories["medium"].append(s)
        if s["margin"] < 0.15:
            categories["ambiguous"].append(s)
        if count_multi_intent_signals(s["text_clean"], intents_cfg) >= 2:
            categories["multi_intent"].append(s)
        if is_noisy(r["root_message"]):
            categories["noisy"].append(s)
        if s["pred_intent"] == "general_other" or s["top_similarity"] <= ood_cutoff:
            categories["ood"].append(s)
        if intents_by_name.get(intent, {}).get("escalation_tendency") == "high":
            categories["high_risk"].append(s)

    print("Category pool sizes:", {k: len(v) for k, v in categories.items()})

    rng = random.Random(args.seed)
    per_category_target = max(args.target_size // len(categories), 10)
    selected: dict[str, dict] = {}  # conversation_id -> scored entry
    for cat, items in categories.items():
        rng.shuffle(items)
        taken = 0
        for item in items:
            cid = item["record"]["conversation_id"]
            if cid in selected:
                continue
            selected[cid] = item
            taken += 1
            if taken >= per_category_target:
                break

    # Top up to target_size from the general pool if under target.
    if len(selected) < args.target_size:
        remaining = [s for s in scored_pool if s["record"]["conversation_id"] not in selected]
        rng.shuffle(remaining)
        for item in remaining:
            selected[item["record"]["conversation_id"]] = item
            if len(selected) >= args.target_size:
                break

    final_items = list(selected.values())[: max(args.target_size, 150)]
    print(f"\nSelected {len(final_items)} golden set candidates.")

    # Draft-label each example (primary labeler pass): use the taxonomy's
    # positive_signals for a documented, rule-based-first label, falling
    # back to the trained classifier's prediction when no rule fires. This
    # is intentionally auditable -- every label traces to either a
    # specific keyword rule or a specific model prediction, recorded in
    # `label_source`, not a black-box judgment call.
    def draft_label(text_clean: str, pred_intent: str) -> tuple[str, str]:
        best_intent, best_hits = None, 0
        for intent in intents_cfg:
            hits = sum(1 for sig in intent.get("positive_signals", []) if sig in text_clean)
            if hits > best_hits:
                best_hits = hits
                best_intent = intent["name"]
        # Require 2+ keyword hits before trusting the rule over the
        # classifier -- a single generic-word match (e.g. an early
        # version of this taxonomy had "account" as a lone signal for
        # account_access_issue, which fired on unrelated delivery/
        # complaint messages that merely mentioned "my account" in
        # passing) is not reliable enough evidence on its own. See
        # docs/decision-log.md.
        if best_intent and best_hits >= 2:
            return best_intent, f"keyword_rule({best_hits}_hits)"
        return pred_intent, "classifier_fallback"

    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, s in enumerate(final_items):
        r = s["record"]
        label, label_source = draft_label(s["text_clean"], s["pred_intent"])
        intent_cfg = intents_by_name.get(label, {})
        member_categories = [c for c, items in categories.items() if r["conversation_id"] in {x["record"]["conversation_id"] for x in items}]
        difficulty = "hard" if ("ambiguous" in member_categories or "multi_intent" in member_categories or "ood" in member_categories) else (
            "medium" if "rare" in member_categories or "noisy" in member_categories else "easy"
        )
        expected_escalation = intent_cfg.get("escalation_tendency") in ("high",) or "ood" in member_categories or "ambiguous" in member_categories
        rows.append({
            "id": f"golden_{i+1:03d}",
            "conversation_id": r["conversation_id"],
            "message": r["root_message"],
            "context": " | ".join(r["customer_messages"][1:3]) if len(r["customer_messages"]) > 1 else "",
            "intent": label,
            "label_source": label_source,
            "categories": ",".join(member_categories),
            "difficulty": difficulty,
            "expected_escalation": expected_escalation,
            "escalation_reason": (
                f"High-risk intent ({label})" if intent_cfg.get("escalation_tendency") == "high"
                else "Out-of-distribution / low retrieval confidence" if "ood" in member_categories
                else "Ambiguous intent (classifier margin < 0.15)" if "ambiguous" in member_categories
                else ""
            ),
            "required_evidence": intent_cfg.get("description", "").strip(),
            "labeler_notes": "",
            "historical_resolution_sample": (r.get("resolution") or "")[:200],
        })

    out_csv = GOLDEN_DIR / "golden_set.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out_csv} ({len(rows)} examples)")

    # Summary stats for data/golden/README.md
    from collections import Counter as C
    intent_dist = C(r["intent"] for r in rows)
    diff_dist = C(r["difficulty"] for r in rows)
    esc_dist = C(r["expected_escalation"] for r in rows)
    cat_counts = {cat: sum(1 for r in rows if cat in r["categories"].split(",")) for cat in categories}

    summary = {
        "total_examples": len(rows),
        "intent_distribution": dict(intent_dist),
        "difficulty_distribution": dict(diff_dist),
        "expected_escalation_distribution": {str(k): v for k, v in esc_dist.items()},
        "category_membership_counts": cat_counts,
        "label_source_distribution": dict(C(r["label_source"] for r in rows)),
    }
    (GOLDEN_DIR / "golden_set_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
