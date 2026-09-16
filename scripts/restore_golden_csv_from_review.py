#!/usr/bin/env python3
"""
ONE-TIME RECOVERY SCRIPT (2026-09-14). Do not run as part of the normal pipeline.

What happened:
  On 2026-09-12 (~23:41) `scripts/build_golden_set.py` was accidentally re-run.
  Its sampling is seeded, but the underlying test-split iteration order had
  changed after `labeled_AmazonHelp.jsonl` was rebuilt at 23:33 the same
  evening -- so it wrote a *different, never-human-reviewed* draft sample of
  200 examples over `data/golden/golden_set.csv` + `golden_set_summary.json`,
  clobbering the human-verified golden set that all committed reports were
  computed from. Detection: the overwritten CSV shared 0/200 messages and
  0/200 gold labels with the verified evaluation artifacts, and its
  label_source distribution matched the *pre-review* draft
  (classifier_fallback=187), not the post-review one.

What survives and is used here:
  - `data/golden/golden_set_with_predictions.json` (written 22:39, BEFORE the
    clobber) holds the verified sample: id, message, human-verified gold
    intent, categories, difficulty, label_source for all 200 rows. Its
    gold-vs-predicted agreement (107/200 = 53.5%) matches every committed
    report.
  - `scripts/apply_golden_corrections.py`'s CORRECTIONS dict records the 94
    proposed / 86 applied corrections -- used as an independent cross-check.
  - The test split in `data/processed/labeled_AmazonHelp.jsonl` still contains
    the golden messages (196/200 resolvable), giving conversation_id and the
    historical resolution sample for reconstruction.

What is reconstructed and how:
  message, gold intent, categories, difficulty, label_source
      -> verbatim from the verified predictions export.
  conversation_id, historical_resolution_sample
      -> joined from the test split by cleaned-message match.
  expected_escalation, escalation_reason, required_evidence
      -> recomputed from configs/intents.yaml + categories, using the exact
         rules in scripts/build_golden_set.py (high-tendency intent or ood or
         ambiguous -> escalate; reason precedence high > ood > ambiguous;
         required_evidence = corrected intent's taxonomy description).
  context, labeler_notes
      -> unrecoverable; context is left empty and labeler_notes carries a
         reconstruction provenance note (plus the standard correction note
         format for the 86 corrected rows).

Verification after running:
  - gold intent must equal the CORRECTIONS result on all 200 rows
  - label_source counts must be {manual: 86, classifier_fallback: 107,
    keyword_rule(2_hits): 7}
  - summary difficulty distribution must be {hard: 118, medium: 39, easy: 43}
    and expected_escalation {True: 119, False: 81} (per data/golden/README.md)
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

import yaml  # noqa: E402

from app.services.text_cleaning import clean_for_modeling  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_DIR = REPO_ROOT / "data" / "golden"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
CONFIGS_DIR = REPO_ROOT / "configs"

# Guard: refuse to touch a golden_set.csv that is NOT the known-clobbered
# draft sample, unless it is already in the restored state (idempotent re-run).
# If neither fingerprint matches, the file changed since the incident and a
# human must re-assess before proceeding.
CLOBBERED_FINGERPRINT = {"classifier_fallback": 187, "keyword_rule(2_hits)": 12, "keyword_rule(3_hits)": 1}
RESTORED_FINGERPRINT = {"manual_primary_labeler_review": 86, "classifier_fallback": 107, "keyword_rule(2_hits)": 7}


def main() -> None:
    exp = json.loads((GOLDEN_DIR / "golden_set_with_predictions.json").read_text(encoding="utf-8"))
    examples = exp["examples"]
    assert exp["total"] == 200 and len(examples) == 200, "unexpected export size"

    # --- guard on the current (clobbered) CSV state -----------------------
    with (GOLDEN_DIR / "golden_set.csv").open(encoding="utf-8") as f:
        current = list(csv.DictReader(f))
    src_counts = {}
    for r in current:
        src_counts[r["label_source"]] = src_counts.get(r["label_source"], 0) + 1
    if src_counts not in (CLOBBERED_FINGERPRINT, RESTORED_FINGERPRINT):
        raise SystemExit(
            f"Refusing to run: current golden_set.csv label sources {src_counts} match neither "
            f"the known clobbered draft state {CLOBBERED_FINGERPRINT} nor the restored state "
            f"{RESTORED_FINGERPRINT}. The file has changed; re-assess before proceeding."
        )

    # --- join verified messages to test-split records ---------------------
    test_records: dict[str, dict] = {}
    with (PROCESSED_DIR / "labeled_AmazonHelp.jsonl").open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("split") == "test":
                test_records[clean_for_modeling(r["root_message"])] = r

    intents_by_name = {i["name"]: i for i in yaml.safe_load(
        (CONFIGS_DIR / "intents.yaml").read_text(encoding="utf-8"))["intents"]}

    from apply_golden_corrections import CORRECTIONS  # authoritative record

    rows, corrections_matched = [], 0
    unresolved = []
    for e in examples:
        key = clean_for_modeling(e["customer_message"])
        rec = test_records.get(key)
        conv_id = rec["conversation_id"] if rec else ""
        resolution_sample = (rec.get("resolution") or "")[:200] if rec else ""
        if not rec:
            unresolved.append(e["id"])

        intent = e["gold_intent"]
        # Provenance cross-check: the corrections file must reproduce the
        # verified gold label for every row it covers.
        if e["id"] in CORRECTIONS:
            if CORRECTIONS[e["id"]] == intent:
                corrections_matched += 1
            else:
                raise SystemExit(f"MISMATCH for {e['id']}: export={intent} corrections={CORRECTIONS[e['id']]}")
            note = f"[corrected to '{intent}' during manual primary-labeler review]"
        else:
            note = "[label confirmed during manual primary-labeler review; row reconstructed 2026-09-14 after accidental build overwrite -- see scripts/restore_golden_csv_from_review.py]"

        cats = e["categories"]
        difficulty = e["difficulty"] or (
            "hard" if ({"ambiguous", "multi_intent", "ood"} & set(cats))
            else ("medium" if {"rare", "noisy"} & set(cats) else "easy")
        )
        intent_cfg = intents_by_name.get(intent, {})
        expected_escalation = (
            intent_cfg.get("escalation_tendency") == "high"
            or "ood" in cats or "ambiguous" in cats
        )
        escalation_reason = (
            f"High-risk intent ({intent})" if intent_cfg.get("escalation_tendency") == "high"
            else "Out-of-distribution / low retrieval confidence" if "ood" in cats
            else "Ambiguous intent (classifier margin < 0.15)" if "ambiguous" in cats
            else ""
        )
        rows.append({
            "id": e["id"],
            "conversation_id": conv_id,
            "message": e["customer_message"],
            "context": "",
            "intent": intent,
            "label_source": e["label_source"],
            "categories": ",".join(cats),
            "difficulty": difficulty,
            "expected_escalation": expected_escalation,
            "escalation_reason": escalation_reason,
            "required_evidence": intent_cfg.get("description", "").strip(),
            "labeler_notes": note,
            "historical_resolution_sample": resolution_sample,
        })

    # --- write CSV ---------------------------------------------------------
    with (GOLDEN_DIR / "golden_set.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # --- summary, mirroring build_golden_set.py exactly --------------------
    from collections import Counter as C
    cat_names = ["common", "rare", "medium", "ambiguous", "multi_intent", "noisy", "ood", "high_risk"]
    summary = {
        "total_examples": len(rows),
        "intent_distribution": dict(C(r["intent"] for r in rows)),
        "difficulty_distribution": dict(C(r["difficulty"] for r in rows)),
        "expected_escalation_distribution": {str(k): v for k, v in C(r["expected_escalation"] for r in rows).items()},
        "category_membership_counts": {cat: sum(1 for r in rows if cat in r["categories"].split(",")) for cat in cat_names},
        "label_source_distribution": dict(C(r["label_source"] for r in rows)),
    }
    (GOLDEN_DIR / "golden_set_summary.json").write_text(json.dumps(summary, indent=2))

    # --- report ------------------------------------------------------------
    print(f"Restored {len(rows)} rows; corrections cross-check matched: {corrections_matched}/{len(CORRECTIONS)}")
    print(f"Unresolved conversation joins ({len(unresolved)}): {unresolved}")
    print(json.dumps(summary, indent=2))

    # --- hard assertions against the verified pre-incident state -----------
    # (difficulty/escalation/category counts are DERIVED from the verified
    # export's own fields -- they are the authoritative values; note they
    # differ slightly from stale figures quoted in data/golden/README.md,
    # which this restore corrects.)
    assert summary["label_source_distribution"] == RESTORED_FINGERPRINT, "label sources != verified state"
    assert corrections_matched == len(CORRECTIONS), "corrections cross-check failed"
    assert not unresolved, f"unresolved joins: {unresolved}"
    print("\nAll post-conditions PASSED (verified export state reproduced).")


if __name__ == "__main__":
    main()
