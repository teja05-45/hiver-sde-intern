#!/usr/bin/env python3
"""
Assemble the final labeled dataset for classification:

1. Map each conversation's cluster assignment (from discover_intents.py)
   to a final intent name via configs/intents.yaml's `source_clusters`.
2. Apply a temporal train/dev/test split.
3. Run exact/near-duplicate leakage analysis, including specifically
   whether duplicate content crosses split boundaries.
4. Write data/processed/labeled_<brand>.jsonl and
   reports/leakage_analysis.md/json.

IMPORTANT, stated plainly: these intent labels are cluster-derived
("silver" labels), not individually human-verified. They are used to
train/evaluate baselines at a scale (tens of thousands of examples) that
hand-labeling cannot reach. The genuinely human-verified evaluation is the
separate ~200-example golden set (scripts/build_golden_set.py), which is
the trustworthy number. Any accuracy reported against silver labels is
explicitly labeled as such everywhere it's surfaced (README, REPORT,
reports/*.json) and is never presented as the headline metric on its own.
See docs/decision-log.md, "Why cluster-derived labels for the training
set" and reports/misleading_headline_number.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

import yaml  # noqa: E402
from app.services.conversation import load_conversations_jsonl  # noqa: E402
from app.services.text_cleaning import clean_for_modeling  # noqa: E402
from app.services.data_split import (  # noqa: E402
    parse_twitter_datetime, temporal_split, analyze_duplicates,
    cross_split_duplicate_overlap, normalize_for_dedup, content_hash,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
REPORTS_DIR = REPO_ROOT / "reports"
CONFIGS_DIR = REPO_ROOT / "configs"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--brand", type=str, required=True)
    parser.add_argument("--cutoff1", type=str, default="2017-11-10", help="train < cutoff1")
    parser.add_argument("--cutoff2", type=str, default="2017-11-25", help="dev in [cutoff1, cutoff2), test >= cutoff2")
    args = parser.parse_args()

    conversations = load_conversations_jsonl(PROCESSED_DIR / f"conversations_{args.brand}.jsonl")
    cluster_assignments = json.loads((PROCESSED_DIR / f"cluster_assignments_{args.brand}.json").read_text(encoding="utf-8"))
    intents_cfg = yaml.safe_load((CONFIGS_DIR / "intents.yaml").read_text(encoding="utf-8"))["intents"]

    cluster_to_intents: dict[int, list[dict]] = {}
    for intent in intents_cfg:
        for c in intent["source_clusters"]:
            cluster_to_intents.setdefault(c, []).append(intent)

    def assign_intent(cluster: int, cleaned_text: str) -> str | None:
        """Resolve a conversation's final intent label given its raw cluster.

        Most clusters map to exactly one intent (a straight merge). Where
        configs/intents.yaml documents a cluster as split across multiple
        intents (the raw cluster blended two real topics), disambiguate
        per-message using each candidate intent's `positive_signals`
        keyword list -- the intent with the most keyword hits in the
        message wins; ties/no-hits fall back to the first-listed
        (documented) candidate for that cluster.
        """
        candidates = cluster_to_intents.get(cluster)
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]["name"]
        best_intent, best_score = candidates[0]["name"], -1
        for cand in candidates:
            score = sum(1 for sig in cand.get("positive_signals", []) if sig in cleaned_text)
            if score > best_score:
                best_score = score
                best_intent = cand["name"]
        return best_intent

    # Build per-conversation records: id, root message, date, intent label
    # (None if filtered out during clustering -- non-English/too-short --
    # or if its cluster wasn't mapped to any final intent).
    records = []
    n_no_cluster = 0
    n_unmapped_cluster = 0
    for c in conversations:
        conv_id = c["conversation_id"]
        cust_msgs = [m["text"] for m in c["messages"] if m["role"] == "customer"]
        agent_msgs = [m["text"] for m in c["messages"] if m["role"] == "agent"]
        if not cust_msgs:
            continue
        cluster = cluster_assignments.get(conv_id)
        if cluster is None:
            n_no_cluster += 1
            continue
        root_msg = cust_msgs[0]
        intent = assign_intent(cluster, clean_for_modeling(root_msg))
        if intent is None:
            n_unmapped_cluster += 1
            continue
        first_dt = None
        for m in c["messages"]:
            if m["role"] == "customer":
                try:
                    first_dt = parse_twitter_datetime(m["created_at"])
                except ValueError:
                    pass
                break
        if first_dt is None:
            continue
        records.append({
            "conversation_id": conv_id,
            "root_message": root_msg,
            "customer_messages": cust_msgs,
            "agent_messages": agent_msgs,
            "resolution": agent_msgs[-1] if agent_msgs else None,
            "intent": intent,
            "cluster": cluster,
            "created_at": m["created_at"],
            "_dt": first_dt,
        })

    print(f"Labeled {len(records)} conversations (skipped {n_no_cluster} not clustered, "
          f"{n_unmapped_cluster} unmapped clusters).")

    # Temporal split
    from datetime import datetime, timezone
    cutoff1 = datetime.strptime(args.cutoff1, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    cutoff2 = datetime.strptime(args.cutoff2, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    dates_by_id = {r["conversation_id"]: r["_dt"] for r in records}
    split = temporal_split(dates_by_id, cutoff1, cutoff2)

    split_of_id = {}
    for cid in split.train_ids:
        split_of_id[cid] = "train"
    for cid in split.dev_ids:
        split_of_id[cid] = "dev"
    for cid in split.test_ids:
        split_of_id[cid] = "test"

    # Leakage analysis
    texts_by_id = {r["conversation_id"]: r["root_message"] for r in records}
    dup_analysis = analyze_duplicates(texts_by_id)

    hash_to_ids = {}
    for cid, text in texts_by_id.items():
        h = content_hash(normalize_for_dedup(text))
        hash_to_ids.setdefault(h, []).append(cid)
    cross_split = cross_split_duplicate_overlap(hash_to_ids, split_of_id)

    # Compare against what a NAIVE RANDOM split would have looked like, to
    # make the "why temporal is more honest" argument concrete rather than
    # asserted.
    import random
    rng = random.Random(42)
    all_ids = list(texts_by_id.keys())
    shuffled = all_ids[:]
    rng.shuffle(shuffled)
    n_test_random = len(split.test_ids)
    random_test_ids = set(shuffled[:n_test_random])
    random_split_of_id = {cid: ("test" if cid in random_test_ids else "train") for cid in all_ids}
    random_cross_split = cross_split_duplicate_overlap(hash_to_ids, random_split_of_id)

    # Class distribution per split
    from collections import Counter
    intent_by_id = {r["conversation_id"]: r["intent"] for r in records}
    train_dist = Counter(intent_by_id[i] for i in split.train_ids)
    dev_dist = Counter(intent_by_id[i] for i in split.dev_ids)
    test_dist = Counter(intent_by_id[i] for i in split.test_ids)

    leakage_report = {
        "brand": args.brand,
        "total_labeled_conversations": len(records),
        "split_sizes": {
            "train": len(split.train_ids),
            "dev": len(split.dev_ids),
            "test": len(split.test_ids),
        },
        "split_cutoffs": {"cutoff1": split.cutoff1, "cutoff2": split.cutoff2},
        "duplicate_analysis": dup_analysis,
        "temporal_split_cross_split_duplicates": cross_split,
        "naive_random_split_cross_split_duplicates": random_cross_split,
        "intent_distribution": {
            "train": dict(train_dist), "dev": dict(dev_dist), "test": dict(test_dist),
        },
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "leakage_analysis.json").write_text(json.dumps(leakage_report, indent=2, default=str), encoding="utf-8")

    md = [
        "# Leakage Analysis",
        "",
        f"Brand: {args.brand}",
        f"Total labeled conversations: {len(records)}",
        f"Split sizes: train={len(split.train_ids)}, dev={len(split.dev_ids)}, test={len(split.test_ids)}",
        f"Split cutoffs: train < {args.cutoff1}, dev in [{args.cutoff1}, {args.cutoff2}), test >= {args.cutoff2}",
        "",
        "## Duplicate analysis (normalized-text exact matches)",
        f"- Duplicate rate: {dup_analysis['duplicate_rate']:.2%} of documents belong to a duplicate group",
        f"- {dup_analysis['n_duplicate_groups']} duplicate groups found "
        f"({dup_analysis['n_documents_in_duplicate_groups']} documents total)",
        "",
        "## Cross-split leakage comparison",
        "",
        "| Split strategy | Duplicate groups spanning splits | Documents involved |",
        "|---|---|---|",
        f"| Temporal (this project) | {cross_split['duplicate_groups_spanning_multiple_splits']} | "
        f"{cross_split['documents_in_cross_split_duplicate_groups']} |",
        f"| Naive random (for comparison) | {random_cross_split['duplicate_groups_spanning_multiple_splits']} | "
        f"{random_cross_split['documents_in_cross_split_duplicate_groups']} |",
        "",
    ]
    (REPORTS_DIR / "leakage_analysis.md").write_text("\n".join(md), encoding="utf-8")

    # Write final labeled dataset
    out_path = PROCESSED_DIR / f"labeled_{args.brand}.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for r in records:
            r = dict(r)
            r.pop("_dt")
            r["split"] = split_of_id.get(r["conversation_id"], "unassigned")
            f.write(json.dumps(r) + "\n")

    print(f"Wrote {out_path}")
    print(f"Wrote reports/leakage_analysis.json and reports/leakage_analysis.md")
    print(json.dumps(leakage_report["split_sizes"], indent=2))
    print("Cross-split duplicate comparison:", cross_split, "vs random:", random_cross_split)


if __name__ == "__main__":
    main()
