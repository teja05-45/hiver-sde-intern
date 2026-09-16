#!/usr/bin/env python3
"""
Two commands in one file:

    python scripts/compare_judge_to_human.py prepare --brand AmazonHelp --n 50
        Selects ~50 golden-set examples, generates a response for each
        (using whatever LLM_PROVIDER is configured -- mock by default),
        and writes data/golden/human_review_subset.csv with empty score
        columns for a human to fill in.

        IMPORTANT: if run in mock mode, the generated replies are
        placeholder mock output, NOT suitable for real human quality
        judgment -- rerun this with a real LLM_PROVIDER configured before
        actually asking a human to score the "generated_reply" column.

    python scripts/compare_judge_to_human.py compare --input <filled_csv>
        Given a human-filled copy of that CSV (with score columns filled
        in) AND a judge_scores.csv (from running the judge on the same
        examples), computes agreement statistics per dimension and writes
        reports/judge_human_agreement.json/.md.

This script intentionally does not fabricate either the human scores or
the judge scores -- both must be real inputs.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.core.config import get_settings  # noqa: E402
from app.providers.llm.factory import get_llm_provider  # noqa: E402
from app.evaluation.judge import JUDGE_DIMENSIONS, judge_response  # noqa: E402
from app.evaluation.agreement import compute_agreement  # noqa: E402
from app.generation.generator import generate_response  # noqa: E402
from app.retrieval.retriever import HistoricalRetriever  # noqa: E402
from app.retrieval.embeddings import TfidfEmbeddingProvider  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_DIR = REPO_ROOT / "data" / "golden"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
REPORTS_DIR = REPO_ROOT / "reports"

HUMAN_SCORE_COLUMNS = [f"human_{d}" for d in JUDGE_DIMENSIONS]
JUDGE_SCORE_COLUMNS = [f"judge_{d}" for d in JUDGE_DIMENSIONS]


def cmd_prepare(args: argparse.Namespace) -> None:
    settings = get_settings()
    provider = get_llm_provider(settings)

    with (GOLDEN_DIR / "golden_set.csv").open(encoding="utf-8") as f:
        golden = list(csv.DictReader(f))
    import random
    subset = random.Random(42).sample(golden, min(args.n, len(golden)))

    train_dev = []
    with (PROCESSED_DIR / f"labeled_{args.brand}.jsonl").open() as f:
        for line in f:
            r = json.loads(line)
            if r.get("split") in ("train", "dev"):
                train_dev.append(r)
    retriever = HistoricalRetriever(TfidfEmbeddingProvider()).build_index(train_dev)

    rows = []
    for r in subset:
        evidence = retriever.retrieve(r["message"], k=5)
        generated = generate_response(provider, r["message"], r["intent"], evidence)
        rows.append({
            "id": r["id"], "message": r["message"], "true_intent": r["intent"],
            "generated_reply": generated.draft_reply, "is_mock": generated.is_mock,
            **{c: "" for c in HUMAN_SCORE_COLUMNS},
        })

    out_path = GOLDEN_DIR / "human_review_subset.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    mock_warning = " -- REPLIES ARE MOCK PLACEHOLDERS, rerun with a real LLM_PROVIDER before scoring" \
        if settings.is_mock_mode() else ""
    print(f"Wrote {out_path} ({len(rows)} examples){mock_warning}")


def cmd_compare(args: argparse.Namespace) -> None:
    with Path(args.input).open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    settings = get_settings()
    provider = get_llm_provider(settings)

    results = []
    for dim in JUDGE_DIMENSIONS:
        human_scores, judge_scores = [], []
        for row in rows:
            human_val = row.get(f"human_{dim}", "").strip()
            if not human_val:
                continue
            score = judge_response(provider, row["message"], "", row["generated_reply"], row.get("true_intent"))
            if score.parse_error:
                continue
            human_scores.append(int(human_val))
            judge_scores.append(getattr(score, dim))
        if human_scores:
            results.append(compute_agreement(human_scores, judge_scores, dim))

    if not results:
        print("No human-scored rows found (all human_* columns empty). Fill in "
              f"{args.input} first.")
        return

    report = {"n_examples": len(rows), "is_mock_judge": settings.is_mock_mode(),
              "per_dimension": [r.__dict__ for r in results]}
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "judge_human_agreement.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    p_prepare = sub.add_parser("prepare")
    p_prepare.add_argument("--brand", type=str, required=True)
    p_prepare.add_argument("--n", type=int, default=50)
    p_prepare.set_defaults(func=cmd_prepare)

    p_compare = sub.add_parser("compare")
    p_compare.add_argument("--input", type=str, required=True)
    p_compare.set_defaults(func=cmd_compare)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
