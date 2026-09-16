#!/usr/bin/env python3
"""Run the LLM judge over golden-set examples with the CONFIGURED provider.

This is an EXPLICIT evaluation command — never run at startup. For each
selected golden example it:
  1. retrieves historical evidence (train+dev index; no leakage),
  2. GENERATES a reply with the live provider (so the judge scores real
     generation output, not stale mock placeholders),
  3. calls the provider as judge (all 7 dimensions, one call),
  4. validates the judge's JSON strictly (missing/malformed = recorded failure),
  5. writes reports/judge_live_judge_run.json with per-example artifacts,
     failure records, and run metadata.

Honesty rules enforced here:
  - Requires a live provider; refuses to run in mock mode (mock judge scores
    are fixed placeholders and must never be aggregated into a report).
  - Generation and judging failures are recorded per example, never skipped
    silently and never fabricated.
  - The "human agreement" story is unchanged: this run produces JUDGE scores
    only. Human agreement remains NOT AVAILABLE until a human actually rates
    these replies (see docs/decision-log.md #9).

Usage:
    python scripts/run_llm_judge.py --n 10          # small verified run
    python scripts/run_llm_judge.py --n 50          # full subset
    python scripts/run_llm_judge.py --n 5 --dry-run # validate inputs only
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.core.config import get_settings
from app.providers.llm.factory import get_llm_provider
from app.generation.generator import generate_response
from app.retrieval.embeddings import TfidfEmbeddingProvider
from app.retrieval.retriever import HistoricalRetriever
from app.evaluation.agreement import compute_agreement

REPO_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_DIR = REPO_ROOT / "data" / "golden"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
REPORTS_DIR = REPO_ROOT / "reports"

JUDGE_DIMENSIONS = ["correctness", "groundedness", "helpfulness",
                    "completeness", "actionability", "brand_consistency", "safety"]

JUDGE_SYSTEM_PROMPT = """You are an impartial evaluator of customer support responses. Score the given
response on each dimension using an integer scale 0-4 (0=very poor, 4=excellent):
- correctness: does the response accurately address the customer's actual issue?
- groundedness: is every claim in the response supported by the provided historical evidence?
- helpfulness: does it move the customer toward resolution?
- completeness: does it address all parts of the customer's message?
- actionability: are next steps (if any) clear and concrete?
- brand_consistency: does the tone match professional customer support?
- safety: does the response avoid making unsafe/inappropriate promises or disclosures?

Respond with ONLY a JSON object:
{"correctness": int, "groundedness": int, "helpfulness": int, "completeness": int,
 "actionability": int, "brand_consistency": int, "safety": int, "overall": float, "reason": string}
"""


def judge_single(provider, message: str, reply: str,
                 expected_intent: str | None = None) -> dict | None:
    """One judge call; returns the 7 dimension scores or None on any failure."""
    payload = json.dumps({
        "task": "judge_response",
        "customer_message": message,
        "evidence_summary": "The historical evidence supplied to the generator (not shown to save tokens).",
        "generated_reply": reply,
        "expected_intent": expected_intent,
    })
    try:
        response = provider.complete(JUDGE_SYSTEM_PROMPT, payload, temperature=0.0, max_tokens=600)
        text = response.text.strip().strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
        data = json.loads(text)  # malformed/missing fields -> failure, never guessed
        return {d: int(data[d]) for d in JUDGE_DIMENSIONS}
    except Exception as e:  # noqa: BLE001 — record the failure, never crash the run
        print(f"    [judge error: {type(e).__name__}: {str(e)[:120]}]", file=sys.stderr)
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50, help="number of golden examples")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true", help="validate inputs, make no API calls")
    args = parser.parse_args()

    settings = get_settings()
    if settings.is_mock_mode():
        print("REFUSING TO RUN: mock mode. The mock judge returns fixed placeholder")
        print("scores that must never be aggregated into an evaluation report.")
        print("Configure LLM_PROVIDER=groq (or gemini) with a real API key, then rerun.")
        return 1

    golden_path = GOLDEN_DIR / "golden_set.csv"
    if not golden_path.exists():
        print(f"ERROR: {golden_path} not found.")
        return 1
    with golden_path.open(encoding="utf-8") as f:
        golden = list(csv.DictReader(f))
    selected = random.Random(args.seed).sample(golden, min(args.n, len(golden)))

    if args.dry_run:
        print(f"DRY RUN: would generate+judge {len(selected)} examples with "
              f"{settings.llm_provider} ({settings.llm_model_name or 'default model'}). No calls made.")
        return 0

    # Retrieval index over train+dev only (same rule as the runtime agent).
    print("Building retrieval index (train+dev)...")
    train_dev = []
    with (PROCESSED_DIR / "labeled_AmazonHelp.jsonl").open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("split") in ("train", "dev"):
                train_dev.append(r)
    retriever = HistoricalRetriever(TfidfEmbeddingProvider()).build_index(train_dev)

    provider = get_llm_provider(settings)
    model_name = settings.llm_model_name or "(provider default)"
    print(f"LLM JUDGE RUN — provider={settings.llm_provider} model={model_name}")
    print(f"Generate + judge {len(selected)} examples...\n")

    per_example, dim_pairs = [], {d: [] for d in JUDGE_DIMENSIONS}
    n_gen_ok = n_gen_failed = n_judged = n_judge_failed = 0
    t0 = time.time()

    for i, row in enumerate(selected):
        entry = {"id": row["id"], "message": row["message"][:300]}

        gen = generate_response(provider, row["message"], row["intent"], retriever.retrieve(row["message"], k=5))
        if gen.parse_error or not gen.draft_reply:
            n_gen_failed += 1
            entry.update({"generation_status": "FAILED", "generation_error": gen.parse_error or "empty draft",
                          "judge_status": "SKIPPED"})
            per_example.append(entry)
            print(f"  [{i+1}/{len(selected)}] {row['id']}: GENERATION FAILED (recorded)")
            continue
        n_gen_ok += 1
        entry.update({
            "generation_status": "PASS",
            "generated_reply": gen.draft_reply[:600],
            "provider": gen.provider,
            "model": gen.model or model_name,
            "is_mock": gen.is_mock,
        })

        judge_vals = judge_single(provider, row["message"], gen.draft_reply, row["intent"])
        if judge_vals is None:
            n_judge_failed += 1
            entry["judge_status"] = "FAILED"
        else:
            n_judged += 1
            entry.update({"judge_status": "OK", "judge_scores": judge_vals})
        per_example.append(entry)

        elapsed = time.time() - t0
        rate = (i + 1) / elapsed if elapsed > 0 else 0
        eta = (len(selected) - i - 1) / rate if rate > 0 else 0
        print(f"  [{i+1}/{len(selected)}] {row['id']}: gen={entry['generation_status']} "
              f"judge={entry.get('judge_status')}  ({elapsed:.0f}s, ETA {eta:.0f}s)")

    # Aggregate judge score means (only over successfully judged examples).
    score_means = {}
    for d in JUDGE_DIMENSIONS:
        vals = [e["judge_scores"][d] for e in per_example if e.get("judge_scores")]
        score_means[d] = round(sum(vals) / len(vals), 3) if vals else None

    report = {
        "run_id": f"live_judge_{time.strftime('%Y%m%d_%H%M%S')}",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "provider": settings.llm_provider,
        "model": model_name,
        "n_examples": len(selected),
        "n_generation_ok": n_gen_ok,
        "n_generation_failed": n_gen_failed,
        "n_judged": n_judged,
        "n_judge_failed": n_judge_failed,
        "is_mock_judge": False,
        "mean_judge_scores": score_means,
        "human_agreement": "NOT_AVAILABLE",
        "human_agreement_note": (
            "This run produced JUDGE scores only. No human has rated these replies, "
            "so no human-agreement number exists and none is claimed."
        ),
        "per_example": per_example,
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / "judge_live_judge_run.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")
    print(f"Result: {n_gen_ok} generated, {n_judged} judged, "
          f"{n_gen_failed} generation failures, {n_judge_failed} judge failures.")
    return 0 if n_judged > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
