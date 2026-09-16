#!/usr/bin/env python3
"""
Fast version of compare_judge_to_human.py that makes ONE Groq call per
example (instead of 7 — one per dimension), asking the judge to score all
7 dimensions in a single response. This cuts API calls from 350 to 50.

Usage:
    python scripts/compare_judge_to_human_fast.py --input data/golden/human_review_subset.csv

Requires LLM_PROVIDER=groq (or gemini) with a real API key configured in .env.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.core.config import get_settings
from app.providers.llm.factory import get_llm_provider
from app.providers.llm.base import LLMResponse
from app.evaluation.agreement import compute_agreement

REPO_ROOT = Path(__file__).resolve().parents[1]
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


def build_judge_prompt(customer_message: str, evidence_summary: str,
                        generated_reply: str, expected_intent: str | None = None) -> str:
    return json.dumps({
        "task": "judge_response",
        "customer_message": customer_message,
        "evidence_summary": evidence_summary if evidence_summary else "No evidence provided.",
        "generated_reply": generated_reply,
        "expected_intent": expected_intent,
    })


def judge_single(provider, message: str, reply: str, expected_intent: str | None = None) -> dict | None:
    """Ask the LLM judge for all 7 scores in one call."""
    prompt = build_judge_prompt(message, "", reply, expected_intent)
    try:
        response = provider.complete(JUDGE_SYSTEM_PROMPT, prompt, temperature=0.0)
        # NOTE: LLMResponse has no parse_error attribute (fixed AttributeError
        # bug that marked every successful call as failed -- see
        # scripts/run_live_judge.py and docs/production-audit.md BUG 3).
        text = response.text.strip().strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        data = json.loads(text)
        return {d: int(data[d]) for d in JUDGE_DIMENSIONS}
    except Exception as e:
        print(f"  [judge error: {e}]", file=sys.stderr)
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: {input_path} not found.")
        sys.exit(1)

    settings = get_settings()
    if settings.is_mock_mode():
        print("ERROR: judge-human agreement requires a LIVE LLM provider.")
        print("Set LLM_PROVIDER=groq (or gemini) with a real API key in .env")
        sys.exit(1)

    provider = get_llm_provider(settings)

    with input_path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    human_cols = [f"human_{d}" for d in JUDGE_DIMENSIONS]

    # Collect per-dimension human vs judge pairs
    dim_pairs: dict[str, list[tuple[int, int]]] = {d: [] for d in JUDGE_DIMENSIONS}

    t0 = time.time()
    for i, row in enumerate(rows):
        human_vals = {}
        for d in JUDGE_DIMENSIONS:
            v = row.get(f"human_{d}", "").strip()
            if v:
                human_vals[d] = int(v)

        if not human_vals:
            continue

        judge_vals = judge_single(provider, row["message"], row.get("generated_reply", ""),
                                   row.get("true_intent"))
        if judge_vals is None:
            continue

        for d in JUDGE_DIMENSIONS:
            if d in human_vals:
                dim_pairs[d].append((human_vals[d], judge_vals[d]))

        elapsed = time.time() - t0
        rate = (i + 1) / elapsed if elapsed > 0 else 0
        eta = (len(rows) - i - 1) / rate if rate > 0 else 0
        print(f"  [{i+1}/{len(rows)}] {row['id']} — {elapsed:.0f}s elapsed, "
              f"{rate:.1f} calls/s, ETA {eta:.0f}s", end="\r")

    print()  # newline after progress

    # Compute agreement per dimension
    results = []
    for d in JUDGE_DIMENSIONS:
        pairs = dim_pairs[d]
        if not pairs:
            continue
        human_scores = [h for h, j in pairs]
        judge_scores = [j for h, j in pairs]
        agreement = compute_agreement(human_scores, judge_scores, d)
        results.append(agreement)
        print(f"  {d:20s} n={agreement.n:2d}  exact={agreement.exact_agreement_rate:.1%}  "
              f"adjacent={agreement.adjacent_agreement_rate:.1%}  "
              f"kappa={agreement.weighted_kappa}  spearman_r={agreement.spearman_r}")

    report = {
        "n_examples": len(rows),
        "is_mock_judge": False,
        "per_dimension": [r.__dict__ for r in results],
        "notes": "Generated with Groq live LLM judge. Human scores are proxy scores derived from golden labels."
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / "judge_human_agreement.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
