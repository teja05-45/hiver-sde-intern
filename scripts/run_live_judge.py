#!/usr/bin/env python3
"""
Run the live Groq judge on the 50 human_review_subset examples.

Makes ONE Groq call per example (all 7 dimensions in one response).
Writes reports/judge_human_agreement.json with the results.

NOTE: human scores in the CSV are proxy scores derived from golden labels
(see scripts/score_human_proxy.py). This is a validation of the judge
pipeline, not a real human agreement study.
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.core.config import get_settings
from app.providers.llm.factory import get_llm_provider
from app.evaluation.agreement import compute_agreement

REPO_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
GOLDEN_DIR = REPO_ROOT / "data" / "golden"
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


def build_judge_prompt(customer_message: str, generated_reply: str,
                        expected_intent: str | None = None) -> str:
    return json.dumps({
        "task": "judge_response",
        "customer_message": customer_message,
        "evidence_summary": "See generated reply and customer message for context.",
        "generated_reply": generated_reply,
        "expected_intent": expected_intent,
    })


def judge_single(provider, message: str, reply: str,
                  expected_intent: str | None = None) -> dict | None:
    prompt = build_judge_prompt(message, reply, expected_intent)
    try:
        response = provider.complete(JUDGE_SYSTEM_PROMPT, prompt, temperature=0.0, max_tokens=300)
        # NOTE: LLMResponse has no parse_error attribute -- checking for one here
        # raised AttributeError on every successful call and the bare `except`
        # below counted real successes as failures (50/50 "failed" in the
        # committed report). Parsing failures are handled by the json.loads guard.
        text = response.text.strip().strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        data = json.loads(text)
        return {d: int(data[d]) for d in JUDGE_DIMENSIONS}
    except Exception as e:
        print(f"  [judge error: {e}]", file=sys.stderr)
        return None


def main():
    settings = get_settings()
    if settings.is_mock_mode():
        print("ERROR: judge-human agreement requires a LIVE LLM provider.")
        print(f"Current: LLM_PROVIDER={settings.llm_provider}, mock={settings.is_mock_mode()}")
        print("Set LLM_PROVIDER=groq (or gemini) with a real API key in .env")
        sys.exit(1)

    provider = get_llm_provider(settings)
    print(f"Provider: {settings.llm_provider} | Model: {settings.llm_model_name or 'default'}")
    print(f"Mock mode: {settings.is_mock_mode()}")

    input_path = GOLDEN_DIR / "human_review_subset.csv"
    if not input_path.exists():
        print(f"ERROR: {input_path} not found. Run compare_judge_to_human.py prepare first.")
        sys.exit(1)

    with input_path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print(f"Reviewing {len(rows)} examples with live Groq judge...\n")

    # Collect per-dimension human vs judge pairs
    dim_pairs: dict[str, list[tuple[int, int]]] = {d: [] for d in JUDGE_DIMENSIONS}
    success_count = 0
    fail_count = 0

    t0 = time.time()
    for i, row in enumerate(rows):
        row_id = row["id"]
        human_vals = {}
        for d in JUDGE_DIMENSIONS:
            v = row.get(f"human_{d}", "").strip()
            if v:
                human_vals[d] = int(v)

        if not human_vals:
            print(f"  [{i+1}/{len(rows)}] {row_id}: SKIP (no human scores)")
            continue

        judge_vals = judge_single(provider, row["message"], row.get("generated_reply", ""),
                                   row.get("true_intent"))
        if judge_vals is None:
            fail_count += 1
            print(f"  [{i+1}/{len(rows)}] {row_id}: JUDGE FAILED")
            continue

        success_count += 1
        for d in JUDGE_DIMENSIONS:
            if d in human_vals:
                dim_pairs[d].append((human_vals[d], judge_vals[d]))

        elapsed = time.time() - t0
        rate = (i + 1) / elapsed if elapsed > 0 else 0
        eta = (len(rows) - i - 1) / rate if rate > 0 else 0
        print(f"  [{i+1}/{len(rows)}] {row_id} — {elapsed:.0f}s, {rate:.1f}/s, ETA {eta:.0f}s  "
              f"judge={judge_vals}", end="\r")

    print()  # newline
    total_time = time.time() - t0
    print(f"\nDone: {success_count} judged, {fail_count} failed, {total_time:.0f}s total")

    # Compute agreement per dimension
    results = []
    print("\n=== Agreement results ===")
    for d in JUDGE_DIMENSIONS:
        pairs = dim_pairs[d]
        if not pairs:
            print(f"  {d:20s} n=0  (no data)")
            continue
        human_scores = [h for h, j in pairs]
        judge_scores = [j for h, j in pairs]
        agreement = compute_agreement(human_scores, judge_scores, d)
        results.append(agreement)
        print(f"  {d:20s} n={agreement.n:2d}  "
              f"exact={agreement.exact_agreement_rate:.1%}  "
              f"adjacent={agreement.adjacent_agreement_rate:.1%}  "
              f"kappa={agreement.weighted_kappa}  "
              f"spearman_r={agreement.spearman_r}")

    report = {
        "n_examples": len(rows),
        "n_judged": success_count,
        "n_failed": fail_count,
        "is_mock_judge": False,
        "judge_model": settings.llm_model_name or "unknown",
        "judge_provider": settings.llm_provider,
        "human_scoring_method": "proxy_scores_from_golden_labels (scripts/score_human_proxy.py)",
        "human_scoring_note": "Human scores are proxy scores derived from golden labels, evidence, "
                               "and generated replies. NOT real human judgments. This validates the "
                               "judge pipeline end-to-end, not human-judge agreement.",
        "per_dimension": [r.__dict__ for r in results],
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / "judge_human_agreement.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
