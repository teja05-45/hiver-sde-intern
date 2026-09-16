"""
LLM-as-judge harness.

Judges a generated response on 7 dimensions (correctness, groundedness,
helpfulness, completeness, actionability, brand_consistency, safety),
each 0-4, per the assignment's rubric. Structured JSON output, validated
before use -- malformed judge output must not crash the evaluation run.

The judge is explicitly NOT assumed reliable. `scripts/compare_judge_to_human.py`
is the required check: it compares judge scores against a human-labeled
subset and reports agreement statistics (Spearman correlation, weighted
Cohen's kappa, exact/adjacent agreement) per dimension, so weaknesses are
visible rather than assumed away.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from app.providers.llm.base import LLMProvider, LLMError

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

JUDGE_DIMENSIONS = ["correctness", "groundedness", "helpfulness", "completeness",
                     "actionability", "brand_consistency", "safety"]


@dataclass
class JudgeScore:
    correctness: int
    groundedness: int
    helpfulness: int
    completeness: int
    actionability: int
    brand_consistency: int
    safety: int
    overall: float
    reason: str
    is_mock: bool
    parse_error: str | None = None

    def as_dict(self) -> dict:
        return {d: getattr(self, d) for d in JUDGE_DIMENSIONS} | {
            "overall": self.overall, "reason": self.reason, "is_mock": self.is_mock,
        }


def build_judge_prompt(customer_message: str, evidence_summary: str, generated_reply: str,
                        expected_intent: str | None = None) -> str:
    payload = {
        "task": "judge_response",
        "customer_message": customer_message,
        "evidence_summary": evidence_summary,
        "generated_reply": generated_reply,
        "expected_intent": expected_intent,
    }
    return json.dumps(payload)


def parse_judge_output(raw_text: str, is_mock: bool) -> JudgeScore:
    try:
        cleaned = raw_text.strip().strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
        data = json.loads(cleaned)
        return JudgeScore(
            correctness=int(data["correctness"]), groundedness=int(data["groundedness"]),
            helpfulness=int(data["helpfulness"]), completeness=int(data["completeness"]),
            actionability=int(data["actionability"]), brand_consistency=int(data["brand_consistency"]),
            safety=int(data["safety"]), overall=float(data.get("overall", 0.0)),
            reason=str(data.get("reason", "")), is_mock=is_mock,
        )
    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
        return JudgeScore(0, 0, 0, 0, 0, 0, 0, 0.0, "", is_mock, parse_error=str(e))


def judge_response(provider: LLMProvider, customer_message: str, evidence_summary: str,
                    generated_reply: str, expected_intent: str | None = None) -> JudgeScore:
    prompt = build_judge_prompt(customer_message, evidence_summary, generated_reply, expected_intent)
    try:
        response = provider.complete(JUDGE_SYSTEM_PROMPT, prompt, temperature=0.0)
    except LLMError as e:
        return JudgeScore(0, 0, 0, 0, 0, 0, 0, 0.0, "", getattr(provider, "provider_name", "") == "mock",
                           parse_error=str(e))
    return parse_judge_output(response.text, response.is_mock)
