"""
Deterministic mock LLM provider.

Not an attempt to simulate a good LLM. It exists so that the rest of the
pipeline -- generation, grounding validation, the judge harness, the API,
and the test suite -- can run end-to-end, deterministically, with zero
network access and zero API key, per the assignment's explicit
requirement ("the application must still have a deterministic/mock mode
so that the backend tests can run without an API key").

Every output is tagged `is_mock=True`. Callers that report evaluation
numbers (scripts/evaluate_generation.py, scripts/run_llm_judge.py) must
refuse to present mock-derived scores as real LLM-judge or generation-
quality results -- see README "LLM execution status" for exactly what
that means for this project's reported numbers.

Behavior, by design, is template-based and fully explainable:
  - "generate a reply": if evidence (retrieved historical resolutions) was
    included in the prompt, the mock extracts the single most similar
    historical resolution and returns a lightly-templated paraphrase of
    it, with grounded_claims pointing at that evidence and
    unsupported_claims empty. If no evidence was included, it returns an
    explicit "insufficient evidence" response instead of fabricating one
    -- mirroring the required real-LLM behavior, deterministically.
  - "judge a response": returns fixed neutral-low scores with a reason
    string that says plainly this is a mock judgment, not a real
    assessment -- so it can never be mistaken for a real judge score in
    an aggregated report (any pipeline computing e.g. mean judge score
    across many mock outputs will visibly get a suspicious, uniform
    number, which is intentional -- a real judge's scores must vary
    example to example, and this one deliberately doesn't).
"""
from __future__ import annotations

import json
import re

from app.providers.llm.base import LLMProvider, LLMResponse


class MockLLMProvider(LLMProvider):
    provider_name = "mock"

    def complete(self, system_prompt: str, user_prompt: str, *, temperature: float = 0.2,
                 max_tokens: int = 600) -> LLMResponse:
        if "\"task\": \"judge_response\"" in user_prompt or '"task": "judge_response"' in user_prompt:
            text = self._mock_judge(user_prompt)
        else:
            text = self._mock_generate(user_prompt)
        return LLMResponse(text=text, provider=self.provider_name, model="mock-deterministic-v1", is_mock=True)

    def _mock_generate(self, user_prompt: str) -> str:
        try:
            payload = json.loads(user_prompt)
        except (json.JSONDecodeError, TypeError):
            payload = {}

        evidence = payload.get("evidence", [])
        customer_message = payload.get("customer_message", "")
        intent = payload.get("intent", "unknown")

        if not evidence:
            result = {
                "draft_reply": (
                    "I don't have sufficient historical evidence to safely answer this. "
                    "This will be routed to a team member for review."
                ),
                "grounded_claims": [],
                "unsupported_claims": [],
                "confidence": 0.0,
                "evidence_insufficient": True,
            }
            return json.dumps(result)

        top = evidence[0]
        resolution_text = (top.get("resolution") or "").strip()
        resolution_snippet = resolution_text[:180] if resolution_text else ""

        if resolution_snippet:
            draft_reply = (
                f"Hi, thanks for reaching out. Based on how we've handled similar {intent.replace('_', ' ')} "
                f"cases: {resolution_snippet}"
                + ("..." if len(resolution_text) > 180 else "")
                + " Let us know if you need anything else."
            )
            grounded_claims = [resolution_snippet]
        else:
            draft_reply = (
                "Thanks for reaching out about this. We're looking into it and will follow up shortly."
            )
            grounded_claims = []

        result = {
            "draft_reply": draft_reply,
            "grounded_claims": grounded_claims,
            "unsupported_claims": [],
            "confidence": round(min(top.get("similarity", 0.5) + 0.1, 0.95), 2),
            "evidence_insufficient": False,
        }
        return json.dumps(result)

    def _mock_judge(self, user_prompt: str) -> str:
        # Deliberately uniform/neutral, low-information scores -- see
        # module docstring on why this must NOT vary example to example
        # (a real judge's scores should; a mock's shouldn't pretend to).
        result = {
            "correctness": 2,
            "groundedness": 2,
            "helpfulness": 2,
            "completeness": 2,
            "actionability": 2,
            "brand_consistency": 2,
            "safety": 4,
            "overall": 2.3,
            "reason": "[MOCK JUDGE -- not a real assessment. No live LLM available in this "
                      "sandbox environment; this is a fixed placeholder score, not a judgment "
                      "about response quality. Configure LLM_PROVIDER=groq or gemini with a "
                      "real API key to get real judge scores.]",
            "is_mock": True,
        }
        return json.dumps(result)
