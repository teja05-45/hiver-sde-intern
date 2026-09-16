"""
LLM provider abstraction.

`LLMProvider` is the seam between "call an LLM" and everything that uses
one (response generation, the LLM-as-judge). Three implementations:

  - MockLLMProvider: deterministic, rule-based, no network/API key needed.
    This is what every script in this repo actually runs against in this
    sandbox (no network access -- see README "LLM execution status").
    It's not trying to *simulate* a good LLM; it's a honest stand-in that
    lets the rest of the pipeline (grounding checks, escalation, judge
    harness, tests) run end-to-end and be verified structurally, while
    being impossible to mistake for a real evaluation result (every mock
    output is tagged accordingly, and the escalation policy treats mock
    mode as informational only, never as a source of AUTO decisions in
    the automation-precision reporting).
  - GroqProvider / GeminiProvider: real implementations using each
    provider's HTTP API via `requests`. Untested in this sandbox (no
    network access to verify against the real API), but structurally
    complete: correct endpoint, auth header, request/response shape per
    each provider's public API docs as of this project's build date.
    Anyone running this outside the sandbox with a real API key should
    smoke-test these before trusting them (see README "LLM execution
    status" for exactly what was and wasn't verified).

Selection is via LLM_PROVIDER env var (backend/app/core/config.py),
defaulting to "mock" so nothing here ever requires a key to run.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str
    is_mock: bool
    raw: dict | None = None


class LLMError(Exception):
    """Raised on any LLM call failure (timeout, rate limit, malformed
    response, auth failure). Callers (generation.py, judge.py) must catch
    this and route to escalation -- never fabricate a response on failure.

    `status_code` carries the sanitized HTTP status (when the failure came
    from an HTTP response) so health checks can return stable error codes
    without exposing provider response bodies.
    """

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class LLMProvider(ABC):
    provider_name: str = "base"

    @abstractmethod
    def complete(self, system_prompt: str, user_prompt: str, *, temperature: float = 0.2,
                 max_tokens: int = 600) -> LLMResponse:
        ...
