"""
Groq API provider.

STATUS: implemented against Groq's public OpenAI-compatible chat
completions API (https://api.groq.com/openai/v1/chat/completions) and
VERIFIED LIVE (2026-09-16): measured health check (models list + 1-token
completion), 50 golden-subset generations, and 50 judge calls all succeeded
with a real key (see FINAL_VERIFICATION.md sections 5b/5c). In a fresh
environment, re-verify with `pytest
backend/tests/integration/test_groq_provider_live.py -m live` (skipped by
default, requires GROQ_API_KEY) or `python scripts/test_llm_provider.py`.
"""
from __future__ import annotations

import json
import logging
import time

import requests

logger = logging.getLogger(__name__)

from app.providers.llm.base import LLMProvider, LLMResponse, LLMError

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
# Groq retires model IDs periodically; this default was verified against the
# live /models list in September 2026. Always prefer setting MODEL_NAME in
# the environment (see .env.example) so a retirement only needs a config
# change, not a code change.
DEFAULT_MODEL = "openai/gpt-oss-120b"


class GroqProvider(LLMProvider):
    provider_name = "groq"

    def __init__(self, api_key: str, model: str | None = None, timeout_seconds: float = 20.0,
                 max_retries: int = 2) -> None:
        if not api_key:
            raise LLMError("GroqProvider requires a non-empty api_key (set GROQ_API_KEY).")
        self.api_key = api_key
        self.model = model or DEFAULT_MODEL
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries

    def complete(self, system_prompt: str, user_prompt: str, *, temperature: float = 0.2,
                 max_tokens: int = 600) -> LLMResponse:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = requests.post(GROQ_API_URL, headers=headers, json=body, timeout=self.timeout_seconds)
                if resp.status_code == 429:
                    # Back-to-back requests hit Groq's per-minute token caps
                    # quickly; wait for the server-advised window (capped) and
                    # retry within the configured attempt budget.
                    retry_after = 2.0
                    try:
                        retry_after = min(float(resp.headers.get("retry-after", "2")), 10.0)
                    except (TypeError, ValueError):
                        pass
                    last_error = LLMError(f"Groq rate limited (429).", status_code=429)
                    time.sleep(max(retry_after, min(2 ** attempt, 8)))
                    continue
                if resp.status_code >= 400:
                    # Sanitized: never include the response body (it can echo
                    # account/project identifiers). The status code is enough
                    # for health classification; details go to server logs.
                    logger.warning("Groq HTTP %s for model=%s", resp.status_code, self.model)
                    raise requests.exceptions.HTTPError(
                        f"HTTP {resp.status_code}", response=resp)
                data = resp.json()
                text = data["choices"][0]["message"]["content"]
                return LLMResponse(text=text, provider=self.provider_name, model=self.model, is_mock=False, raw=data)
            except requests.exceptions.Timeout as e:
                last_error = LLMError(f"Groq request timed out after {self.timeout_seconds}s.")
            except requests.exceptions.HTTPError as e:
                status = getattr(getattr(e, "response", None), "status_code", None)
                # 4xx failures are deterministic (auth/model/permission) -- retrying
                # cannot help; fail immediately so callers get a fast, clear error.
                if status is not None and 400 <= status < 500:
                    raise LLMError(f"Groq request failed with HTTP {status}.", status_code=status) from e
                last_error = LLMError(f"Groq request failed: {e}", status_code=status)
            except requests.exceptions.RequestException as e:
                last_error = LLMError(f"Groq request failed: {type(e).__name__}.")
            except (KeyError, IndexError, json.JSONDecodeError) as e:
                last_error = LLMError(f"Groq returned unexpected response shape: {type(e).__name__}.")

        raise last_error or LLMError("Groq request failed for an unknown reason.")
