"""
Gemini API provider.

STATUS: same caveat as groq_provider.py -- implemented against Google's
public Generative Language API
(https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent)
as documented at project build time, but NOT executed or verified in this
sandbox (no network access). Verify against the real API before trusting
in production.
"""
from __future__ import annotations

import logging
import time

import requests

logger = logging.getLogger(__name__)

from app.providers.llm.base import LLMProvider, LLMResponse, LLMError

DEFAULT_MODEL = "gemini-1.5-flash"


class GeminiProvider(LLMProvider):
    provider_name = "gemini"

    def __init__(self, api_key: str, model: str | None = None, timeout_seconds: float = 20.0,
                 max_retries: int = 2) -> None:
        if not api_key:
            raise LLMError("GeminiProvider requires a non-empty api_key (set GEMINI_API_KEY).")
        self.api_key = api_key
        self.model = model or DEFAULT_MODEL
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries

    def complete(self, system_prompt: str, user_prompt: str, *, temperature: float = 0.2,
                 max_tokens: int = 600) -> LLMResponse:
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent?key={self.api_key}"
        )
        body = {
            "system_instruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens},
        }

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = requests.post(url, json=body, timeout=self.timeout_seconds)
                if resp.status_code == 429:
                    last_error = LLMError(f"Gemini rate limited (429).", status_code=429)
                    time.sleep(min(2 ** attempt, 8))
                    continue
                if resp.status_code >= 400:
                    logger.warning("Gemini HTTP %s for model=%s", resp.status_code, self.model)
                    raise requests.exceptions.HTTPError(
                        f"HTTP {resp.status_code}", response=resp)
                data = resp.json()
                text = data["candidates"][0]["content"]["parts"][0]["text"]
                return LLMResponse(text=text, provider=self.provider_name, model=self.model, is_mock=False, raw=data)
            except requests.exceptions.Timeout as e:
                last_error = LLMError(f"Gemini request timed out after {self.timeout_seconds}s.")
            except requests.exceptions.HTTPError as e:
                status = getattr(getattr(e, "response", None), "status_code", None)
                if status is not None and 400 <= status < 500:
                    raise LLMError(f"Gemini request failed with HTTP {status}.", status_code=status) from e
                last_error = LLMError(f"Gemini request failed: {e}", status_code=status)
            except requests.exceptions.RequestException as e:
                last_error = LLMError(f"Gemini request failed: {type(e).__name__}.")
            except (KeyError, IndexError) as e:
                last_error = LLMError(f"Gemini returned unexpected response shape: {type(e).__name__}.")

        raise last_error or LLMError("Gemini request failed for an unknown reason.")


def get_llm_provider(settings) -> LLMProvider:
    """DEPRECATED shim: the provider factory now lives in
    app.providers.llm.factory. Kept so existing imports
    (`from app.providers.llm.gemini_provider import get_llm_provider`)
    keep working; new code should import from the factory module."""
    import warnings

    from app.providers.llm.factory import get_llm_provider as _factory_get_llm_provider

    warnings.warn(
        "Importing get_llm_provider from gemini_provider is deprecated; "
        "import from app.providers.llm.factory instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _factory_get_llm_provider(settings)
