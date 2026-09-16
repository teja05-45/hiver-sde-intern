"""
LLM provider factory.

Selection logic (all driven by app.core.config.Settings, i.e. environment
variables -- never hardcoded):

  LLM_PROVIDER=mock   -> MockLLMProvider (deterministic, offline, tagged is_mock=True)
  LLM_PROVIDER=groq   -> GroqProvider if GROQ_API_KEY is set, else mock (graceful fallback)
  LLM_PROVIDER=gemini -> GeminiProvider if GEMINI_API_KEY is set, else mock (graceful fallback)
  anything else       -> MockLLMProvider (with a logged warning, never a silent guess)

The fallback-to-mock-on-missing-key behavior is deliberate: the assignment
requires the app to run without an API key. But a silent fallback can hide a
configuration mistake, so callers that care should check
`settings.is_mock_mode()` (or /health's `mock_mode`) and surface it in the UI,
which the frontend does via the persistent MOCK MODE indicator.

Raising on invalid config (rather than falling back) is available explicitly:
construct GroqProvider/GeminiProvider directly -- they raise LLMError on an
empty key. The tests cover both behaviors.
"""
from __future__ import annotations

import logging

from app.core.config import Settings, get_settings
from app.providers.llm.base import LLMProvider, LLMError, LLMResponse
from app.providers.llm.gemini_provider import GeminiProvider
from app.providers.llm.groq_provider import GroqProvider
from app.providers.llm.mock_provider import MockLLMProvider

__all__ = [
    "GeminiProvider",
    "GroqProvider",
    "LLMError",
    "LLMProvider",
    "LLMResponse",
    "MockLLMProvider",
    "get_llm_provider",
]

logger = logging.getLogger(__name__)

_KNOWN_PROVIDERS = ("mock", "groq", "gemini")


def get_llm_provider(settings: Settings | None = None) -> LLMProvider:
    """Build the configured LLMProvider from settings (env-driven)."""
    s = settings or get_settings()

    if s.llm_provider not in _KNOWN_PROVIDERS:
        logger.warning("Unknown LLM_PROVIDER=%r; falling back to mock provider.", s.llm_provider)
        return MockLLMProvider()

    if s.is_mock_mode():
        return MockLLMProvider()

    if s.llm_provider == "groq":
        return GroqProvider(
            s.groq_api_key,
            model=s.llm_model_name or None,
            timeout_seconds=s.llm_timeout_seconds,
            max_retries=s.llm_max_retries,
        )
    # s.llm_provider == "gemini" (and not mock mode => key must be present)
    return GeminiProvider(
        s.gemini_api_key,
        model=s.llm_model_name or None,
        timeout_seconds=s.llm_timeout_seconds,
        max_retries=s.llm_max_retries,
    )
