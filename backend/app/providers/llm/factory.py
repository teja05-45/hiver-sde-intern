"""
LLM provider factory.

Selection is driven entirely by app.core.config.Settings (environment
variables) and is EXPLICIT -- there is no hidden fallback:

  LLM_PROVIDER=mock (or LLM_MODE=mock)      -> MockLLMProvider (deterministic,
                                               offline, tagged is_mock=True)
  LLM_PROVIDER=groq|gemini + key present    -> GroqProvider / GeminiProvider
  LLM_PROVIDER=groq|gemini + key MISSING    -> ProviderConfigError. The truthful
                                               state is NOT_CONFIGURED; the
                                               system must never quietly switch
                                               to mock, because mock output
                                               presented as live output is the
                                               exact failure mode this
                                               assignment forbids. Callers that
                                               want a deliberate offline run
                                               must set LLM_PROVIDER=mock (or
                                               LLM_MODE=mock) explicitly.
  unknown LLM_PROVIDER                      -> ProviderConfigError as well: a
                                               typo'd provider name is a
                                               configuration mistake, not a
                                               reason to run something else.

Raising on invalid config (rather than falling back) is the fail-loud
contract; the previous behavior (silent mock fallback on missing key) is
preserved nowhere -- tests that want mock construct MockLLMProvider directly
or set LLM_PROVIDER=mock.
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
    "ProviderConfigError",
    "get_llm_provider",
]

logger = logging.getLogger(__name__)

_KNOWN_PROVIDERS = ("mock", "groq", "gemini")


class ProviderConfigError(LLMError):
    """Raised when the provider configuration cannot support a live call
    (missing key, unknown provider name). This is a configuration error the
    operator must fix -- never silently worked around by degrading to mock."""


def get_llm_provider(settings: Settings | None = None) -> LLMProvider:
    """Build the configured LLMProvider from settings (env-driven).

    Never substitutes a different provider than the one configured: if the
    configuration cannot support the requested mode it raises
    ProviderConfigError with a sanitized, actionable message.
    """
    s = settings or get_settings()

    if s.llm_provider not in _KNOWN_PROVIDERS:
        raise ProviderConfigError(
            f"LLM_PROVIDER={s.llm_provider!r} is not a known provider "
            f"(expected one of {', '.join(_KNOWN_PROVIDERS)})."
        )

    if s.is_mock_mode():
        return MockLLMProvider()

    if s.llm_provider == "groq":
        if not s.groq_api_key:
            raise ProviderConfigError(
                "LLM_PROVIDER=groq but GROQ_API_KEY is not set. "
                "Set the key, or explicitly select offline mode with LLM_PROVIDER=mock."
            )
        return GroqProvider(
            s.groq_api_key,
            model=s.llm_model_name or None,
            timeout_seconds=s.llm_timeout_seconds,
            max_retries=s.llm_max_retries,
        )

    # s.llm_provider == "gemini"
    if not s.gemini_api_key:
        raise ProviderConfigError(
            "LLM_PROVIDER=gemini but GEMINI_API_KEY is not set. "
            "Set the key, or explicitly select offline mode with LLM_PROVIDER=mock."
        )
    return GeminiProvider(
        s.gemini_api_key,
        model=s.llm_model_name or None,
        timeout_seconds=s.llm_timeout_seconds,
        max_retries=s.llm_max_retries,
    )
