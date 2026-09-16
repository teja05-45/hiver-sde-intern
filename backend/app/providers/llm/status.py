"""LLM provider status / health checks.

Distinguishes CONFIGURED (an API key exists) from HEALTHY (a real, minimal API
call succeeded) — an environment variable existing proves nothing about the
provider actually working. This module is the single source of truth for
provider status; the API and the frontend render what it measures and never
derive "live" from an env name.

All health probes are cheap, time-boxed, and safe to call repeatedly:
  - reachability: GET /openai/v1/models (no tokens consumed)
  - end-to-end:   POST chat/completions with max_tokens=1 ("Reply with OK.")

Every returned field is sanitized: no keys, no raw provider response bodies,
no filesystem paths.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import requests

from app.core.config import Settings
from app.providers.llm.base import LLMProvider
from app.providers.llm.factory import get_llm_provider

logger = logging.getLogger(__name__)

MODELS_URLS = {
    "groq": "https://api.groq.com/openai/v1/models",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/models",
}

TIMEOUT_SECONDS = 10.0


@dataclass
class ProviderStatus:
    provider: str
    mode: str                      # "mock" | "live"
    configured: bool
    healthy: bool | None           # None = not checked / not applicable
    reachable: bool | None
    model: str | None
    latency_ms: int | None = None
    checked_at: float | None = None
    error_code: str | None = None
    error_message: str | None = None
    checks: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "provider": self.provider,
            "mode": self.mode,
            "configured": self.configured,
            "healthy": self.healthy,
            "reachable": self.reachable,
            "model": self.model,
            "latency_ms": self.latency_ms,
            "checked_at": self.checked_at,
            "error_code": self.error_code,
            "error_message": self.error_message,
            "checks": self.checks,
        }


def classify_http_error(status_code: int) -> str:
    """Sanitized, stable error codes for the known provider failure modes."""
    return {
        400: "PROVIDER_BAD_REQUEST",
        401: "PROVIDER_AUTH_FAILED",
        403: "PROVIDER_FORBIDDEN",
        404: "PROVIDER_MODEL_NOT_FOUND",
        429: "PROVIDER_RATE_LIMITED",
        500: "PROVIDER_SERVER_ERROR",
        502: "PROVIDER_SERVER_ERROR",
        503: "PROVIDER_SERVER_ERROR",
    }.get(status_code, f"PROVIDER_HTTP_{status_code}")


def _classify_exception(exc: Exception) -> str:
    import requests as _r
    if isinstance(exc, _r.exceptions.Timeout):
        return "PROVIDER_TIMEOUT"
    if isinstance(exc, _r.exceptions.ConnectionError):
        return "PROVIDER_UNREACHABLE"
    return "PROVIDER_UNAVAILABLE"


def get_provider_status(settings: Settings | None = None,
                        provider: LLMProvider | None = None,
                        check_health: bool = False) -> dict:
    """Return the provider status dict.

    check_health=False (default): configuration-only status, no network calls.
    check_health=True: performs a real, minimal verification (models list, then a
    1-token completion) and reports measured health. Never raises.
    """
    s = settings
    if s is None:
        from app.core.config import get_settings
        s = get_settings()

    # provider_mode() is the single config-level source of truth:
    # "mock" | "live" | "not_configured". A selected-but-keyless provider is
    # NOT_CONFIGURED -- never mock, never "live that might secretly fall back".
    mode = s.provider_mode()
    effective = s.llm_provider if mode != "mock" else "mock"

    if mode == "mock":
        status = ProviderStatus(
            provider="mock", mode="mock", configured=True, healthy=True, reachable=True,
            model="mock-deterministic-v1", checks={"note": "deterministic local provider; no external API used"},
        )
        return status.as_dict()

    key_present = mode == "live"
    model = s.llm_model_name or None

    status = ProviderStatus(
        provider=s.llm_provider, mode=mode,
        configured=key_present, healthy=None, reachable=None, model=model,
    )
    if not key_present:
        status.error_code = "PROVIDER_NOT_CONFIGURED"
        status.error_message = (
            f"LLM_PROVIDER={s.llm_provider} is selected but no API key is configured. "
            "Generation requests will fail with PROVIDER_NOT_CONFIGURED until a key is set "
            "(or offline mode is explicitly selected with LLM_PROVIDER=mock)."
        )
        return status.as_dict()

    if not check_health:
        return status.as_dict()

    # --- real, minimal verification ------------------------------------
    api_key = s.groq_api_key if s.llm_provider == "groq" else s.gemini_api_key
    base_url = MODELS_URLS.get(s.llm_provider)
    if base_url is None:
        status.error_code = "PROVIDER_UNKNOWN"
        return status.as_dict()

    # 1) reachability + auth via the models list (no tokens consumed)
    t0 = time.perf_counter()
    try:
        resp = requests.get(base_url, headers={"Authorization": f"Bearer {api_key}"},
                            timeout=TIMEOUT_SECONDS)
        status.latency_ms = round((time.perf_counter() - t0) * 1000)
        status.reachable = True
        if resp.status_code != 200:
            status.error_code = classify_http_error(resp.status_code)
            status.error_message = f"Models list returned HTTP {resp.status_code}."
            status.healthy = False
            return status.as_dict()
        status.checks["models_list"] = "ok"
    except Exception as e:  # noqa: BLE001 — health must never raise
        status.latency_ms = round((time.perf_counter() - t0) * 1000)
        status.reachable = False
        status.healthy = False
        status.error_code = _classify_exception(e)
        status.error_message = str(e)[:200]
        return status.as_dict()

    # 2) end-to-end minimal completion with the configured model
    p = provider or get_llm_provider(s)
    t1 = time.perf_counter()
    try:
        out = p.complete("You are a health check.", "Reply with the single word OK.",
                         temperature=0.0, max_tokens=300)
        gen_ms = round((time.perf_counter() - t1) * 1000)
        if not out.text or not out.text.strip():
            status.healthy = False
            status.error_code = "PROVIDER_EMPTY_RESPONSE"
            return status.as_dict()
        status.healthy = True
        status.latency_ms = gen_ms
        status.model = out.model or status.model
        status.checks["completion"] = "ok"
        status.checks["completion_latency_ms"] = gen_ms
    except Exception as e:  # noqa: BLE001
        status.healthy = False
        status.error_code = classify_http_error(getattr(e, "status_code", 0) or 0) \
            if getattr(e, "status_code", None) else "PROVIDER_UNAVAILABLE"
        status.error_message = str(e)[:200]

    status.checked_at = time.time()
    return status.as_dict()
