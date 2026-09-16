"""
Application configuration.

All configuration is environment-driven per the assignment requirement:
"Never hard-code API keys" / "Make the LLM provider configurable using
environment variables." This module is intentionally dependency-light
(stdlib `dataclasses` + `os.environ`) rather than `pydantic-settings`,
so it works identically whether or not `pydantic` is installed in the
target environment. If `pydantic` is available, `backend/app/schemas/`
uses it for request/response validation; this module only handles
process-level configuration.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_dotenv() -> None:
    """Load repo-root `.env` when python-dotenv is installed (local dev / demo)."""
    env_path = REPO_ROOT / ".env"
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(env_path, encoding="utf-8")
    except ImportError:
        pass


_load_dotenv()


def _env_int(name: str, default: int) -> int:
    val = os.environ.get(name)
    return int(val) if val not in (None, "") else default


def _env_float(name: str, default: float) -> float:
    val = os.environ.get(name)
    return float(val) if val not in (None, "") else default


def _env_bool(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None or val == "":
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Settings:
    app_env: str = field(default_factory=lambda: os.environ.get("APP_ENV", "development"))

    # LLM provider selection. "mock" is the deterministic mode that requires
    # no API key and no network access, per the assignment's requirement that
    # tests/evaluation must run without an API key. It must be EXPLICITLY
    # selected (LLM_PROVIDER=mock or LLM_MODE=mock) -- a missing API key is
    # NOT_CONFIGURED, never a silent switch to mock.
    llm_provider: str = field(default_factory=lambda: os.environ.get("LLM_PROVIDER", "mock"))
    # Explicit mode intent ("mock" | "live"). LLM_PROVIDER names the
    # implementation; LLM_MODE names the intended behavior. When LLM_MODE=live
    # is set but no usable provider/key exists, the system reports an error --
    # it never falls back to mock (hidden live->mock fallback is forbidden).
    llm_mode: str = field(default_factory=lambda: os.environ.get("LLM_MODE", "").strip().lower())
    groq_api_key: str = field(default_factory=lambda: os.environ.get("GROQ_API_KEY", ""))
    gemini_api_key: str = field(default_factory=lambda: os.environ.get("GEMINI_API_KEY", ""))
    # MODEL_NAME is the generic knob; GROQ_MODEL/GEMINI_MODEL are the
    # provider-specific aliases (GROQ_MODEL wins for Groq).
    llm_model_name: str = field(default_factory=lambda: (
        os.environ.get("GROQ_MODEL") if os.environ.get("LLM_PROVIDER") == "groq"
        else os.environ.get("GEMINI_MODEL") if os.environ.get("LLM_PROVIDER") == "gemini"
        else None
    ) or os.environ.get("MODEL_NAME", ""))
    llm_timeout_seconds: float = field(default_factory=lambda: _env_float("LLM_TIMEOUT_SECONDS", 20.0))
    llm_max_retries: int = field(default_factory=lambda: _env_int("LLM_MAX_RETRIES", 2))

    brand_name: str = field(default_factory=lambda: os.environ.get("BRAND_NAME", ""))
    data_sample_size: int = field(default_factory=lambda: _env_int("DATA_SAMPLE_SIZE", 60000))
    random_seed: int = field(default_factory=lambda: _env_int("RANDOM_SEED", 42))

    cors_origins: str = field(default_factory=lambda: os.environ.get("CORS_ORIGINS", "*"))
    backend_url: str = field(default_factory=lambda: os.environ.get("BACKEND_URL", "http://localhost:8000"))
    frontend_url: str = field(default_factory=lambda: os.environ.get("FRONTEND_URL", "http://localhost:3000"))

    raw_data_path: Path = field(default_factory=lambda: REPO_ROOT / "data" / "raw" / "twcs.csv")
    processed_dir: Path = field(default_factory=lambda: REPO_ROOT / "data" / "processed")
    golden_dir: Path = field(default_factory=lambda: REPO_ROOT / "data" / "golden")
    configs_dir: Path = field(default_factory=lambda: REPO_ROOT / "configs")
    reports_dir: Path = field(default_factory=lambda: REPO_ROOT / "reports")

    def provider_mode(self) -> str:
        """The truthful provider state, derived ONLY from configuration:

            "mock"           mock was explicitly selected (LLM_MODE=mock or
                              LLM_PROVIDER=mock)
            "not_configured" a real provider is selected but no API key is
                              set (or the provider name is unknown)
            "live"           a real provider is selected AND its key is set

        There is deliberately no "live with fallback to mock" state: a live
        request that fails surfaces as an error (mode stays "live", health
        check reports healthy=False), never as mock output. Whether the live
        provider is actually REACHABLE is a measured property (see
        providers/llm/status.py), not a configuration property.
        """
        if self.llm_mode == "mock":
            return "mock"
        if self.llm_mode == "live":
            # Operator explicitly demanded live. A missing key is still
            # NOT_CONFIGURED (the truthful cause), but it is never mock.
            key = self.groq_api_key if self.llm_provider == "groq" else self.gemini_api_key
            if self.llm_provider in ("groq", "gemini") and key:
                return "live"
            return "not_configured"
        # LLM_MODE unset: derive from the provider selection alone.
        if self.llm_provider == "mock":
            return "mock"
        key = self.groq_api_key if self.llm_provider == "groq" else self.gemini_api_key
        if self.llm_provider in ("groq", "gemini") and key:
            return "live"
        return "not_configured"

    def is_mock_mode(self) -> bool:
        """True only when mock was EXPLICITLY selected. A missing API key is
        not_configured, not mock -- callers must surface that distinction
        instead of silently degrading to deterministic output."""
        return self.provider_mode() == "mock"


def get_settings() -> Settings:
    """Return a fresh Settings instance read from the current environment.

    Not cached (unlike a typical `lru_cache`-wrapped settings getter) so that
    tests can freely monkeypatch `os.environ` between cases without stale
    state leaking across tests.
    """
    return Settings()
