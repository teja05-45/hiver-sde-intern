#!/usr/bin/env python3
"""Real LLM provider test.

Loads the environment, initializes the configured provider, checks
configuration, makes ONE minimal real generation request, and prints a
sanitized result with latency. Exit code 0 = pass, non-zero = fail. Never
prints API keys or provider response bodies.

Usage:
    python scripts/test_llm_provider.py            # uses .env / environment
    python scripts/test_llm_provider.py --json     # machine-readable output
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", help="emit JSON instead of text")
    args = parser.parse_args()

    # Importing app.core.config loads the repo-root .env (if any).
    from app.core.config import get_settings
    from app.providers.llm.factory import get_llm_provider
    from app.providers.llm.status import get_provider_status

    settings = get_settings()
    provider = get_llm_provider(settings)
    is_mock = settings.is_mock_mode()

    print("PROVIDER TEST")
    print("-" * 40)
    print(f"Provider:  {'Groq' if settings.llm_provider == 'groq' else settings.llm_provider}")
    print(f"Mode:      {'live' if not is_mock else 'mock'}")
    print(f"Model:     {settings.llm_model_name or '(provider default)'}")
    print(f"Configured: {'YES' if (settings.groq_api_key or settings.gemini_api_key or is_mock) else 'NO'}")

    if is_mock:
        print("Status:    MOCK MODE — no external API call is made.")
        print("Reachable: n/a (deterministic local provider)")
        print("Latency:   n/a")
        print("Generation: SKIPPED (mock provider; not a real API test)")
        print("\nTo run a REAL API test: set LLM_PROVIDER=groq and GROQ_API_KEY in .env")
        if args.json:
            print(json.dumps({"mode": "mock", "generation": "SKIPPED"}))
        return 2  # non-zero: this is NOT a real provider verification

    # Real check: models list + one minimal completion, measured and sanitized.
    status = get_provider_status(settings, provider=provider, check_health=True)
    d = status
    print(f"Reachable: {'YES' if d.get('reachable') else 'NO'}")
    print(f"Latency:   {d.get('latency_ms')} ms")
    print(f"Generation: {'PASS' if d.get('healthy') else 'FAIL'}")
    if d.get("error_code"):
        print(f"Error code: {d['error_code']}")
        if d.get("error_message"):
            print(f"Error:      {d['error_message']}")
    if args.json:
        print(json.dumps(d, indent=2))

    if d.get("healthy"):
        print("\nResult: the configured provider answered a real minimal request.")
        return 0
    print("\nResult: the configured provider FAILED the real check — see error above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
