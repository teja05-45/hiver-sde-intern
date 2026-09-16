import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.core.config import Settings
from app.providers.llm.base import LLMError
from app.providers.llm.factory import ProviderConfigError, get_llm_provider
from app.providers.llm.groq_provider import GroqProvider
from app.providers.llm.gemini_provider import GeminiProvider
from app.providers.llm.mock_provider import MockLLMProvider


def make_settings(**overrides) -> Settings:
    defaults = dict(
        app_env="development",
        llm_provider="mock",
        llm_mode="",
        groq_api_key="",
        gemini_api_key="",
        llm_model_name="",
        llm_timeout_seconds=5.0,
        llm_max_retries=0,
        brand_name="AmazonHelp",
    )
    defaults.update(overrides)
    return Settings(**defaults)


class TestProviderFactory(unittest.TestCase):
    def test_mock_provider_selected_by_default(self):
        self.assertIsInstance(get_llm_provider(make_settings(llm_provider="mock")), MockLLMProvider)

    def test_groq_provider_selected_when_configured_with_key(self):
        p = get_llm_provider(make_settings(llm_provider="groq", groq_api_key="test-key-123"))
        self.assertIsInstance(p, GroqProvider)
        self.assertEqual(p.provider_name, "groq")

    def test_gemini_provider_selected_when_configured_with_key(self):
        p = get_llm_provider(make_settings(llm_provider="gemini", gemini_api_key="test-key-456"))
        self.assertIsInstance(p, GeminiProvider)
        self.assertEqual(p.provider_name, "gemini")

    # --- Fail-loud contract: a missing key or unknown provider is a
    # configuration error (NOT_CONFIGURED), NEVER a silent switch to mock.
    # Mock must be explicitly selected (LLM_PROVIDER=mock / LLM_MODE=mock).
    def test_groq_without_key_raises_not_falls_back(self):
        with self.assertRaises(ProviderConfigError) as ctx:
            get_llm_provider(make_settings(llm_provider="groq", groq_api_key=""))
        self.assertIn("GROQ_API_KEY", str(ctx.exception))
        self.assertIn("LLM_PROVIDER=mock", str(ctx.exception))

    def test_gemini_without_key_raises_not_falls_back(self):
        with self.assertRaises(ProviderConfigError):
            get_llm_provider(make_settings(llm_provider="gemini", gemini_api_key=""))

    def test_invalid_provider_name_raises(self):
        with self.assertRaises(ProviderConfigError) as ctx:
            get_llm_provider(make_settings(llm_provider="claude"))
        self.assertIn("claude", str(ctx.exception))

    def test_llm_mode_live_without_key_is_not_mock(self):
        """LLM_MODE=live with a missing key must raise (NOT_CONFIGURED truth),
        never quietly become a mock provider."""
        with self.assertRaises(ProviderConfigError):
            get_llm_provider(make_settings(llm_provider="groq", llm_mode="live", groq_api_key=""))

    def test_llm_mode_mock_forces_mock_even_with_key(self):
        """Explicit mock selection wins even when a key exists -- mode is what
        the operator asked for."""
        p = get_llm_provider(make_settings(llm_provider="groq", llm_mode="mock",
                                           groq_api_key="k"))
        self.assertIsInstance(p, MockLLMProvider)

    def test_provider_mode_states(self):
        s = make_settings(llm_provider="groq", groq_api_key="k")
        self.assertEqual(s.provider_mode(), "live")
        s = make_settings(llm_provider="groq", groq_api_key="")
        self.assertEqual(s.provider_mode(), "not_configured")
        self.assertFalse(s.is_mock_mode())
        s = make_settings(llm_provider="mock")
        self.assertEqual(s.provider_mode(), "mock")
        self.assertTrue(s.is_mock_mode())

    def test_constructing_groq_without_key_raises(self):
        with self.assertRaises(LLMError):
            GroqProvider("")

    def test_constructing_gemini_without_key_raises(self):
        with self.assertRaises(LLMError):
            GeminiProvider("")


def _resp(status=200, json_body=None, text=""):
    import requests as _requests
    m = MagicMock()
    m.status_code = status
    m.text = text
    if status >= 400:
        m.raise_for_status.side_effect = _requests.exceptions.HTTPError(f"{status} error")
    if json_body is not None:
        m.json.return_value = json_body
    return m


class TestGroqProviderRequests(unittest.TestCase):
    """All HTTP interactions are mocked -- these tests verify request shape,
    response parsing, and failure handling, never the real API."""

    def test_successful_completion_parses_text(self):
        p = GroqProvider("k", max_retries=0)
        body = {"choices": [{"message": {"content": "hello from groq"}}]}
        with patch("app.providers.llm.groq_provider.requests.post", return_value=_resp(200, body)) as post:
            out = p.complete("sys", "user")
        self.assertEqual(out.text, "hello from groq")
        self.assertFalse(out.is_mock)
        self.assertEqual(post.call_args.kwargs["headers"]["Authorization"], "Bearer k")

    def test_http_error_raises_llmerror(self):
        p = GroqProvider("k", max_retries=0)
        with patch("app.providers.llm.groq_provider.requests.post",
                   return_value=_resp(500, text="boom")):
            with self.assertRaises(LLMError):
                p.complete("sys", "user")

    def test_malformed_response_shape_raises_llmerror(self):
        p = GroqProvider("k", max_retries=0)
        with patch("app.providers.llm.groq_provider.requests.post",
                   return_value=_resp(200, {"unexpected": "shape"})):
            with self.assertRaises(LLMError):
                p.complete("sys", "user")

    def test_timeout_raises_llmerror(self):
        import requests as _requests
        p = GroqProvider("k", max_retries=0)
        with patch("app.providers.llm.groq_provider.requests.post",
                   side_effect=_requests.exceptions.Timeout("timed out")):
            with self.assertRaises(LLMError):
                p.complete("sys", "user")


class TestGeminiProviderRequests(unittest.TestCase):
    def test_successful_completion_parses_text(self):
        p = GeminiProvider("k", max_retries=0)
        body = {"candidates": [{"content": {"parts": [{"text": "hello from gemini"}]}}]}
        with patch("app.providers.llm.gemini_provider.requests.post", return_value=_resp(200, body)):
            out = p.complete("sys", "user")
        self.assertEqual(out.text, "hello from gemini")
        self.assertFalse(out.is_mock)

    def test_http_error_raises_llmerror(self):
        p = GeminiProvider("k", max_retries=0)
        with patch("app.providers.llm.gemini_provider.requests.post",
                   return_value=_resp(403, text="forbidden")):
            with self.assertRaises(LLMError):
                p.complete("sys", "user")

    def test_malformed_response_shape_raises_llmerror(self):
        p = GeminiProvider("k", max_retries=0)
        with patch("app.providers.llm.gemini_provider.requests.post",
                   return_value=_resp(200, {"candidates": []})):
            with self.assertRaises(LLMError):
                p.complete("sys", "user")


class TestMockProviderBehavior(unittest.TestCase):
    def test_outputs_are_always_tagged_mock(self):
        out = MockLLMProvider().complete("sys", '{"task": "generate_response"}')
        self.assertTrue(out.is_mock)
        self.assertEqual(out.provider, "mock")

    def test_no_evidence_produces_abstention_not_fabrication(self):
        import json
        from app.providers.llm.mock_provider import MockLLMProvider
        prompt = json.dumps({"task": "generate_response", "customer_message": "hi",
                             "evidence": []})
        out = MockLLMProvider().complete("sys", prompt)
        data = json.loads(out.text)
        self.assertTrue(data["evidence_insufficient"])
        self.assertEqual(data["grounded_claims"], [])


if __name__ == "__main__":
    unittest.main()
