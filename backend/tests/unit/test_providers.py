import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.core.config import Settings
from app.providers.llm.base import LLMError
from app.providers.llm.factory import get_llm_provider
from app.providers.llm.groq_provider import GroqProvider
from app.providers.llm.gemini_provider import GeminiProvider
from app.providers.llm.mock_provider import MockLLMProvider


def make_settings(**overrides) -> Settings:
    defaults = dict(
        app_env="development",
        llm_provider="mock",
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

    def test_groq_without_key_falls_back_to_mock(self):
        self.assertIsInstance(get_llm_provider(make_settings(llm_provider="groq", groq_api_key="")), MockLLMProvider)

    def test_gemini_without_key_falls_back_to_mock(self):
        self.assertIsInstance(get_llm_provider(make_settings(llm_provider="gemini", gemini_api_key="")), MockLLMProvider)

    def test_invalid_provider_name_falls_back_to_mock(self):
        self.assertIsInstance(get_llm_provider(make_settings(llm_provider="claude")), MockLLMProvider)

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
