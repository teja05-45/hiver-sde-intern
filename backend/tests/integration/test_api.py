import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.api.app import app


class TestClassifyEndpoint(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_valid_message_returns_intent(self):
        r = self.client.post("/api/v1/agent/classify", json={"message": "my package never arrived"})
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        self.assertIn("intent", data)
        self.assertIn("confidence", data)

    def test_missing_message_returns_400(self):
        r = self.client.post("/api/v1/agent/classify", json={})
        self.assertEqual(r.status_code, 400)

    def test_empty_message_returns_400(self):
        r = self.client.post("/api/v1/agent/classify", json={"message": ""})
        self.assertEqual(r.status_code, 400)

    def test_non_string_message_returns_400(self):
        r = self.client.post("/api/v1/agent/classify", json={"message": 12345})
        self.assertEqual(r.status_code, 400)

    def test_very_long_message_returns_400_on_respond(self):
        r = self.client.post("/api/v1/agent/respond", json={"message": "x" * 6000})
        self.assertEqual(r.status_code, 400)


class TestRetrieveEndpoint(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_valid_request_returns_cases(self):
        r = self.client.post("/api/v1/agent/retrieve", json={"message": "where is my order", "k": 3})
        self.assertEqual(r.status_code, 200)
        self.assertIn("cases", r.get_json())

    def test_invalid_k_returns_400(self):
        r = self.client.post("/api/v1/agent/retrieve", json={"message": "hello", "k": 999})
        self.assertEqual(r.status_code, 400)

    def test_missing_message_returns_400(self):
        r = self.client.post("/api/v1/agent/retrieve", json={})
        self.assertEqual(r.status_code, 400)


class TestRespondEndpoint(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_valid_request_returns_full_result(self):
        r = self.client.post("/api/v1/agent/respond", json={"message": "my package never arrived"})
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        for key in ("request_id", "intent", "evidence", "response", "decision"):
            self.assertIn(key, data)

    def test_decision_is_auto_or_escalate(self):
        r = self.client.post("/api/v1/agent/respond", json={"message": "my card was charged twice for one order"})
        data = r.get_json()
        self.assertIn(data["decision"]["decision"], ("AUTO", "ESCALATE"))

    def test_escalation_always_has_a_reason(self):
        r = self.client.post("/api/v1/agent/respond", json={"message": "I need to reset my password urgently"})
        data = r.get_json()
        if data["decision"]["decision"] == "ESCALATE":
            self.assertTrue(data["decision"]["reason"])


class TestIntentsEndpoint(unittest.TestCase):
    def test_returns_intents_list(self):
        client = app.test_client()
        r = client.get("/api/v1/intents")
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        self.assertIn("intents", data)
        self.assertGreaterEqual(len(data["intents"]), 8)


class TestEvaluationEndpoints(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_summary_returns_200_or_404(self):
        r = self.client.get("/api/v1/evaluation/summary")
        self.assertIn(r.status_code, (200, 404))


class TestNewEndpoints(unittest.TestCase):
    """Endpoints added for the operations-console frontend."""

    def setUp(self):
        self.client = app.test_client()

    def test_every_response_carries_request_id(self):
        r = self.client.get("/health")
        self.assertTrue(r.headers.get("X-Request-ID"))

    def test_error_shape_is_consistent(self):
        r = self.client.post("/api/v1/agent/classify", json={"message": ""})
        err = r.get_json()["error"]
        for key in ("code", "message", "request_id"):
            self.assertIn(key, err)
        self.assertEqual(err["code"], "INVALID_REQUEST")

    def test_unknown_api_path_returns_404_error_shape(self):
        r = self.client.get("/api/v1/definitely_not_a_path")
        self.assertEqual(r.status_code, 404)
        self.assertEqual(r.get_json()["error"]["code"], "NOT_FOUND")

    def test_automation_endpoint(self):
        r = self.client.get("/api/v1/evaluation/automation")
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        self.assertTrue("curve" in data or "golden_check" in data)

    def test_intent_metrics_endpoint(self):
        r = self.client.get("/api/v1/evaluation/intents")
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        self.assertIn("silver_per_intent", data)
        self.assertIn("golden_per_intent", data)

    def test_golden_examples_filter_incorrect(self):
        r = self.client.get("/api/v1/golden-set/examples?outcome=incorrect")
        if r.status_code == 404:
            self.skipTest("golden_set_with_predictions.json not generated")
        data = r.get_json()
        self.assertLessEqual(data["returned"], data["total"])
        for ex in data["examples"]:
            self.assertFalse(ex["correct"])

    def test_golden_examples_text_search(self):
        r = self.client.get("/api/v1/golden-set/examples?q=refund")
        if r.status_code == 404:
            self.skipTest("golden_set_with_predictions.json not generated")
        data = r.get_json()
        for ex in data["examples"]:
            self.assertIn("refund", ex["customer_message"].lower())

    def test_judge_summary_is_explicitly_not_validated_in_mock_mode(self):
        r = self.client.get("/api/v1/llm-judge/summary")
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        if data["mock_mode"]:
            self.assertEqual(data["status"], "NOT_VALIDATED")

    def test_decisions_endpoint_parses_decision_log(self):
        r = self.client.get("/api/v1/decisions")
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        self.assertGreaterEqual(len(data["decisions"]), 15)
        first = data["decisions"][0]
        for key in ("number", "title", "decision", "reason"):
            self.assertIn(key, first)

    def test_system_endpoint_never_returns_secrets(self):
        r = self.client.get("/api/v1/system")
        self.assertEqual(r.status_code, 200)
        raw = r.get_data(as_text=True).lower()
        # no key material, only boolean presence flags
        self.assertNotIn("apikey", raw.replace("api_key_configured", "").replace("api_key", ""))
        self.assertIn("api_key_configured", raw)

    def test_respond_includes_ambiguity_and_features(self):
        r = self.client.post("/api/v1/agent/respond", json={"message": "my package never arrived"})
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        self.assertIsNotNone(data["ambiguity"])
        self.assertIn("top2_margin", data["ambiguity"])
        self.assertIn("features", data["evidence"])
        self.assertIn("all_scores", data["intent"])


if __name__ == "__main__":
    unittest.main()
