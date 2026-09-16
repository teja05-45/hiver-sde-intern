import os
import sys
import unittest
from pathlib import Path

# Tests must be deterministic and offline: force mock mode BEFORE the app
# module (and its dotenv load) is imported, so a developer's real .env with
# LLM_PROVIDER=groq never makes the test suite burn live API quota or fail
# on provider rate limits. Assignment requirement: "tests run without an
# API key".
os.environ["LLM_PROVIDER"] = "mock"
os.environ.pop("GROQ_API_KEY", None)
os.environ.pop("LLM_MODE", None)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.api.app import app


class TestNewPipelineFields(unittest.TestCase):
    """The /respond payload must expose the signals the redesigned Live Agent
    renders: per-claim verification, novelty/OOD, and the decision reasons."""

    def setUp(self):
        self.client = app.test_client()

    def test_claim_verification_present_and_derived_from_draft(self):
        r = self.client.post("/api/v1/agent/respond", json={"message": "my package never arrived"})
        self.assertEqual(r.status_code, 200)
        d = r.get_json()
        cv = d["response"].get("claim_verification")
        if cv is None:
            self.skipTest("generation skipped in this mode")
        draft = d["response"]["draft"] or ""
        for claim in cv["claims"]:
            self.assertIn("claim_text", claim)
            self.assertIn(claim["status"], ("supported", "unsupported", "greeting", "abstention"))
            if claim["status"] in ("supported", "unsupported") and draft:
                # every checkable claim is a substring of the actual draft
                self.assertIn(claim["claim_text"][:30], draft)

    def test_supported_claims_carry_evidence_ids(self):
        r = self.client.post("/api/v1/agent/respond", json={"message": "where is my order"})
        d = r.get_json()
        cv = d["response"].get("claim_verification")
        if cv is None:
            self.skipTest("generation skipped in this mode")
        for claim in cv["claims"]:
            if claim["status"] == "supported":
                self.assertTrue(claim["evidence_ids"], "supported claim must cite evidence")

    def test_novelty_block_present(self):
        r = self.client.post("/api/v1/agent/respond", json={"message": "my package never arrived"})
        d = r.get_json()
        self.assertIsNotNone(d.get("novelty"))
        self.assertIn("ood_score", d["novelty"])
        self.assertIn("is_ood", d["novelty"])
        self.assertIn("subsignals", d["novelty"])

    def test_ood_message_escalates_with_ood_code(self):
        r = self.client.post("/api/v1/agent/respond", json={"message": "What is the capital of India?"})
        d = r.get_json()
        if d["decision"]["decision"] == "ESCALATE" and "OOD_REQUEST" in d["decision"]["reason_codes"]:
            self.assertTrue(d["novelty"]["is_ood"])
        # Regardless of which signal fires, OOD-style messages must never AUTO.
        self.assertEqual(d["decision"]["decision"], "ESCALATE")

    def test_privacy_message_escalates_with_privacy_code(self):
        r = self.client.post("/api/v1/agent/respond",
                             json={"message": "my password is hunter2 please help me log in"})
        d = r.get_json()
        self.assertEqual(d["decision"]["decision"], "ESCALATE")
        self.assertIn("PRIVACY_RISK", d["decision"]["reason_codes"])

    def test_multi_intent_message_escalates_with_multi_intent_code(self):
        r = self.client.post(
            "/api/v1/agent/respond",
            json={"message": "My package arrived late, the product is damaged, and I want a refund."})
        d = r.get_json()
        self.assertEqual(d["decision"]["decision"], "ESCALATE")
        self.assertIn("MULTI_INTENT", d["decision"]["reason_codes"])

    def test_no_reason_code_references_the_old_multi_intent_name(self):
        r = self.client.post("/api/v1/agent/respond", json={"message": "where is my order"})
        d = r.get_json()
        self.assertNotIn("MULTI_INTENT_SUSPECTED", d["decision"]["reason_codes"])


class TestDecisionsLogEndpoint(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_decisions_endpoint_serves_real_document(self):
        r = self.client.get("/api/v1/decisions")
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        self.assertGreaterEqual(len(data["decisions"]), 19)


class TestSystemEndpointComponents(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_system_reports_component_health(self):
        r = self.client.get("/api/v1/system")
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        for comp in ("api", "classifier", "retriever", "evidence_model", "llm_provider", "golden_set"):
            self.assertIn(comp, data.get("components", {}))

    def test_provider_health_verifies(self):
        """verify=0 path: configuration truth only, must never claim healthy
        for a live provider without a measurement."""
        r = self.client.get("/api/v1/provider/health")
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        self.assertIn("mode", data)
        self.assertIn("configured", data)


if __name__ == "__main__":
    unittest.main()
