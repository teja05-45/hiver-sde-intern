"""Integration coverage for the support-workspace endpoints added in the
EvidenceDesk redesign: inbox, conversations, escalations, resolved, runtime
decision log, retrieval explorer. Offline + mock-forced, like all API tests.
"""
import os
import sys
import unittest
from pathlib import Path

os.environ["LLM_PROVIDER"] = "mock"
os.environ.pop("GROQ_API_KEY", None)
os.environ.pop("LLM_MODE", None)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.api.app import app  # noqa: E402

HAS_DATASET = (Path(__file__).resolve().parents[3] / "data" / "processed" / "labeled_AmazonHelp.jsonl").exists()


@unittest.skipUnless(HAS_DATASET, "labeled dataset not generated")
class TestInboxEndpoints(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_inbox_returns_paginated_tickets(self):
        r = self.client.get("/api/v1/inbox?page=1&page_size=5")
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        for key in ("total", "page", "pages", "tickets"):
            self.assertIn(key, data)
        self.assertLessEqual(len(data["tickets"]), 5)
        for t in data["tickets"]:
            for key in ("conversation_id", "customer_ref", "preview", "intent", "status", "priority"):
                self.assertIn(key, t)
            self.assertTrue(t["customer_ref"].startswith("Customer #"))
            self.assertIn(t["status"], ("open", "resolved"))

    def test_inbox_filters_by_intent(self):
        r = self.client.get("/api/v1/inbox?intent=delivery_delay&page_size=10")
        data = r.get_json()
        for t in data["tickets"]:
            self.assertEqual(t["intent"], "delivery_delay")

    def test_inbox_search_matches_preview(self):
        r = self.client.get("/api/v1/inbox?q=refund&page_size=10")
        data = r.get_json()
        for t in data["tickets"]:
            self.assertIn("refund", t["preview"].lower())

    def test_inbox_status_filter_valid_values(self):
        r = self.client.get("/api/v1/inbox?status=resolved&page_size=10")
        data = r.get_json()
        for t in data["tickets"]:
            self.assertEqual(t["status"], "resolved")

    def test_conversation_detail_roundtrip(self):
        listing = self.client.get("/api/v1/inbox?page_size=1").get_json()
        cid = listing["tickets"][0]["conversation_id"]
        r = self.client.get(f"/api/v1/conversations/{cid}")
        self.assertEqual(r.status_code, 200)
        conv = r.get_json()
        self.assertEqual(conv["conversation_id"], cid)
        self.assertTrue(conv["messages"])
        for m in conv["messages"]:
            self.assertIn(m["role"], ("customer", "agent"))
            self.assertTrue(m["text"])

    def test_conversation_detail_unknown_id_is_404_shape(self):
        r = self.client.get("/api/v1/conversations/AmazonHelp_does_not_exist")
        self.assertEqual(r.status_code, 404)
        self.assertEqual(r.get_json()["error"]["code"], "NOT_FOUND")

    def test_previews_are_sanitized(self):
        r = self.client.get("/api/v1/inbox?page_size=50&page_size=50")
        data = r.get_json()
        for t in data["tickets"]:
            self.assertNotIn("http", t["preview"].lower())
            self.assertNotIn("@", t["preview"])


class TestRuntimeDecisionEndpoints(unittest.TestCase):
    """These must be honest empty states when nothing has been decided yet."""

    def setUp(self):
        self.client = app.test_client()

    def test_escalations_shape(self):
        r = self.client.get("/api/v1/escalations")
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        self.assertIn("escalations", data)
        self.assertIsInstance(data["escalations"], list)

    def test_resolved_shape(self):
        r = self.client.get("/api/v1/resolved")
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        self.assertIn("resolved", data)
        self.assertIsInstance(data["resolved"], list)

    def test_agent_decisions_shape(self):
        r = self.client.get("/api/v1/agent/decisions")
        self.assertEqual(r.status_code, 200)
        self.assertIn("decisions", r.get_json())

    def test_agent_decision_detail_unknown_id_is_404(self):
        r = self.client.get("/api/v1/agent/decisions/no-such-request-id")
        self.assertEqual(r.status_code, 404)

    def test_respond_persists_a_queryable_decision(self):
        resp = self.client.post("/api/v1/agent/respond", json={"message": "my package never arrived"})
        self.assertEqual(resp.status_code, 200)
        rid = resp.get_json()["request_id"]
        detail = self.client.get(f"/api/v1/agent/decisions/{rid}")
        self.assertEqual(detail.status_code, 200)
        rec = detail.get_json()
        self.assertEqual(rec["request_id"], rid)
        self.assertIn(rec["decision"], ("AUTO", "ESCALATE"))
        self.assertIn(rec["mode"], ("mock", "live"))
        self.assertIsInstance(rec["reason_codes"], list)


class TestRetrievalExplorer(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_requires_q(self):
        r = self.client.get("/api/v1/retrieval/explorer")
        self.assertEqual(r.status_code, 400)
        self.assertEqual(r.get_json()["error"]["code"], "INVALID_REQUEST")

    def test_returns_cases_without_generation(self):
        r = self.client.get("/api/v1/retrieval/explorer?q=where%20is%20my%20refund&k=3")
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        self.assertIn("cases", data)
        self.assertLessEqual(len(data["cases"]), 3)
        # Retrieval-only: no response/decision keys leak in.
        self.assertNotIn("response", data)
        self.assertNotIn("decision", data)

    def test_invalid_k_rejected(self):
        r = self.client.get("/api/v1/retrieval/explorer?q=refund&k=99")
        self.assertEqual(r.status_code, 400)


class TestDecisionsEndpointBackwardCompat(unittest.TestCase):
    def test_design_decisions_still_present_with_runtime_block(self):
        client = app.test_client()
        r = client.get("/api/v1/decisions")
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        self.assertGreaterEqual(len(data["decisions"]), 15)
        self.assertIn("runtime_decisions", data)
        self.assertIn("runtime_counts", data)


if __name__ == "__main__":
    unittest.main()
