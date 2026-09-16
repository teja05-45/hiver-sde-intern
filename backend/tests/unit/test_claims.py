import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.generation.claims import extract_claims, verify_claims
from app.retrieval.retriever import EvidenceResult, EvidenceCase


def make_evidence(cases):
    return EvidenceResult(cases=[
        EvidenceCase(conversation_id=f"c{i}", similarity=sim, customer_message=msg,
                     resolution=res, intent="delivery_delay", created_at="")
        for i, (msg, res, sim) in enumerate(cases)
    ])


class TestClaimExtraction(unittest.TestCase):
    def test_claims_are_literal_substrings_of_the_draft(self):
        draft = ("Hi, thanks for reaching out. You can check your latest tracking status. "
                 "Amazon will refund you within 24 hours.")
        claims = extract_claims(draft)
        for c in claims:
            self.assertIn(c, draft)

    def test_sentence_boundaries_split(self):
        draft = "First claim here. Second claim follows! Third asks a question?"
        self.assertEqual(len(extract_claims(draft)), 3)

    def test_empty_draft_yields_no_claims(self):
        self.assertEqual(extract_claims(""), [])
        self.assertEqual(extract_claims("   \n "), [])

    def test_newlines_split(self):
        draft = "Line one with a claim\nLine two with another claim"
        self.assertEqual(len(extract_claims(draft)), 2)


class TestPerClaimVerification(unittest.TestCase):
    def test_supported_claim_records_evidence_ids(self):
        evidence = make_evidence([("pkg late", "please check your tracking page for updates", 0.8)])
        draft = "You can check your tracking page for updates."
        result = verify_claims(draft, evidence)
        self.assertTrue(result.supported)
        self.assertEqual(result.supported[0].evidence_ids, ["c0"])
        self.assertTrue(result.passed)

    def test_verbatim_quoted_evidence_is_supported_despite_scaffolding(self):
        """Template scaffolding around a verbatim evidence quote must not
        dilute a directly-quoted factual core below the threshold."""
        evidence = make_evidence([("pkg", "please check your tracking page for updates", 0.8)])
        draft = ("Hi, thanks for reaching out. Based on how we've handled similar delivery delay "
                 "cases: please check your tracking page for updates. Let us know if you need anything else.")
        result = verify_claims(draft, evidence)
        statuses = {c.status for c in result.claims}
        self.assertIn("supported", statuses)
        self.assertTrue(result.passed)

    def test_unsupported_claim_has_no_evidence_ids(self):
        evidence = make_evidence([("pkg late", "please check your tracking page", 0.8)])
        draft = "Amazon will refund you within 24 hours, guaranteed."
        result = verify_claims(draft, evidence)
        self.assertEqual(len(result.unsupported), 1)
        self.assertEqual(result.unsupported[0].evidence_ids, [])
        self.assertFalse(result.passed)

    def test_mixed_draft_scores_partially(self):
        evidence = make_evidence([("pkg", "check your tracking page for updates", 0.8)])
        draft = ("You can check your tracking page for updates. "
                 "A refund will be issued within one hour automatically.")
        result = verify_claims(draft, evidence)
        self.assertEqual(len(result.supported), 1)
        self.assertEqual(len(result.unsupported), 1)
        self.assertEqual(result.score, 0.5)
        self.assertFalse(result.passed)

    def test_greeting_and_abstention_are_policy_neutral(self):
        evidence = make_evidence([])
        draft = ("Hi, thanks for reaching out. I don't have enough historical evidence "
                 "to safely answer this. This will be routed to a team member for review.")
        result = verify_claims(draft, evidence)
        statuses = {c.status for c in result.claims}
        self.assertIn("abstention", statuses)
        self.assertIn("greeting", statuses)
        self.assertEqual(len(result.checkable), 0)
        self.assertTrue(result.passed)  # abstention never fails grounding

    def test_no_checkable_content_without_abstention_is_neutral_half(self):
        evidence = make_evidence([("a", "b", 0.5)])
        draft = "Let us know if you need anything else."
        result = verify_claims(draft, evidence)
        self.assertEqual(len(result.checkable), 0)
        self.assertEqual(result.score, 0.5)

    def test_best_evidence_case_wins(self):
        evidence = make_evidence([
            ("pkg", "we apologize for the inconvenience", 0.9),
            ("pkg", "please check your tracking page for the latest delivery estimate", 0.7),
        ])
        draft = "Please check your tracking page for the latest delivery estimate."
        result = verify_claims(draft, evidence)
        self.assertEqual(result.supported[0].evidence_ids, ["c1"])

    def test_result_is_json_serializable(self):
        import json
        evidence = make_evidence([("pkg", "check tracking page", 0.8)])
        result = verify_claims("You can check your tracking page.", evidence)
        json.dumps(result.as_dict())


if __name__ == "__main__":
    unittest.main()
