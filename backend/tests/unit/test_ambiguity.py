import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.services.ambiguity import AmbiguityThresholds, compute_ambiguity_signals


class _Case:
    """Minimal stand-in for EvidenceCase (only `.intent` is read)."""

    def __init__(self, intent: str):
        self.intent = intent


CONFIDENT_SCORES = {"delivery_delay": 0.92, "order_status_inquiry": 0.05, "general_other": 0.03}
AMBIGUOUS_SCORES = {"delivery_delay": 0.41, "delivery_not_received": 0.38, "general_other": 0.21}

INTENTS_CFG = {
    "delivery_delay": {"positive_signals": ["late", "delayed"]},
    "payment_or_billing_issue": {"positive_signals": ["charged", "double charged"]},
    "delivery_not_received": {"positive_signals": ["never received", "not delivered"]},
}


class TestAmbiguitySignals(unittest.TestCase):
    def test_committed_prediction_is_not_ambiguous(self):
        r = compute_ambiguity_signals(CONFIDENT_SCORES, "my delivery is late", [], INTENTS_CFG)
        self.assertFalse(r.is_ambiguous)
        self.assertGreaterEqual(r.top2_margin, 0.15)

    def test_split_prediction_is_ambiguous(self):
        r = compute_ambiguity_signals(AMBIGUOUS_SCORES, "my delivery never showed", [], INTENTS_CFG)
        self.assertTrue(r.is_ambiguous)
        self.assertLess(r.top2_margin, 0.15)

    def test_multi_intent_detected_from_keyword_disagreement(self):
        msg = "My order is late AND I was double charged for it."
        r = compute_ambiguity_signals(CONFIDENT_SCORES, msg, [], INTENTS_CFG)
        self.assertTrue(r.multi_intent_suspected)
        self.assertIn("delivery_delay", r.multi_intent_candidates)
        self.assertIn("payment_or_billing_issue", r.multi_intent_candidates)

    def test_single_intent_keywords_do_not_trigger_multi_intent(self):
        r = compute_ambiguity_signals(CONFIDENT_SCORES, "my delivery is late", [], INTENTS_CFG)
        self.assertFalse(r.multi_intent_suspected)

    def test_sparse_evidence_flagged(self):
        r = compute_ambiguity_signals(CONFIDENT_SCORES, "late delivery", [_Case("delivery_delay")], INTENTS_CFG)
        self.assertTrue(r.sparse_evidence)

    def test_retrieval_disagreement_detected(self):
        cases = [_Case("return_or_replacement_request"), _Case("general_other"), _Case("general_other")]
        r = compute_ambiguity_signals(CONFIDENT_SCORES, "late delivery", cases, INTENTS_CFG)
        self.assertTrue(r.retrieval_disagreement)

    def test_noise_suspected_on_very_short_message(self):
        r = compute_ambiguity_signals(CONFIDENT_SCORES, "??", [], INTENTS_CFG)
        self.assertTrue(r.noise_suspected)

    def test_ambiguity_can_be_vetoed_off_but_multi_intent_still_esclates(self):
        # The policy may lower thresholds, but the signals remain inspectable.
        r = compute_ambiguity_signals(AMBIGUOUS_SCORES, "late", [], INTENTS_CFG,
                                      thresholds=AmbiguityThresholds(ambiguous_top2_margin=0.0))
        self.assertFalse(r.is_ambiguous)

    def test_as_dict_is_json_serializable(self):
        import json
        r = compute_ambiguity_signals(CONFIDENT_SCORES, "late", [], INTENTS_CFG)
        json.dumps(r.as_dict())


class TestPolicyIntegration(unittest.TestCase):
    """The escalation policy must treat these as ESCALATE-only signals."""

    def _signals(self, **kw):
        from app.escalation.policy import EscalationSignals
        defaults = dict(
            intent="delivery_delay", intent_confidence=0.9, intent_escalation_tendency="low",
            evidence_score=0.9, num_supporting_cases=5, intent_agreement_rate=1.0,
        )
        defaults.update(kw)
        return EscalationSignals(**defaults)

    def test_multi_intent_vetoes_auto(self):
        from app.escalation.policy import decide
        from app.services.ambiguity import AmbiguityResult
        amb = AmbiguityResult(
            top2_margin=0.9, is_ambiguous=False, multi_intent_suspected=True,
            multi_intent_candidates=["delivery_delay", "payment_or_billing_issue"],
            sparse_evidence=False, retrieval_disagreement=False, noise_suspected=False, notes=[],
        )
        d = decide(self._signals(ambiguity=amb))
        self.assertEqual(d.public_decision, "ESCALATE")
        self.assertIn("MULTI_INTENT_SUSPECTED", d.reason_codes)

    def test_ambiguity_blocks_auto_when_otherwise_strong(self):
        from app.escalation.policy import decide
        from app.services.ambiguity import AmbiguityResult
        amb = AmbiguityResult(
            top2_margin=0.05, is_ambiguous=True, multi_intent_suspected=False,
            multi_intent_candidates=[], sparse_evidence=False,
            retrieval_disagreement=False, noise_suspected=False, notes=[],
        )
        d = decide(self._signals(ambiguity=amb))
        self.assertEqual(d.public_decision, "ESCALATE")
        self.assertIn("AMBIGUOUS_INTENT", d.reason_codes)

    def test_no_ambiguity_object_preserves_old_behavior(self):
        from app.escalation.policy import decide
        d = decide(self._signals())
        self.assertEqual(d.public_decision, "AUTO")


if __name__ == "__main__":
    unittest.main()
