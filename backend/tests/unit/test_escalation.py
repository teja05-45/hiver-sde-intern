import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.escalation.policy import EscalationSignals, EscalationThresholds, decide, InternalDecision


def base_signals(**overrides) -> EscalationSignals:
    defaults = dict(
        intent="delivery_delay", intent_confidence=0.9, intent_escalation_tendency="low",
        evidence_score=0.8, num_supporting_cases=5, intent_agreement_rate=1.0,
    )
    defaults.update(overrides)
    return EscalationSignals(**defaults)


class TestEscalationPolicy(unittest.TestCase):
    def test_strong_low_risk_evidence_is_auto(self):
        decision = decide(base_signals())
        self.assertEqual(decision.public_decision, "AUTO")
        self.assertEqual(decision.internal_decision, InternalDecision.AUTO)

    def test_high_risk_intent_always_escalates_even_with_perfect_evidence(self):
        decision = decide(base_signals(intent="payment_or_billing_issue", intent_escalation_tendency="high",
                                        intent_confidence=0.99, evidence_score=0.99, num_supporting_cases=20))
        self.assertEqual(decision.public_decision, "ESCALATE")
        self.assertIn("HIGH_RISK_INTENT", decision.reason_codes)

    def test_low_intent_confidence_escalates(self):
        decision = decide(base_signals(intent_confidence=0.2))
        self.assertEqual(decision.public_decision, "ESCALATE")
        self.assertIn("LOW_INTENT_CONFIDENCE", decision.reason_codes)

    def test_insufficient_evidence_count_escalates(self):
        decision = decide(base_signals(num_supporting_cases=0))
        self.assertEqual(decision.public_decision, "ESCALATE")
        self.assertIn("INSUFFICIENT_EVIDENCE_COUNT", decision.reason_codes)

    def test_low_evidence_score_escalates(self):
        decision = decide(base_signals(evidence_score=0.1))
        self.assertEqual(decision.public_decision, "ESCALATE")
        self.assertIn("LOW_EVIDENCE_SCORE", decision.reason_codes)

    def test_retrieval_failure_always_escalates_regardless_of_other_signals(self):
        decision = decide(base_signals(retrieval_failed=True, intent_confidence=0.99, evidence_score=0.99))
        self.assertEqual(decision.public_decision, "ESCALATE")
        self.assertIn("RETRIEVAL_FAILED", decision.reason_codes)

    def test_generation_failure_always_escalates(self):
        decision = decide(base_signals(generation_failed=True, intent_confidence=0.99, evidence_score=0.99))
        self.assertEqual(decision.public_decision, "ESCALATE")
        self.assertIn("GENERATION_FAILED", decision.reason_codes)

    def test_unsupported_claims_always_escalates(self):
        decision = decide(base_signals(unsupported_claims_present=True, intent_confidence=0.99, evidence_score=0.99))
        self.assertEqual(decision.public_decision, "ESCALATE")
        self.assertIn("UNSUPPORTED_CLAIMS", decision.reason_codes)

    def test_one_weak_signal_is_review_internally_but_escalate_publicly(self):
        decision = decide(base_signals(evidence_score=0.1))  # only one signal fails
        self.assertEqual(decision.internal_decision, InternalDecision.REVIEW)
        self.assertEqual(decision.public_decision, "ESCALATE")

    def test_multiple_weak_signals_is_escalate_internally_too(self):
        decision = decide(base_signals(evidence_score=0.1, intent_confidence=0.1))
        self.assertEqual(decision.internal_decision, InternalDecision.ESCALATE)

    def test_reason_is_never_empty(self):
        for sig in [base_signals(), base_signals(evidence_score=0.0), base_signals(retrieval_failed=True)]:
            decision = decide(sig)
            self.assertTrue(decision.reason)

    def test_custom_thresholds_are_respected(self):
        strict = EscalationThresholds(min_evidence_score_for_auto=0.95, min_intent_confidence_for_auto=0.95,
                                       min_supporting_cases_for_auto=10, min_grounding_score_for_auto=0.9)
        decision = decide(base_signals(), strict)
        self.assertEqual(decision.public_decision, "ESCALATE")


if __name__ == "__main__":
    unittest.main()
