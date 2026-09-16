import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.services.novelty import compute_novelty_signals, NoveltyThresholds
from app.retrieval.retriever import EvidenceCase


def cases(specs):
    """specs: list of (similarity, intent)."""
    return [EvidenceCase(conversation_id=f"c{i}", similarity=s, customer_message="m",
                         resolution="r", intent=intent, created_at="")
            for i, (s, intent) in enumerate(specs)]


class TestNoveltyDetection(unittest.TestCase):
    def test_off_topic_message_with_saturated_confidence_and_no_evidence_is_ood(self):
        """The measured failure: 'What is the capital of India?' -> 1.00
        confidence, near-zero retrieval similarity, scattered intents."""
        scores = {"content_availability_inquiry": 1.0, "delivery_delay": 0.0}
        result = compute_novelty_signals(scores, cases([(0.08, "general_other"), (0.06, "app_or_device_technical_issue")]))
        self.assertTrue(result.is_ood)
        self.assertGreaterEqual(result.ood_score, 0.5)

    def test_confident_in_domain_message_with_agreeing_evidence_is_not_ood(self):
        scores = {"delivery_not_received": 0.99, "delivery_delay": 0.005}
        result = compute_novelty_signals(scores, cases([
            (0.55, "delivery_not_received"), (0.48, "delivery_not_received"), (0.42, "delivery_not_received"),
        ]))
        self.assertFalse(result.is_ood)

    def test_empty_retrieval_is_strong_novelty_evidence(self):
        scores = {"general_other": 0.6, "delivery_delay": 0.2}
        result = compute_novelty_signals(scores, [])
        self.assertTrue(result.is_ood)

    def test_low_margin_between_intents_raises_score(self):
        scores = {"delivery_delay": 0.45, "delivery_not_received": 0.44}
        result = compute_novelty_signals(scores, cases([(0.35, "delivery_delay"), (0.33, "delivery_delay")]))
        self.assertGreater(result.ood_score, 0.2)

    def test_score_is_bounded_and_deterministic(self):
        scores = {"a": 1.0}
        r1 = compute_novelty_signals(scores, cases([(0.0, "x")]))
        r2 = compute_novelty_signals(scores, cases([(0.0, "x")]))
        self.assertGreaterEqual(r1.ood_score, 0.0)
        self.assertLessEqual(r1.ood_score, 1.0)
        self.assertEqual(r1.ood_score, r2.ood_score)

    def test_custom_thresholds_respected(self):
        strict = NoveltyThresholds(ood_threshold=0.99)
        scores = {"content_availability_inquiry": 1.0}
        result = compute_novelty_signals(scores, cases([(0.05, "general_other")]), strict)
        self.assertFalse(result.is_ood)


if __name__ == "__main__":
    unittest.main()
