import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.classification.baselines import TfidfLogisticRegressionClassifier
from app.retrieval.embeddings import TfidfEmbeddingProvider
from app.retrieval.retriever import HistoricalRetriever
from app.providers.llm.mock_provider import MockLLMProvider
from app.services.agent import SupportAgent
from app.escalation.policy import EscalationThresholds

TRAIN_TEXTS = ["my package never arrived", "where is my order", "package delivered but not received"] * 5
TRAIN_LABELS = ["delivery_not_received", "order_status_inquiry", "delivery_not_received"] * 5

RECORDS = [
    {"conversation_id": f"c{i}", "root_message": t, "resolution": "please check your tracking page for updates",
     "intent": l, "created_at": ""}
    for i, (t, l) in enumerate(zip(TRAIN_TEXTS, TRAIN_LABELS))
]

INTENTS_CFG = {
    "delivery_not_received": {"escalation_tendency": "medium"},
    "order_status_inquiry": {"escalation_tendency": "low"},
}


def build_agent(thresholds=None) -> SupportAgent:
    clf = TfidfLogisticRegressionClassifier().fit(TRAIN_TEXTS, TRAIN_LABELS)
    retriever = HistoricalRetriever(TfidfEmbeddingProvider()).build_index(RECORDS)
    weights = {"retrieval_score": 0.75, "intent_agreement_rate": 0.25, "resolution_consistency": 0.0,
               "evidence_count_score": 0.0}
    return SupportAgent(clf, retriever, MockLLMProvider(), weights, INTENTS_CFG, thresholds or EscalationThresholds())


class TestSupportAgentEndToEnd(unittest.TestCase):
    def test_respond_returns_well_formed_result(self):
        agent = build_agent()
        result = agent.respond("my package never arrived, where is it")
        self.assertTrue(result.request_id)
        self.assertIn(result.decision.public_decision, ("AUTO", "ESCALATE"))
        self.assertIsNotNone(result.generated)

    def test_as_dict_is_json_serializable(self):
        import json
        agent = build_agent()
        result = agent.respond("where is my package")
        json.dumps(result.as_dict())  # raises if not serializable

    def test_high_confidence_strong_evidence_can_auto_handle(self):
        lenient = EscalationThresholds(min_evidence_score_for_auto=0.0, min_intent_confidence_for_auto=0.0,
                                        min_supporting_cases_for_auto=1, min_grounding_score_for_auto=0.0)
        agent = build_agent(lenient)
        result = agent.respond("my package never arrived")
        self.assertEqual(result.decision.public_decision, "AUTO")

    def test_strict_thresholds_force_escalation(self):
        strict = EscalationThresholds(min_evidence_score_for_auto=0.999, min_intent_confidence_for_auto=0.999,
                                       min_supporting_cases_for_auto=50, min_grounding_score_for_auto=0.999)
        agent = build_agent(strict)
        result = agent.respond("my package never arrived")
        self.assertEqual(result.decision.public_decision, "ESCALATE")

    def test_empty_retrieval_index_still_produces_a_decision(self):
        clf = TfidfLogisticRegressionClassifier().fit(TRAIN_TEXTS, TRAIN_LABELS)
        empty_retriever = HistoricalRetriever(TfidfEmbeddingProvider()).build_index([])
        weights = {"retrieval_score": 1.0}
        agent = SupportAgent(clf, empty_retriever, MockLLMProvider(), weights, INTENTS_CFG, EscalationThresholds())
        result = agent.respond("anything")
        self.assertEqual(result.decision.public_decision, "ESCALATE")

    def test_latency_is_recorded_for_each_stage(self):
        agent = build_agent()
        result = agent.respond("where is my order")
        for stage in ("classification_ms", "retrieval_ms", "evidence_scoring_ms", "generation_ms"):
            self.assertIn(stage, result.latency_ms)


if __name__ == "__main__":
    unittest.main()
