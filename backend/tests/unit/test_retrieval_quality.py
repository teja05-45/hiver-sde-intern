import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.retrieval.embeddings import TfidfEmbeddingProvider
from app.retrieval.lexical import BM25Index, tokenize
from app.retrieval.quality import (
    HybridRetrievalConfig,
    resolution_quality,
    score_candidates,
)
from app.retrieval.retriever import EvidenceCase, EvidenceResult, HistoricalRetriever

RECORDS = [
    {"conversation_id": "c1", "root_message": "my package never arrived", "resolution": "please check tracking",
     "intent": "delivery_not_received", "created_at": "t1"},
    {"conversation_id": "c2", "root_message": "package is late again", "resolution": "we apologize for the delay",
     "intent": "delivery_delay", "created_at": "t2"},
    {"conversation_id": "c3", "root_message": "I want my money back", "resolution": "refund has been issued",
     "intent": "cancellation_or_refund_request", "created_at": "t3"},
]

# Two candidates with near-identical similarity; only quality and intent
# differ. Used to prove each channel can flip the ranking on its own.
def _case(cid: str, sim: float, intent: str, resolution: str) -> EvidenceCase:
    return EvidenceCase(conversation_id=cid, similarity=sim, customer_message=f"msg {cid}",
                        resolution=resolution, intent=intent, created_at="")


GOOD_RESOLUTION = ("We issued a refund to the original payment method and the customer "
                   "confirmed the money arrived within three business days.")
DEFERRAL_RESOLUTION = "Please contact support so we can help you with this issue."
EMPTY_RESOLUTION = ""


class TestResolutionQuality(unittest.TestCase):
    def test_empty_is_zero(self):
        self.assertEqual(resolution_quality(None), 0.0)
        self.assertEqual(resolution_quality(""), 0.0)
        self.assertEqual(resolution_quality("   "), 0.0)

    def test_too_short_is_placeholder(self):
        self.assertEqual(resolution_quality("Thanks!"), 0.2)

    def test_pure_deferral_is_downweighted(self):
        self.assertEqual(resolution_quality(DEFERRAL_RESOLUTION), 0.4)

    def test_deferral_with_substance_is_between(self):
        long_deferral = ("Please contact support with the details of what happened here "
                         "so we can look into it and get everything sorted out for you.")
        self.assertEqual(resolution_quality(long_deferral), 0.5)

    def test_generic_substantive(self):
        text = ("The customer explained the situation and we walked through the details of "
                "what happened step by step until things were clear.")
        self.assertEqual(resolution_quality(text), 0.6)

    def test_topic_specific_beats_generic(self):
        self.assertGreater(resolution_quality(GOOD_RESOLUTION), 0.7)

    def test_monotone_in_usefulness(self):
        order = [resolution_quality(x) for x in
                 (None, "ok", DEFERRAL_RESOLUTION,
                  "We reviewed the account and reset the password for the customer.")]
        self.assertEqual(order, sorted(order))


class TestBM25(unittest.TestCase):
    def test_tokenize_basic(self):
        self.assertEqual(tokenize("My package NEVER arrived!"), ["my", "package", "never", "arrived"])
        self.assertEqual(tokenize("a I"), [])

    def test_matching_doc_scores_higher(self):
        idx = BM25Index().fit(["refund issued for order", "password reset done", "refund refund late"])
        scores = idx.score("where is my refund")
        self.assertGreater(scores[0], scores[1])
        self.assertGreater(scores[2], scores[1])  # term frequency helps

    def test_unknown_terms_contribute_zero(self):
        idx = BM25Index().fit(["refund issued", "password reset"])
        self.assertEqual(idx.score("quantum entanglement"), [0.0, 0.0])

    def test_empty_index(self):
        self.assertEqual(BM25Index().fit([]).score("anything"), [])


class TestHybridScoring(unittest.TestCase):
    def _bm25(self, docs):
        return BM25Index().fit(docs)

    def test_quality_channel_flips_ranking(self):
        """Same similarity, same intent: the case with a usable resolution must win."""
        a = _case("a", 0.60, "delivery_not_received", GOOD_RESOLUTION)
        b = _case("b", 0.60, "delivery_not_received", DEFERRAL_RESOLUTION)
        bm25 = self._bm25([a.customer_message, b.customer_message])
        scored = score_candidates([a, b], "package never arrived", None, bm25,
                                  {"a": resolution_quality(GOOD_RESOLUTION),
                                   "b": resolution_quality(DEFERRAL_RESOLUTION)},
                                  HybridRetrievalConfig())
        self.assertEqual(scored[0].case.conversation_id, "a")
        self.assertGreater(scored[0].components["resolution_quality"],
                           scored[1].components["resolution_quality"])

    def test_intent_channel_flips_ranking(self):
        """Same similarity and quality: classifier-compatible intent wins over lexically
        similar wrong-intent case (the audit's failure mode #1)."""
        good = _case("good", 0.55, "delivery_not_received", GOOD_RESOLUTION)
        wrong = _case("wrong", 0.60, "general_other", GOOD_RESOLUTION)
        bm25 = self._bm25([good.customer_message, wrong.customer_message])
        probs = {"delivery_not_received": 0.9, "general_other": 0.05}
        scored = score_candidates([wrong, good], "package never arrived", probs, bm25,
                                  {"good": 0.8, "wrong": 0.8}, HybridRetrievalConfig())
        self.assertEqual(scored[0].case.conversation_id, "good")

    def test_contradiction_penalty_applies_to_minority_intent(self):
        a = _case("a", 0.5, "delivery_not_received", GOOD_RESOLUTION)
        b = _case("b", 0.5, "delivery_not_received", GOOD_RESOLUTION)
        c = _case("c", 0.5, "general_other", GOOD_RESOLUTION)
        bm25 = self._bm25([x.customer_message for x in (a, b, c)])
        scored = score_candidates([a, b, c], "package never arrived", None, bm25,
                                  {x.conversation_id: 0.6 for x in (a, b, c)},
                                  HybridRetrievalConfig())
        by_id = {s.case.conversation_id: s for s in scored}
        self.assertGreater(by_id["a"].final_score, by_id["c"].final_score)
        self.assertGreater(by_id["c"].components["contradiction"], 0.0)

    def test_explanations_and_components_present(self):
        a = _case("a", 0.7, "delivery_not_received", GOOD_RESOLUTION)
        bm25 = self._bm25([a.customer_message])
        scored = score_candidates([a], "package never arrived", {"delivery_not_received": 0.9},
                                  bm25, {"a": 0.8}, HybridRetrievalConfig())
        s = scored[0]
        self.assertIn("semantic", s.components)
        self.assertIn("lexical", s.components)
        self.assertIn("intent_compatibility", s.components)
        self.assertIn("resolution_quality", s.components)
        self.assertTrue(s.explanation)
        self.assertEqual(s.rank_final, 1)
        self.assertEqual(s.rank_raw, 1)
        ex = s.explain()
        self.assertEqual(ex["conversation_id"], "a")
        self.assertIn("weighted", ex)

    def test_bm25_lexical_channel_normalizes(self):
        a = _case("a", 0.1, "delivery_not_received", GOOD_RESOLUTION)  # low semantic, high lexical
        b = _case("b", 0.1, "delivery_not_received", GOOD_RESOLUTION)
        a.customer_message = "tracking number shows delivered but package missing"
        b.customer_message = "prime video subtitle problem"
        bm25 = self._bm25([a.customer_message, b.customer_message])
        scored = score_candidates([a, b], "tracking number shows delivered but package missing",
                                  None, bm25, {"a": 0.8, "b": 0.8}, HybridRetrievalConfig())
        self.assertEqual(scored[0].case.conversation_id, "a")
        self.assertGreater(scored[0].components["lexical"], 0.9)
        self.assertLess(scored[1].components["lexical"], 0.5)

    def test_empty_candidates(self):
        bm25 = self._bm25(["x"])
        self.assertEqual(score_candidates([], "q", None, bm25, {}, HybridRetrievalConfig()), [])


class TestHybridRetriever(unittest.TestCase):
    def test_retrieve_sets_hybrid_fields(self):
        r = HistoricalRetriever(TfidfEmbeddingProvider()).build_index(RECORDS)
        result = r.retrieve("my package never showed up", k=2)
        self.assertTrue(result.hybrid_enabled)
        for c in result.cases:
            self.assertIsNotNone(c.final_score)
            self.assertTrue(c.explanation)
            self.assertEqual(c.rank_raw, result.baseline_cases.index(c) + 1 if c in result.baseline_cases else c.rank_raw)

    def test_baseline_order_preserved_for_ab(self):
        r = HistoricalRetriever(TfidfEmbeddingProvider()).build_index(RECORDS)
        result = r.retrieve("refund my money", k=3)
        self.assertEqual(len(result.baseline_cases), 3)
        # Baseline is pure cosine order: non-increasing similarity.
        sims = [c.similarity for c in result.baseline_cases]
        self.assertEqual(sims, sorted(sims, reverse=True))

    def test_hybrid_disabled_by_env(self):
        r = HistoricalRetriever(TfidfEmbeddingProvider()).build_index(RECORDS)
        with patch.dict(os.environ, {"RETRIEVAL_HYBRID": "0"}):
            result = r.retrieve("my package never showed up", k=2)
        self.assertFalse(result.hybrid_enabled)
        self.assertIsNone(result.cases[0].final_score)
        # Cases come back in cosine order.
        self.assertEqual([c.conversation_id for c in result.cases],
                         [c.conversation_id for c in result.baseline_cases][:2])

    def test_old_artifact_without_hybrid_attrs_still_works(self):
        """Pickles restore __dict__ without __init__: an artifact from before
        hybrid scoring must not crash, and must fall back to cosine ranking."""
        r = HistoricalRetriever(TfidfEmbeddingProvider()).build_index(RECORDS)
        for attr in ("config", "bm25", "_quality_by_id", "_hybrid_capable"):
            delattr(r, attr)
        result = r.retrieve("my package never showed up", k=2)
        self.assertFalse(result.hybrid_enabled)
        self.assertEqual(result.num_cases, 2)

    def test_resolution_agreement_rate(self):
        cases = [
            _case("a", 0.5, "i1", GOOD_RESOLUTION),
            _case("b", 0.5, "i2", DEFERRAL_RESOLUTION),
            _case("c", 0.5, "i3", EMPTY_RESOLUTION),
        ]
        res = EvidenceResult(cases=cases)
        self.assertAlmostEqual(res.resolution_agreement_rate, 1.0 / 3.0)
        self.assertEqual(EvidenceResult(cases=[]).resolution_agreement_rate, 0.0)

    def test_empty_index_returns_empty_result(self):
        r = HistoricalRetriever(TfidfEmbeddingProvider()).build_index([])
        result = r.retrieve("anything", k=5)
        self.assertEqual(result.num_cases, 0)
        self.assertFalse(result.hybrid_enabled)


if __name__ == "__main__":
    unittest.main()
