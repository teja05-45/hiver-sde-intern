import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.retrieval.embeddings import TfidfEmbeddingProvider
from app.retrieval.vector_store import VectorStore
from app.retrieval.retriever import HistoricalRetriever

RECORDS = [
    {"conversation_id": "c1", "root_message": "my package never arrived", "resolution": "please check tracking",
     "intent": "delivery_not_received", "created_at": "t1"},
    {"conversation_id": "c2", "root_message": "package is late again", "resolution": "we apologize for the delay",
     "intent": "delivery_delay", "created_at": "t2"},
    {"conversation_id": "c3", "root_message": "I want my money back", "resolution": "refund has been issued",
     "intent": "cancellation_or_refund_request", "created_at": "t3"},
]


class TestTfidfEmbeddingProvider(unittest.TestCase):
    def test_encode_before_fit_raises(self):
        provider = TfidfEmbeddingProvider()
        with self.assertRaises(RuntimeError):
            provider.encode(["hello"])

    def test_encode_returns_correct_row_count(self):
        provider = TfidfEmbeddingProvider().fit(["cat dog bird", "fish tree rock", "sun moon star"])
        vecs = provider.encode(["cat dog bird", "car bus train"])
        self.assertEqual(vecs.shape[0], 2)


class TestVectorStore(unittest.TestCase):
    def test_search_before_build_raises(self):
        store = VectorStore()
        import numpy as np
        with self.assertRaises(RuntimeError):
            store.search(np.array([[1.0, 0.0]]))

    def test_search_returns_most_similar_first(self):
        import numpy as np
        store = VectorStore()
        vectors = np.array([[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]])
        store.build(vectors, [{"id": "a"}, {"id": "b"}, {"id": "c"}])
        results = store.search(np.array([[1.0, 0.0]]), k=2)
        self.assertEqual(results[0].metadata["id"], "a")


class TestHistoricalRetriever(unittest.TestCase):
    def test_retrieve_before_build_raises(self):
        retriever = HistoricalRetriever(TfidfEmbeddingProvider())
        with self.assertRaises(RuntimeError):
            retriever.retrieve("hello")

    def test_retrieve_returns_relevant_case_first(self):
        retriever = HistoricalRetriever(TfidfEmbeddingProvider()).build_index(RECORDS)
        result = retriever.retrieve("my package never showed up", k=3)
        self.assertGreater(result.num_cases, 0)
        self.assertEqual(result.cases[0].conversation_id, "c1")

    def test_empty_index_returns_empty_result(self):
        retriever = HistoricalRetriever(TfidfEmbeddingProvider()).build_index([])
        result = retriever.retrieve("anything", k=5)
        self.assertEqual(result.num_cases, 0)
        self.assertEqual(result.top_similarity, 0.0)

    def test_intent_agreement_rate_is_one_when_all_agree(self):
        records = [
            {"conversation_id": "a", "root_message": "package late", "resolution": "r1",
             "intent": "delivery_delay", "created_at": ""},
            {"conversation_id": "b", "root_message": "package is late too", "resolution": "r2",
             "intent": "delivery_delay", "created_at": ""},
        ]
        retriever = HistoricalRetriever(TfidfEmbeddingProvider()).build_index(records)
        result = retriever.retrieve("my package is really late", k=2)
        self.assertEqual(result.intent_agreement_rate, 1.0)


if __name__ == "__main__":
    unittest.main()
