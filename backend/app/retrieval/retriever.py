"""
Historical evidence retriever.

CRITICAL temporal-leakage rule, enforced structurally here (not just by
convention): the evidence index is built only from conversations dated
*before* the query period. This mirrors deployment reality -- a live
system can only retrieve evidence from the past, never from conversations
that haven't happened yet. Concretely:

  - Evaluating against the `dev` split -> index built from `train` only.
  - Evaluating against the `test` split -> index built from `train` + `dev`
    (both strictly earlier than the test cutoff).

This module deliberately takes an explicit list of allowed source splits
rather than "all data" as an argument, so a caller cannot accidentally
build an index that includes the split being evaluated.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.retrieval.embeddings import EmbeddingProvider
from app.retrieval.vector_store import VectorStore
from app.services.text_cleaning import clean_for_modeling


@dataclass
class EvidenceCase:
    conversation_id: str
    similarity: float
    customer_message: str
    resolution: str | None
    intent: str
    created_at: str


@dataclass
class EvidenceResult:
    cases: list[EvidenceCase] = field(default_factory=list)

    @property
    def top_similarity(self) -> float:
        return self.cases[0].similarity if self.cases else 0.0

    @property
    def num_cases(self) -> int:
        return len(self.cases)

    @property
    def intent_agreement_rate(self) -> float:
        """Fraction of retrieved cases sharing the SAME intent as the top
        result -- one input to the evidence-quality score: high similarity
        scores that disagree on intent are a weaker signal than similar
        scores that agree."""
        if not self.cases:
            return 0.0
        top_intent = self.cases[0].intent
        agree = sum(1 for c in self.cases if c.intent == top_intent)
        return agree / len(self.cases)


class HistoricalRetriever:
    def __init__(self, embedding_provider: EmbeddingProvider) -> None:
        self.embedding_provider = embedding_provider
        self.vector_store = VectorStore()
        self._built = False

    def build_index(self, records: list[dict]) -> "HistoricalRetriever":
        """`records` must only contain conversations from allowed
        (earlier-than-query) splits -- see module docstring."""
        texts = [clean_for_modeling(r["root_message"]) for r in records]
        self.embedding_provider.fit(texts)
        vectors = self.embedding_provider.encode(texts)
        metadata = [
            {
                "conversation_id": r["conversation_id"],
                "customer_message": r["root_message"],
                "resolution": r.get("resolution"),
                "intent": r["intent"],
                "created_at": r.get("created_at", ""),
            }
            for r in records
        ]
        self.vector_store.build(vectors, metadata)
        self._built = True
        return self

    def retrieve(self, query_text: str, k: int = 5) -> EvidenceResult:
        if not self._built:
            raise RuntimeError("HistoricalRetriever.build_index() must be called before retrieve().")
        query_vec = self.embedding_provider.encode([clean_for_modeling(query_text)])
        raw_results = self.vector_store.search(query_vec, k=k)
        cases = [
            EvidenceCase(
                conversation_id=r.metadata["conversation_id"],
                similarity=r.similarity,
                customer_message=r.metadata["customer_message"],
                resolution=r.metadata["resolution"],
                intent=r.metadata["intent"],
                created_at=r.metadata["created_at"],
            )
            for r in raw_results
        ]
        return EvidenceResult(cases=cases)
