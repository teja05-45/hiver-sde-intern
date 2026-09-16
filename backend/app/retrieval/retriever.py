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

Hybrid retrieval (see retrieval/quality.py): candidates are pulled from the
raw TF-IDF cosine search at pool size (~4x k), then reranked by a weighted
combination of semantic similarity, BM25 lexical relevance, classifier-based
intent compatibility, and resolution quality, minus a contradiction penalty.
Set config.hybrid_enabled=False (or RETRIEVAL_HYBRID=0) to get the plain
cosine ranking -- both rankings are computed and kept, so the evaluation
script can A/B them instead of asserting the improvement.

Intent compatibility uses the classifier's probability for the case's
intent given the query. An IntentProbsProvider is any callable
query_text -> {intent: probability}; the agent passes its fitted
classifier (no retraining, no keyword rules). Without a provider the
intent channel is simply 0 for every case and the ranking degrades to
text+quality signals.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Callable

from app.retrieval.embeddings import EmbeddingProvider
from app.retrieval.lexical import BM25Index
from app.retrieval.quality import (
    HybridRetrievalConfig,
    candidate_pool_size,
    resolution_quality,
    score_candidates,
    score_candidates_full,
)
from app.retrieval.vector_store import VectorStore
from app.services.text_cleaning import clean_for_modeling

# Provider of per-query intent probabilities: callable(query) -> {intent: p}.
IntentProbsProvider = Callable[[str], dict[str, float]]

logger = logging.getLogger(__name__)


def _env_flag(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class EvidenceCase:
    conversation_id: str
    similarity: float
    customer_message: str
    resolution: str | None
    intent: str
    created_at: str
    # Hybrid-scoring extras (empty when hybrid reranking is disabled):
    final_score: float | None = None
    components: dict[str, float] = field(default_factory=dict)
    explanation: str = ""
    rank_raw: int | None = None


@dataclass
class EvidenceResult:
    cases: list[EvidenceCase] = field(default_factory=list)
    # Pre-rerank cosine ranking (baseline order), kept for evaluation A/B
    # and for the UI's "why this case?" promotion display.
    baseline_cases: list[EvidenceCase] = field(default_factory=list)
    hybrid_enabled: bool = False
    # Aggregate quality of the FINAL top-k (pool-level when hybrid is on):
    # intent agreement, usable-resolution share, top similarity, mean final
    # score. Empty dict when hybrid reranking did not run.
    retrieval_quality: dict = field(default_factory=dict)

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

    @property
    def resolution_agreement_rate(self) -> float:
        """Fraction of retrieved cases whose resolutions are actually usable
        (resolution_quality >= 0.5): "please contact support" placeholders and
        empty resolutions are not evidence of HOW the problem gets resolved.
        Consumed by the evidence scorer and surfaced in the UI."""
        if not self.cases:
            return 0.0
        usable = sum(1 for c in self.cases if resolution_quality(c.resolution) >= 0.5)
        return usable / len(self.cases)


class HistoricalRetriever:
    def __init__(self, embedding_provider: EmbeddingProvider,
                 config: HybridRetrievalConfig | None = None) -> None:
        self.embedding_provider = embedding_provider
        self.vector_store = VectorStore()
        self.config = config or HybridRetrievalConfig()
        # Backward compat: artifacts pickled before hybrid scoring exist and
        # must still load. Treated as "reranking available but off" until the
        # first build_index(); retrieve() then falls back to plain cosine.
        self.bm25: BM25Index | None = None
        self._quality_by_id: dict[str, float] = {}
        self._hybrid_capable = False
        self._built = False

    # -- index construction -------------------------------------------------

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
        # Lexical channel + resolution-quality table, built once here and
        # pickled with the artifact.
        self.bm25 = BM25Index().fit(texts)
        self._quality_by_id = {
            r["conversation_id"]: resolution_quality(r.get("resolution"))
            for r in records
        }
        self._hybrid_capable = len(records) > 0
        self._built = True
        return self

    # -- backward compatibility with artifacts pickled before hybrid scoring
    # (pickle restores __dict__ without running __init__, so new fields are
    # simply absent on old instances). The old ranking behavior is preserved,
    # with a loud log so nobody mistakes a pre-hybrid artifact for the
    # current retriever.
    def _ensure_hybrid_fields(self) -> None:
        if not hasattr(self, "config"):
            self.config = HybridRetrievalConfig()
            logger.warning(
                "Retriever artifact predates hybrid scoring (no config attr); "
                "reranking disabled for this instance. Rebuild with "
                "scripts/train_and_save_agent.py to enable it."
            )
        for name, default in (("bm25", None), ("_quality_by_id", {}),
                              ("_hybrid_capable", False)):
            if not hasattr(self, name):
                setattr(self, name, default)

    # -- retrieval ------------------------------------------------------------

    def retrieve(self, query_text: str, k: int = 5,
                 intent_probs: dict[str, float] | None = None,
                 intent_provider: IntentProbsProvider | None = None) -> EvidenceResult:
        """Return the top-k evidence cases under the configured ranking.

        intent_probs / intent_provider: classifier distribution for this
        query (provider called lazily only when reranking needs it).
        """
        if not self._built:
            raise RuntimeError("HistoricalRetriever.build_index() must be called before retrieve().")
        self._ensure_hybrid_fields()

        pool_k = candidate_pool_size(k) if self._hybrid_enabled() else k
        query_vec = self.embedding_provider.encode([clean_for_modeling(query_text)])
        raw_results = self.vector_store.search(query_vec, k=pool_k)

        def to_case(r, pos: int) -> EvidenceCase:
            m = r.metadata
            return EvidenceCase(
                conversation_id=m["conversation_id"],
                similarity=r.similarity,
                customer_message=m["customer_message"],
                resolution=m["resolution"],
                intent=m["intent"],
                created_at=m.get("created_at", ""),
                rank_raw=pos,
            )

        baseline_cases = [to_case(r, i + 1) for i, r in enumerate(raw_results)]
        if not baseline_cases:
            return EvidenceResult(cases=[], baseline_cases=[], hybrid_enabled=False)

        if not self._hybrid_enabled():
            return EvidenceResult(cases=baseline_cases[:k], baseline_cases=baseline_cases,
                                  hybrid_enabled=False)

        if intent_probs is None and intent_provider is not None:
            intent_probs = intent_provider(query_text)

        outcome = score_candidates_full(
            baseline_cases, query_text, intent_probs,
            self.bm25, self._quality_by_id, self.config,
        )
        cases = []
        for s in outcome.cases[:k]:
            c = s.case
            c.final_score = s.final_score
            c.components = s.components
            c.explanation = s.explanation
            c.rank_raw = s.rank_raw
            cases.append(c)
        return EvidenceResult(cases=cases, baseline_cases=baseline_cases, hybrid_enabled=True,
                              retrieval_quality=outcome.quality_summary)

    def _hybrid_enabled(self) -> bool:
        self._ensure_hybrid_fields()
        return self._hybrid_capable and bool(self.bm25) and _env_flag("RETRIEVAL_HYBRID", True)