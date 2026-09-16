"""
Evidence quality scoring.

A high top-1 similarity score alone is not proof of useful evidence (the
assignment is explicit about this). This module computes several
independent signals and combines them into a single `evidence_score`:

  - retrieval_score: top-1 cosine similarity from the retriever
  - intent_confidence: the classifier's confidence in its own prediction
  - intent_agreement_rate: what fraction of the top-k retrieved cases
    share the same intent as the top result (do the neighbors agree with
    each other, not just with the query?)
  - resolution_consistency: among the agreeing-intent retrieved cases, how
    similar are their historical resolutions to each other? (Are they
    telling the same story about how this gets resolved, or are they
    contradictory?)
  - evidence_count_score: diminishing-returns function of how many
    retrieved cases actually exist (0 supporting cases is a hard signal
    of "no evidence" regardless of any similarity number).

The combination weights are NOT hand-picked. `scripts/calibrate_evidence_score.py`
fits a small logistic regression on the dev split, predicting "will the
classifier's intent prediction turn out correct" from exactly these five
features, and the resulting coefficients (normalized to sum to 1, all
clipped to be non-negative -- a feature that's *negatively* associated
with correctness has no principled place in a "more evidence = more
trustworthy" score) become the evidence-score weights, persisted to
configs/evidence_score_weights.json. This keeps the formula empirically
grounded rather than arbitrary, and auditable (the calibration script
reports the fitted model's own validation accuracy).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from app.retrieval.retriever import EvidenceResult
from app.retrieval.embeddings import TfidfEmbeddingProvider
import numpy as np

DEFAULT_WEIGHTS_PATH = Path(__file__).resolve().parents[3] / "configs" / "evidence_score_weights.json"

DEFAULT_WEIGHTS = {
    "retrieval_score": 0.40,
    "intent_agreement_rate": 0.30,
    "resolution_consistency": 0.20,
    "evidence_count_score": 0.10,
}


@dataclass
class EvidenceFeatures:
    """Retrieval-only evidence-quality features. Deliberately excludes the
    classifier's intent_confidence -- see module docstring on why blending
    self-reported classifier confidence into "evidence quality" made the
    score degenerate into just restating confidence (an empirically
    measured finding, see reports/evidence_score_calibration.json), which
    defeats the "evidence-first, not confidence-first" design principle.
    intent_confidence is used as its own separate, explicit input to the
    escalation policy instead (backend/app/escalation/).
    """
    retrieval_score: float
    intent_agreement_rate: float
    resolution_consistency: float
    evidence_count_score: float

    def as_dict(self) -> dict[str, float]:
        return {
            "retrieval_score": self.retrieval_score,
            "intent_agreement_rate": self.intent_agreement_rate,
            "resolution_consistency": self.resolution_consistency,
            "evidence_count_score": self.evidence_count_score,
        }


def load_weights(path: Path = DEFAULT_WEIGHTS_PATH) -> dict[str, float]:
    if path.exists():
        return json.loads(path.read_text())
    return dict(DEFAULT_WEIGHTS)


def _resolution_consistency(evidence: EvidenceResult, resolution_embedder: TfidfEmbeddingProvider | None) -> float:
    """Average pairwise cosine similarity among the resolutions of
    top-intent-agreeing evidence cases. Returns 0.0 if fewer than 2 such
    cases exist (can't measure "consistency" from a single data point)."""
    if not evidence.cases or resolution_embedder is None:
        return 0.0
    top_intent = evidence.cases[0].intent
    resolutions = [c.resolution for c in evidence.cases if c.intent == top_intent and c.resolution]
    if len(resolutions) < 2:
        return 0.0
    vecs = resolution_embedder.encode(resolutions)
    sims = (vecs @ vecs.T).toarray() if hasattr(vecs, "toarray") else vecs @ vecs.T
    n = sims.shape[0]
    total, count = 0.0, 0
    for i in range(n):
        for j in range(i + 1, n):
            total += float(sims[i, j])
            count += 1
    return total / count if count else 0.0


def compute_evidence_features(
    evidence: EvidenceResult,
    resolution_embedder: TfidfEmbeddingProvider | None = None,
) -> EvidenceFeatures:
    evidence_count_score = min(evidence.num_cases / 5.0, 1.0)
    resolution_consistency = _resolution_consistency(evidence, resolution_embedder)
    return EvidenceFeatures(
        retrieval_score=evidence.top_similarity,
        intent_agreement_rate=evidence.intent_agreement_rate,
        resolution_consistency=resolution_consistency,
        evidence_count_score=evidence_count_score,
    )


def compute_evidence_score(features: EvidenceFeatures, weights: dict[str, float] | None = None) -> float:
    w = weights or load_weights()
    total_weight = sum(w.values()) or 1.0
    score = sum(w[k] * v for k, v in features.as_dict().items() if k in w)
    return round(score / total_weight, 4)
