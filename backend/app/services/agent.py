"""
Agent orchestrator: customer message -> intent -> retrieval -> evidence
scoring -> generation -> grounding -> escalation decision.

This is the one place that wires every module together end to end. It's
deliberately thin -- each stage's actual logic lives in its own module
(classification/, retrieval/, generation/, escalation/) and this class
just sequences them and assembles the final structured result, per the
assignment's "each stage should be modular, do NOT put everything inside
one massive LLM prompt" requirement.

Fail-safe behavior is enforced here: any stage failure (retrieval
exception, generation failure) is caught and converted into the
appropriate escalation signal rather than propagating as an unhandled
exception or silently producing a degraded answer.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from time import perf_counter

from app.classification.baselines import TfidfLogisticRegressionClassifier
from app.retrieval.embeddings import TfidfEmbeddingProvider
from app.retrieval.retriever import HistoricalRetriever, EvidenceResult
from app.services.ambiguity import AmbiguityResult, compute_ambiguity_signals
from app.services.novelty import compute_novelty_signals, NoveltyResult
from app.services.evidence_scoring import compute_evidence_features, compute_evidence_score
from app.services.text_cleaning import clean_for_modeling
from app.generation.generator import generate_response, check_grounding, GeneratedResponse
from app.generation.claims import verify_claims, ClaimVerificationResult
from app.escalation.policy import EscalationSignals, EscalationThresholds, decide, EscalationDecision
from app.providers.llm.base import LLMProvider

logger = logging.getLogger(__name__)

# Credential-shaped strings that must never be echoed by an automated draft.
# Deliberately narrow (few false positives): full card numbers, OTP/passcode
# phrasing, and explicit passwords. Order numbers and tracking IDs are NOT
# secrets — flagging them would escalate most of the corpus.
import re as _re

_CREDENTIAL_PATTERNS = [
    _re.compile(r"\b\d{16}\b"),                                  # 16-digit card number
    _re.compile(r"\bmy password (is|:)?\s*\S+", _re.IGNORECASE),
    _re.compile(r"\b(password|passcode|otp|one[- ]time (code|password)|pin)\s*(is|:|=)\s*\S+", _re.IGNORECASE),
    _re.compile(r"\b\d{6}\b(?=.*\b(otp|code|passcode)\b)", _re.IGNORECASE),
]


def _contains_credential_secrets(message: str) -> bool:
    return any(p.search(message or "") for p in _CREDENTIAL_PATTERNS)


@dataclass
class AgentResult:
    request_id: str
    customer_message: str
    intent: str
    intent_confidence: float
    evidence: EvidenceResult
    evidence_score: float
    generated: GeneratedResponse | None
    grounding_score: float | None
    decision: EscalationDecision
    latency_ms: dict[str, float] = field(default_factory=dict)
    evidence_features: dict[str, float] = field(default_factory=dict)
    ambiguity: AmbiguityResult | None = None
    novelty: NoveltyResult | None = None
    all_scores: dict[str, float] = field(default_factory=dict)
    claim_verification: ClaimVerificationResult | None = None

    def as_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "customer_message": self.customer_message,
            "intent": {
                "name": self.intent,
                "confidence": round(self.intent_confidence, 4),
                "all_scores": {k: round(v, 4) for k, v in
                               sorted(self.all_scores.items(), key=lambda kv: -kv[1])[:5]},
            },
            "evidence": {
                "quality": self.evidence_score,
                "features": {k: round(v, 4) for k, v in self.evidence_features.items()},
                "cases": [
                    {
                        "conversation_id": c.conversation_id,
                        "similarity": round(c.similarity, 4),
                        "customer_message": c.customer_message,
                        "resolution": c.resolution,
                        "intent": c.intent,
                        # Hybrid-reranking extras (None/empty when disabled):
                        "final_score": (round(c.final_score, 4)
                                         if c.final_score is not None else None),
                        "rank_raw": c.rank_raw,
                        "components": ({k: round(v, 4) for k, v in c.components.items()}
                                        if c.components else {}),
                        "explanation": c.explanation,
                    }
                    for c in self.evidence.cases
                ],
                # Retrieval-quality summary for the UI panel:
                "hybrid_enabled": self.evidence.hybrid_enabled,
                "top_similarity": round(self.evidence.top_similarity, 4),
                "intent_agreement_rate": round(self.evidence.intent_agreement_rate, 4),
                "resolution_agreement_rate": round(self.evidence.resolution_agreement_rate, 4),
            },
            "ambiguity": self.ambiguity.as_dict() if self.ambiguity else None,
            "novelty": self.novelty.as_dict() if self.novelty else None,
            "response": {
                "draft": self.generated.draft_reply if self.generated else None,
                "grounded": self.grounding_score is not None and self.grounding_score >= 0.6,
                "grounding_score": self.grounding_score,
                "is_mock": self.generated.is_mock if self.generated else None,
                # Self-reported claim lists (kept for audit/comparison only —
                # the UI renders claim_verification, whose claims are derived
                # from the actual draft text).
                "grounded_claims": self.generated.grounded_claims if self.generated else [],
                "unsupported_claims": self.generated.unsupported_claims if self.generated else [],
                # Per-claim verification of the ACTUAL draft text. Every
                # claim_text here is a literal substring of `draft`.
                "claim_verification": (self.claim_verification.as_dict()
                                        if self.claim_verification else None),
                # Stage-level provenance for the Live Agent page: generation is a
                # first-class pipeline stage with its own PASS/FAILED status.
                "generation_status": "FAILED" if (self.generated and self.generated.parse_error)
                                     else ("SKIPPED" if self.generated is None else "PASS"),
                "generation_error_code": (
                    ("GENERATION_FAILED" if self.generated and self.generated.parse_error else None)
                ),
                "generation_error": self.generated.parse_error if self.generated else None,
                "provider": self.generated.provider if self.generated else None,
                "model": self.generated.model if self.generated else None,
            },
            "decision": self.decision.as_dict(),
            "latency_ms": self.latency_ms,
        }


class SupportAgent:
    def __init__(self, classifier: TfidfLogisticRegressionClassifier, retriever: HistoricalRetriever,
                 llm_provider: LLMProvider, evidence_weights: dict[str, float],
                 intents_cfg: dict[str, dict], thresholds: EscalationThresholds | None = None) -> None:
        self.classifier = classifier
        self.retriever = retriever
        self.llm_provider = llm_provider
        self.evidence_weights = evidence_weights
        self.intents_cfg = intents_cfg
        self.thresholds = thresholds or EscalationThresholds()

    def respond(self, customer_message: str, k: int = 5,
                request_id: str | None = None) -> AgentResult:
        # Adopt the caller's request ID (the API layer passes the incoming
        # X-Request-ID) so the HTTP response, the structured decision log,
        # and the runtime decision store all reference the SAME id. A locally
        # generated id is only a fallback for direct service use.
        request_id = request_id or str(uuid.uuid4())
        latency: dict[str, float] = {}

        t0 = perf_counter()
        all_scores: dict[str, float] = {}
        try:
            pred = self.classifier.predict([clean_for_modeling(customer_message)])[0]
            intent, intent_confidence = pred.intent, pred.confidence
            all_scores = dict(pred.all_scores)
        except Exception as e:  # noqa: BLE001 - classification must never crash the request
            logger.exception("Classification failed")
            intent, intent_confidence = "general_other", 0.0
        latency["classification_ms"] = round((perf_counter() - t0) * 1000, 1)

        t0 = perf_counter()
        retrieval_failed = False
        try:
            # The fitted classifier doubles as the intent-compatibility
            # channel for hybrid retrieval: its full distribution for THIS
            # query is exactly P(case_intent | query) the reranker needs.
            # No separate model, no retraining, no keyword rules.
            evidence = self.retriever.retrieve(
                customer_message, k=k,
                intent_provider=lambda q: self.classifier.predict([clean_for_modeling(q)])[0].all_scores,
            )
        except Exception:  # noqa: BLE001
            logger.exception("Retrieval failed")
            evidence = EvidenceResult(cases=[])
            retrieval_failed = True
        latency["retrieval_ms"] = round((perf_counter() - t0) * 1000, 1)

        t0 = perf_counter()
        # resolution_consistency needs an embedder for the retrieved
        # resolutions. It is fitted on the fly over the retrieved cases'
        # resolutions (pairwise cosine among them is exactly the quantity
        # being measured). Without this, the feature was mechanically 0.0 at
        # runtime -- see docs/decision-log.md #16 for the full story.
        resolution_embedder = None
        resolutions = [c.resolution for c in evidence.cases if c.resolution]
        if len(resolutions) >= 2:
            resolution_embedder = TfidfEmbeddingProvider().fit(resolutions)
        features = compute_evidence_features(evidence, resolution_embedder)
        evidence_score = compute_evidence_score(features, self.evidence_weights)
        latency["evidence_scoring_ms"] = round((perf_counter() - t0) * 1000, 1)

        # Ambiguity and novelty must measure CLASSIFIER-vs-WORLD disagreement.
        # With hybrid reranking on, `cases` is intent-steered by construction
        # (the reranker boosts cases compatible with the predicted intent),
        # so its agreement/scatter is no longer independent evidence: a
        # confidently-wrong classifier would manufacture retrieval agreement
        # and suppress its own OOD signal (measured regression: "What is the
        # capital of India?" went 0.9 -> 0.27 OOD through exactly this loop).
        # The pre-rerank baseline ranking is the unsteered world evidence,
        # truncated to the same k the evidence window uses (the candidate
        # pool is ~4x k; agreement statistics are not comparable across
        # different window sizes).
        unsteered_cases = (evidence.baseline_cases or evidence.cases)[:k]
        ambiguity = compute_ambiguity_signals(all_scores, customer_message, unsteered_cases, self.intents_cfg)
        novelty = compute_novelty_signals(all_scores, unsteered_cases)

        t0 = perf_counter()
        generated = None
        grounding_score = None
        generation_failed = False
        unsupported_claims_present = False
        claim_verification: ClaimVerificationResult | None = None
        if not retrieval_failed:
            generated = generate_response(self.llm_provider, customer_message, intent, evidence)
            if generated.parse_error:
                generation_failed = True
            else:
                grounding_result = check_grounding(generated, evidence)
                grounding_score = grounding_result.grounding_score
                unsupported_claims_present = not grounding_result.grounded and not generated.evidence_insufficient
                # Independent, draft-derived per-claim verification. Runs even
                # when the model self-reported no unsupported claims — the
                # self-report is never trusted.
                claim_verification = verify_claims(generated.draft_reply, evidence)
                if claim_verification.unsupported:
                    unsupported_claims_present = True
        latency["generation_ms"] = round((perf_counter() - t0) * 1000, 1)

        escalation_tendency = self.intents_cfg.get(intent, {}).get("escalation_tendency", "high")
        signals = EscalationSignals(
            intent=intent, intent_confidence=intent_confidence,
            intent_escalation_tendency=escalation_tendency, evidence_score=evidence_score,
            num_supporting_cases=evidence.num_cases, intent_agreement_rate=evidence.intent_agreement_rate,
            unsupported_claims_present=unsupported_claims_present, grounding_score=grounding_score,
            generation_failed=generation_failed, retrieval_failed=retrieval_failed,
            privacy_risk_suspected=_contains_credential_secrets(customer_message),
            ambiguity=ambiguity, novelty=novelty,
        )
        decision = decide(signals, self.thresholds)

        return AgentResult(
            request_id=request_id, customer_message=customer_message, intent=intent,
            intent_confidence=intent_confidence, evidence=evidence, evidence_score=evidence_score,
            generated=generated, grounding_score=grounding_score, decision=decision, latency_ms=latency,
            evidence_features=features.as_dict(), ambiguity=ambiguity, novelty=novelty, all_scores=all_scores,
            claim_verification=claim_verification,
        )
