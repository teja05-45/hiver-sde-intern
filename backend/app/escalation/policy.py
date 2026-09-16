"""
Escalation / auto-handling decision policy.

Explicitly NOT `if confidence > 0.7: auto`. Per the assignment, the policy
must weigh multiple independent signals. This implementation keeps each
signal as its own named, inspectable field (rather than collapsing
everything into one opaque blended number) because the whole point of the
UI's "evidence trail" and this module's `reason` field is that a human
reviewer -- or a Hiver interviewer -- can see exactly which signal drove
an ESCALATE decision, not just a final score.

Internal three-state model (AUTO / REVIEW / ESCALATE), collapsed to the
assignment's required two-state public output (AUTO / ESCALATE) at the
boundary -- REVIEW maps to ESCALATE. Keeping REVIEW internally distinct is
useful for the automation-precision/coverage curve (scripts/evaluate_automation.py):
it lets us describe *how close* a case was to being auto-handled, not
just whether it crossed the line.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from app.services.ambiguity import AmbiguityResult
from app.services.novelty import NoveltyResult


class InternalDecision(str, Enum):
    AUTO = "AUTO"
    REVIEW = "REVIEW"
    ESCALATE = "ESCALATE"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class EscalationSignals:
    """All inputs to the policy, each independently inspectable."""
    intent: str
    intent_confidence: float
    intent_escalation_tendency: str  # "low" | "medium" | "high", from configs/intents.yaml
    evidence_score: float
    num_supporting_cases: int
    intent_agreement_rate: float
    unsupported_claims_present: bool = False   # set by the grounding checker, post-generation
    grounding_score: float | None = None       # set by the grounding checker, post-generation
    generation_failed: bool = False             # LLM call failed / malformed output
    retrieval_failed: bool = False
    privacy_risk_suspected: bool = False        # customer shared sensitive credentials
    # Experimental ambiguity/multi-intent/novelty signals
    # (backend/app/services/ambiguity.py, backend/app/services/novelty.py).
    # ESCALATE-only: they can veto an AUTO decision, never force one.
    ambiguity: "AmbiguityResult | None" = None
    novelty: "NoveltyResult | None" = None


@dataclass
class EscalationDecision:
    internal_decision: InternalDecision
    public_decision: str  # "AUTO" | "ESCALATE"
    risk_level: RiskLevel
    reason: str
    reason_codes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "internal_decision": self.internal_decision.value,
            "decision": self.public_decision,
            "risk_level": self.risk_level.value,
            "reason": self.reason,
            "reason_codes": self.reason_codes,
        }


@dataclass
class EscalationThresholds:
    """Operating point. Defaults are placeholders until
    scripts/calibrate_thresholds.py sets them from real precision/coverage
    tradeoffs on the dev split (see docs/decision-log.md, "why this
    threshold" for the business rationale, not just "best-looking
    number")."""
    min_evidence_score_for_auto: float = 0.55
    min_intent_confidence_for_auto: float = 0.75
    min_supporting_cases_for_auto: int = 2
    min_grounding_score_for_auto: float = 0.8


def decide(signals: EscalationSignals, thresholds: EscalationThresholds | None = None) -> EscalationDecision:
    t = thresholds or EscalationThresholds()
    reason_codes: list[str] = []

    # --- Hard fail-safes: always escalate, no matter how good the other
    # signals look. These map directly to the assignment's "fail safely"
    # requirement (LLM/retrieval failure -> escalate, never fabricate).
    if signals.retrieval_failed:
        reason_codes.append("RETRIEVAL_FAILED")
        return EscalationDecision(
            InternalDecision.ESCALATE, "ESCALATE", RiskLevel.HIGH,
            "Historical evidence retrieval failed; cannot verify a grounded response is possible.",
            reason_codes,
        )
    if signals.generation_failed:
        reason_codes.append("GENERATION_FAILED")
        return EscalationDecision(
            InternalDecision.ESCALATE, "ESCALATE", RiskLevel.HIGH,
            "Response generation failed or returned invalid output; escalating rather than guessing.",
            reason_codes,
        )
    # --- Privacy risk: the customer posted credentials/secrets in the
    # message. Auto-handling could echo them; a human must handle rotation.
    # Checked BEFORE output-level claim checks: input sensitivity outranks
    # output quality as a reason (the draft may be fine; the context is not).
    if signals.privacy_risk_suspected:
        reason_codes.append("PRIVACY_RISK")
        return EscalationDecision(
            InternalDecision.ESCALATE, "ESCALATE", RiskLevel.HIGH,
            "The message appears to contain sensitive credentials (password, OTP, card number); "
            "a human must handle credential rotation and the draft must never echo them.",
            reason_codes,
        )

    if signals.unsupported_claims_present:
        reason_codes.append("UNSUPPORTED_CLAIMS")
        return EscalationDecision(
            InternalDecision.ESCALATE, "ESCALATE", RiskLevel.HIGH,
            "The drafted response contains claims not supported by retrieved historical evidence.",
            reason_codes,
        )

    # --- OOD veto (experimental novelty signal, escalate-by-default). A
    # message outside the support taxonomy gets a human, not a forced fit —
    # even when the classifier saturates at high confidence on the wrong
    # intent (measured: "What is the capital of India?" at p=1.00).
    if signals.novelty is not None and signals.novelty.is_ood:
        reason_codes.append("OOD_REQUEST")
        return EscalationDecision(
            InternalDecision.ESCALATE, "ESCALATE", RiskLevel.MEDIUM,
            "The message does not match the supported support-intent taxonomy "
            f"(novelty score {signals.novelty.ood_score:.2f}); escalating rather than "
            "forcing it into a guessed intent.",
            reason_codes,
        )

    # --- Multi-intent veto (experimental signal, escalate-by-default).
    # A message that raises multiple issues gets ESCALATE rather than a
    # confidently wrong single intent ("MULTI-INTENT -> ESCALATE" is safer
    # than guessing which issue the customer means).
    if signals.ambiguity is not None and signals.ambiguity.multi_intent_suspected:
        reason_codes.append("MULTI_INTENT")
        return EscalationDecision(
            InternalDecision.ESCALATE, "ESCALATE", RiskLevel.HIGH,
            "Message appears to raise multiple distinct issues ("
            + ", ".join(signals.ambiguity.multi_intent_candidates)
            + "); routing to a human rather than resolving one of them in isolation.",
            reason_codes,
        )

    # --- Intent-level risk floor. High-risk intents (financial, account
    # security) never auto-handle regardless of how strong the evidence
    # looks -- this is a deliberate business-rule ceiling, not something
    # evidence strength should be able to override.
    if signals.intent_escalation_tendency == "high":
        reason_codes.append("HIGH_RISK_INTENT")
        return EscalationDecision(
            InternalDecision.ESCALATE, "ESCALATE", RiskLevel.HIGH,
            f"Intent '{signals.intent}' is classified as high-risk (financial, account-security, or "
            f"no-clear-resolution-path) and always requires human handling regardless of evidence strength.",
            reason_codes,
        )

    # --- Evidence sufficiency checks.
    if signals.num_supporting_cases < t.min_supporting_cases_for_auto:
        reason_codes.append("INSUFFICIENT_EVIDENCE_COUNT")
    if signals.evidence_score < t.min_evidence_score_for_auto:
        reason_codes.append("LOW_EVIDENCE_SCORE")
    if signals.intent_confidence < t.min_intent_confidence_for_auto:
        reason_codes.append("LOW_INTENT_CONFIDENCE")
    if signals.grounding_score is not None and signals.grounding_score < t.min_grounding_score_for_auto:
        reason_codes.append("LOW_GROUNDING_SCORE")
    if signals.ambiguity is not None and signals.ambiguity.is_ambiguous:
        reason_codes.append("AMBIGUOUS_INTENT")
    # Explicit classifier-vs-retrieval disagreement: hidden disagreement is
    # worse than visible disagreement — the reviewer must see that history
    # points somewhere else than the classifier's argmax.
    if signals.ambiguity is not None and signals.ambiguity.retrieval_disagreement:
        reason_codes.append("RETRIEVAL_DISAGREEMENT")
    if signals.ambiguity is not None and signals.ambiguity.sparse_evidence:
        reason_codes.append("WEAK_EVIDENCE")

    if not reason_codes:
        risk = RiskLevel.MEDIUM if signals.intent_escalation_tendency == "medium" else RiskLevel.LOW
        return EscalationDecision(
            InternalDecision.AUTO, "AUTO", risk,
            f"Strong historical evidence ({signals.num_supporting_cases} supporting cases, "
            f"evidence_score={signals.evidence_score:.2f}) and high intent confidence "
            f"({signals.intent_confidence:.2f}) support an automated response for this "
            f"{signals.intent_escalation_tendency}-risk intent.",
            reason_codes,
        )

    # One weak signal alone -> REVIEW (borderline, still maps to ESCALATE
    # publicly, but distinguished internally for the coverage curve).
    # Two or more weak signals -> clearly ESCALATE.
    internal = InternalDecision.REVIEW if len(reason_codes) == 1 else InternalDecision.ESCALATE
    risk = RiskLevel.MEDIUM if len(reason_codes) == 1 else RiskLevel.HIGH
    reason = "Escalating due to: " + "; ".join(_describe_reason_code(c, signals) for c in reason_codes)
    return EscalationDecision(internal, "ESCALATE", risk, reason, reason_codes)


def _describe_reason_code(code: str, s: EscalationSignals) -> str:
    # Lambdas (not f-strings in a dict literal): every value in a dict
    # literal is evaluated eagerly, and several descriptions reference
    # optional fields (e.g. s.ambiguity) that may be None for codes that
    # never use them. Lazy evaluation means each description only touches
    # the fields it actually needs.
    descriptions = {
        "INSUFFICIENT_EVIDENCE_COUNT": lambda: f"only {s.num_supporting_cases} historical supporting case(s) found",
        "LOW_EVIDENCE_SCORE": lambda: f"evidence score {s.evidence_score:.2f} below the auto-handling threshold",
        "LOW_INTENT_CONFIDENCE": lambda: f"intent confidence {s.intent_confidence:.2f} below the auto-handling threshold",
        "LOW_GROUNDING_SCORE": lambda: "generated response grounding score below threshold",
        "AMBIGUOUS_INTENT": lambda: (
            f"top-2 intent probability margin {s.ambiguity.top2_margin:.2f} "
            "indicates a genuinely ambiguous message"
            if s.ambiguity else "message is genuinely ambiguous between multiple intents"
        ),
        "RETRIEVAL_DISAGREEMENT": lambda: (
            "retrieved historical cases mostly belong to a different intent than the classifier's prediction"
        ),
        "LOW_CLASSIFICATION_CONFIDENCE": lambda: (
            f"intent confidence {s.intent_confidence:.2f} below the auto-handling threshold"
        ),
        "WEAK_EVIDENCE": lambda: (
            f"only {s.num_supporting_cases} retrieved case(s) corroborate this reply"
        ),
        "PRIVACY_RISK": lambda: "message may contain credentials that must never be echoed",
        "OOD_REQUEST": lambda: (
            f"novelty score {s.novelty.ood_score:.2f}: message resembles no historical case"
            if s.novelty else "message appears out-of-distribution for the support taxonomy"
        ),
        "WEAK_RETRIEVAL_CORROBORATION": lambda: (
            "retrieved historical evidence is sparse or disagrees with the predicted intent"
        ),
    }
    factory = descriptions.get(code)
    return factory() if factory else code
