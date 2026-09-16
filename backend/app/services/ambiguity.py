"""
Ambiguity and multi-intent detection.

Why this exists: the golden-set failure analysis (reports/failure_analysis.json)
shows the two largest failure categories are `ambiguous_intent` (23.7% of
failures) and `multi_intent_message` (15.1%). The escalation policy used to
see only the classifier's confidence and the evidence score -- both of which
can look healthy on a message that genuinely fits two intents about equally
well. These detectors give the policy a direct, inspectable signal for both
failure modes instead of relying on low confidence to catch them (it often
doesn't: see golden_104 in the failure analysis, predicted with 0.986
confidence while wrong).

Status: EXPERIMENTAL. The signals are computed from measured quantities
(top-2 probability margin, retrieval-intent disagreement, sparse evidence,
message shape) and are unit-tested for direction (ambiguous input scores
higher than clean input), but they have NOT been evaluated against a
labeled ambiguity ground truth -- there is none in this project. The
escalation policy treats them as escalat*ing* signals only: they can turn
an AUTO into an ESCALATE, never the reverse. The golden set's `ambiguous`
and `multi_intent` category flags let a future evaluation measure them
properly.

Design notes:
  - top2_margin: probability gap between the classifier's best and second
    intent. A tiny margin is the classic signature of ambiguity; large
    margins indicate a committed prediction. Threshold chosen from the
    golden-set methodology itself, which defined `ambiguous` as top-2
    margin < 0.15 (see data/golden/README.md) -- so this detector is
    consistent with how the project's own evaluation data was stratified.
  - multi-intent: keyword-rule disagreement -- different intents'
    positive_signals matching the same message is direct evidence of
    multiple issue types present (the same signal the golden set used).
  - All thresholds are dataclass fields, not magic numbers inline.
"""
from __future__ import annotations

from dataclasses import dataclass

AMBIGUOUS_MARGIN_THRESHOLD = 0.15  # same definition the golden set used


@dataclass
class AmbiguityResult:
    """Ambiguity/novelty signals for one message. Every field is independently
    inspectable and logged -- no blended 'ambiguity score' hiding the inputs."""
    top2_margin: float                 # p(top1) - p(top2); small = ambiguous
    is_ambiguous: bool                 # top2_margin < threshold
    multi_intent_suspected: bool       # 2+ intents' keyword signals fired
    multi_intent_candidates: list[str]
    sparse_evidence: bool              # fewer than `min_cases` supporting cases
    retrieval_disagreement: bool       # top-1 intent vs majority of retrieved
    noise_suspected: bool              # very short / low-signal message
    notes: list[str]

    def as_dict(self) -> dict:
        return {
            "top2_margin": round(self.top2_margin, 4),
            "is_ambiguous": self.is_ambiguous,
            "multi_intent_suspected": self.multi_intent_suspected,
            "multi_intent_candidates": self.multi_intent_candidates,
            "sparse_evidence": self.sparse_evidence,
            "retrieval_disagreement": self.retrieval_disagreement,
            "noise_suspected": self.noise_suspected,
            "notes": self.notes,
        }


@dataclass
class AmbiguityThresholds:
    ambiguous_top2_margin: float = AMBIGUOUS_MARGIN_THRESHOLD
    min_cases_for_confident_retrieval: int = 2
    min_message_tokens: int = 3


def compute_ambiguity_signals(
    all_scores: dict[str, float],
    message: str,
    evidence_cases: list,
    intents_cfg: dict[str, dict],
    thresholds: AmbiguityThresholds | None = None,
) -> AmbiguityResult:
    """Compute ambiguity/multi-intent signals from measured quantities.

    `all_scores`: classifier's full probability distribution {intent: p}.
    `evidence_cases`: retrieved EvidenceCase objects (or an empty list).
    `intents_cfg`: taxonomy config with each intent's positive_signals.
    """
    t = thresholds or AmbiguityThresholds()
    notes: list[str] = []

    ranked = sorted(all_scores.items(), key=lambda kv: -kv[1])
    top1_p = ranked[0][1] if ranked else 0.0
    top2_p = ranked[1][1] if len(ranked) > 1 else 0.0
    margin = top1_p - top2_p
    is_ambiguous = margin < t.ambiguous_top2_margin
    if is_ambiguous:
        notes.append(
            f"top-2 intent probability margin {margin:.3f} < {t.ambiguous_top2_margin} "
            f"(the project's own 'ambiguous' definition)"
        )

    # Multi-intent: distinct intents' keyword signal sets matching the message.
    msg_lower = message.lower()
    fired: dict[str, list[str]] = {}
    for intent_name, cfg in (intents_cfg or {}).items():
        hits = [s for s in (cfg.get("positive_signals") or []) if s and s.lower() in msg_lower]
        if hits:
            fired[intent_name] = hits
    multi_candidates = sorted(fired.keys())
    multi_intent_suspected = len(multi_candidates) >= 2
    if multi_intent_suspected:
        notes.append(
            "positive_signals from multiple intents matched: " + ", ".join(multi_candidates)
        )

    sparse_evidence = len(evidence_cases) < t.min_cases_for_confident_retrieval
    if sparse_evidence:
        notes.append(f"only {len(evidence_cases)} retrieved case(s) support this reply")

    retrieval_disagreement = False
    if evidence_cases and all_scores:
        pred_intent = ranked[0][0]
        agree = sum(1 for c in evidence_cases if c.intent == pred_intent)
        retrieval_disagreement = agree < len(evidence_cases) / 2
        if retrieval_disagreement:
            notes.append("retrieved historical intents mostly disagree with the predicted intent")

    tokens = [w for w in message.split() if w.strip()]
    noise_suspected = len(tokens) < t.min_message_tokens
    if noise_suspected:
        notes.append(f"message is very short ({len(tokens)} tokens); little signal to classify")

    return AmbiguityResult(
        top2_margin=margin,
        is_ambiguous=is_ambiguous,
        multi_intent_suspected=multi_intent_suspected,
        multi_intent_candidates=multi_candidates,
        sparse_evidence=sparse_evidence,
        retrieval_disagreement=retrieval_disagreement,
        noise_suspected=noise_suspected,
        notes=notes,
    )
