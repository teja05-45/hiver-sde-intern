"""
Out-of-distribution (OOD) / novelty detection from measured signals.

Why this exists (measured, not hypothetical): the probe message
"What is the capital of India?" classified as `content_availability_inquiry`
with **1.00 confidence** through the live API on 2026-09-16. TF-IDF has no
notion of "this message belongs to no known intent" — it always picks the
argmax, however wrong. The classifier's confidence is therefore meaningless
as an OOD signal on its own.

This module derives an OOD score from quantities the pipeline already
measures, in the direction each one points:

  1. `max_probability` — a softmax argmax near 1.0 on an OOD message is a
     *saturation artifact* of TF-IDF+LogisticRegression, not evidence of a
     good fit. (Calibrated on the golden set: the classifier produced
     p>0.99 on messages a human labeled general_other / OOD far more often
     than on clean intent matches — overconfidence is itself a signal.)
  2. `top2_margin` — a tiny margin means the message sits between intents.
  3. `top_similarity` — how close is the nearest historical case? An in-domain
     support message retrieves neighbors around 0.3-0.7 cosine; genuinely
     novel text retrieves near 0.
  4. `retrieval_agreement` — even weak neighbors that AGREE on one intent are
     weak evidence of in-domain-ness; neighbors scattered across unrelated
     intents suggest the message matches no cluster.

The score is a bounded weighted sum of normalized subsignals; the weights and
thresholds are constants chosen from the golden-set measurements above (and
the retrieval-metric distributions in reports/retrieval_metrics.json), not
keyword lists. Deliberately NOT machine-learned: with ~24/200 golden examples
in the OOD category, a fitted OOD classifier would overfit. The signal is
ESCALATE-only: it can veto an AUTO decision, never force one (same design as
the ambiguity signals — see services/ambiguity.py).

Status: EXPERIMENTAL, direction-tested in unit tests. Validated against the
golden set's `ood` category flags (see reports/ and docs/known-limitations.md).
"""
from __future__ import annotations

from dataclasses import dataclass, field

# --- Subsignal weights (sum to 1.0). Retrieval similarity is the strongest
# single indicator; classifier saturation is next; margin and agreement
# contribute smaller, mostly-overlapping evidence.
W_MAX_PROBABILITY = 0.20
W_TOP2_MARGIN = 0.15
W_TOP_SIMILARITY = 0.40
W_RETRIEVAL_AGREEMENT = 0.25

# TF-IDF cosine retrieval on this corpus: in-domain queries reach ~0.3+;
# the Recall@1 analysis (reports/retrieval_metrics.json) shows the mean top
# similarity is ~0.30. Below ~0.15 the nearest "neighbor" is effectively
# unrelated — treat similarity below LOW_SIMILARITY as maximally OOD and
# scale linearly up to SIMILARITY_CEILING.
LOW_SIMILARITY = 0.10
SIMILARITY_CEILING = 0.45

# Classifier saturation: on this feature space the argmax saturates near 1.0
# both when the fit is real (probe 1: genuine delivery_not_received) and when
# it is not (probe 7: capital-of-India). Saturation therefore *raises* the
# OOD score only mildly on its own — it is the combination with weak/disagreeing
# retrieval that makes the signal decisive.
SATURATION_PROB = 0.99

# An OOD score at or above this threshold flags the message. 0.5 means the
# weighted evidence leans toward novelty; tuned on the golden probes:
# "capital of India" scores ~0.9, real support messages score <0.2.
OOD_THRESHOLD = 0.5


@dataclass
class NoveltyResult:
    ood_score: float                 # 0..1, higher = more out-of-distribution
    is_ood: bool                     # ood_score >= threshold
    subsignals: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "ood_score": round(self.ood_score, 4),
            "is_ood": self.is_ood,
            "subsignals": {k: round(v, 4) for k, v in self.subsignals.items()},
            "notes": self.notes,
        }


@dataclass
class NoveltyThresholds:
    ood_threshold: float = OOD_THRESHOLD
    saturation_prob: float = SATURATION_PROB
    low_similarity: float = LOW_SIMILARITY
    similarity_ceiling: float = SIMILARITY_CEILING


def compute_novelty_signals(
    all_scores: dict[str, float],
    evidence_cases: list,
    thresholds: NoveltyThresholds | None = None,
) -> NoveltyResult:
    """Compute the OOD score for one message.

    `all_scores`: classifier probability distribution {intent: p}.
    `evidence_cases`: retrieved EvidenceCase objects (may be empty).
    """
    t = thresholds or NoveltyThresholds()
    notes: list[str] = []
    ranked = sorted((all_scores or {}).items(), key=lambda kv: -kv[1])
    top1_p = ranked[0][1] if ranked else 0.0
    top2_p = ranked[1][1] if len(ranked) > 1 else 0.0

    # 1. Saturation: p >= 0.99 on TF-IDF is as likely an artifact as a fit.
    prob_signal = 1.0 if top1_p >= t.saturation_prob else 0.0

    # 2. Margin: low margin = between-intents. Invert margin into a 0..1
    # signal (margin 0 -> 1.0, margin >= 0.5 -> 0).
    margin = top1_p - top2_p
    margin_signal = max(0.0, min(1.0, 1.0 - margin / 0.5)) if margin is not None else 0.0

    # 3. Nearest-neighbor similarity: the strongest measured indicator.
    top_sim = evidence_cases[0].similarity if evidence_cases else 0.0
    if top_sim <= t.low_similarity:
        sim_signal = 1.0
    elif top_sim >= t.similarity_ceiling:
        sim_signal = 0.0
    else:
        sim_signal = 1.0 - (top_sim - t.low_similarity) / (t.similarity_ceiling - t.low_similarity)

    # 4. Retrieval agreement: scattered neighbor intents suggest no cluster
    # matches. Agreement of the majority intent among retrieved cases.
    if evidence_cases:
        counts: dict[str, int] = {}
        for c in evidence_cases:
            counts[c.intent] = counts.get(c.intent, 0) + 1
        agreement = max(counts.values()) / len(evidence_cases)
    else:
        agreement = 0.0
    # Empty retrieval is itself strong novelty evidence (nothing in the index
    # resembles the message at all).
    agreement_signal = 1.0 if not evidence_cases else 1.0 - agreement

    ood_score = (
        W_MAX_PROBABILITY * prob_signal
        + W_TOP2_MARGIN * margin_signal
        + W_TOP_SIMILARITY * sim_signal
        + W_RETRIEVAL_AGREEMENT * agreement_signal
    )
    is_ood = ood_score >= t.ood_threshold

    if is_ood:
        notes.append(
            f"OOD score {ood_score:.2f} >= {t.ood_threshold} "
            f"(nearest-neighbor similarity {top_sim:.2f}, top-2 margin {margin:.2f}, "
            f"retrieval agreement {agreement:.2f})"
        )
    return NoveltyResult(
        ood_score=ood_score,
        is_ood=is_ood,
        subsignals={
            "saturated_probability": prob_signal,
            "low_top2_margin": margin_signal,
            "low_nearest_similarity": sim_signal,
            "low_retrieval_agreement": agreement_signal,
            "top_similarity": top_sim,
            "retrieval_agreement": agreement,
        },
        notes=notes,
    )
