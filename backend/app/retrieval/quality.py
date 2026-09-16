"""
Retrieval quality layer: hybrid scoring, intent-aware reranking,
resolution-quality weighting, and per-case retrieval explanations.

Why this module exists: plain TF-IDF top-k answers "which historical
messages are textually nearest" but not "which cases are actually useful
resolutions for this customer's problem." The measured Recall@1 of 0.574
(intent-proxy relevance) leaves real headroom, and inspection of
mis-ranked queries shows two systematic failure modes that pure cosine
cannot fix:

  1. A lexically similar case from a DIFFERENT intent outranks the true
     intent ("I need help with my order" beating a real
     delivery_not_received case for a not-received query).
  2. Cases with no usable resolution (placeholder "please contact
     support", empty) rank high purely on message similarity and waste
     half the evidence window.

The hybrid score is a weighted, configurable combination of four
independently-inspectable channels (assignment Phase 4):

    final = w_semantic  * semantic_similarity     (TF-IDF cosine)
          + w_lexical   * lexical_relevance       (BM25, query-normalized)
          + w_intent    * intent_compatibility    (classifier P(case.intent | query))
          + w_quality   * resolution_quality      (usable, specific resolution)
          - contradiction_penalty                 (retrieval-majority disagreement)

Design rules:
  - Every weight is justified (docstrings below) and overridable via env /
    constructor -- no unexplained magic numbers baked into ranking code.
  - The classifier's probability for the CASE's intent given the QUERY is
    the intent-compatibility signal. This is a soft prior from the already-
    trained classifier -- NOT keyword rules and NOT a hard filter, so rare
    intents and near-boundary messages degrade gracefully instead of
    being excluded.
  - Resolution quality is corpus-derived at index time (documented heuristics,
    monotone in usefulness), never per-query tuned.
  - `explain()` records every component per case so the UI can answer
    "why was this case retrieved?" from real numbers.
  - The retriever keeps the raw semantic ranking available
    (`baseline_cases`) so A/B evaluation (reports/retrieval_metrics.json)
    can measure the delta instead of asserting it.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from app.retrieval.lexical import BM25Index, tokenize

if TYPE_CHECKING:  # circular-free: retriever imports this module at runtime
    from app.retrieval.retriever import EvidenceCase

# ---------------------------------------------------------------------------
# Resolution-quality heuristics (corpus-level, computed at index time).
#
# Justification for each rule, from the dataset build report
# (reports/dataset_build_report.json) and manual inspection of low-quality
# resolutions during the audit:
#   - Tweets are short; a resolution under ~40 chars is a placeholder
#     ("Thanks!", "DM us") that carries no resolution pattern.
#   - "please contact support/DM us/call us" responses defer the customer
#     elsewhere instead of resolving; they are not evidence of HOW a problem
#     gets resolved. They are down-weighted, not deleted (they may still be
#     the only history available -- documented decision, decision-log #23).
#   - Very long resolutions are often concatenated multi-turn threads with
#     boilerplate; they get a mild cap, not a penalty, because length alone
#     does not make them bad.
# ---------------------------------------------------------------------------

_MIN_RESOLUTION_CHARS = 40
_LONG_RESOLUTION_CHARS = 600
_DEFERRAL_RE = re.compile(
    r"(please (dm|contact|call)|dm us|direct message|reach out to|"
    r"contact (us|support|customer (service|care))|call (us|our)|"
    r"check (your )?(dm|direct message)s?)",
)
_SPECIFICITY_HINTS = (
    "refund", "resend", "replacement", "tracking", "order", "account",
    "password", "reset", "prime", "cancel", "return", "label", "voucher",
    "credit", "shipment", "delivery", "dispatch", "re-ship", "reimburse",
)


def resolution_quality(text: str | None) -> float:
    """0..1 usefulness estimate of a historical resolution, from corpus-level
    heuristics fixed at index time (never query-dependent):
      0.0  missing/empty
      0.2  too short to contain a resolution pattern
      0.4  pure deferral ("please contact support") with no substance
      0.6  generic but substantive
      0.8+ substantive AND topic-specific (mentions a concrete action/object)
    Length above _LONG_RESOLUTION_CHARS neither adds nor subtracts (mild cap
    implemented as a floor of the long-text bonus).
    """
    if not text or not text.strip():
        return 0.0
    t = text.strip().lower()
    if len(t) < _MIN_RESOLUTION_CHARS:
        return 0.2
    score = 0.6
    if _DEFERRAL_RE.search(t):
        # Deferral with substance still ranks between generic and deferral.
        score = 0.5 if len(t) > 120 else 0.4
    hits = sum(1 for h in _SPECIFICITY_HINTS if h in t)
    if hits:
        score = max(score, 0.8 if hits >= 2 else 0.75)
    return min(score, 1.0)


# ---------------------------------------------------------------------------
# Hybrid weights. Defaults are engineering choices with stated rationale;
# every one is overridable by env (RETRIEVAL_W_*) or constructor so the
# evaluation script can sweep them instead of the code hardcoding a guess.
# ---------------------------------------------------------------------------

def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    try:
        return float(v) if v not in (None, "") else default
    except ValueError:
        return default


@dataclass
class HybridRetrievalConfig:
    """Weights for the hybrid retrieval score. All values in [0, 1].

    Defaults and why:
      w_semantic=1.0   TF-IDF cosine stays the primary channel: it is the
                       signal the evidence scorer and grounding were
                       calibrated against, and the retrieval metrics
                       (Recall@5 0.86) show it is genuinely informative.
      w_lexical=0.35   BM25 adds term-frequency saturation + length
                       normalization; secondary because TF-IDF 1-2 grams
                       already covers most lexical overlap on short text.
      w_intent=0.30    The classifier's probability for the case's intent
                       is a strong, already-trained prior. Kept BELOW the
                       text channels deliberately: classifier errors
                       (golden macro-F1 0.18!) must not dominate ranking.
      w_quality=0.25   Resolution quality separates "useful resolution"
                       from "contact support" cases. Modest weight because
                       it is heuristic, not learned.
      contradiction_penalty=0.15
                       Applied when the case's intent disagrees with the
                       retrieval-majority intent among the top candidates
                       (classifier-vs-retrieval disagreement is already
                       escalated downstream by the policy; here it only
                       demotes a likely-off-topic case within the pool).
    Weights are additive on a 0..1 semantic scale, so the final score is
    comparable across queries after per-channel normalization below.
    """
    w_semantic: float = field(default_factory=lambda: _env_float("RETRIEVAL_W_SEMANTIC", 1.0))
    w_lexical: float = field(default_factory=lambda: _env_float("RETRIEVAL_W_LEXICAL", 0.35))
    w_intent: float = field(default_factory=lambda: _env_float("RETRIEVAL_W_INTENT", 0.30))
    w_quality: float = field(default_factory=lambda: _env_float("RETRIEVAL_W_QUALITY", 0.25))
    contradiction_penalty: float = field(default_factory=lambda: _env_float("RETRIEVAL_CONTRADICTION_PENALTY", 0.15))

    def as_dict(self) -> dict:
        return {
            "w_semantic": self.w_semantic,
            "w_lexical": self.w_lexical,
            "w_intent": self.w_intent,
            "w_quality": self.w_quality,
            "contradiction_penalty": self.contradiction_penalty,
        }


# Candidate pool pulled from the raw vector search before reranking. The
# final k is cut from this pool, so reranking has room to promote cases the
# cosine ranking buried. 4x k, bounded to keep the BM25/pipeline cost small.
def candidate_pool_size(k: int) -> int:
    return max(20, min(4 * k, 100))


@dataclass
class ScoredCase:
    """One retrieved case with its full scoring breakdown."""
    case: EvidenceCase
    final_score: float
    components: dict[str, float]      # named channel -> normalized 0..1 value
    weight_contributions: dict[str, float]  # named channel -> weight * value
    rank_raw: int                     # position in the pre-rerank (cosine) order
    rank_final: int                   # position in the final order
    explanation: str = ""

    def explain(self) -> dict:
        return {
            "conversation_id": self.case.conversation_id,
            "final_score": round(self.final_score, 4),
            "rank_raw": self.rank_raw,
            "rank_final": self.rank_final,
            "components": {k: round(v, 4) for k, v in self.components.items()},
            "weighted": {k: round(v, 4) for k, v in self.weight_contributions.items()},
            "explanation": self.explanation,
        }


def score_candidates(
    candidates: list[EvidenceCase],
    query_text: str,
    intent_probs: dict[str, float] | None,
    bm25: BM25Index,
    quality_by_id: dict[str, float],
    cfg: HybridRetrievalConfig,
) -> list[ScoredCase]:
    """Score one query's candidate pool and return it ranked by final score.

    Normalization contract (keeps channels comparable):
      semantic  : raw cosine in [0, 1] already.
      lexical   : BM25 max-normalized within THIS query's candidate pool
                  (BM25 magnitudes are query-dependent; raw scores kept in
                  components as bm25_raw).
      intent    : classifier P(case_intent | query) in [0, 1] directly.
      quality   : resolution_quality() in [0, 1] directly.
    """
    if not candidates:
        return []

    raw_bm25 = bm25.score(query_text)
    # Retrieval-majority intent among the pool (for the contradiction signal).
    intent_counts: dict[str, int] = {}
    for c in candidates:
        intent_counts[c.intent] = intent_counts.get(c.intent, 0) + 1
    majority_intent = max(intent_counts, key=intent_counts.get)

    max_bm25 = max(raw_bm25) if raw_bm25 else 0.0

    scored: list[ScoredCase] = []
    for pos, case in enumerate(candidates):
        semantic = max(0.0, min(1.0, case.similarity))
        lex_raw = raw_bm25[pos] if pos < len(raw_bm25) else 0.0
        lexical = (lex_raw / max_bm25) if max_bm25 > 0 else 0.0
        intent_compat = (intent_probs or {}).get(case.intent, 0.0)
        quality = quality_by_id.get(case.conversation_id, 0.0)

        penalty = 0.0
        if (case.intent != majority_intent and intent_counts[majority_intent] > len(candidates) / 2):
            penalty = cfg.contradiction_penalty

        components = {
            "semantic": semantic,
            "lexical": lexical,
            "intent_compatibility": intent_compat,
            "resolution_quality": quality,
            "contradiction": penalty,
            "bm25_raw": lex_raw,
        }
        weighted = {
            "semantic": cfg.w_semantic * semantic,
            "lexical": cfg.w_lexical * lexical,
            "intent": cfg.w_intent * intent_compat,
            "quality": cfg.w_quality * quality,
        }
        final = sum(weighted.values()) - penalty
        scored.append(ScoredCase(
            case=case, final_score=final, components=components,
            weight_contributions=weighted, rank_raw=pos + 1, rank_final=0,
        ))

    scored.sort(key=lambda s: (-s.final_score, -s.case.similarity))
    for i, s in enumerate(scored):
        s.rank_final = i + 1
        s.explanation = _explain(s, majority_intent)
    return scored


def _explain(s: ScoredCase, majority_intent: str) -> str:
    c = s.components
    bits: list[str] = []
    bits.append(f"semantic similarity {c['semantic']:.2f}")
    if c["lexical"] > 0:
        bits.append(f"lexical relevance {c['lexical']:.2f}")
    if c["intent_compatibility"] > 0:
        bits.append(f"classifier assigns {s.case.intent} p={c['intent_compatibility']:.2f}")
    bits.append(f"resolution quality {c['resolution_quality']:.1f}/1.0")
    if c["contradiction"] > 0:
        bits.append(f"demoted: pool majority intent is {majority_intent}")
    promoted = s.rank_raw - s.rank_final
    if promoted > 0:
        bits.append(f"promoted {promoted} place(s) by reranking")
    elif promoted < 0:
        bits.append(f"demoted {-promoted} place(s) by reranking")
    return "; ".join(bits)
