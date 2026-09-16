"""
Claim extraction and per-claim verification against retrieved evidence.

Why this module exists: grounding used to verify only the LLM's
*self-reported* `grounded_claims` list. Two failure modes followed:
(a) the LLM could mislabel its own claims, and (b) claims displayed in the
UI were self-reported strings that did not literally correspond to the
generated draft text — the UI could show "unsupported claim: refund/timeline"
for a draft that contained no such claim.

The fix is structural: claims are extracted from the **actual draft reply**
(sentence segmentation), then each extracted claim is independently verified
against the retrieved evidence text. The verification method stays the same
deliberately-literal word-overlap rule as `check_grounding` (no semantic
entailment available without an embedding/LLM check — documented limitation),
but now every claim shown downstream is guaranteed to be a substring of the
generated response, and every claim carries the IDs of the evidence cases
that support it.

`status` values:
  - "supported":   word-overlap >= threshold against >= 1 evidence case
  - "unsupported": no evidence case reaches the threshold
  - "greeting":    social filler / thanks / sign-off with no checkable content
  - "abstention":  the draft declined to answer (evidence_insufficient path)

Greeting/abstention claims are reported for UI completeness but are
policy-neutral: they neither pass nor fail grounding on their own.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.retrieval.retriever import EvidenceResult

# Same verification threshold as check_grounding's historical rule.
SUPPORTED_OVERLAP_THRESHOLD = 0.6

# Sentence splitter: split on sentence-ending punctuation followed by
# whitespace+capital, or newlines. Deliberately simple and deterministic —
# a heavyweight NLP segmenter would be overengineering for tweet-length
# replies, and determinism keeps the UI stable across identical drafts.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])|\n+")

# Social filler that contains no checkable factual claim. Matched against the
# whole sentence (case-insensitive, punctuation-stripped).
_GREETING_RE = re.compile(
    r"^(hi|hello|hey|thanks|thank you|greetings|good (morning|afternoon|evening)"
    r"|sorry (to hear|for the|you'?re)|we'?re sorry|i'?m (really )?sorry"
    r"|best regards|regards|sincerely|let us know|hope this helps"
    r"|(we|ll) (look|are looking) (forward|into)|have a (great|good|nice))"
)
_ABSTENTION_RE = re.compile(
    r"(don'?t have (enough |sufficient )?(historical )?(evidence|information)"
    r"|cannot (safely )?answer|routed to (a )?(team member|human|agent)"
    r"|unable to (help|answer|confirm))"
)


@dataclass
class ClaimCheck:
    """One extracted claim with its verification result. `claim_text` is
    always a literal substring of the draft it was extracted from."""
    claim_text: str
    status: str                 # supported | unsupported | greeting | abstention
    evidence_ids: list[str] = field(default_factory=list)
    confidence: float = 0.0     # best word-overlap ratio across evidence cases
    explanation: str = ""

    def as_dict(self) -> dict:
        return {
            "claim_text": self.claim_text,
            "status": self.status,
            "evidence_ids": self.evidence_ids,
            "confidence": round(self.confidence, 3),
            "explanation": self.explanation,
        }


@dataclass
class ClaimVerificationResult:
    claims: list[ClaimCheck] = field(default_factory=list)

    @property
    def checkable(self) -> list[ClaimCheck]:
        return [c for c in self.claims if c.status in ("supported", "unsupported")]

    @property
    def unsupported(self) -> list[ClaimCheck]:
        return [c for c in self.claims if c.status == "unsupported"]

    @property
    def supported(self) -> list[ClaimCheck]:
        return [c for c in self.claims if c.status == "supported"]

    @property
    def score(self) -> float:
        """Share of checkable claims that are supported. Returns 1.0 when
        there are no checkable claims but the draft abstained or was pure
        greeting (nothing to contradict), 0.5 when a draft had neither
        checkable claims nor abstention (nothing verifiable was said)."""
        n = len(self.checkable)
        if n:
            return round(len(self.supported) / n, 3)
        if any(c.status == "abstention" for c in self.claims):
            return 1.0
        return 0.5

    @property
    def passed(self) -> bool:
        return not self.unsupported and self.score >= 0.6

    def as_dict(self) -> dict:
        return {
            "claims": [c.as_dict() for c in self.claims],
            "score": self.score,
            "passed": self.passed,
            "n_supported": len(self.supported),
            "n_unsupported": len(self.unsupported),
        }


def extract_claims(draft: str) -> list[str]:
    """Split a draft reply into sentence-level claims. Empty sentences and
    pure whitespace are dropped; every returned string is a substring of
    `draft`."""
    if not draft or not draft.strip():
        return []
    sentences = _SENTENCE_RE.split(draft.strip())
    claims = []
    for s in sentences:
        s = s.strip().strip('"').strip()
        if len(s) >= 3:
            claims.append(s)
    return claims


def _word_overlap_ratio(claim: str, evidence_text: str) -> tuple[float, int]:
    """Returns (ratio, matched-word-count) of claim words present in the
    evidence text. Same rule as check_grounding: words of length > 3,
    case-insensitive."""
    claim_lower = claim.lower().strip()
    words = [w for w in re.findall(r"\w+", claim_lower) if len(w) > 3]
    if not words:
        return 0.0, 0
    matched = sum(1 for w in words if w in evidence_text)
    return matched / len(words), matched


def verify_claims(draft: str, evidence: EvidenceResult,
                  threshold: float = SUPPORTED_OVERLAP_THRESHOLD) -> ClaimVerificationResult:
    """Extract claims from the actual draft and verify each against the
    evidence text of every retrieved case, tracking which case IDs support
    each claim."""
    result = ClaimVerificationResult()
    for claim in extract_claims(draft):
        stripped = claim.lower()
        if _ABSTENTION_RE.search(stripped):
            result.claims.append(ClaimCheck(
                claim_text=claim, status="abstention", confidence=1.0,
                explanation="Draft declined to answer due to insufficient evidence; no factual claim to verify."))
            continue
        if _GREETING_RE.search(stripped):
            result.claims.append(ClaimCheck(
                claim_text=claim, status="greeting", confidence=0.0,
                explanation="Social filler with no checkable factual content."))
            continue

        best_ratio, best_ids = 0.0, []
        for case in evidence.cases:
            ev_text = (case.resolution or "").lower()
            if not ev_text:
                continue
            ratio, _ = _word_overlap_ratio(claim, ev_text)
            if ratio > best_ratio:
                best_ratio = ratio
                best_ids = [case.conversation_id]
            elif ratio == best_ratio and ratio > 0 and case.conversation_id not in best_ids:
                best_ids.append(case.conversation_id)

        supported = best_ratio >= threshold
        result.claims.append(ClaimCheck(
            claim_text=claim,
            status="supported" if supported else "unsupported",
            evidence_ids=best_ids if supported else [],
            confidence=best_ratio,
            explanation=(
                f"{int(best_ratio * 100)}% word overlap with historical case {best_ids[0]}."
                if supported and best_ids else
                f"Best evidence match reached only {int(best_ratio * 100)}% word overlap "
                f"(below the {int(threshold * 100)}% threshold) — no historical case supports this claim."
            ),
        ))
    return result
