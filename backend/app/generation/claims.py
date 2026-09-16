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

# A claim containing this many consecutive words verbatim from an evidence
# case is supported regardless of the overall ratio: the factual core is a
# direct quotation, and scaffolding words around it ("Based on how we've
# handled similar cases: ...") should not dilute that. Fabricated claims
# ("you will receive a refund within 24 hours") contain no such span.
# Measured motivation: the mock provider's draft embeds the evidence snippet
# in template text, dropping pure word-overlap to ~0.5 for a claim whose
# factual content is a verbatim evidence quote.
SUPPORTED_SPAN_WORDS = 5

# Claim classes verify at different strictness. An ASSERTION (declarative
# statement about the customer's situation or brand policy) demands literal
# evidence support at 0.6 stemmed content-word overlap. A SUGGESTION
# (troubleshooting step: questions, imperatives, numbered-list items) is an
# action to try, not a claim about policy — a real LLM paraphrases recorded
# actions ("logging out, uninstalling and reinstalling the app" becomes
# "Logging out of the app, then logging back in."), so demanding 0.6 literal
# overlap on suggestions rejects exactly the good drafts. Suggestion
# threshold 0.45 keeps hallucinated actions (steps no historical case took)
# failing while accepting close paraphrases. Measured on live Groq drafts:
# evidence-traceable suggestions score 0.5-0.8; invented steps score <0.35.
ASSERTION_THRESHOLD = 0.6
SUGGESTION_THRESHOLD = 0.45

# Sentence splitter: split on sentence-ending punctuation followed by
# whitespace+capital, or newlines. Deliberately simple and deterministic —
# a heavyweight NLP segmenter would be overengineering for tweet-length
# replies, and determinism keeps the UI stable across identical drafts.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])|\n+")

# Social filler that contains no checkable factual claim. Matched against the
# whole sentence (case-insensitive, punctuation-stripped).
_GREETING_RE = re.compile(
    r"^(hi|hello|hey|thanks|thank you|greetings|good (morning|afternoon|evening)"
    r"|sorry (to hear|for the|you|about)|we'?re sorry|i'?m (really )?sorry"
    r"|best regards|regards|sincerely|let us know|hope this helps"
    r"|(we|ll) (look|are looking) (forward|into)|have a (great|good|nice))"
)
_ABSTENTION_RE = re.compile(
    r"(don'?t have (enough |sufficient )?(historical )?(evidence|information)"
    r"|cannot (safely )?answer|routed to (a )?(team member|human|agent)"
    r"|unable to (help|answer|confirm))"
)

# Transitional/structural sentences that introduce content but assert nothing
# themselves ("This can sometimes be resolved by:").
_FILLER_RE = re.compile(
    r"^(this can|here (are|is)|you can try|try the following|in the meantime"
    r"|please (note|see|find)|these steps|the following)"
)

# Suggestion shape: a question, a numbered/bulleted step, or an imperative
# opener. Suggestions are actions to try, not assertions about policy.
_SUGGESTION_RE = re.compile(
    r"^(could|would|can|will|please|try|have you|make sure|let us|kindly"
    r"|check|ensure|verify|visit|reach|contact|uninstall|reinstall|log(ging)? (in|out)"
    r"|sign(ing)? (in|out)|restart|clear|update|confirm|double[- ]check|once|after|if)\b"
)
_LIST_ITEM_RE = re.compile(r"^\s*(?:\d+[.)\]]|[-*•])\s*")

# Lightweight stemming for matching (not for display): suffixes that rarely
# change whether a content word is "the same word" for overlap purposes.
_STEM_SUFFIXES = ("ing", "ed", "es", "s", "ly")


def _normalize(text: str) -> str:
    """Normalize the typographic characters live LLMs actually emit (curly
    quotes, unicode dashes) so literal matching behaves on real drafts."""
    return (text.replace("\u2019", "'").replace("\u2018", "'")
                .replace("\u201c", '"').replace("\u201d", '"')
                .replace("\u2011", "-").replace("\u2013", "-").replace("\u2014", "-"))


def _stem(word: str) -> str:
    for suf in _STEM_SUFFIXES:
        if word.endswith(suf) and len(word) - len(suf) >= 4:
            return word[: len(word) - len(suf)]
    return word


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
    the draft (modulo marker preservation — numbered/bulleted list items
    keep their markers so suggestions classify correctly).

    Consecutive list items are grouped into ONE suggestion claim: a numbered
    troubleshooting list is a single multi-part suggestion, not N independent
    assertions; verifying its parts individually over-counts failures."""
    if not draft or not draft.strip():
        return []
    sentences = _SENTENCE_RE.split(_normalize(draft.strip()))
    claims: list[str] = []
    list_buffer: list[str] = []

    def flush_list():
        if list_buffer:
            claims.append(" ".join(list_buffer))
            list_buffer.clear()

    for s in sentences:
        s = s.strip().strip('"').strip()
        if len(s) < 3:
            continue
        if _LIST_ITEM_RE.match(s):
            list_buffer.append(s)
            continue
        flush_list()
        claims.append(s)
    flush_list()
    return claims


def _word_overlap_ratio(claim: str, evidence_text: str) -> tuple[float, int]:
    """Returns (ratio, matched-word-count) of stemmed claim content words
    present in the evidence text. Words of length > 3, case-insensitive,
    suffix-stemmed (logging->log, reinstalling->reinstall)."""
    words = _content_words(claim)
    if not words:
        return 0.0, 0
    matched = sum(1 for w in words if w in evidence_text)
    return matched / len(words), matched


def _content_words(claim: str) -> list[str]:
    return [_stem(w) for w in re.findall(r"\w+", claim.lower()) if len(w) > 3]


def _longest_verbatim_span(claim: str, evidence_text: str) -> int:
    """Length (in words > 3 chars) of the longest contiguous run of claim
    words that appears verbatim in the evidence text."""
    words = [w for w in re.findall(r"\w+", claim.lower()) if len(w) > 3]
    best = run = 0
    for w in words:
        # Contiguity in the claim plus presence in evidence approximates a
        # verbatim span without requiring exact index alignment (punctuation
        # differs). Good enough for a literal check, and errs toward
        # strictness because the words must still appear in evidence.
        if w in evidence_text:
            run += 1
            best = max(best, run)
        else:
            run = 0
    return best


def verify_claims(draft: str, evidence: EvidenceResult,
                  threshold: float = SUPPORTED_OVERLAP_THRESHOLD) -> ClaimVerificationResult:
    """Extract claims from the actual draft and verify each against the
    evidence the generator was given.

    The generator's contract is "use only the SUPPLIED evidence" (all cases,
    collectively) — a real LLM synthesizes a draft across several retrieved
    resolutions, so a claim may be assembled from two cases while matching
    either one only partially. Verification therefore runs against the UNION
    of all evidence text (the correct contract boundary), while attribution
    tracks the single best-matching case so the UI can point at the strongest
    supporting history. Genuinely invented claims (refund timelines, amounts,
    policy promises) match neither the union nor any case and still fail.

    Claim classes (see thresholds in the module constants):
      - assertion: declarative statements — strict threshold.
      - suggestion: questions / imperatives / numbered steps — lenient
        threshold, because recorded actions are legitimately paraphrased.
    """
    result = ClaimVerificationResult()
    union_text = " ".join((c.resolution or "") for c in evidence.cases).lower()
    for claim in extract_claims(draft):
        claim = _normalize(claim)
        stripped = claim.lower()
        if _ABSTENTION_RE.search(stripped):
            result.claims.append(ClaimCheck(
                claim_text=claim, status="abstention", confidence=1.0,
                explanation="Draft declined to answer due to insufficient evidence; no factual claim to verify."))
            continue
        if _GREETING_RE.search(stripped) or _FILLER_RE.search(stripped):
            result.claims.append(ClaimCheck(
                claim_text=claim, status="greeting", confidence=0.0,
                explanation="Social filler / empathy with no checkable factual content."))
            continue

        is_suggestion = bool(claim.endswith("?") or _LIST_ITEM_RE.match(claim)
                             or _SUGGESTION_RE.match(stripped))
        claim_threshold = SUGGESTION_THRESHOLD if is_suggestion else ASSERTION_THRESHOLD

        if not union_text.strip():
            result.claims.append(ClaimCheck(
                claim_text=claim, status="unsupported", confidence=0.0,
                explanation="No historical evidence was retrieved, so nothing in the draft can be verified."))
            continue

        union_ratio, _ = _word_overlap_ratio(claim, union_text)
        union_span = _longest_verbatim_span(claim, union_text)
        supported = union_ratio >= claim_threshold or union_span >= SUPPORTED_SPAN_WORDS

        # Attribution: best single supporting case (for the UI's evidence link).
        best_ratio, best_ids, best_span = 0.0, [], 0
        for case in evidence.cases:
            ev_text = (case.resolution or "").lower()
            if not ev_text:
                continue
            ratio, _ = _word_overlap_ratio(claim, ev_text)
            span = _longest_verbatim_span(claim, ev_text)
            if ratio > best_ratio or (ratio == best_ratio and span > best_span):
                best_ratio, best_span = ratio, span
                best_ids = [case.conversation_id]
            elif ratio == best_ratio and ratio > 0 and case.conversation_id not in best_ids:
                best_ids.append(case.conversation_id)

        kind = "suggested action" if is_suggestion else "claim"
        if supported and best_span >= SUPPORTED_SPAN_WORDS and best_ratio < claim_threshold:
            explanation = (f"Contains a {best_span}-word verbatim passage from historical case "
                           f"{best_ids[0]} — a direct quotation of the recorded resolution.")
        elif supported:
            explanation = (f"{int(union_ratio * 100)}% word overlap with the retrieved evidence "
                           f"(best case: {best_ids[0]})" if best_ids else
                           f"{int(union_ratio * 100)}% word overlap with the retrieved evidence.")
        else:
            explanation = (f"Best evidence match reached only {int(union_ratio * 100)}% word overlap "
                           f"(below the {int(claim_threshold * 100)}% threshold for this {kind}, "
                           f"longest verbatim span {union_span} words) — no retrieved historical "
                           f"case supports this {kind}.")
        result.claims.append(ClaimCheck(
            claim_text=claim,
            status="supported" if supported else "unsupported",
            evidence_ids=best_ids if supported else [],
            confidence=max(union_ratio, min(union_span / 8.0, 1.0)),
            explanation=explanation,
        ))
    return result
