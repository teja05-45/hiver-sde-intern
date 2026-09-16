"""
Lexical (BM25) scoring for hybrid retrieval.

Why lexical on top of TF-IDF cosine: TF-IDF similarity is dominated by
shared *rare* terms and is length-sensitive in a different way than BM25.
BM25 adds a saturating term-frequency model plus document-length
normalization, which surfaces cases sharing distinctive support vocabulary
("refund", "charged twice", "tracking number") even when the TF-IDF cosine
ranks them slightly lower. The hybrid score treats the two as independent
evidence channels; the weights live in retrieval/quality.py
(HybridRetrievalConfig) with their justification.

Implementation notes:
  - Pure stdlib + math. An inverted index (term -> [(doc, tf), ...]) means
    scoring touches only documents containing at least one query term,
    which on a 36k-document corpus is the difference between ~300k dict
    lookups and a few dozen per query.
  - k1=1.2, b=0.75 are the standard Robertson/Sparck-Jones defaults,
    used unchanged: they are the most-studied setting and this corpus is
    short-text (b=0.75 mildly corrects for length; no evidence-driven
    reason to deviate).
  - Scores are NOT directly comparable across queries (BM25 magnitudes
    depend on query length and IDF distribution), so the hybrid scorer
    normalizes BM25 within each query's candidate pool (max-normalize).
    Raw BM25 is kept on each case's component breakdown for inspection.
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

_TOKEN_RE = re.compile(r"\w+")

# Tokens too short or too generic to carry support-topic signal. Kept tiny
# and conservative: dropping content words loses recall, and TF-IDF's own
# IDF already handles most boilerplate.
_MIN_TOKEN_LEN = 2


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens, length >= 2. No stemming (BM25 in the hybrid
    score is a secondary channel; TF-IDF 1-2 grams provides the morphology
    sensitivity, and unstemmed tokens keep the component breakdown legible)."""
    return [t for t in _TOKEN_RE.findall((text or "").lower()) if len(t) >= _MIN_TOKEN_LEN]


class BM25Index:
    """Okapi BM25 over a fixed corpus, backed by an inverted index."""

    def __init__(self, k1: float = 1.2, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._doc_lens: list[int] = []
        self._avgdl: float = 0.0
        self._n_docs = 0
        self._df: Counter = Counter()                     # term -> document frequency
        self._postings: dict[str, list[tuple[int, int]]] = defaultdict(list)  # term -> [(doc, tf)]

    def fit(self, texts: list[str]) -> "BM25Index":
        self._doc_lens = []
        self._df = Counter()
        self._postings = defaultdict(list)
        for i, text in enumerate(texts):
            tokens = tokenize(text)
            self._doc_lens.append(len(tokens))
            for term, tf in Counter(tokens).items():
                self._postings[term].append((i, tf))
                self._df[term] += 1
        self._n_docs = len(texts)
        self._avgdl = (sum(self._doc_lens) / self._n_docs) if self._n_docs else 0.0
        return self

    def _ensure_postings(self) -> None:
        """Backward compatibility: artifacts pickled from the first BM25
        version (per-doc Counter lists) predate the inverted index. Rebuild
        the postings once, in place, instead of crashing at query time."""
        if hasattr(self, "_postings"):
            return
        self._postings = defaultdict(list)
        self._df = Counter()
        for i, counts in enumerate(getattr(self, "_doc_freqs", []) or []):
            for term, tf in counts.items():
                self._postings[term].append((i, tf))
                self._df[term] += 1

    def score(self, query: str) -> list[float]:
        """BM25 score of every document against `query`, in corpus order.
        Documents containing no query term keep score 0.0."""
        self._ensure_postings()
        scores = [0.0] * self._n_docs
        if not self._n_docs:
            return scores
        avgdl = self._avgdl or 1.0
        for term in set(tokenize(query)):
            postings = self._postings.get(term)
            if not postings:
                continue
            df = self._df[term]
            # Standard IDF with the +1 inside the log to keep non-negative
            # scores when a term appears in more than half the corpus.
            idf = math.log(1.0 + (self._n_docs - df + 0.5) / (df + 0.5))
            for i, tf in postings:
                denom = tf + self.k1 * (1.0 - self.b + self.b * (self._doc_lens[i] / avgdl))
                scores[i] += idf * (tf * (self.k1 + 1.0)) / denom
        return scores
