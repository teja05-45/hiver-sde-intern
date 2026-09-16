"""
Embedding provider abstraction.

`EmbeddingProvider` is the seam between "how we turn text into vectors"
and everything downstream (retrieval, evidence scoring). The only
implementation available in this sandbox is TF-IDF-based (no network
access to `pip install sentence-transformers`), but retrieval/evidence
code depends only on this interface, so swapping in a real sentence
embedding model in a normal deployment environment is a ~10-line change
in one place, not a rewrite. See docs/decision-log.md.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class EmbeddingProvider(ABC):
    @abstractmethod
    def fit(self, texts: list[str]) -> "EmbeddingProvider":
        ...

    @abstractmethod
    def encode(self, texts: list[str]) -> np.ndarray:
        """Return an (n_texts, dim) array. Rows should be L2-normalized so
        cosine similarity reduces to a dot product."""
        ...


class TfidfEmbeddingProvider(EmbeddingProvider):
    """TF-IDF vectors, L2-normalized. Not semantic (no synonym/paraphrase
    matching -- "package late" and "order delayed" get no credit for
    meaning the same thing), which is a real, documented limitation
    relative to sentence-transformer embeddings. Adequate for this
    sandbox; see docs/decision-log.md for the swap-out path.
    """

    def __init__(self, max_features: int = 30000, ngram_range: tuple[int, int] = (1, 2)) -> None:
        self.vectorizer = TfidfVectorizer(
            max_features=max_features, min_df=2, max_df=0.5, ngram_range=ngram_range, norm="l2"
        )
        self._fitted = False

    def fit(self, texts: list[str]) -> "TfidfEmbeddingProvider":
        if not texts:
            # Nothing to fit on (e.g. a brand-new brand with zero history
            # yet). Leave unfitted; encode()/downstream retrieval will
            # correctly treat this as "no evidence available" rather than
            # crashing on an empty-vocabulary vectorizer fit.
            self._fitted = False
            self._empty = True
            return self
        self._empty = False
        # Adaptive min_df/max_df: the configured defaults (min_df=2,
        # max_df=0.5) are sensible at production scale (tens of thousands
        # of documents) but raise on tiny corpora (e.g. unit tests with a
        # handful of fixture documents, or a brand new/low-volume brand in
        # practice) where they can eliminate the entire vocabulary. Falling
        # back to min_df=1/max_df=1.0 below a small-corpus threshold keeps
        # this robust without changing production behavior.
        vectorizer = self.vectorizer
        if len(texts) < 20:
            vectorizer.set_params(min_df=1, max_df=1.0)
        vectorizer.fit(texts)
        self._fitted = True
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        if getattr(self, "_empty", False):
            # Fit on zero documents -> zero-dimensional feature space;
            # return an all-zero sparse row per query so downstream
            # cosine-similarity search sees "no similarity to anything"
            # rather than crashing.
            from scipy.sparse import csr_matrix
            return csr_matrix((len(texts), 0))
        if not self._fitted:
            raise RuntimeError("TfidfEmbeddingProvider.fit() must be called before encode().")
        # Deliberately sparse (not .toarray()): a dense (n_texts, max_features)
        # float64 array blows past this sandbox's memory budget once the
        # corpus reaches tens of thousands of documents (measured:
        # ~37K x 30K float64 needs ~8.2GB, which OOMs a ~3GB container).
        # sklearn's NearestNeighbors handles sparse input natively for the
        # cosine metric, so there's no need to densify.
        return self.vectorizer.transform(texts)
