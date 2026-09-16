"""Simple in-memory vector store (cosine similarity via normalized dot product).

No FAISS/Chroma available in this sandbox (no network to install). For the
corpus sizes here (tens of thousands of vectors, not millions), a plain
`sklearn.neighbors.NearestNeighbors` with cosine metric is fast enough and
has zero extra dependencies. The `VectorStore` interface is narrow enough
that swapping in FAISS/Chroma for a larger production corpus is a
same-shape replacement, not a redesign. See docs/decision-log.md.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.neighbors import NearestNeighbors


@dataclass
class SearchResult:
    index: int
    similarity: float
    metadata: dict[str, Any]


class VectorStore:
    def __init__(self) -> None:
        self._vectors: np.ndarray | None = None
        self._metadata: list[dict[str, Any]] = []
        self._nn: NearestNeighbors | None = None
        self._built: bool = False

    def build(self, vectors: np.ndarray, metadata: list[dict[str, Any]]) -> "VectorStore":
        assert vectors.shape[0] == len(metadata), "vectors/metadata length mismatch"
        self._vectors = vectors
        self._metadata = metadata
        self._built = True
        if len(metadata) == 0:
            # Nothing to index (e.g. a brand-new brand with no historical
            # data yet). Leave _nn unset; search() short-circuits to [].
            self._nn = None
            return self
        # cosine distance = 1 - cosine similarity; vectors are expected
        # pre-normalized by the embedding provider, but we don't rely on
        # that here -- 'cosine' metric normalizes internally too.
        self._nn = NearestNeighbors(n_neighbors=min(50, len(metadata)), metric="cosine")
        self._nn.fit(vectors)
        return self

    def search(self, query_vector, k: int = 5) -> list[SearchResult]:
        if not self._built:
            raise RuntimeError("VectorStore.build() must be called before search().")
        if len(self._metadata) == 0:
            return []
        k = min(k, len(self._metadata))
        if k == 0:
            return []
        distances, indices = self._nn.kneighbors(query_vector, n_neighbors=k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            similarity = 1.0 - float(dist)
            results.append(SearchResult(index=int(idx), similarity=similarity, metadata=self._metadata[idx]))
        return results

    def __len__(self) -> int:
        return len(self._metadata)
