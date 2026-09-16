"""
Human-vs-LLM-judge agreement statistics.

Implements the specific metrics the assignment asks for: Spearman
correlation, weighted Cohen's kappa, and exact/adjacent agreement rate,
computed per judge dimension (not just on the "overall" score, since a
judge can be reliable on one dimension and unreliable on another -- the
assignment explicitly wants that breakdown, not a single number).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import spearmanr


@dataclass
class AgreementResult:
    dimension: str
    n: int
    spearman_r: float | None
    spearman_p: float | None
    weighted_kappa: float | None
    exact_agreement_rate: float
    adjacent_agreement_rate: float  # within +/-1 point


def _weighted_kappa(y1: list[int], y2: list[int], max_score: int = 4) -> float | None:
    """Quadratic-weighted Cohen's kappa, implemented directly (not assuming
    a specific sklearn version's `cohen_kappa_score(weights=...)` signature
    is available), so this module has no fragile version dependency."""
    n = len(y1)
    if n == 0:
        return None
    categories = list(range(max_score + 1))
    k = len(categories)

    observed = np.zeros((k, k))
    for a, b in zip(y1, y2):
        observed[a, b] += 1
    observed /= n

    hist1 = observed.sum(axis=1)
    hist2 = observed.sum(axis=0)
    expected = np.outer(hist1, hist2)

    weights = np.zeros((k, k))
    for i in categories:
        for j in categories:
            weights[i, j] = ((i - j) ** 2) / ((k - 1) ** 2)

    numerator = (weights * observed).sum()
    denominator = (weights * expected).sum()
    if denominator == 0:
        return None
    return round(1 - (numerator / denominator), 4)


def compute_agreement(human_scores: list[int], judge_scores: list[int], dimension: str,
                       max_score: int = 4) -> AgreementResult:
    n = len(human_scores)
    assert n == len(judge_scores), "human/judge score length mismatch"

    if n < 2:
        return AgreementResult(dimension, n, None, None, None, 0.0, 0.0)

    exact = sum(1 for h, j in zip(human_scores, judge_scores) if h == j) / n
    adjacent = sum(1 for h, j in zip(human_scores, judge_scores) if abs(h - j) <= 1) / n

    try:
        r, p = spearmanr(human_scores, judge_scores)
        if np.isnan(r):
            r, p = None, None
    except Exception:
        r, p = None, None

    kappa = _weighted_kappa(human_scores, judge_scores, max_score=max_score)

    return AgreementResult(
        dimension=dimension, n=n,
        spearman_r=round(float(r), 4) if r is not None else None,
        spearman_p=round(float(p), 4) if p is not None else None,
        weighted_kappa=kappa,
        exact_agreement_rate=round(exact, 4),
        adjacent_agreement_rate=round(adjacent, 4),
    )
