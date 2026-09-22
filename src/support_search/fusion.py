"""Score normalization and fusion strategies (Gate 4).

Three fusion strategies are implemented deliberately, not exhaustively
(Section 4 forbids blindly implementing every possible method):

- ``weighted_sum``: per-query min-max (or z-score) normalized weighted
  linear combination. This is the strategy the original repository already
  used (with hardcoded alpha=0.55, min-max only) — kept and generalized to
  arbitrary feature sets and an explicit normalization choice.
- ``reciprocal_rank_fusion``: rank-based, does not require score
  normalization at all (scores from different models are not commensurable;
  ranks always are). Standard choice to compare against weighted-sum when
  score distributions are suspected to be incomparable across components.
- ``learned_linear``: a linear combination whose weights come from a fitted
  model (``ranking_model.py``) instead of a hand-picked constant. This is
  the "machine-learning-based ranking" fusion option — deliberately still a
  linear model (whiteboard-explainable), not a black box, per Section 14's
  instruction not to overengineer.

All three are compared on the *validation* split before one is frozen
(Section 18); see ``docs/BENCHMARK_REPORT.md`` for the outcome.
"""
from __future__ import annotations

import numpy as np


def normalize(values: np.ndarray, method: str = "minmax") -> np.ndarray:
    if len(values) == 0:
        return values
    if method == "none":
        return values.astype(np.float32)
    if method == "minmax":
        lo, hi = float(values.min()), float(values.max())
        if hi - lo < 1e-12:
            return np.zeros_like(values, dtype=np.float32)
        return ((values - lo) / (hi - lo)).astype(np.float32)
    if method == "zscore":
        mean, std = float(values.mean()), float(values.std())
        if std < 1e-12:
            return np.zeros_like(values, dtype=np.float32)
        return ((values - mean) / std).astype(np.float32)
    raise ValueError(f"Unknown normalization method: {method}")


def weighted_sum(
    feature_matrix: dict[str, np.ndarray],
    weights: dict[str, float],
    normalization: str = "minmax",
) -> np.ndarray:
    total = None
    for name, weight in weights.items():
        if name not in feature_matrix or weight == 0:
            continue
        normed = normalize(feature_matrix[name], normalization)
        contribution = weight * normed
        total = contribution if total is None else total + contribution
    if total is None:
        n = len(next(iter(feature_matrix.values())))
        return np.zeros(n, dtype=np.float32)
    return total


def reciprocal_rank_fusion(feature_matrix: dict[str, np.ndarray], k: int = 60) -> np.ndarray:
    """Standard RRF: score(doc) = sum over component rankings of 1/(k + rank)."""
    if not feature_matrix:
        raise ValueError("reciprocal_rank_fusion requires at least one feature")
    n = len(next(iter(feature_matrix.values())))
    total = np.zeros(n, dtype=np.float32)
    for values in feature_matrix.values():
        order = np.argsort(-values)
        ranks = np.empty(n, dtype=np.int64)
        ranks[order] = np.arange(1, n + 1)
        total += 1.0 / (k + ranks)
    return total


def learned_linear(
    feature_matrix: dict[str, np.ndarray],
    coefficients: dict[str, float],
    intercept: float = 0.0,
    normalization: str = "minmax",
) -> np.ndarray:
    total = None
    for name, coef in coefficients.items():
        if name not in feature_matrix:
            continue
        normed = normalize(feature_matrix[name], normalization)
        contribution = coef * normed
        total = contribution if total is None else total + contribution
    if total is None:
        n = len(next(iter(feature_matrix.values())))
        total = np.zeros(n, dtype=np.float32)
    return total + intercept


FUSION_STRATEGIES = {
    "weighted_sum": weighted_sum,
    "reciprocal_rank_fusion": reciprocal_rank_fusion,
    "learned_linear": learned_linear,
}
