"""A minimal, dependency-free logistic-regression ranking model.

This is the project's "machine-learning-based ranking model": given the same
diversified feature set the rest of the system uses (bm25, dense,
title_overlap, freshness, source_authority — and optionally cross_encoder),
it learns a linear weight per feature by gradient descent on binary
relevance labels (grade > 0) from the **dev** split only, then those
learned weights are used as the ``learned_linear`` fusion strategy and
evaluated on validation/test like every other configuration.

Deliberately not scikit-learn: the model is a handful of lines, easy to
explain on a whiteboard (Section 14), and adding a heavy ML-tuning
dependency for a linear model with <10 features would be disproportionate
(Section 18's "do not introduce complex tuning libraries unnecessarily").
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .fusion import normalize


@dataclass
class LogisticRankModel:
    feature_names: list[str]
    weights: np.ndarray
    bias: float
    normalization: str = "minmax"
    train_rows: int = 0
    train_positive_rate: float = 0.0
    iterations: int = 0
    learning_rate: float = 0.0
    l2: float = 0.0
    seed: int = 0

    def coefficients(self) -> dict[str, float]:
        return {name: float(w) for name, w in zip(self.feature_names, self.weights)}

    def to_dict(self) -> dict:
        return {
            "feature_names": self.feature_names,
            "weights": self.weights.tolist(),
            "bias": self.bias,
            "normalization": self.normalization,
            "train_rows": self.train_rows,
            "train_positive_rate": self.train_positive_rate,
            "iterations": self.iterations,
            "learning_rate": self.learning_rate,
            "l2": self.l2,
            "seed": self.seed,
        }

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> LogisticRankModel:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            feature_names=raw["feature_names"],
            weights=np.array(raw["weights"], dtype=np.float64),
            bias=raw["bias"],
            normalization=raw.get("normalization", "minmax"),
            train_rows=raw.get("train_rows", 0),
            train_positive_rate=raw.get("train_positive_rate", 0.0),
            iterations=raw.get("iterations", 0),
            learning_rate=raw.get("learning_rate", 0.0),
            l2=raw.get("l2", 0.0),
            seed=raw.get("seed", 0),
        )


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def fit_logistic_rank_model(
    rows: list[dict[str, float]],
    labels: list[int],
    feature_names: list[str],
    learning_rate: float = 0.5,
    iterations: int = 2000,
    l2: float = 1e-3,
    seed: int = 42,
    normalization: str = "minmax",
) -> LogisticRankModel:
    """Batch gradient descent, per-feature min-max normalized over the
    training rows (the same normalization is then re-derived at scoring
    time per-query, matching how every other fusion strategy normalizes)."""
    rng = np.random.default_rng(seed)
    x = np.array([[row.get(name, 0.0) for name in feature_names] for row in rows], dtype=np.float64)
    for col in range(x.shape[1]):
        x[:, col] = normalize(x[:, col], normalization)
    y = np.array(labels, dtype=np.float64)

    n, d = x.shape
    weights = rng.normal(scale=0.01, size=d)
    bias = 0.0
    for _ in range(iterations):
        z = x @ weights + bias
        pred = _sigmoid(z)
        error = pred - y
        grad_w = (x.T @ error) / n + l2 * weights
        grad_b = float(error.mean())
        weights -= learning_rate * grad_w
        bias -= learning_rate * grad_b

    return LogisticRankModel(
        feature_names=feature_names,
        weights=weights,
        bias=bias,
        normalization=normalization,
        train_rows=n,
        train_positive_rate=float(y.mean()) if n else 0.0,
        iterations=iterations,
        learning_rate=learning_rate,
        l2=l2,
        seed=seed,
    )
