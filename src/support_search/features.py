"""Diversified ranking-feature framework.

A "feature" is a small, named, independently enable/disable-able function
from (query, candidate document, already-computed raw retrieval scores) to
a single float. Features are registered once in ``FEATURE_REGISTRY`` and an
experiment config (see ``experiment.py``) selects a subset by name — adding
a new feature never requires touching the fusion or experiment-running code.

Two stages exist, matching real retrieval-system practice (Section 15):

- ``retrieval`` features are cheap enough to compute for every candidate in
  the first-stage candidate pool (BM25 score, dense score, title overlap,
  freshness, source authority).
- ``rerank`` features are expensive and are only computed for the narrower
  set of candidates that survive to the reranking stage (currently just the
  cross-encoder score).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Protocol

from .models import Document
from .text import tokenize

DEFAULT_REFERENCE_DATE = date(2026, 1, 1)


@dataclass(frozen=True)
class FeatureContext:
    query: str
    document: Document
    raw_scores: dict[str, float]
    reference_date: date = DEFAULT_REFERENCE_DATE


class Feature(Protocol):
    name: str
    stage: str  # "retrieval" | "rerank"

    def extract(self, ctx: FeatureContext) -> float: ...


class BM25Feature:
    name = "bm25"
    stage = "retrieval"

    def extract(self, ctx: FeatureContext) -> float:
        return float(ctx.raw_scores.get("bm25", 0.0))


class DenseFeature:
    name = "dense"
    stage = "retrieval"

    def extract(self, ctx: FeatureContext) -> float:
        return float(ctx.raw_scores.get("dense", 0.0))


class TitleOverlapFeature:
    """Fraction of query tokens that also appear in the document title.

    Rationale: cheap, interpretable, query-and-document dependent lexical
    signal that is *not* the same computation as BM25 (no IDF weighting, no
    length normalization, title-only rather than whole document) — a
    genuinely different feature family, not a restatement of BM25.
    """

    name = "title_overlap"
    stage = "retrieval"

    def extract(self, ctx: FeatureContext) -> float:
        q_tokens = set(tokenize(ctx.query))
        if not q_tokens:
            return 0.0
        title_tokens = set(tokenize(ctx.document.title))
        return len(q_tokens & title_tokens) / len(q_tokens)


class FreshnessFeature:
    """Exponential recency decay from ``metadata.last_updated``.

    Rationale: a support KB article that was reviewed/updated recently is
    more likely to reflect the current product than a stale one; this is a
    static, document-only, always-available signal with a clear real-world
    justification (Section 13 candidate A). Half-life chosen at ~180 days —
    an article updated 6 months before the reference date scores 0.5, one
    updated a year before scores 0.25, one from today scores 1.0. This is a
    document property, never derived from query text or any user behavior.
    """

    name = "freshness"
    stage = "retrieval"
    half_life_days = 180.0

    def extract(self, ctx: FeatureContext) -> float:
        raw = ctx.document.metadata.get("last_updated")
        if not raw:
            return 0.0
        try:
            updated = date.fromisoformat(str(raw))
        except ValueError:
            return 0.0
        age_days = max((ctx.reference_date - updated).days, 0)
        return math.exp(-age_days * math.log(2) / self.half_life_days)


class AuthorityFeature:
    """Maps the document's disclosed ``source_authority`` metadata to a score.

    Rationale: an officially-authored support procedure is a priori more
    trustworthy than a community-sourced tip for the same symptom; this is
    static metadata already present on every document, not a popularity or
    click signal (Section 13 candidate B / metadata-authority family).
    """

    name = "source_authority"
    stage = "retrieval"
    weights = {"official": 1.0, "community": 0.5}

    def extract(self, ctx: FeatureContext) -> float:
        value = ctx.document.metadata.get("source_authority")
        return self.weights.get(str(value), 0.5)


class CrossEncoderFeature:
    """Cross-encoder relevance score. Rerank-stage: expensive, only computed
    for the narrower reranking candidate set, never for the full corpus."""

    name = "cross_encoder"
    stage = "rerank"

    def extract(self, ctx: FeatureContext) -> float:
        return float(ctx.raw_scores.get("cross_encoder", 0.0))


FEATURE_REGISTRY: dict[str, Feature] = {
    f.name: f
    for f in (
        BM25Feature(),
        DenseFeature(),
        TitleOverlapFeature(),
        FreshnessFeature(),
        AuthorityFeature(),
        CrossEncoderFeature(),
    )
}


def retrieval_stage_features() -> list[Feature]:
    return [f for f in FEATURE_REGISTRY.values() if f.stage == "retrieval"]


def rerank_stage_features() -> list[Feature]:
    return [f for f in FEATURE_REGISTRY.values() if f.stage == "rerank"]
