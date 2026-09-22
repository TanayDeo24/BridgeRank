from datetime import date

from support_search.features import (
    FEATURE_REGISTRY,
    AuthorityFeature,
    BM25Feature,
    FeatureContext,
    FreshnessFeature,
    TitleOverlapFeature,
    rerank_stage_features,
    retrieval_stage_features,
)
from support_search.models import Document


def test_registry_contains_expected_features():
    assert set(FEATURE_REGISTRY) == {
        "bm25", "dense", "title_overlap", "freshness", "source_authority", "cross_encoder",
    }
    assert {f.name for f in retrieval_stage_features()} == {
        "bm25", "dense", "title_overlap", "freshness", "source_authority",
    }
    assert {f.name for f in rerank_stage_features()} == {"cross_encoder"}


def test_title_overlap_is_fraction_of_query_tokens():
    doc = Document("d1", "Reset a forgotten account password", "body text", "account")
    ctx = FeatureContext(query="forgotten account password reset", document=doc, raw_scores={})
    value = TitleOverlapFeature().extract(ctx)
    assert 0.0 < value <= 1.0
    # every query token appears in the title -> full overlap
    assert value == 1.0


def test_title_overlap_zero_for_disjoint_vocabulary():
    doc = Document("d1", "Printer troubleshooting", "toner jam", "hardware")
    ctx = FeatureContext(query="wireless signal drops constantly", document=doc, raw_scores={})
    assert TitleOverlapFeature().extract(ctx) == 0.0


def test_freshness_decays_with_age_and_is_bounded():
    doc_recent = Document("d1", "t", "b", metadata={"last_updated": "2026-01-01"})
    doc_old = Document("d2", "t", "b", metadata={"last_updated": "2024-01-01"})
    doc_missing = Document("d3", "t", "b", metadata={})
    feature = FreshnessFeature()
    ref = date(2026, 1, 1)
    recent = feature.extract(FeatureContext(query="q", document=doc_recent, raw_scores={}, reference_date=ref))
    old = feature.extract(FeatureContext(query="q", document=doc_old, raw_scores={}, reference_date=ref))
    missing = feature.extract(FeatureContext(query="q", document=doc_missing, raw_scores={}, reference_date=ref))
    assert recent == 1.0
    assert 0.0 <= old < recent
    assert missing == 0.0


def test_authority_maps_known_and_unknown_values():
    feature = AuthorityFeature()
    official = Document("d1", "t", "b", metadata={"source_authority": "official"})
    community = Document("d2", "t", "b", metadata={"source_authority": "community"})
    unknown = Document("d3", "t", "b", metadata={})
    assert feature.extract(FeatureContext(query="q", document=official, raw_scores={})) == 1.0
    assert feature.extract(FeatureContext(query="q", document=community, raw_scores={})) == 0.5
    assert feature.extract(FeatureContext(query="q", document=unknown, raw_scores={})) == 0.5


def test_bm25_feature_reads_raw_scores():
    doc = Document("d1", "t", "b")
    ctx = FeatureContext(query="q", document=doc, raw_scores={"bm25": 3.2})
    assert BM25Feature().extract(ctx) == 3.2


def test_features_are_deterministic():
    doc = Document("d1", "Reset a forgotten account password", "body", metadata={"last_updated": "2025-06-01", "source_authority": "official"})
    ctx = FeatureContext(query="forgot password", document=doc, raw_scores={"bm25": 1.0, "dense": 0.5})
    for feature in FEATURE_REGISTRY.values():
        if feature.stage == "rerank":
            continue
        first = feature.extract(ctx)
        second = feature.extract(ctx)
        assert first == second
