import json

from support_search.experiment import (
    ExperimentConfig,
    build_engine_bundle,
    evaluate_config,
    score_query,
)
from support_search.models import Document

DOCS = [
    Document("wifi", "Fix wireless connection drops", "Restart the router and reset network settings", "network", "router"),
    Document("password", "Reset a forgotten account password", "Use the account recovery email link", "account", "portal"),
    Document("battery", "Laptop battery drains quickly", "Check battery health and firmware updates", "hardware", "laptop"),
]

QUERIES = [
    {"query": "wifi keeps disconnecting", "relevance": {"wifi": 3}, "category": "network"},
    {"query": "forgot my password", "relevance": {"password": 3}, "category": "account"},
]


def fast_config(**overrides) -> ExperimentConfig:
    base = {"id": "test-config", "embedding_model": "hashing", "retrieval_features": {"bm25": 0.5, "dense": 0.5}, "top_k": 2, "split": "dev"}
    base.update(overrides)
    return ExperimentConfig(**base)


def test_config_round_trips_through_dict():
    config = fast_config(candidate_k=10, rerank_depth=0)
    restored = ExperimentConfig.from_dict(config.to_dict())
    assert restored == config


def test_config_loads_from_json_file(tmp_path):
    config = fast_config()
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config.to_dict()))
    loaded = ExperimentConfig.load(path)
    assert loaded.id == "test-config"
    assert loaded.retrieval_features == {"bm25": 0.5, "dense": 0.5}


def test_score_query_ranks_the_relevant_document_first():
    config = fast_config()
    bundle = build_engine_bundle(DOCS, config, cache=False)
    doc_ids, _, latency = score_query(config, bundle, "wifi keeps disconnecting")
    assert doc_ids[0] == "wifi"
    assert latency["total_ms"] >= 0


def test_score_query_is_deterministic():
    config = fast_config()
    bundle = build_engine_bundle(DOCS, config, cache=False)
    first, _, _ = score_query(config, bundle, "wifi keeps disconnecting")
    second, _, _ = score_query(config, bundle, "wifi keeps disconnecting")
    assert first == second


def test_bm25_only_weight_ignores_dense_component():
    config = fast_config(retrieval_features={"bm25": 1.0, "dense": 0.0})
    result = evaluate_config(config, DOCS, QUERIES)
    assert result["summary"]["query_count"] == 2
    assert result["summary"]["ndcg_at_k"] > 0


def test_evaluate_config_reports_failed_queries_when_nothing_relevant_returned():
    unsolvable_queries = [{"query": "completely unrelated nonsense zzz", "relevance": {"nonexistent-doc": 3}, "category": "x"}]
    config = fast_config(top_k=2)
    result = evaluate_config(config, DOCS, unsolvable_queries)
    assert result["summary"]["failed_query_rate"] == 1.0
    assert result["rows"][0]["failed"] is True


def test_learned_linear_without_model_path_fails_every_query_cleanly():
    # evaluate_config treats a per-query scoring exception as a failed query
    # (Section 25: failed-query rate is a reliability metric), not a crash --
    # so a misconfigured learned_linear fusion surfaces as error_rate=1.0,
    # never as an unhandled exception escaping evaluate_config.
    config = fast_config(fusion="learned_linear", learned_model_path=None)
    result = evaluate_config(config, DOCS, QUERIES)
    assert result["summary"]["error_rate"] == 1.0
    assert all(row["failed"] for row in result["rows"])
