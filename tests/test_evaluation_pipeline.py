import argparse
import json

import pytest

from support_search.cli import run_quality_gate
from support_search.embeddings import HashingEmbedder
from support_search.evaluation import evaluate, percentile
from support_search.failure_analysis import create_summary
from support_search.models import Document
from support_search.retrieval import HybridSearchEngine
from support_search.text import chunks


def test_evaluate_returns_quality_latency_and_slices():
    documents = [
        Document("vpn", "VPN DNS", "Resolve internal sites over a connected VPN", "network"),
        Document("email", "Delayed email", "Inspect message trace and mailbox quota", "messaging"),
    ]
    engine = HybridSearchEngine(documents, HashingEmbedder(64))
    metrics = evaluate(
        engine,
        [{"query": "VPN internal DNS", "relevance": {"vpn": 3}, "category": "network"}],
        k=1,
    )
    assert metrics["mrr"] == 1
    assert metrics["ndcg_at_k"] == 1
    assert metrics["failed_query_rate"] == 0
    assert metrics["by_category"]["network"]["count"] == 1
    assert metrics["p95_latency_ms"] >= 0


def test_percentile_and_chunk_validation():
    assert percentile([], 0.95) == 0
    assert percentile([1, 2, 3], 0.5) == 2
    assert chunks("one two three four", size=3, overlap=1) == ["one two three", "three four"]
    assert chunks("", size=3, overlap=0) == []
    with pytest.raises(ValueError):
        chunks("text", size=3, overlap=3)


def test_failure_report_without_api_key(monkeypatch, tmp_path):
    source = tmp_path / "evaluation.json"
    output = tmp_path / "report.md"
    source.write_text(json.dumps({"hybrid": {"mrr": 0.7}}), encoding="utf-8")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    create_summary(source, output)
    report = output.read_text(encoding="utf-8")
    assert report.startswith("# Claude evaluation prompt")
    assert '"mrr": 0.7' in report


def test_quality_gate_passes_and_fails(tmp_path):
    source = tmp_path / "evaluation.json"
    source.write_text(
        json.dumps(
            {
                "hybrid": {
                    "ndcg_at_k": 0.8,
                    "mrr": 0.75,
                    "p95_latency_ms": 20,
                    "failed_query_rate": 0.1,
                }
            }
        ),
        encoding="utf-8",
    )
    args = argparse.Namespace(
        input=source,
        mode="hybrid",
        min_ndcg=0.7,
        min_mrr=0.7,
        max_p95_ms=100,
        max_failed_rate=0.2,
    )
    run_quality_gate(args)
    args.min_ndcg = 0.9
    with pytest.raises(SystemExit):
        run_quality_gate(args)
