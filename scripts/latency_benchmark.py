"""Section 24: production-relevant latency benchmark, explicit protocol.

Timing boundary (measured inside experiment.score_query, per request):
  START -> BM25.scores() + DenseIndex.scores() (first-stage retrieval)
        -> feature extraction + fusion (candidate ranking)
        -> [optional] cross-encoder inference over rerank_depth candidates
  END   -> ranked doc-id list ready to serialize

Explicitly EXCLUDED: model loading/download (measured once, separately, and
reported as "cold start", never mixed into the per-request P50/P95), HTTP
serialization, network transport (no FastAPI/network round trip is in this
benchmark — this measures the in-process ranking pipeline exactly like the
original repository's README latency numbers did, not a live HTTP load
test; `loadtest/locustfile.py` covers HTTP-level load separately and is
unchanged by this project).

Protocol:
  1. Build the engine bundle once (this is the model-loading step; timed
     and reported separately as cold-start, excluded from P50/P95).
  2. Warm up with 15 queries (discarded).
  3. Cycle through the fixed 30-query TEST set until >= 300 measured
     requests have been collected per configuration (single-process,
     concurrency=1 — this is a per-request latency benchmark, not a
     throughput/concurrency load test).
  4. Report P50, P95, P99 (n>=300 supports a P99 estimate), and mean as a
     supporting-only statistic, per configuration, split by stage.
"""
from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import numpy as np

from support_search import __name__ as _pkg  # noqa: F401 - ensures package import works
from support_search.experiment import ExperimentConfig, build_engine_bundle, load_corpus, queries_for_split, score_query

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "experiments"
OUT_DIR = ROOT / "results" / "latency"
CONFIGS = ["bm25_only", "hybrid_weighted", "cross_encoder_rerank", "final_frozen"]
TARGET_SAMPLES = 300
WARMUP = 15


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(round((len(ordered) - 1) * p), len(ordered) - 1)
    return ordered[idx]


def software_versions() -> dict:
    versions = {"python": platform.python_version()}
    for module_name in ("torch", "sentence_transformers", "numpy", "faiss"):
        try:
            mod = __import__(module_name)
            versions[module_name] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[module_name] = "not installed"
    return versions


def main() -> None:
    documents = load_corpus()
    test_queries = queries_for_split("test")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    hardware = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "cpu_count_logical": __import__("os").cpu_count(),
    }
    software = software_versions()

    all_reports = {}
    for config_id in CONFIGS:
        config = ExperimentConfig.load(CONFIG_DIR / f"{config_id}.json")
        config.split = "test"

        cold_start = time.perf_counter()
        bundle = build_engine_bundle(documents, config, cache=False)
        cold_start_s = time.perf_counter() - cold_start

        for i in range(WARMUP):
            score_query(config, bundle, test_queries[i % len(test_queries)]["query"])

        totals, retrievals, features, reranks = [], [], [], []
        i = 0
        while len(totals) < TARGET_SAMPLES:
            item = test_queries[i % len(test_queries)]
            _, _, latency = score_query(config, bundle, item["query"])
            totals.append(latency["total_ms"])
            retrievals.append(latency["retrieval_ms"])
            features.append(latency["feature_fusion_ms"])
            reranks.append(latency["rerank_ms"])
            i += 1

        report = {
            "config_id": config_id,
            "cold_start_s": round(cold_start_s, 4),
            "sample_count": len(totals),
            "warmup_requests": WARMUP,
            "concurrency": 1,
            "hardware": hardware,
            "software": software,
            "total_ms": {
                "p50": round(percentile(totals, 0.5), 4),
                "p95": round(percentile(totals, 0.95), 4),
                "p99": round(percentile(totals, 0.99), 4),
                "mean": round(float(np.mean(totals)), 4),
            },
            "retrieval_ms": {"p50": round(percentile(retrievals, 0.5), 4), "p95": round(percentile(retrievals, 0.95), 4)},
            "feature_fusion_ms": {"p50": round(percentile(features, 0.5), 4), "p95": round(percentile(features, 0.95), 4)},
            "rerank_ms": {"p50": round(percentile(reranks, 0.5), 4), "p95": round(percentile(reranks, 0.95), 4)},
        }
        all_reports[config_id] = report
        (OUT_DIR / f"{config_id}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(
            f"{config_id:22s} cold_start={report['cold_start_s']:.3f}s  n={report['sample_count']:4d}  "
            f"P50={report['total_ms']['p50']:7.3f}ms  P95={report['total_ms']['p95']:7.3f}ms  P99={report['total_ms']['p99']:7.3f}ms"
        )

    (OUT_DIR / "all_latency_results.json").write_text(json.dumps(all_reports, indent=2), encoding="utf-8")
    print(f"\nwrote -> {OUT_DIR}")


if __name__ == "__main__":
    main()
