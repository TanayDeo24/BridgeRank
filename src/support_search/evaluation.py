from __future__ import annotations

import math
import statistics
import time
from collections import defaultdict
from typing import Iterable

from .retrieval import HybridSearchEngine


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    return sum(doc in relevant for doc in retrieved[:k]) / k


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    return sum(doc in relevant for doc in retrieved[:k]) / max(len(relevant), 1)


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    return next((1 / rank for rank, doc in enumerate(retrieved, 1) if doc in relevant), 0.0)


def ndcg_at_k(retrieved: list[str], relevance: dict[str, int], k: int) -> float:
    dcg = sum(
        (2 ** relevance.get(doc, 0) - 1) / math.log2(i + 2) for i, doc in enumerate(retrieved[:k])
    )
    ideal = sorted(relevance.values(), reverse=True)[:k]
    idcg = sum((2**rel - 1) / math.log2(i + 2) for i, rel in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(round((len(ordered) - 1) * p), len(ordered) - 1)]


def evaluate(
    engine: HybridSearchEngine, queries: Iterable[dict], k: int = 5, mode: str = "hybrid"
) -> dict:
    rows, latencies, categories = [], [], defaultdict(list)
    for item in queries:
        started = time.perf_counter()
        results = engine.search(item["query"], top_k=k, mode=mode)
        latencies.append((time.perf_counter() - started) * 1000)
        ids = [r.document.id for r in results]
        relevance = item["relevance"]
        relevant = {doc_id for doc_id, grade in relevance.items() if grade > 0}
        row = {
            "precision_at_k": precision_at_k(ids, relevant, k),
            "recall_at_k": recall_at_k(ids, relevant, k),
            "mrr": reciprocal_rank(ids, relevant),
            "ndcg_at_k": ndcg_at_k(ids, relevance, k),
            "failed": not bool(relevant.intersection(ids)),
        }
        rows.append(row)
        categories[item.get("category", "unknown")].append(row)
    summary = {
        key: statistics.fmean(r[key] for r in rows)
        for key in ("precision_at_k", "recall_at_k", "mrr", "ndcg_at_k")
    }
    summary.update(
        {
            "p50_latency_ms": percentile(latencies, 0.5),
            "p95_latency_ms": percentile(latencies, 0.95),
            "failed_query_rate": statistics.fmean(float(r["failed"]) for r in rows),
            "query_count": len(rows),
            "by_category": {
                cat: {"mrr": statistics.fmean(r["mrr"] for r in vals), "count": len(vals)}
                for cat, vals in categories.items()
            },
        }
    )
    return summary
