import pytest

from support_search.evaluation import ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank


def test_ranking_metrics():
    retrieved = ["b", "a", "c"]
    relevant = {"a", "d"}
    assert precision_at_k(retrieved, relevant, 2) == 0.5
    assert recall_at_k(retrieved, relevant, 2) == 0.5
    assert reciprocal_rank(retrieved, relevant) == 0.5
    assert ndcg_at_k(retrieved, {"a": 2, "d": 1}, 3) == pytest.approx(0.521296)
