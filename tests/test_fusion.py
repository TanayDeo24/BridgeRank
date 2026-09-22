import numpy as np
import pytest

from support_search.fusion import learned_linear, normalize, reciprocal_rank_fusion, weighted_sum


def test_minmax_normalize_bounds():
    values = np.array([1.0, 5.0, 10.0])
    normed = normalize(values, "minmax")
    assert normed.min() == pytest.approx(0.0)
    assert normed.max() == pytest.approx(1.0)


def test_minmax_normalize_constant_input_is_zero_not_nan():
    values = np.array([4.0, 4.0, 4.0])
    normed = normalize(values, "minmax")
    assert np.all(normed == 0.0)


def test_zscore_normalize_mean_zero():
    values = np.array([2.0, 4.0, 6.0])
    normed = normalize(values, "zscore")
    assert normed.mean() == pytest.approx(0.0, abs=1e-6)


def test_normalize_rejects_unknown_method():
    with pytest.raises(ValueError):
        normalize(np.array([1.0, 2.0]), "bogus")


def test_weighted_sum_respects_weights():
    matrix = {"a": np.array([1.0, 0.0]), "b": np.array([0.0, 1.0])}
    fused = weighted_sum(matrix, {"a": 1.0, "b": 0.0}, normalization="none")
    assert list(fused) == [1.0, 0.0]


def test_weighted_sum_ignores_zero_weight_features():
    matrix = {"a": np.array([1.0, 2.0]), "b": np.array([100.0, 100.0])}
    fused = weighted_sum(matrix, {"a": 1.0, "b": 0.0}, normalization="none")
    assert list(fused) == [1.0, 2.0]


def test_reciprocal_rank_fusion_prefers_consistently_high_ranked_items():
    matrix = {"a": np.array([3.0, 2.0, 1.0]), "b": np.array([3.0, 1.0, 2.0])}
    fused = reciprocal_rank_fusion(matrix, k=60)
    # item 0 is rank-1 in both components and must score highest
    assert np.argmax(fused) == 0


def test_learned_linear_applies_coefficients_and_bias():
    matrix = {"a": np.array([1.0, 0.0])}
    fused = learned_linear(matrix, {"a": 2.0}, intercept=1.0, normalization="none")
    assert list(fused) == [3.0, 1.0]
