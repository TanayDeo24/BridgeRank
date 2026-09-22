from support_search.ranking_model import fit_logistic_rank_model


def test_logistic_model_learns_separable_signal_with_correct_sign():
    # feature "a" perfectly predicts the label; "b" is pure noise.
    rows = [
        {"a": 1.0, "b": 0.1}, {"a": 1.0, "b": 0.9}, {"a": 1.0, "b": 0.5},
        {"a": 0.0, "b": 0.2}, {"a": 0.0, "b": 0.8}, {"a": 0.0, "b": 0.4},
    ]
    labels = [1, 1, 1, 0, 0, 0]
    model = fit_logistic_rank_model(rows, labels, ["a", "b"], iterations=500, seed=1)
    coeffs = model.coefficients()
    assert coeffs["a"] > 0
    assert model.train_rows == 6
    assert model.train_positive_rate == 0.5


def test_model_save_and_load_round_trip(tmp_path):
    rows = [{"a": 1.0}, {"a": 0.0}]
    model = fit_logistic_rank_model(rows, [1, 0], ["a"], iterations=50, seed=7)
    path = tmp_path / "model.json"
    model.save(path)
    from support_search.ranking_model import LogisticRankModel

    reloaded = LogisticRankModel.load(path)
    assert reloaded.coefficients() == model.coefficients()
    assert reloaded.seed == 7
