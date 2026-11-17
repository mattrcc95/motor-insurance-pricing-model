import numpy as np
import pytest
from sklearn.metrics import mean_gamma_deviance, mean_poisson_deviance

from motor_pricing.metrics import (
    double_lift_table, gamma_deviance, gini_index, lift_table, lorenz_curve,
    normalised_gini, poisson_deviance,
)

rng = np.random.default_rng(1)
Y = rng.poisson(0.3, 1000) / 1.0
MU = rng.uniform(0.1, 0.6, 1000)
W = rng.uniform(0.1, 1.0, 1000)


def test_poisson_deviance_matches_sklearn():
    assert poisson_deviance(Y, MU, W) == pytest.approx(mean_poisson_deviance(Y, MU, sample_weight=W))


def test_gamma_deviance_matches_sklearn():
    y = rng.gamma(2, 1000, 500)
    mu = rng.uniform(500, 3000, 500)
    w = rng.integers(1, 4, 500)
    assert gamma_deviance(y, mu, w) == pytest.approx(mean_gamma_deviance(y, mu, sample_weight=w))


def test_deviance_zero_for_perfect_prediction():
    y = np.array([0.5, 1.0, 2.0])
    assert poisson_deviance(y, y) == pytest.approx(0.0)
    assert gamma_deviance(y, y) == pytest.approx(0.0)


def test_poisson_deviance_rejects_non_positive_predictions():
    with pytest.raises(ValueError):
        poisson_deviance([1.0], [0.0])


def test_lorenz_curve_endpoints():
    x, y = lorenz_curve(Y, MU, W)
    assert x[0] == 0 and y[0] == 0
    assert x[-1] == pytest.approx(1) and y[-1] == pytest.approx(1)
    assert np.all(np.diff(x) >= 0)


def test_gini_random_vs_oracle():
    random_gini = gini_index(Y, rng.uniform(size=Y.size), W)
    oracle = gini_index(Y, Y, W)
    assert abs(random_gini) < 0.1
    assert oracle > 0.5
    assert normalised_gini(Y, Y, W) == pytest.approx(1.0)


def test_gini_of_constant_prediction_is_zero_regardless_of_row_order():
    y = np.sort(Y)  # data ordered by outcome: a tie-breaking bug would show up here
    assert gini_index(y, np.ones_like(y), np.ones_like(y)) == pytest.approx(0.0, abs=1e-12)
    assert gini_index(y[::-1], np.ones_like(y), np.ones_like(y)) == pytest.approx(0.0, abs=1e-12)


def test_gini_invariant_to_monotone_transform():
    assert gini_index(Y, MU, W) == pytest.approx(gini_index(Y, np.log(MU) * 3 + 7, W))


def test_lift_table_balanced_weights_and_mean():
    t = lift_table(Y, MU, W, n_bins=5)
    assert len(t) == 5
    assert t["weight"].sum() == pytest.approx(W.sum())
    overall = np.sum(Y * W) / np.sum(W)
    assert np.average(t["observed"], weights=t["weight"]) == pytest.approx(overall)
    assert t["predicted"].is_monotonic_increasing


def test_double_lift_shape():
    t = double_lift_table(Y, MU, MU * rng.uniform(0.8, 1.2, MU.size), W, n_bins=4)
    assert list(t.columns) == ["weight", "observed", "model_a", "model_b"]
    assert len(t) == 4
