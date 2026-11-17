import numpy as np
import pandas as pd
import pytest

from motor_pricing.eda import cramers_v, one_way_frequency, one_way_severity


def test_one_way_frequency_totals(clean):
    t = one_way_frequency(clean, "Area")
    assert t["exposure"].sum() == pytest.approx(clean["Exposure"].sum())
    assert t["claims"].sum() == clean["ClaimNb"].sum()
    assert (t["ci_low"] <= t["frequency"]).all() and (t["frequency"] <= t["ci_high"]).all()
    overall = clean["ClaimNb"].sum() / clean["Exposure"].sum()
    assert np.average(t["relative_to_portfolio"], weights=t["exposure"]) == pytest.approx(1.0)
    assert np.average(t["frequency"], weights=t["exposure"]) == pytest.approx(overall)


def test_one_way_severity_totals(clean):
    t = one_way_severity(clean, "VehGas")
    claims = clean.loc[clean["ClaimNb"] > 0]
    assert t["claims"].sum() == claims["ClaimNb"].sum()
    overall = claims["ClaimAmountCapped"].sum() / claims["ClaimNb"].sum()
    assert np.average(t["avg_cost"], weights=t["claims"]) == pytest.approx(overall)


def test_cramers_v_bounds():
    a = pd.Series(list("aabbccaabbcc"))
    assert cramers_v(a, a) == pytest.approx(1.0)
    rng = np.random.default_rng(0)
    x = pd.Series(rng.choice(list("ab"), 5000))
    y = pd.Series(rng.choice(list("cd"), 5000))
    assert cramers_v(x, y) < 0.05
