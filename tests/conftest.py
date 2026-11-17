"""Synthetic freMTPL2-like data so the test-suite runs offline and fast."""

import numpy as np
import pandas as pd
import pytest


def make_raw(n: int = 4000, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    freq = pd.DataFrame({
        "IDpol": np.arange(1, n + 1, dtype=float),
        "Exposure": rng.uniform(0.05, 1.2, n).round(2),
        "Area": pd.Categorical(rng.choice(list("ABCDEF"), n)),
        "VehPower": rng.integers(4, 16, n),
        "VehAge": rng.integers(0, 25, n),
        "DrivAge": rng.integers(18, 90, n),
        "BonusMalus": rng.integers(50, 200, n),
        "VehBrand": pd.Categorical(rng.choice(["B1", "B2", "B12"], n)),
        "VehGas": rng.choice(["'Regular'", "'Diesel'"], n),
        "Density": rng.integers(1, 20000, n),
        "Region": pd.Categorical(rng.choice(["R11", "R24", "R82"], n)),
    })
    lam = 0.08 * np.exp(0.01 * (freq["BonusMalus"] - 50)) * freq["Exposure"].clip(upper=1)
    true_n = rng.poisson(lam)
    freq["ClaimNb"] = true_n
    # Inject an inconsistency: a policy with claims but no severity record
    freq.loc[0, "ClaimNb"] = 2
    rows = [(pid, rng.gamma(2.0, 1000.0)) for pid, k in zip(freq["IDpol"].astype(int)[1:], true_n[1:]) for _ in range(k)]
    sev = pd.DataFrame(rows, columns=["IDpol", "ClaimAmount"])
    # A huge claim to test capping and a severity row with unknown policy
    sev.loc[len(sev)] = [int(freq.loc[5, "IDpol"]), 1_000_000.0]
    sev.loc[len(sev)] = [99_999_999, 500.0]
    sev["IDpol"] = sev["IDpol"].astype("int64")
    freq = freq[["IDpol", "ClaimNb", "Exposure", "Area", "VehPower", "VehAge", "DrivAge",
                 "BonusMalus", "VehBrand", "VehGas", "Density", "Region"]]
    return freq, sev


@pytest.fixture(scope="session")
def raw():
    return make_raw()


@pytest.fixture(scope="session")
def clean(raw):
    from motor_pricing.data import add_targets, build_modelling_table

    df, _ = build_modelling_table(*raw, large_claim_cap=50_000)
    return add_targets(df)
