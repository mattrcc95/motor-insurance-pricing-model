"""One-way (univariate) analysis helpers for exploratory data analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd

Z95 = 1.959964


def one_way_frequency(df: pd.DataFrame, by: pd.Series | str) -> pd.DataFrame:
    """Observed claim frequency per level of a factor, with a 95% confidence interval.

    frequency = sum(ClaimNb) / sum(Exposure). Under a Poisson assumption the number
    of claims N in a group has Var(N) = E[N], so the standard error of the
    frequency N/E is sqrt(N)/E (normal approximation).
    """
    key = df[by] if isinstance(by, str) else by
    g = df.groupby(key, observed=True).agg(exposure=("Exposure", "sum"), claims=("ClaimNb", "sum"),
                                           policies=("Exposure", "size"))
    g["frequency"] = g["claims"] / g["exposure"]
    se = np.sqrt(g["claims"]) / g["exposure"]
    g["ci_low"] = (g["frequency"] - Z95 * se).clip(lower=0)
    g["ci_high"] = g["frequency"] + Z95 * se
    g["exposure_share"] = g["exposure"] / g["exposure"].sum()
    g["relative_to_portfolio"] = g["frequency"] / (df["ClaimNb"].sum() / df["Exposure"].sum())
    return g


def one_way_severity(df: pd.DataFrame, by: pd.Series | str, amount: str = "ClaimAmountCapped") -> pd.DataFrame:
    """Average cost per claim per level of a factor, with a 95% confidence interval.

    avg = sum(amount) / sum(ClaimNb) over policies with claims. The standard error
    uses the claim-weighted standard deviation of the per-policy average cost
    divided by sqrt(number of claims).
    """
    d = df[df["ClaimNb"] > 0]
    key = d[by] if isinstance(by, str) else by[d.index]
    avg = d[amount] / d["ClaimNb"]
    tmp = pd.DataFrame({"key": key, "n": d["ClaimNb"], "amt": d[amount], "avg": avg})
    rows = {}
    for k, grp in tmp.groupby("key", observed=True):
        n = grp["n"].sum()
        mean = grp["amt"].sum() / n
        var = np.average((grp["avg"] - mean) ** 2, weights=grp["n"])
        se = np.sqrt(var / n)
        rows[k] = {"claims": n, "avg_cost": mean, "ci_low": max(mean - Z95 * se, 0), "ci_high": mean + Z95 * se}
    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index.name = key.name if hasattr(key, "name") else None
    out["relative_to_portfolio"] = out["avg_cost"] / (d[amount].sum() / d["ClaimNb"].sum())
    return out


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Cramér's V association between two categorical variables (0 = none, 1 = perfect)."""
    from scipy.stats import chi2_contingency

    table = pd.crosstab(x, y)
    chi2 = chi2_contingency(table, correction=False)[0]
    n = table.to_numpy().sum()
    r, k = table.shape
    return float(np.sqrt(chi2 / (n * (min(r, k) - 1))))
