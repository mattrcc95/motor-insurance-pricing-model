"""Model-agnostic explanation helpers (partial dependence and ICE).

They only use the common ``model.predict(df)`` interface, so the very same
computation applies to the GLM and to the GBM and their curves are directly
comparable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def ice_curves(model, df: pd.DataFrame, feature: str, grid, n_samples: int | None = 1000,
               random_state: int = 0) -> pd.DataFrame:
    """Individual Conditional Expectation curves.

    For each sampled policy, the prediction is recomputed while ``feature`` is
    forced to every value of ``grid`` (all other characteristics unchanged).
    Returns a DataFrame (rows = policies, columns = grid values).
    """
    d = df if n_samples is None or n_samples >= len(df) else df.sample(n_samples, random_state=random_state)
    out = np.empty((len(d), len(grid)))
    for j, v in enumerate(grid):
        tmp = d.copy()
        if isinstance(d[feature].dtype, pd.CategoricalDtype):
            tmp[feature] = pd.Categorical([v] * len(d), categories=d[feature].cat.categories)
        else:
            tmp[feature] = v
        out[:, j] = model.predict(tmp)
    return pd.DataFrame(out, index=d.index, columns=list(grid))


def partial_dependence(model, df: pd.DataFrame, feature: str, grid, n_samples: int | None = 5000,
                       random_state: int = 0, weight: str | None = "Exposure") -> pd.Series:
    """Partial dependence: (exposure-weighted) average of the ICE curves.

    PD(v) = average over policies of f(x_i with feature := v).
    """
    ice = ice_curves(model, df, feature, grid, n_samples, random_state)
    if weight is None:
        return ice.mean(axis=0)
    w = df.loc[ice.index, weight].to_numpy()
    return pd.Series(np.average(ice.to_numpy(), axis=0, weights=w), index=ice.columns)
