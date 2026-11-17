"""Feature engineering.

Two different representations of the same rating factors are built:

* **GLM features** - a GLM is linear in its parameters on the log scale, so any
  non-linear effect (e.g. the U-shape of claim frequency in driver age) must be
  given to it explicitly. We therefore bin some continuous variables into
  categories, cap others and log-transform skewed ones. The binning follows the
  reference study of Noll, Salzmann & Wuthrich (2018).
* **GBM features** - gradient-boosted trees split on thresholds, so they learn
  non-linearities and interactions on their own and are invariant to monotone
  transforms (log or not makes no difference). They get the (almost) raw columns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

AREA_ORDER = ["A", "B", "C", "D", "E", "F"]

DRIVAGE_BINS = [17, 20, 25, 30, 40, 50, 70, np.inf]
DRIVAGE_LABELS = ["18-20", "21-25", "26-30", "31-40", "41-50", "51-70", "71+"]

VEHAGE_BINS = [-1, 0, 10, np.inf]
VEHAGE_LABELS = ["0", "1-10", "11+"]

VEHPOWER_CAP = 9
BONUSMALUS_CAP = 150

GLM_CATEGORICAL = ["VehPowerGLM", "VehAgeGLM", "DrivAgeGLM", "VehBrand", "VehGas", "Region"]
GLM_NUMERIC = ["AreaGLM", "BonusMalusGLM", "LogDensity"]

GBM_FEATURES = [
    "VehPower", "VehAge", "DrivAge", "BonusMalus", "Density",
    "AreaCode", "VehBrand", "VehGas", "Region",
]
GBM_CATEGORICAL = ["VehBrand", "VehGas", "Region"]


# --------------------------------------------------------------------------- #
# GLM representation
# --------------------------------------------------------------------------- #
def glm_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return the engineered rating factors used by the GLMs."""
    out = pd.DataFrame(index=df.index)
    # Area A..F is ordered (rural -> urban); Noll et al. found a log-linear trend
    out["AreaGLM"] = df["Area"].astype(str).map({a: i + 1 for i, a in enumerate(AREA_ORDER)}).astype(float)

    vp = df["VehPower"].clip(upper=VEHPOWER_CAP).astype(int).astype(str)
    vp = vp.where(vp != str(VEHPOWER_CAP), f"{VEHPOWER_CAP}+")
    vp_levels = [str(v) for v in range(4, VEHPOWER_CAP)] + [f"{VEHPOWER_CAP}+"]
    out["VehPowerGLM"] = pd.Categorical(vp, categories=vp_levels)

    out["VehAgeGLM"] = pd.cut(df["VehAge"], VEHAGE_BINS, labels=VEHAGE_LABELS)
    out["DrivAgeGLM"] = pd.cut(df["DrivAge"], DRIVAGE_BINS, labels=DRIVAGE_LABELS)
    out["BonusMalusGLM"] = df["BonusMalus"].clip(upper=BONUSMALUS_CAP).astype(float)
    out["LogDensity"] = np.log(df["Density"].astype(float))
    for c in ["VehBrand", "VehGas", "Region"]:
        out[c] = df[c]
    return out


class GLMDesign:
    """Build a GLM design matrix with explicit, data-driven reference levels.

    For each categorical factor the *reference (base) level* is the level with
    the largest exposure in the training data. This is the actuarial convention:
    the base level gets relativity 1.00 and every other level is expressed as a
    multiplier relative to the most common risk, which keeps standard errors of
    the remaining levels small.
    """

    def __init__(self, categorical=GLM_CATEGORICAL, numeric=GLM_NUMERIC):
        self.categorical = list(categorical)
        self.numeric = list(numeric)
        self.reference_: dict[str, str] = {}
        self.levels_: dict[str, list[str]] = {}
        self.columns_: list[str] = []

    def fit(self, df: pd.DataFrame, weight: pd.Series | np.ndarray | None = None) -> "GLMDesign":
        feats = glm_features(df)
        w = np.ones(len(df)) if weight is None else np.asarray(weight, dtype=float)
        for c in self.categorical:
            levels = [str(x) for x in feats[c].cat.categories]
            by_level = pd.Series(w, index=feats.index).groupby(feats[c].astype(str)).sum()
            self.levels_[c] = levels
            self.reference_[c] = str(by_level.idxmax())
        self.columns_ = list(self.transform(df).columns)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        feats = glm_features(df)
        parts = [pd.Series(1.0, index=feats.index, name="const"), feats[self.numeric].astype(float)]
        for c in self.categorical:
            vals = feats[c].astype(str)
            for lev in self.levels_[c]:
                if lev == self.reference_[c]:
                    continue
                parts.append((vals == lev).astype(float).rename(f"{c}[{lev}]"))
        X = pd.concat(parts, axis=1)
        if self.columns_:
            X = X.reindex(columns=self.columns_, fill_value=0.0)
        return X


# --------------------------------------------------------------------------- #
# GBM representation
# --------------------------------------------------------------------------- #
def gbm_features(df: pd.DataFrame) -> pd.DataFrame:
    """Raw rating factors for LightGBM (categoricals as pandas 'category')."""
    out = df[[c for c in GBM_FEATURES if c != "AreaCode"]].copy()
    out["AreaCode"] = df["Area"].astype(str).map({a: i + 1 for i, a in enumerate(AREA_ORDER)}).astype(int)
    for c in GBM_CATEGORICAL:
        if not isinstance(out[c].dtype, pd.CategoricalDtype):
            out[c] = out[c].astype("category")
    return out[GBM_FEATURES]
