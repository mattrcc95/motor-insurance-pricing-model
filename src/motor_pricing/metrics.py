"""Evaluation metrics for insurance pricing models.

All metrics are *weighted*: for frequency the weight is the exposure (a policy
observed for one full year carries more information than one observed for a
month); for severity the weight is the number of claims behind each average.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Deviances
# --------------------------------------------------------------------------- #
def _xlogy_ratio(y: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """y * log(y / mu) with the convention 0 * log(0) = 0."""
    out = np.zeros_like(y, dtype=float)
    pos = y > 0
    out[pos] = y[pos] * np.log(y[pos] / mu[pos])
    return out


def poisson_deviance(y_true, y_pred, weight=None) -> float:
    """Weighted mean Poisson deviance.

    d(y, mu) = 2 * [ y log(y/mu) - (y - mu) ]

    With y = frequency and weight = exposure this equals the deviance of the
    claim counts divided by total exposure.
    """
    y, mu = np.asarray(y_true, float), np.asarray(y_pred, float)
    w = np.ones_like(y) if weight is None else np.asarray(weight, float)
    if np.any(mu <= 0):
        raise ValueError("Predictions must be strictly positive")
    dev = 2.0 * (_xlogy_ratio(y, mu) - (y - mu))
    return float(np.sum(w * dev) / np.sum(w))


def gamma_deviance(y_true, y_pred, weight=None) -> float:
    """Weighted mean Gamma deviance:  d(y, mu) = 2 * [ (y - mu)/mu - log(y/mu) ]."""
    y, mu = np.asarray(y_true, float), np.asarray(y_pred, float)
    w = np.ones_like(y) if weight is None else np.asarray(weight, float)
    if np.any(mu <= 0) or np.any(y <= 0):
        raise ValueError("Gamma deviance needs strictly positive y and predictions")
    dev = 2.0 * ((y - mu) / mu - np.log(y / mu))
    return float(np.sum(w * dev) / np.sum(w))


def deviance_explained(dev_model: float, dev_null: float) -> float:
    """D^2 = 1 - D_model / D_null: share of the null deviance explained (a pseudo R^2)."""
    return 1.0 - dev_model / dev_null


# --------------------------------------------------------------------------- #
# Ranking: Lorenz curve and Gini index
# --------------------------------------------------------------------------- #
def lorenz_curve(y_true, y_pred, weight) -> tuple[np.ndarray, np.ndarray]:
    """Ordered Lorenz curve.

    Policies are sorted from the *lowest* to the *highest* predicted risk. The
    x-axis is the cumulative share of weight (exposure), the y-axis the
    cumulative share of the observed loss (y_true * weight).
    """
    y, p, w = (np.asarray(a, float) for a in (y_true, y_pred, weight))
    order = np.argsort(p, kind="mergesort")
    w_sorted = w[order]
    loss_sorted = (y * w)[order]
    cum_w = np.concatenate([[0.0], np.cumsum(w_sorted)]) / w_sorted.sum()
    cum_loss = np.concatenate([[0.0], np.cumsum(loss_sorted)]) / loss_sorted.sum()
    return cum_w, cum_loss


def gini_index(y_true, y_pred, weight) -> float:
    """Gini = 1 - 2 * area under the ordered Lorenz curve.

    0 means the model ranks no better than random; the higher the better.
    The maximum attainable value is the Gini of an 'oracle' that sorts by the
    true outcome (see :func:`normalised_gini`).
    """
    x, y = lorenz_curve(y_true, y_pred, weight)
    return float(1.0 - 2.0 * np.trapezoid(y, x))


def normalised_gini(y_true, y_pred, weight) -> float:
    """Gini of the model divided by the Gini of the oracle (sorting by y_true)."""
    return gini_index(y_true, y_pred, weight) / gini_index(y_true, y_true, weight)


# --------------------------------------------------------------------------- #
# Lift
# --------------------------------------------------------------------------- #
def _weighted_bins(score: np.ndarray, weight: np.ndarray, n_bins: int) -> np.ndarray:
    """Assign each row to one of ``n_bins`` buckets of (roughly) equal total weight."""
    order = np.argsort(score, kind="mergesort")
    cum_w = np.cumsum(weight[order]) / weight.sum()
    bins_sorted = np.minimum((cum_w * n_bins - 1e-12).astype(int), n_bins - 1)
    bins = np.empty_like(bins_sorted)
    bins[order] = bins_sorted
    return bins + 1


def lift_table(y_true, y_pred, weight, n_bins: int = 10) -> pd.DataFrame:
    """Simple lift chart data: equal-weight buckets of predicted risk.

    For each bucket returns the weighted mean prediction and the weighted mean
    observed outcome. A good model has observed values that increase steeply
    from bucket 1 to bucket ``n_bins`` and track the predictions closely.
    """
    y, p, w = (np.asarray(a, float) for a in (y_true, y_pred, weight))
    b = _weighted_bins(p, w, n_bins)
    df = pd.DataFrame({"bin": b, "w": w, "yw": y * w, "pw": p * w})
    g = df.groupby("bin").sum()
    out = pd.DataFrame({
        "weight": g["w"],
        "predicted": g["pw"] / g["w"],
        "observed": g["yw"] / g["w"],
    })
    out["lift"] = out["observed"] / (df["yw"].sum() / df["w"].sum())
    return out


def double_lift_table(y_true, pred_a, pred_b, weight, n_bins: int = 10) -> pd.DataFrame:
    """Double lift chart: buckets sorted by the ratio pred_a / pred_b.

    In the buckets where the two models disagree the most (first and last), we
    check which model's prediction is closer to the observed outcome.
    All quantities are expressed relative to the overall observed mean.
    """
    y, a, bb, w = (np.asarray(v, float) for v in (y_true, pred_a, pred_b, weight))
    bins = _weighted_bins(a / bb, w, n_bins)
    df = pd.DataFrame({"bin": bins, "w": w, "yw": y * w, "aw": a * w, "bw": bb * w})
    g = df.groupby("bin").sum()
    overall = df["yw"].sum() / df["w"].sum()
    return pd.DataFrame({
        "weight": g["w"],
        "observed": g["yw"] / g["w"] / overall,
        "model_a": g["aw"] / g["w"] / overall,
        "model_b": g["bw"] / g["w"] / overall,
    })
