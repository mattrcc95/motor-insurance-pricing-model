"""Train/test split and exposure-weighted K-fold cross-validation."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from . import RANDOM_STATE
from .metrics import deviance_explained, gamma_deviance, gini_index, poisson_deviance


def _claim_strata(df: pd.DataFrame) -> np.ndarray:
    """Stratification label: 0, 1 or 2+ claims.

    Claims are rare (~4% of policies). Stratifying makes every fold contain the
    same share of claimants, so fold-to-fold differences reflect the model and
    not the luck of the split.
    """
    return df["ClaimNb"].clip(upper=2).to_numpy()


def fold_ids(df: pd.DataFrame, n_splits: int = 5, random_state: int = RANDOM_STATE) -> np.ndarray:
    """Fold number (1..n_splits) of every row, identical to the folds of :func:`cross_validate`."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    out = np.zeros(len(df), dtype=int)
    for fold, (_, va_idx) in enumerate(skf.split(df, _claim_strata(df)), start=1):
        out[va_idx] = fold
    return out


def split_train_test(df: pd.DataFrame, test_size: float = 0.2,
                     random_state: int = RANDOM_STATE) -> tuple[pd.DataFrame, pd.DataFrame]:
    train, test = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=_claim_strata(df)
    )
    return train.copy(), test.copy()


def frequency_scores(df: pd.DataFrame, pred: np.ndarray, null_pred: np.ndarray) -> dict:
    y, w = df["Frequency"].to_numpy(), df["Exposure"].to_numpy()
    dev = poisson_deviance(y, pred, w)
    dev0 = poisson_deviance(y, null_pred, w)
    return {
        "poisson_deviance": dev,
        "deviance_explained": deviance_explained(dev, dev0),
        "gini": gini_index(y, pred, w),
        "predicted_claims / observed_claims": float(np.sum(pred * w) / np.sum(y * w)),
    }


def severity_scores(df: pd.DataFrame, pred: np.ndarray, null_pred: np.ndarray) -> dict:
    mask = df["ClaimNb"].to_numpy() > 0
    y, w = df["AvgClaim"].to_numpy()[mask], df["ClaimNb"].to_numpy()[mask]
    p, p0 = pred[mask], null_pred[mask]
    dev = gamma_deviance(y, p, w)
    dev0 = gamma_deviance(y, p0, w)
    return {
        "gamma_deviance": dev,
        "deviance_explained": deviance_explained(dev, dev0),
        "gini": gini_index(y, p, w),
        "predicted_cost / observed_cost": float(np.sum(p * w) / np.sum(y * w)),
    }


def cross_validate(
    df: pd.DataFrame,
    model_factories: dict[str, Callable[[], object]],
    scorer: Callable[[pd.DataFrame, np.ndarray, np.ndarray], dict],
    null_factory: Callable[[], object],
    n_splits: int = 5,
    random_state: int = RANDOM_STATE,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified K-fold CV with weighted metrics.

    Returns
    -------
    scores : long DataFrame (model, fold, metric, value)
    oof    : out-of-fold predictions, one column per model (index = df.index)
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    rows, oof = [], pd.DataFrame(index=df.index)
    for name in ["Null", *model_factories]:
        oof[name] = np.nan

    for fold, (tr_idx, va_idx) in enumerate(skf.split(df, _claim_strata(df)), start=1):
        tr, va = df.iloc[tr_idx], df.iloc[va_idx]
        null_pred = null_factory().fit(tr).predict(va)
        oof.loc[va.index, "Null"] = null_pred
        for name, factory in {"Null": null_factory, **model_factories}.items():
            pred = null_pred if name == "Null" else factory().fit(tr).predict(va)
            oof.loc[va.index, name] = pred
            for metric, value in scorer(va, pred, null_pred).items():
                rows.append({"model": name, "fold": fold, "metric": metric, "value": value})
        if verbose:
            print(f"fold {fold}/{n_splits} done")

    return pd.DataFrame(rows), oof


def summarise_cv(scores: pd.DataFrame) -> pd.DataFrame:
    """Mean and standard deviation of every metric across folds."""
    return (
        scores.groupby(["metric", "model"])["value"]
        .agg(["mean", "std"])
        .unstack("metric")
        .swaplevel(axis=1)
        .sort_index(axis=1)
    )
