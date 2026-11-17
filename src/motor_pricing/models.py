"""Model wrappers with a common interface: ``fit(df)`` / ``predict(df)``.

Every model takes the full policy-level DataFrame and knows itself which
columns are the target, the weight and the features. ``predict`` returns the
prediction on the *target scale*:

* frequency models -> expected claims per policy-year
* severity models  -> expected cost of one claim
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.model_selection import train_test_split

from . import RANDOM_STATE
from .features import GLM_CATEGORICAL, GLM_NUMERIC, GLMDesign, gbm_features

FREQ_TARGET, FREQ_WEIGHT = "Frequency", "Exposure"
SEV_TARGET, SEV_WEIGHT = "AvgClaim", "ClaimNb"


# --------------------------------------------------------------------------- #
# Baseline
# --------------------------------------------------------------------------- #
class NullModel:
    """Intercept-only model: predicts the weighted average of the target.

    It is the benchmark every other model must beat and the denominator of the
    'deviance explained' (D^2) metric.
    """

    def __init__(self, target: str, weight: str):
        self.target, self.weight = target, weight

    def fit(self, df: pd.DataFrame) -> "NullModel":
        d = df[df[self.weight] > 0]  # e.g. severity: only policies with claims
        self.mean_ = float(np.average(d[self.target], weights=d[self.weight]))
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.full(len(df), self.mean_)


# --------------------------------------------------------------------------- #
# GLMs (statsmodels -> full inference: standard errors, p-values, CIs)
# --------------------------------------------------------------------------- #
class _GLMBase:
    """Shared GLM logic. ``categorical`` / ``numeric`` select the rating factors
    (default: all GLM factors, see ``features.GLM_CATEGORICAL`` / ``GLM_NUMERIC``)."""

    def __init__(self, categorical=None, numeric=None):
        self.categorical = list(GLM_CATEGORICAL if categorical is None else categorical)
        self.numeric = list(GLM_NUMERIC if numeric is None else numeric)

    def _design(self, df: pd.DataFrame, fit: bool, weight=None) -> pd.DataFrame:
        if fit:
            self.design_ = GLMDesign(self.categorical, self.numeric).fit(df, weight)
        return self.design_.transform(df)

    def relativities(self, alpha: float = 0.05) -> pd.DataFrame:
        """exp(beta) with confidence interval, i.e. multiplicative rating factors."""
        params = self.result_.params
        ci = self.result_.conf_int(alpha)
        out = pd.DataFrame({
            "coef": params,
            "std_err": self.result_.bse,
            "p_value": self.result_.pvalues,
            "relativity": np.exp(params),
            "ci_low": np.exp(ci[0]),
            "ci_high": np.exp(ci[1]),
        })
        return out

    def summary(self):
        return self.result_.summary()


class FrequencyGLM(_GLMBase):
    """Poisson GLM with log link and log(exposure) offset.

        ClaimNb_i ~ Poisson(mu_i),   log(mu_i) = log(Exposure_i) + x_i' beta

    so that  E[ClaimNb_i / Exposure_i] = exp(x_i' beta)  is the annual claim frequency.
    """

    def fit(self, df: pd.DataFrame) -> "FrequencyGLM":
        X = self._design(df, fit=True, weight=df["Exposure"])
        model = sm.GLM(
            df["ClaimNb"].to_numpy(float), X,
            family=sm.families.Poisson(),
            offset=np.log(df["Exposure"].to_numpy(float)),
        )
        self.result_ = model.fit()
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X = self._design(df, fit=False)
        return np.exp(X.to_numpy() @ self.result_.params.to_numpy())


class SeverityGLM(_GLMBase):
    """Gamma GLM with log link on the average claim cost.

        AvgClaim_i ~ Gamma(mu_i, phi / ClaimNb_i),   log(mu_i) = x_i' beta

    A policy with n claims has an average of n independent costs, so its
    variance is n times smaller: ClaimNb enters as a *variance weight*.
    Only policies with at least one claim are used.
    """

    def fit(self, df: pd.DataFrame) -> "SeverityGLM":
        d = df[df["ClaimNb"] > 0]
        X = self._design(d, fit=True, weight=d["ClaimNb"])
        model = sm.GLM(
            d["AvgClaim"].to_numpy(float), X,
            family=sm.families.Gamma(link=sm.families.links.Log()),
            var_weights=d["ClaimNb"].to_numpy(float),
        )
        self.result_ = model.fit()
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X = self._design(df, fit=False)
        return np.exp(X.to_numpy() @ self.result_.params.to_numpy())


# --------------------------------------------------------------------------- #
# Gradient boosting (LightGBM)
# --------------------------------------------------------------------------- #
FREQ_GBM_PARAMS = dict(
    objective="poisson",
    learning_rate=0.05,
    n_estimators=2000,
    num_leaves=31,
    min_child_samples=500,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    cat_smooth=50,
    verbose=-1,
)

SEV_GBM_PARAMS = dict(
    objective="gamma",
    learning_rate=0.02,
    n_estimators=2000,
    num_leaves=7,
    min_child_samples=200,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=10.0,
    cat_smooth=100,
    verbose=-1,
)


class _GBMBase:
    target: str
    weight: str
    default_params: dict
    eval_metric: str

    def __init__(self, early_stopping_rounds: int = 100, valid_size: float = 0.1,
                 random_state: int = RANDOM_STATE, **params):
        self.params = {**self.default_params, "random_state": random_state, **params}
        self.early_stopping_rounds = early_stopping_rounds
        self.valid_size = valid_size
        self.random_state = random_state

    def _rows(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def fit(self, df: pd.DataFrame) -> "_GBMBase":
        d = self._rows(df)
        X, y, w = gbm_features(d), d[self.target].to_numpy(float), d[self.weight].to_numpy(float)
        self.model_ = LGBMRegressor(**self.params)
        if self.early_stopping_rounds:
            # Inner hold-out (taken from the training data only) to pick the number of trees
            X_tr, X_va, y_tr, y_va, w_tr, w_va = train_test_split(
                X, y, w, test_size=self.valid_size, random_state=self.random_state
            )
            self.model_.fit(
                X_tr, y_tr, sample_weight=w_tr,
                eval_X=X_va, eval_y=y_va, eval_sample_weight=[w_va],
                eval_metric=self.eval_metric,
                callbacks=[early_stopping(self.early_stopping_rounds, verbose=False), log_evaluation(0)],
            )
            self.best_iteration_ = self.model_.best_iteration_
        else:
            self.model_.fit(X, y, sample_weight=w)
            self.best_iteration_ = self.params["n_estimators"]
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.model_.predict(gbm_features(df), num_iteration=self.best_iteration_)


class FrequencyGBM(_GBMBase):
    """LightGBM with Poisson objective on Frequency, weighted by Exposure.

    Minimising the exposure-weighted Poisson deviance on ClaimNb/Exposure is
    mathematically identical to Poisson likelihood on ClaimNb with a
    log(Exposure) offset - the same assumption as the GLM.
    """

    target, weight = FREQ_TARGET, FREQ_WEIGHT
    default_params = FREQ_GBM_PARAMS
    eval_metric = "poisson"


class SeverityGBM(_GBMBase):
    """LightGBM with Gamma objective on AvgClaim, weighted by ClaimNb."""

    target, weight = SEV_TARGET, SEV_WEIGHT
    default_params = SEV_GBM_PARAMS
    eval_metric = "gamma"

    def _rows(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[df["ClaimNb"] > 0]
