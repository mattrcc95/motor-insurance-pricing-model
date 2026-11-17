import numpy as np
import pytest

from motor_pricing.features import GLMDesign, gbm_features, glm_features
from motor_pricing.models import (
    FrequencyGBM, FrequencyGLM, NullModel, SeverityGBM, SeverityGLM,
)
from motor_pricing.validation import (
    cross_validate, frequency_scores, split_train_test, summarise_cv,
)


def test_glm_features_bins(clean):
    f = glm_features(clean)
    assert f["DrivAgeGLM"].isna().sum() == 0
    assert f["VehAgeGLM"].isna().sum() == 0
    assert f["BonusMalusGLM"].max() <= 150


def test_glm_design_reference_is_largest_exposure(clean):
    d = GLMDesign().fit(clean, clean["Exposure"])
    f = glm_features(clean)
    expected = clean.groupby(f["DrivAgeGLM"].astype(str))["Exposure"].sum().idxmax()
    assert d.reference_["DrivAgeGLM"] == expected
    X = d.transform(clean)
    assert f"DrivAgeGLM[{expected}]" not in X.columns
    assert "const" in X.columns
    # transform on a subset keeps exactly the same columns
    assert list(d.transform(clean.head(10)).columns) == list(X.columns)


def test_gbm_features_types(clean):
    X = gbm_features(clean)
    assert X["Region"].dtype == "category"
    assert "Area" not in X.columns  # redundant with Density


def test_frequency_glm_balance_property(clean):
    """A Poisson GLM with log link and intercept reproduces total observed claims."""
    m = FrequencyGLM().fit(clean)
    pred_claims = (m.predict(clean) * clean["Exposure"]).sum()
    assert pred_claims == pytest.approx(clean["ClaimNb"].sum(), rel=1e-6)
    rel = m.relativities()
    assert {"relativity", "ci_low", "ci_high"} <= set(rel.columns)
    # BonusMalus drives frequency in the synthetic data -> positive coefficient
    assert rel.loc["BonusMalusGLM", "coef"] > 0


def test_severity_glm_positive_predictions(clean):
    m = SeverityGLM().fit(clean)
    assert (m.predict(clean) > 0).all()


@pytest.mark.parametrize("cls", [FrequencyGBM, SeverityGBM])
def test_gbm_fit_predict(clean, cls):
    m = cls(n_estimators=50, early_stopping_rounds=10).fit(clean)
    p = m.predict(clean)
    assert p.shape == (len(clean),)
    assert np.all(p > 0)


def test_null_model(clean):
    m = NullModel("Frequency", "Exposure").fit(clean)
    assert m.predict(clean[:3]) == pytest.approx([clean["ClaimNb"].sum() / clean["Exposure"].sum()] * 3)


def test_split_and_cv(clean):
    tr, te = split_train_test(clean, test_size=0.25)
    assert len(tr) + len(te) == len(clean)
    assert set(tr.index).isdisjoint(te.index)
    scores, oof = cross_validate(
        tr, {"GLM": FrequencyGLM}, frequency_scores,
        null_factory=lambda: NullModel("Frequency", "Exposure"), n_splits=3, verbose=False,
    )
    assert oof.notna().all().all()
    summary = summarise_cv(scores)
    null_d2 = scores.query("model == 'Null' and metric == 'deviance_explained'")["value"]
    assert np.allclose(null_d2, 0)
    assert ("poisson_deviance", "mean") in summary.columns
