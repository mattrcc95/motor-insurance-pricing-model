import numpy as np
import pytest

from motor_pricing.explain import ice_curves, partial_dependence
from motor_pricing.models import FrequencyGLM


@pytest.fixture(scope="module")
def glm(clean):
    return FrequencyGLM(categorical=["DrivAgeGLM", "Region"], numeric=["BonusMalusGLM"]).fit(clean)


def test_ice_shape(clean, glm):
    ice = ice_curves(glm, clean, "BonusMalus", [50, 100, 150], n_samples=200)
    assert ice.shape == (200, 3)
    assert (ice > 0).all().all()


def test_glm_ice_curves_are_parallel_on_log_scale(clean, glm):
    """In a log-link GLM without interactions, changing one factor multiplies every
    policy's prediction by the same relativity -> parallel log-ICE curves."""
    ice = ice_curves(glm, clean, "BonusMalus", [50, 80, 120], n_samples=300)
    ratios = ice[120] / ice[50]
    np.testing.assert_allclose(ratios, ratios.iloc[0], rtol=1e-10)
    beta = glm.result_.params["BonusMalusGLM"]
    assert ratios.iloc[0] == pytest.approx(np.exp(beta * 70))


def test_pd_flat_for_unused_feature(clean, glm):
    pd_ = partial_dependence(glm, clean, "VehPower", [4, 8, 12], n_samples=300)
    assert np.allclose(pd_.to_numpy(), pd_.iloc[0])


def test_pd_categorical_feature(clean, glm):
    pd_ = partial_dependence(glm, clean, "Region", list(clean["Region"].cat.categories), n_samples=300)
    assert len(pd_) == clean["Region"].cat.categories.size
