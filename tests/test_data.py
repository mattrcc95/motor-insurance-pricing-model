import numpy as np
import pytest

from motor_pricing.data import (
    CLAIMNB_CAP, EXPOSURE_CAP, add_targets, build_modelling_table, validate_raw,
)


def test_validate_raw_accepts_good_data(raw):
    validate_raw(*raw)


def test_validate_raw_rejects_duplicate_ids(raw):
    freq, sev = raw
    bad = freq.copy()
    bad.loc[1, "IDpol"] = bad.loc[0, "IDpol"]
    with pytest.raises(ValueError, match="not unique"):
        validate_raw(bad, sev)


def test_types_and_quotes_cleaned(clean):
    assert clean["IDpol"].dtype == "int64"
    assert set(clean["VehGas"].cat.categories) == {"Diesel", "Regular"}


def test_claimnb_consistent_with_severity(raw):
    freq, sev = raw
    df, rep = build_modelling_table(freq, sev)
    # Policy 1 had ClaimNb=2 but no severity record -> recounted to 0
    assert df.loc[df["IDpol"] == 1, "ClaimNb"].item() == 0
    assert rep.sev_ids_not_in_freq == 1
    assert (df.loc[df["ClaimNb"] == 0, "ClaimAmount"] == 0).all()
    assert (df.loc[df["ClaimNb"] > 0, "ClaimAmount"] > 0).all()


def test_caps_applied(clean):
    assert clean["Exposure"].max() <= EXPOSURE_CAP
    assert clean["ClaimNb"].max() <= CLAIMNB_CAP


def test_large_claim_cap_only_affects_capped_column(clean):
    big = clean.loc[clean["ClaimAmount"] > 1e5]
    assert len(big) == 1
    assert big["ClaimAmountCapped"].item() < big["ClaimAmount"].item()
    assert (clean["ClaimAmountCapped"] <= clean["ClaimAmount"] + 1e-9).all()


def test_targets(clean):
    np.testing.assert_allclose(clean["Frequency"], clean["ClaimNb"] / clean["Exposure"])
    assert clean.loc[clean["ClaimNb"] == 0, "AvgClaim"].isna().all()
    assert (clean.loc[clean["ClaimNb"] > 0, "AvgClaim"] > 0).all()


def test_total_amount_preserved(raw):
    freq, sev = raw
    df, _ = build_modelling_table(freq, sev)
    known = sev[sev["IDpol"].isin(freq["IDpol"].astype(int))]
    assert df["ClaimAmount"].sum() == pytest.approx(known["ClaimAmount"].sum())


def test_add_targets_does_not_mutate(clean):
    before = clean.copy()
    add_targets(clean)
    assert before.equals(clean)
