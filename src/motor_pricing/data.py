"""Data ingestion and cleaning for the freMTPL2 dataset.

Source
------
The two tables come from the R package CASdatasets (Dutang & Charpentier) and
are mirrored on OpenML, where we download them over HTTPS with
``sklearn.datasets.fetch_openml``:

* freMTPL2freq  -> OpenML data id 41214 (one row per policy)
* freMTPL2sev   -> OpenML data id 41215 (one row per claim)

Layers
------
``data/raw``        exact copy of what OpenML returned (parquet, never modified)
``data/processed``  cleaned, merged modelling table
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import OPENML_CACHE, PROCESSED_DIR, RAW_DIR

OPENML_FREQ_ID = 41214
OPENML_SEV_ID = 41215

FREQ_COLUMNS = [
    "IDpol", "ClaimNb", "Exposure", "Area", "VehPower", "VehAge",
    "DrivAge", "BonusMalus", "VehBrand", "VehGas", "Density", "Region",
]
SEV_COLUMNS = ["IDpol", "ClaimAmount"]

CATEGORICAL_COLUMNS = ["Area", "VehBrand", "VehGas", "Region"]
NUMERIC_COLUMNS = ["VehPower", "VehAge", "DrivAge", "BonusMalus", "Density"]

# Capping thresholds used in the literature (Noll, Salzmann & Wuthrich, 2018)
EXPOSURE_CAP = 1.0
CLAIMNB_CAP = 4


# --------------------------------------------------------------------------- #
# 1. Ingestion
# --------------------------------------------------------------------------- #
def _fetch_openml_frame(data_id: int) -> pd.DataFrame:
    """Download one OpenML dataset as a DataFrame (cached on disk by sklearn)."""
    from sklearn.datasets import fetch_openml

    bunch = fetch_openml(data_id=data_id, as_frame=True, data_home=str(OPENML_CACHE))
    return bunch.frame


def load_raw(force_download: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the raw (freq, sev) tables.

    The first call downloads from OpenML and stores an untouched parquet copy in
    ``data/raw``; subsequent calls read the parquet files (no internet needed).
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    freq_path = RAW_DIR / "freMTPL2freq.parquet"
    sev_path = RAW_DIR / "freMTPL2sev.parquet"

    if force_download or not (freq_path.exists() and sev_path.exists()):
        freq = _fetch_openml_frame(OPENML_FREQ_ID)
        sev = _fetch_openml_frame(OPENML_SEV_ID)
        validate_raw(freq, sev)
        # Store strings as plain object columns so parquet round-trips cleanly
        for df in (freq, sev):
            for c in df.columns:
                if isinstance(df[c].dtype, pd.CategoricalDtype) or pd.api.types.is_string_dtype(df[c]):
                    df[c] = df[c].astype(str).astype(object)
        freq.to_parquet(freq_path, index=False)
        sev.to_parquet(sev_path, index=False)

    return pd.read_parquet(freq_path), pd.read_parquet(sev_path)


def validate_raw(freq: pd.DataFrame, sev: pd.DataFrame) -> None:
    """Fail loudly if the upstream schema changes."""
    missing_f = set(FREQ_COLUMNS) - set(freq.columns)
    missing_s = set(SEV_COLUMNS) - set(sev.columns)
    if missing_f or missing_s:
        raise ValueError(f"Missing columns: freq={missing_f}, sev={missing_s}")
    if freq["IDpol"].duplicated().any():
        raise ValueError("IDpol is not unique in the frequency table")
    if (freq["Exposure"] <= 0).any():
        raise ValueError("Non-positive exposure found")
    if (sev["ClaimAmount"] <= 0).any():
        raise ValueError("Non-positive claim amount found")


# --------------------------------------------------------------------------- #
# 2. Cleaning
# --------------------------------------------------------------------------- #
@dataclass
class CleaningReport:
    """Book-keeping of every change applied during cleaning."""

    n_policies_raw: int = 0
    n_claims_raw: int = 0
    claims_raw_freq_table: int = 0
    sev_ids_not_in_freq: int = 0
    policies_claimnb_changed: int = 0
    exposure_capped: int = 0
    claimnb_capped: int = 0
    notes: list[str] = field(default_factory=list)

    def as_series(self) -> pd.Series:
        d = {k: v for k, v in self.__dict__.items() if k != "notes"}
        return pd.Series(d, name="value")


def clean_freq_types(freq: pd.DataFrame) -> pd.DataFrame:
    """Fix data types and strip stray quotes from string columns."""
    df = freq.copy()
    df["IDpol"] = df["IDpol"].astype("int64")
    df["ClaimNb"] = df["ClaimNb"].astype("int64")
    df["Exposure"] = df["Exposure"].astype("float64")
    for c in NUMERIC_COLUMNS:
        df[c] = df[c].astype("int64")
    for c in CATEGORICAL_COLUMNS:
        df[c] = df[c].astype(str).str.strip().str.strip("'\"")
        df[c] = pd.Categorical(df[c], categories=sorted(df[c].unique()))
    return df


def aggregate_severity(sev: pd.DataFrame, large_claim_cap: float | None = None) -> pd.DataFrame:
    """One row per policy: number of recorded claims, total amount and, if a
    cap is given, the total amount after capping every *single* claim at
    ``large_claim_cap`` (the part above the cap is the 'large-loss excess')."""
    s = sev.assign(IDpol=sev["IDpol"].astype("int64"))
    cap = np.inf if large_claim_cap is None else large_claim_cap
    s = s.assign(ClaimAmountCapped=s["ClaimAmount"].clip(upper=cap))
    out = s.groupby("IDpol").agg(
        ClaimNbSev=("ClaimAmount", "size"),
        ClaimAmount=("ClaimAmount", "sum"),
        ClaimAmountCapped=("ClaimAmountCapped", "sum"),
    )
    return out


def build_modelling_table(
    freq: pd.DataFrame,
    sev: pd.DataFrame,
    exposure_cap: float = EXPOSURE_CAP,
    claimnb_cap: int = CLAIMNB_CAP,
    large_claim_cap: float | None = None,
) -> tuple[pd.DataFrame, CleaningReport]:
    """Merge freq + sev into a single, consistent policy-level table.

    Steps
    -----
    1. Fix dtypes / strip quotes.
    2. Aggregate the severity table per policy.
    3. Left-join onto the policy table (severity IDs with no policy are dropped).
    4. Re-define ``ClaimNb`` as the number of claims that have a recorded amount,
       so that frequency and severity describe the same set of claims.
    5. Cap ``Exposure`` at ``exposure_cap`` and ``ClaimNb`` at ``claimnb_cap``.
       When ``ClaimNb`` is capped the total amount is kept (the money is real),
       so the average severity of those few policies rises slightly.
    """
    rep = CleaningReport()
    df = clean_freq_types(freq)
    rep.n_policies_raw = len(df)
    rep.n_claims_raw = len(sev)
    rep.claims_raw_freq_table = int(df["ClaimNb"].sum())

    agg = aggregate_severity(sev, large_claim_cap)
    rep.sev_ids_not_in_freq = int((~agg.index.isin(df["IDpol"])).sum())

    df = df.join(agg, on="IDpol")
    df["ClaimNbSev"] = df["ClaimNbSev"].fillna(0).astype("int64")
    df["ClaimAmount"] = df["ClaimAmount"].fillna(0.0)
    df["ClaimAmountCapped"] = df["ClaimAmountCapped"].fillna(0.0)

    df = df.rename(columns={"ClaimNb": "ClaimNbOriginal"})
    rep.policies_claimnb_changed = int((df["ClaimNbOriginal"] != df["ClaimNbSev"]).sum())
    df["ClaimNb"] = df["ClaimNbSev"]
    df = df.drop(columns=["ClaimNbSev"])

    rep.exposure_capped = int((df["Exposure"] > exposure_cap).sum())
    df["Exposure"] = df["Exposure"].clip(upper=exposure_cap)

    rep.claimnb_capped = int((df["ClaimNb"] > claimnb_cap).sum())
    df["ClaimNb"] = df["ClaimNb"].clip(upper=claimnb_cap)

    ordered = ["IDpol", "ClaimNb", "ClaimNbOriginal", "ClaimAmount", "ClaimAmountCapped", "Exposure"]
    df = df[ordered + [c for c in df.columns if c not in ordered]]
    return df.reset_index(drop=True), rep


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Add the modelling targets.

    * ``Frequency``   = ClaimNb / Exposure            (claims per policy-year)
    * ``AvgClaim``    = ClaimAmountCapped / ClaimNb   (average *capped* cost of one claim, NaN if no claim)
    * ``PurePremium`` = ClaimAmount / Exposure        (total observed cost per policy-year, uncapped)
    """
    out = df.copy()
    out["Frequency"] = out["ClaimNb"] / out["Exposure"]
    out["AvgClaim"] = np.where(
        out["ClaimNb"] > 0, out["ClaimAmountCapped"] / out["ClaimNb"].clip(lower=1), np.nan
    )
    out["PurePremium"] = out["ClaimAmount"] / out["Exposure"]
    return out


def save_processed(df: pd.DataFrame, name: str = "fremtpl2_clean.parquet") -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PROCESSED_DIR / name, index=False)


def load_processed(name: str = "fremtpl2_clean.parquet") -> pd.DataFrame:
    path = PROCESSED_DIR / name
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found - run notebooks/01_data_ingestion_and_cleaning.ipynb first"
        )
    df = pd.read_parquet(path)
    for c in CATEGORICAL_COLUMNS:
        df[c] = pd.Categorical(df[c], categories=sorted(df[c].astype(str).unique()))
    return df
