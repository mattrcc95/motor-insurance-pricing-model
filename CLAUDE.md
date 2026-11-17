# CLAUDE.md

Guidance for Claude Code (and humans) working in this repository.

## Project
Motor third-party-liability pricing on the public **freMTPL2** dataset (~678k French policies).
Claim **frequency** (Poisson) and **severity** (Gamma) are modelled with actuarial GLMs and
benchmarked against LightGBM, evaluated with exposure-weighted cross-validation
(Poisson/Gamma deviance, Gini index, lift curves) and explained with SHAP / partial dependence.

## Environment
- Windows, Python 3.14, virtualenv in `.venv/` (not committed).
- Jupyter kernel: **`motor-pricing`** ("Python (motor-pricing)").
- Setup from scratch:
  ```powershell
  python -m venv .venv
  .\.venv\Scripts\python.exe -m pip install -r requirements.txt
  .\.venv\Scripts\python.exe -m pip install -e .
  .\.venv\Scripts\python.exe -m ipykernel install --user --name motor-pricing --display-name "Python (motor-pricing)"
  ```
- Tests: `.\.venv\Scripts\python.exe -m pytest -q` (offline, synthetic data, a few seconds).
- Execute a notebook in place:
  `.\.venv\Scripts\jupyter.exe nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=motor-pricing --ExecutePreprocessor.timeout=-1 notebooks\<nb>.ipynb`

## Layout
- `src/motor_pricing/` - reusable, tested code (the notebooks import from here)
  - `data.py` - OpenML download (ids 41214 / 41215), raw parquet cache, cleaning, targets
  - `features.py` - GLM feature engineering (binning, caps, logs, reference levels) and GBM features
  - `models.py` - `NullModel`, `FrequencyGLM`, `SeverityGLM`, `FrequencyGBM`, `SeverityGBM` (common `fit(df)` / `predict(df)`)
  - `metrics.py` - weighted Poisson/Gamma deviance, Lorenz curve, Gini, lift and double-lift tables
  - `validation.py` - stratified train/test split, K-fold CV, `fold_ids`, scorers
  - `eda.py` - one-way frequency/severity tables with CIs, Cramér's V
  - `explain.py` - model-agnostic partial dependence / ICE (works with any `predict(df)` model)
  - `plotting.py` - shared style (fixed colours: GLM blue, GBM orange) and reusable charts
- `notebooks/` - numbered, run in order (01 -> 06). Each one is the documented narrative.
  01 ingestion/cleaning, 02 EDA, 03 frequency, 04 severity, 05 CV comparison + pure premium,
  06 SHAP/PDP explainability.
- `models/` - tuned GBM hyper-parameters written by notebooks 03/04 and read by 05/06 (git-ignored).
- `tests/` - pytest suite on synthetic data.
- `data/` - `raw/`, `processed/`, `openml_cache/`: generated, git-ignored.
- `reports/figures/` - generated figures, git-ignored. `docs/images/` holds the copies used by the README.

## Conventions (important)
- **Notebook documentation standard**: every step is preceded by a markdown cell stating, in
  plain language, the goal of the step; every modelling choice gets a short theory section;
  every result is interpreted both mathematically and in real-world insurance terms.
  Keep this standard when adding or editing notebooks.
- Business logic lives in `src/`, not in notebooks; add tests for new functions.
- Notebooks are committed **with outputs** (executed top-to-bottom with the `motor-pricing` kernel)
  and the narrative numbers must match the outputs - re-check text after re-running.
- Weights: frequency -> `Exposure`; severity -> `ClaimNb`. Never report unweighted metrics.
- Random seed: `motor_pricing.RANDOM_STATE = 42`.
- Data is never committed.
- One commit per logical step; end commit messages with the Co-Authored-By trailer.

## Key modelling decisions
- `Area` is dropped from both models (Spearman 0.98 with Density; notebook 02).
- Frequency GLM: Poisson, log link, log(Exposure) offset; age/vehicle-age/power binned, BM capped at 150, log density.
- Severity GLM: Gamma, log link, ClaimNb variance weights, **reduced to DrivAge bins + BonusMalus** (F-tests + CV, notebook 04).
- Severity GBM: tree count chosen by 5-fold CV curves (no early stopping - inner split too noisy).
- Gini/Lorenz merge tied predictions (a constant model must have Gini 0 regardless of row order).
- Narrative text in notebooks quotes numbers from the outputs: after changing code, re-run and re-check the text.

## Key data decisions (see notebook 01)
- `ClaimNb` is recomputed as the number of claims with a recorded amount in freMTPL2sev
  (the original counts include ~9k policies with claims but no amounts, almost all with IDpol <= ~25k).
- `Exposure` capped at 1 year, `ClaimNb` capped at 4.
- Single claims capped at a large-loss threshold for severity modelling (`ClaimAmountCapped`);
  the excess is re-allocated as a flat loading when building the pure premium.
