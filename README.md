# Motor insurance pricing model

End-to-end claim frequency and severity modelling on the public French motor third-party liability
dataset **freMTPL2** (678,013 policies, ~358,000 policy-years, ~26,000 claims). Classical actuarial
GLMs are benchmarked against gradient boosting (LightGBM), and the pricing drivers are explained with
SHAP values and partial dependence plots. The work is presented in six documented Jupyter notebooks,
run in order.

## Steps and milestones

1. **Data ingestion (notebook 01)** - The two public tables (one row per policy, one row per claim) were
   downloaded automatically from OpenML, checked for schema changes and stored untouched, so the whole
   analysis can be rebuilt from scratch.

2. **Data cleaning (notebook 01)** - About 9,100 policies declared claims with no recorded cost, almost
   all in one block of policy numbers. Claim counts were recounted from the claim table so that frequency
   and severity describe the same events. Exposure was capped at one year and claim counts at four.

3. **Large-claim treatment (notebook 01)** - Claims above €100,000 are only 0.16% of claims but ~18% of
   total cost. They were capped for model training, and the excess was added back as a portfolio loading
   so that no money is lost from the final price.

4. **Exploratory analysis (notebook 02)** - One-way analyses with confidence intervals showed that
   bonus-malus, driver age and population density drive claim frequency, while claim cost varies little.
   `Area` was dropped because it duplicates `Density` (correlation 0.98), and an age × bonus-malus
   interaction was identified.

5. **Frequency modelling (notebook 03)** - A Poisson GLM with an exposure offset (the actuarial standard,
   read as multiplicative relativities) was compared with a LightGBM model using the same Poisson loss.
   On held-out data the GBM explained 6.1% of the deviance against 4.2%, and ranked risks better
   (Gini 0.320 vs 0.281).

6. **Severity modelling (notebook 04)** - A Gamma GLM was reduced to driver age and bonus-malus using
   statistical tests and cross-validation, then compared with a heavily regularised GBM. Claim cost turned
   out to be almost unpredictable from the rating factors (~1% of deviance), and the two models are tied.

7. **Robust comparison (notebook 05)** - Exposure-weighted, stratified 5-fold cross-validation on the full
   portfolio confirmed the frequency result: the GBM is better in all five folds (Gini 0.330 vs 0.282;
   deviance explained 6.5% vs 4.4%), while for severity there is no significant difference.

8. **Pure premium (notebook 05)** - Frequency × severity × large-loss loading gives the expected annual
   cost per policy. The GBM-based tariff separates cheap and expensive customers better (Gini 0.349 vs
   0.308), and the best practical combination is GBM frequency with GLM severity.

9. **Business impact (notebook 05)** - In a simulated market where customers buy the cheaper price, the
   GLM-priced insurer attracts the risks it under-prices (adverse selection) and reaches a loss ratio of
   ~135-147%, against ~94-102% for the GBM-priced insurer. Large claims also turned out to be concentrated
   among the riskiest customers, so a flat loading under-prices them.

10. **Explainability (notebook 06)** - SHAP values show that the GBM relies on the same factors as the GLM,
    in the same order of importance. Its advantage comes from finer effect shapes and above all from the
    **bonus-malus × driver age interaction**: a high bonus-malus is far less alarming for a young driver
    than for an experienced one. Individual premiums can be broken down factor by factor, like a GLM tariff.

11. **Reproducibility** - All random seeds are fixed. The full pipeline was re-run from a clean state
    (fresh download) and reproduced every result exactly.

## Data source

freMTPL2, from the R package CASdatasets (Dutang & Charpentier), mirrored on OpenML (datasets 41214 and
41215). Reference study: Noll, Salzmann & Wüthrich (2018), *Case Study: French Motor Third-Party Liability
Claims*.
