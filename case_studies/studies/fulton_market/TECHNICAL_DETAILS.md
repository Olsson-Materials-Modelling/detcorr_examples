# Fulton rolling calibration: technical details

[Study overview](README.md) ·
[Registered protocol](REGISTERED_PROTOCOL.json) ·
[Case-study technical contract](../../TECHNICAL_DETAILS.md)

This document consolidates the temporal design, model, measure contract,
protocol locks, audit surface, provenance, and limitations behind the concise
study overview. Generated `summary.json`, claim evaluations, and CSV tables
remain authoritative.

## Evidence status

The locked full-fixture run is a registered retrospective reanalysis. Its
rolling score comparison supports E2 evidence conditional on the training-only
plug-in reference and dependence model; scale-interval and coverage statements
that require full nuisance-parameter propagation remain E1. Prospective
confirmation would require a protocol and data identity frozen before the new
outcomes are observed.

## Fixture and temporal roles

The bundled `fish.csv` is a byte-identical 97-row Rdatasets serialization of
the table distributed with `wooldridge`:

- SHA256:
  `fa51db236cb3f9e7040cdc50b002542bcbb3d51077132848d1960a996ec9b3c9`;
- all 97 complete ordered rows are retained;
- `rownames` is the unique source-row key; and
- rolling evaluation begins at observation 65.

For each test position `t`:

```text
rows before t-16       fit the reduced log-price reference
rows t-16 through t-1  calibrate eta = log(lambda)
row t                  remains untouched and is scored once
```

This produces 33 one-step evaluations. The test outcome is unavailable before
scoring, but its contemporaneous weather, weekday, and trend covariates are
used to construct the reference. The estimand is therefore conditional one-
step prediction or nowcasting, not an unconditional advance forecast when
those covariates would be unavailable.

The fixture manifest, rights basis, schema, source, roles, and hash are fixed
in [`data/FIXTURE_MANIFEST.json`](data/FIXTURE_MANIFEST.json). The CSV is
conveyed separately under GPL-3 to the extent the official `wooldridge` package
licence applies, with the Rdatasets underlying-rights caveat retained in
[`../../THIRD_PARTY_NOTICES.md`](../../THIRD_PARTY_NOTICES.md). The immediate
source is the Rdatasets serialization of the `wooldridge::fish` table; the
manifest retains the Graddy (1995) source attribution, snapshot URL,
transformation statement, and explicit boundary between package licensing and
underlying-data rights.

## Verification spine and scientific model

The source demonstrator's first 64 rows fit a native-price reference and its
last 33 rows form an iid Gaussian calibration likelihood. This study retains
that exact full-state Gaussian calculation only as a regression spine: adding
`log|det D_u c|=n log(lambda)` recovers the reduced graph target, while
omitting it changes the target. It is not claim-bearing Fulton evidence.

The scientific model instead uses log price because prices are positive and
the calibrated scale is multiplicative. With training-only positive reference
`a_t`:

```text
c_t(lambda,u_t) = a_t - lambda u_t = 0
u_t = a_t/lambda
log(y_t) = log(u_t) + epsilon_t
eta = log(lambda) ~ Normal(0, 0.5^2)
```

The reduced reference is OLS on log price using intercept, squared wave height,
squared wind speed, weekday indicators, and trend. It is refit from past-only
rows at every origin.

## Dependence and nuisance treatment

The primary observation covariance is stationary AR(1) in log price. The
stationary marginal residual SD and raw AR coefficient are plug-in estimates
from only that origin's reference-training residuals. The innovation SD is
`sigma_marginal*sqrt(1-phi_used^2)`. The primary coefficient bound is
`[-0.90,0.90]`; `phi` has no posterior.

The analysis also reports training-only residual ACF, Durbin–Watson,
approximate Ljung–Box, and squared-residual diagnostics. AR(1) is a declared
working dependence model, not an asserted data-generating truth.

A coefficient-covariance sensitivity propagates Gaussian OLS uncertainty
analytically, with its own matched identity comparator. This is not full joint
uncertainty propagation over reference coefficients, residual scale, and
dependence.

## Comparators and scores

The locked study reports:

- primary plug-in AR(1) graph and matched scale-one identity;
- iid graph and matched iid identity;
- coefficient-covariance graph and its matched identity;
- the target obtained by omitting the state-volume factor;
- an order-preserving lag-four reference negative control;
- last-price persistence; and
- a one-at-a-time AR-coefficient-bound sensitivity.

Native-price log predictive density and CRPS are the proper scores.
Predictive-mean RMSE, predictive-median MAE, and 90% interval coverage/width
supplement them. All paired contrasts share the same circular moving-block
resamples: block length four, 5th/95th percentile interval, 5,000 full-run
replicates, and seed `20260711`. The 33 overlapping origins are not treated as
iid replications.

The lag-four control replaces contemporaneous reference covariates with those
from four observations earlier without reordering responses or using future
rows. The state-volume omission is a target-measure counterfactual, not a
candidate model selected by predictive score.

## Claim rules and results

`FU-PRED` requires positive cumulative primary graph-minus-matched-identity
Delta ELPD, a positive lower moving-block endpoint, graph CRPS no worse than
identity, and deterioration of the lag-four control. The reference values are:

- cumulative Delta ELPD `+0.408`;
- 90% mean-difference interval `[-0.0417,0.0634]`;
- graph CRPS worse by `0.000069`; and
- persistence cumulative log score `0.745` above graph.

The gate therefore gives `FU-PRED=NOT_SUPPORTED`.

The matched iid Delta ELPD is `+6.180` with a positive interval, whereas the
AR(1) interval crosses zero. `FU-DEP=SUPPORTED` with conclusion
`DEPENDENCE_SENSITIVE`. This identifies sensitivity to a plug-in dependence
assumption, not the true serial process.

For the primary `0.90` bound, raw `phi` estimates span `0.58084–0.68275`, so
no origin is clipped. Bounds `0.75`, `0.90`, and `0.99` are identical. The
post-outcome `0.50` stress clips every origin but retains every terminal claim
category: Delta ELPD `+1.202`, interval `[-0.0224,0.0915]`, and maximum scale-
posterior movement `0.1264` primary posterior SD. This check is non-selecting
and cannot upgrade the evidence class.

## State measure and determinant

Price states are made dimensionless using the geometric mean price of each
origin's reference-training rows. In direct state `u>0`, the declared base is
Lebesgue: `q_u=1` relative to `du`, and `J_u=lambda^n`. Under the genuine
pushforward `v=log(u)`:

```text
q_v(v) = exp(sum(v))
J_v = product_i lambda exp(v_i)
(J_v/q_v)|graph = lambda^n = J_u/q_u
```

Both the base density and residual Jacobian must be pushed. Treating `v` as a
fresh Lebesgue state while multiplying by the pushed `J_v` would define a
different target.

The normalized residual `r=c/lambda=a/lambda-u` is equivalent only when its
residual metric is pushed to `R_r=lambda^2 I`:

```text
log|det D_u r| + 0.5 log det(R_r)
= 0 + n log(lambda)
= log|det D_u c|.
```

Leaving the residual metric equal to identity changes the model and is
reported as a deliberate failure control.

## Independent geometry audit

At the first rolling origin, a scrambled-Sobol integration evaluates the
normalized corrected finite tube over the complete positive state domain at
25 points spanning the central 99% `eta` region. The locked full route uses:

- `rho=10^6`;
- four independent scrambles;
- 16,384 Sobol points per scramble; and
- seed `20260711`.

Maximum centered log error must be below 1% of determinant variation, with QMC
spread reported separately. The audit also verifies the log/direct state
pushforward, residual metric, `lambda`-to-`eta` density transformation, global
tails, and the public `detcorr>=0.10` diagonal Jacobian.

Omitting `J/q` shifts rolling `eta` by a mean `-2.744` graph-posterior SD,
worsens cumulative log score by `3.416`, and raises mean CRPS by `0.02094`.
All geometry gates pass, giving `FU-GEOM=SUPPORTED/QUANTIFIED`. The determinant
is a measure factor, not an economic mechanism or causal variable.

## Non-circularity and provenance

`pre_score_predictions.csv` is written and hashed at each origin before that
origin's price is read for scoring. Its hashes are copied into the final
rolling prediction table. `role_manifest.csv` records every origin/source-row
role and deterministic split. Claim evaluations record protocol, data,
configuration, source, and evidence-artifact hashes.

Quick and full modes use identical rows, models, and endpoints. Full mode uses
4,001 rather than 1,601 verification-grid points and 5,000 rather than 500
moving-block resamples. Only the locked full bundled-fixture run carries
terminal claim status.

[`REGISTERED_PROTOCOL.json`](REGISTERED_PROTOCOL.json) is the machine-readable
lock. Code, data, manifests, locked protocols, environments, and generated
evidence are hash-bound; `README.md` and `TECHNICAL_DETAILS.md` summarize that
evidence.

## Output contract

Every quick or full run writes strict `summary.json`, terminal claim records
under `claims/`, CSV evidence under `tables/`, and four figures under
`figures/`. The fixed surface is:

```text
claims/claim_evaluations.json
tables/{aggregate_predictive_metrics,claim_evaluations,
        initial_reference_residual_acf,
        initial_reference_residual_diagnostics,
        matched_calibration_effects,paired_score_uncertainty,
        phi_clip_sensitivity,pre_score_predictions,
        primary_model_geometry_audit,residual_pushforward_audit,
        role_manifest,rolling_predictions,verification_spine}.csv
figures/{cumulative_log_score_contrasts,initial_reference_residual_acf,
         rolling_eta_volume_effect,rolling_predictions}.png
```

`summary.json` rejects non-finite JSON values and records mode, fixture status,
environment, protocol/configuration/source/data hashes, implementation gates,
and `FU-PRED`, `FU-DEP`, and `FU-GEOM` evaluations. Scientific outcomes are
reported rather than converted into software pass/fail conditions.
`rolling_predictions.csv` contains one row per model and test origin;
`pre_score_predictions.csv` is written and hashed before each outcome is read;
and `role_manifest.csv` makes every temporal assignment inspectable. Null
posterior fields are expected for fixed baselines. Output directories are
immutable by default; `--force` performs an intentional whole-directory
replacement and removes stale products first.

## External data contract

The `--data PATH` route accepts a longer same-schema ordered series without
source edits. It requires at least 65 observations and exactly the analysis
columns `rownames`, `avgprc`, `wave2`, `speed2`, `mon`, `tues`, `wed`,
`thurs`, and `t` (additional columns may be retained). Required values must be
finite and complete, `avgprc` must be positive, `rownames` unique, and `t`
strictly increasing. Rows are not silently sorted, dropped, or imputed. The
selected path, SHA256, dimensions, and time range are recorded, and any input
whose hash or row count differs from the registered fixture is labelled
`exploratory_schema_compatible_external_input`.

The same temporal rule is then applied without tuning: the first test is
position 65, the preceding 16 observations calibrate the scale, all earlier
observations fit the reference, and every subsequent outcome is scored once at
horizon one.

A second market or differently defined price series requires a new registered
adapter documenting price definition, sampling interval, covariate timing,
missingness, units, calendar, provenance, and estimand. Renaming columns is not
sufficient. Prospective status requires the adapter, protocol, and hashes to be
frozen before outcomes are obtained.

## Supported scope and extension requirements

The defensible results are the registered conditional predictive comparison,
the demonstrated sensitivity to dependence treatment, and the quantified
target-measure consequence. They are conditional on the plug-in reference,
plug-in dependence parameters, 16-row calibration window, contemporaneous
covariates, and one market series. A structural economic extension
would need joint nuisance propagation, a stated market model, suitable
instruments or another causal identification strategy, and genuinely new
market/time coverage before estimating supply, demand, elasticity, or market
power.
