# JARVIS positive band gaps: technical details

[Study overview](README.md) ·
[Case-study technical contract](../../TECHNICAL_DETAILS.md)

This document consolidates the method, protocol locks, audit surface,
provenance, rejected-route treatment, and limitations behind the concise study
overview. Generated `summary.json` and CSV tables remain authoritative.

## Evidence status

Protocol `0.2-retrospective` was finalized after the public fixture and
implementation outputs had been inspected. The checksum-pinned locked
`--full` run therefore supports retrospective E1 fixture-conditional evidence.
Quick, standard, custom-input, or resampling-overridden runs are
`EXPLORATORY`; prospective confirmation requires a new outcome-unseen lock.

## Fixture and roles

The fixture is a pinned 54-row positive-gap transcription from Table 3 of
Choudhary et al., as reproduced in the JARVIS-DFT documentation:

- processed SHA256:
  `4b7a5f437b1a41030bb65c31d56eaa9524cc91598bd09ff2825c02df60528631`;
- 50 normalized chemical-formula groups;
- all retained calculated and tabulated experimental values are finite and
  strictly positive; and
- all polymorphs of AlN, CdS, GaN, and TiO2 are held out together.

The upstream snapshot and row-level experimental citations were not retained,
and the table does not resolve fundamental versus optical experimental gaps
row by row. The processed transcription, source, adaptation, CC BY 4.0 notice,
roles, and hashes are fixed in
[`data/FIXTURE_MANIFEST.json`](data/FIXTURE_MANIFEST.json) and
[`../../THIRD_PARTY_NOTICES.md`](../../THIRD_PARTY_NOTICES.md).

## Primary model

For positive calculated gap `x=E_OPT`, tabulated experimental gap `y=E_expt`,
and `mu=log(gamma)`:

```text
log(y/x) = mu + epsilon
epsilon ~ Student-t(df=4, location=0, scale=tau)
mu ~ Normal(0, 0.75^2)
log(tau) ~ Normal(log(0.45), 0.75^2)
```

The density is with respect to `dmu dlog(tau)`. `gamma=exp(mu)` is the sole
scientific target (`graph_primary`); identity fixes `mu=0` under the same
discrepancy model. No naive geometry route is scored predictively.

The log-Student-`t` predictive distribution has no finite raw-scale mean, so
point predictions are posterior predictive medians. Exact finite-mixture CDF
inversion gives medians and 5/25/75/95% quantiles in log-ratio space. Physical-
eV log densities include the required response-density Jacobian.

## Locked evaluation

The full claim-bearing route fixes:

- formula-LOGO roles for every row and fold;
- 2,000 formula-cluster bootstrap draws, seed `20260711`;
- 2,000 formula-level pairing permutations, seed `20260711`;
- aggregate and formula-level graph-minus-identity ELPD;
- median-point RMSE and MAE;
- central 50% and 90% coverage and widths; and
- leave-one-formula influence in full-posterior SD units.

Held-out outcomes enter scoring only. A perturbation sentinel changes every
held-out response and requires fitted folds and predictive quantiles to remain
unchanged. Role manifests verify disjoint sets, exact complements, and one
holdout per row.

A physical-eV Gaussian identity/scale/affine ladder separately screens model
adequacy. Its unbounded support yields negative 5% predictive endpoints in
`42/54` identity, `17/54` scale, and `7/54` affine rows. It is a diagnostic,
not a positive-gap generative model or a replacement primary.

## Retrospective nested threshold subsets

The complete graph-versus-identity LOGO analysis is repeated after applying
`E_OPT >= 0, 0.25, 0.50, 1.00 eV`. Every threshold has its own:

- row predictions and predictive scores;
- formula contributions and 2,000-draw cluster bootstrap;
- posterior scale and retained row/formula counts;
- role and leakage manifests;
- broad-support audit; and
- adjacent-grid numerical audit.

`JG-THRESHOLD` depends on the sign of complete threshold-specific held-out Delta
ELPD and its locked bootstrap interval. Posterior-scale movement alone cannot
pass the claim. The thresholds are protocol-locked retrospective nested
subsets. All four are reported, the full fixture remains primary, and no
threshold selects or upgrades the primary claim. They are not
independent-domain replications.

Reference Delta ELPD values at `0`, `0.25`, `0.50`, and `1.00 eV` are
`28.654`, `35.962`, `39.823`, and `31.985`; their respective 90% intervals are
`[18.591,37.782]`, `[26.254,44.940]`, `[31.600,47.259]`, and
`[24.252,39.402]`. Thus `JG-THRESHOLD=SUPPORTED` as a bounded within-fixture
nested-threshold sensitivity result.

## Predictive claim rule and result

`JG-PRED` requires positive aggregate and median formula ELPD, a positive
bootstrap lower endpoint, improved RMSE and MAE, acceptable 90% coverage,
bounded formula influence, successful pairing control, direct scale better
than identity, and all role, support, density-Jacobian, and grid gates. A
positive scale-versus-identity result remains `MIXED` if the affine screen is
better.

The reference result is:

- aggregate Delta ELPD `28.6539951922`;
- 90% formula-bootstrap interval `[18.5912071164, 37.7820091240]`;
- graph RMSE/MAE `1.013/0.755 eV` versus identity `1.618/1.329 eV`;
- direct affine RMSE `0.729 eV` versus direct scale `0.835 eV`; and
- `JG-PRED=MIXED`.

The largest absolute formula contribution is GaN, `0.0479` of total absolute
formula-level Delta ELPD, below the locked concentration threshold. That does
not establish transport to another formula population.

## Geometry and rejected route

The fixed geometry audit uses a direct-energy Gaussian likelihood with
`sigma=sqrt(0.12^2+(0.15 E_OPT)^2)` and a Gaussian prior on `mu`. Its active
coercive routes are:

- log coordinate `w=log(Y/X0)`: `q=1`, `J=1`;
- genuine direct pushforward `s=Y/X0`: `q=1/s`, `J=1/s`;
- genuine reciprocal pushforward `v=X0/Y`: `q=1/v`, `J=1/v`; and
- independently based direct/reciprocal Lebesgue states with `q=1`, compared
  with the reference only after their own `J/q` correction in the hard limit.

Complete genuine finite tubes are independently integrated at 25 `gamma`
points. Corrected hard routes must recover the graph pointwise; pushed
residual metrics must remain invariant; deliberate double counting must fail;
and the `detcorr>=0.10` callable Jacobian must match an independent analytic
value.

The reciprocal construction `u=E0/E`, `c=1/u-gamma*x` under independent
Lebesgue `du` is rejected before inference because its finite-penalty state
integral diverges as `u` tends to infinity. Under the extension geometry
likelihood and prior, its computed formal naive hard mode is `mu=-60.75`; the
corrected worked demonstrator uses a different prior and has a local formal
target. Neither direction repairs the improper finite tube, and neither is a
claim gate. No truncated posterior, moment, or predictive score is reported.

Reference finite-pushforward errors are below `3.45e-9`, corrected hard-
versus-graph error is zero at reported precision, and all active geometry and
rejected-path gates pass. Thus `JG-GEOM=SUPPORTED` as a measure result.

A uniform global audit over `mu in [-120,20]` records modes, endpoint log
drops, edge mass, and local-window mass for the graph and independently based
uncorrected direct/reciprocal hard targets. The rejected reciprocal route is
therefore assessed globally rather than inferred from a local plotting range.

## Support and numerical gates

Every graph and identity full/LOGO fit, including all threshold fits, is
repeated on `mu in [-4,4]` and
`eta=log(tau) in [log(0.003),log(10)]`. The mode must remain in the local grid,
mass outside it below `1e-4`, broad edge mass below `1e-10`, and minimum
boundary-profile log drop above 30.

Adjacent grids repeat every posterior headline, row log score, exact
predictive quantile, coverage decision, and aggregate ELPD. Claim-bearing log
scores, quantiles, and ELPD must agree within `1e-10`; coverage decisions must
be identical; and posterior log-scale quantiles must agree within `0.001`.
A physical-density audit requires the expected `-log(scale)` eV-density shift
after a known unit rescaling.

## Declared robustness analyses

The sensitivity surface is fixed rather than selected from the result:

- Student-`t4` versus Gaussian log-ratio discrepancy;
- `mu` prior SD `0.40`, `0.75` (primary), and `1.00`;
- direct-eV normal and Student-`t4` likelihoods with
  `sigma=sqrt(0.12^2+(0.15 E_OPT)^2)`;
- a wider direct-eV normal likelihood with
  `sigma=sqrt(0.25^2+(0.20 E_OPT)^2)`; and
- physical-eV identity, scale, and affine prediction.

These diagnose robustness and one-scale adequacy; they do not replace the
registered Student-`t4` log-ratio primary after outcomes are viewed.

## Outputs and provenance

Every run writes strict `summary.json`, CSV evidence under `tables/`, and PNG
figures under `figures/`. The primary scientific surface contains:

- `posterior_targets.csv`, `grouped_predictions.csv`, and
  `predictive_scores.csv` for the sole graph posterior and matched held-out
  graph/identity comparison;
- `predictive_group_elpd_contributions.csv` and
  `predictive_formula_cluster_bootstrap.csv` for formula-level evidence;
- `logo_role_manifest.csv`, `logo_role_validation.csv`, and
  `heldout_outcome_leakage_audit.csv` for split and leakage evidence;
- `primary_grid_support_audit.csv`,
  `predictive_grid_refinement_{gates,details}.csv`, and
  `physical_density_jacobian_audit.csv` for support, convergence, and units;
- `direct_model_ladder_{predictions,scores}.csv`,
  `influence_and_residuals.csv`, `sensitivity.csv`, and
  `pairing_permutation_distribution.csv` for adequacy and robustness; and
- `geometry_route_contract.csv`, `geometry_finite_pushforward_audit.csv`,
  `geometry_hard_limit_audit.csv`, `geometry_global_support.csv`,
  `rejected_reciprocal_diagnostic.csv`, and `detcorr_api_audit.csv` for the
  measure and rejected-route audit.

The nested-threshold analysis adds `opt_gap_domain_sensitivity.csv` plus complete
threshold-specific LOGO predictions, scores, formula contributions,
bootstraps, predictive evidence, support, role, leakage, and adjacent-grid
tables, all prefixed `opt_gap_threshold_`. These tables—not posterior movement
alone—carry `JG-THRESHOLD`.

`claim_evaluations.csv` is the compact `JG-PRED`/`JG-THRESHOLD`/`JG-GEOM` index.
The figures are `grouped_predictions.png`,
`pairing_permutation_control.png`, and `posterior_targets.png`; the last figure
depicts geometry target/support routes, not a second scientific
posterior. `summary.json` rejects non-finite JSON and records every gate,
data/configuration/source hash, role and resampling record, limitation, and
claim-to-artifact hash. The runner refuses to overwrite an existing output
directory unless `--force` requests an auditable whole-directory replacement.

The immutable scientific choices are in
[`../../PROTOCOL_LOCK.json`](../../PROTOCOL_LOCK.json). Code, data, the fixture
manifest, protocol lock, environment constraints, and generated evidence are
hash-bound; `README.md` and `TECHNICAL_DETAILS.md` summarize that evidence.

## External data contract

Custom data are accepted only through explicit `--data` and `--data-manifest`
arguments. The CSV must contain positive finite `bandgap_opt_ev` and
`bandgap_expt_ev`, a unique `jvasp_id`, a nonempty `formula`, and a
`material_label`. Formula is the outer-validation group, so all polymorphs of
one formula are held out together. Metals and zero/censored gaps require a new
two-part model rather than epsilon replacement or silent filtering.

Copy [`data/FIXTURE_MANIFEST.json`](data/FIXTURE_MANIFEST.json) as the manifest
template and replace its source, retrieval, raw/processed identity, hashes,
bytes, row/group counts, selection, deduplication, dependence, transformation,
rights, and redistribution fields with truthful records for the selected
table. The required property declarations are exactly:

```json
{
  "dependent_observation_columns": ["bandgap_expt_ev"],
  "reference_input_columns": ["bandgap_opt_ev"],
  "units": {
    "bandgap_opt_ev": "eV",
    "bandgap_expt_ev": "eV"
  }
}
```

At runtime the runner reconciles the manifest with the selected file's SHA256,
name, byte count, row count, formula-group count, column roles, and units. It
rejects incomplete, blank, or inconsistent records and does not download,
impute, or join data. A replacement table must state its own rights basis; CC
BY 4.0 for the bundled adapted table does not license another source.

For scientific reuse, retain calculation method and structure, experimental
source, measurement type (fundamental or optical), temperature, uncertainty,
and original identifiers where available; record absence explicitly. Do not
pool unlike gap definitions merely because the columns are numerically
compatible. A compatible custom run is exploratory and does not inherit the
fixture's claim status.

```bash
python run_analysis.py \
  --data expanded.csv \
  --data-manifest expanded.manifest.json \
  --full
```

## Supported scope and extension requirements

The results support fixture-conditional positive-gap prediction, within-fixture
sensitivity across four retrospective nested calculated-gap subsets, a
one-scale adequacy diagnosis,
and coordinate/measure consistency. Broader materials use requires data that
resolve calculation settings, experimental source, uncertainty, temperature,
and fundamental-versus-optical measurement type. Metals and zero/censored gaps
need a separately specified two-part model; transport beyond represented
formula groups needs independent population coverage. The affine ladder's
negative tails and the diagnostic (rather than exchangeability-test) role of
the pairing permutation remain part of that interpretation.
