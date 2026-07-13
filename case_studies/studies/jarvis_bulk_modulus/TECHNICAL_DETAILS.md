# JARVIS bulk modulus: technical details

[Study overview](README.md) ·
[Case-study technical contract](../../TECHNICAL_DETAILS.md)

This document consolidates the method, protocol locks, audit surface,
provenance, and limitations behind the concise study overview. Generated
`summary.json` and CSV tables remain authoritative.

## Evidence status

Protocol `0.2-retrospective` was finalized after the public fixture and
implementation outputs had been inspected. The checksum-pinned locked
`--full` run therefore supports retrospective E1 fixture-conditional evidence.
Quick, standard, custom-input, or resampling-overridden runs are
`EXPLORATORY`; prospective confirmation requires a new outcome-unseen lock.

## Fixture and roles

The fixture is a pinned 36-row transcription of the NIST JARVIS-DFT
OPT/experimental bulk-modulus comparison table:

- processed SHA256:
  `6d9c37c52ef1d7eeb62d0cce3bc00f5b4c979b4b5d2b3bdcfcda8ab6a73cae86`;
- 35 normalized chemical-formula groups;
- formula is the outer evaluation unit; and
- every polymorph of one formula is held out together.

The raw upstream HTML snapshot and row-level experimental source metadata were
not retained. The processed transcription, retrieval record, source URL,
rights basis, grouping rule, and hashes are fixed in
[`data/FIXTURE_MANIFEST.json`](data/FIXTURE_MANIFEST.json). Redistribution and
NIST attribution are covered by
[`../../THIRD_PARTY_NOTICES.md`](../../THIRD_PARTY_NOTICES.md).

## Primary model

For calculated modulus `x=B_OPT`, experimental modulus `y=B_expt`, and
`mu=log(gamma)`:

```text
log(y/x) = mu + epsilon
epsilon ~ Student-t(df=4, location=0, scale=tau)
mu ~ Normal(0, 0.35^2)
log(tau) ~ Normal(log(0.12), 0.75^2)
```

The density is with respect to `dmu dlog(tau)`. `gamma=exp(mu)` is the sole
scientific posterior target (`graph_primary`). Identity fixes `mu=0` and uses
the same discrepancy model. No naive determinant tube is used as a predictive
competitor.

Because a log-Student-`t` predictive distribution has no finite raw-scale
mean, point predictions are posterior predictive medians. Exact finite-
mixture CDF inversion provides the medians and 5/25/75/95% quantiles in log-
ratio space. Reported endpoints are transformed to GPa, with the physical-
density Jacobian included in log predictive density.

## Locked evaluation

The full claim-bearing route fixes:

- formula-LOGO roles recorded for every row and fold;
- 2,000 formula-cluster bootstrap draws, seed `20260711`;
- 2,000 formula-level pairing permutations, seed `20260711`;
- aggregate and formula-level graph-minus-identity ELPD;
- median-point RMSE and MAE;
- central 50% and 90% coverage and widths; and
- leave-one-formula influence in full-posterior SD units.

Held-out outcomes are used only for scoring. A perturbation sentinel changes
every held-out response and requires the fitted fold and predictive quantiles
to remain unchanged. The role manifest verifies disjoint training/held-out
sets, exact complements, and one holdout per row.

The separate physical-GPa Gaussian identity/scale/affine ladder screens
one-scale adequacy. Its support is unbounded and gives negative 5% endpoints
for `7/36` rows in each route. It is therefore a disclosed diagnostic, not a
positive-modulus generative model or replacement primary.

## Claim rules and status

`JB-PRED` can be `SUPPORTED` only when aggregate ELPD, the formula-bootstrap
lower endpoint, the median formula contribution, RMSE, MAE, coverage,
influence, pairing-control, direct-scale, role, support, Jacobian, and grid
gates all pass. Non-positive aggregate ELPD forces `NOT_SUPPORTED`, regardless
of point-error improvement.

The reference result is:

- aggregate Delta ELPD `-0.6467295486`;
- 90% formula-bootstrap interval `[-2.067857413, 0.784835509]`;
- graph RMSE/MAE `11.412/8.067 GPa` versus identity `11.720/8.506 GPa`;
- graph/identity 90% coverage `0.944/0.917`; and
- `JB-PRED=NOT_SUPPORTED`.

The largest absolute formula contribution is `V`, only `0.0494` of the total
absolute formula-level Delta ELPD; lack of case concentration does not establish
transport to another formula population.

## Geometry audit

The fixed geometry audit is separate from prediction. It uses a direct-GPa
Gaussian likelihood with
`sigma=sqrt(3^2+(0.055 B_OPT)^2)` and a Gaussian prior on `mu`. Its routes are:

- log coordinate `w=log(Y/X0)`: `q=1`, `J=1`;
- genuine direct pushforward `s=Y/X0`: `q=1/s`, `J=1/s`;
- genuine reciprocal pushforward `v=X0/Y`: `q=1/v`, `J=1/v`; and
- independently based direct/reciprocal Lebesgue states with `q=1`, compared
  with the reference only after their own `J/q` correction in the hard limit.

Complete genuine finite tubes are independently integrated at 25 `gamma`
points over the graph target's 0.5%–99.5% region. Corrected hard routes must
recover the graph pointwise; a pushed residual metric must remain invariant;
and deliberate Jacobian double counting must fail. The public `detcorr>=0.10`
callable Jacobian must match the independent analytic value.

The reference centered direct/log and reciprocal/log errors are each below
`1.05e-11`; corrected hard-versus-graph error is zero at reported precision.
All geometry gates pass, giving `JB-GEOM=SUPPORTED`. This is a measure result,
not evidence for a physical mechanism or a common correction factor.

A uniform global audit over `mu in [-120,20]` records modes, endpoint log
drops, edge mass, and local-window mass for the graph and independently based
uncorrected direct/reciprocal hard targets. This prevents a locally plausible
geometry curve from passing while material posterior mass lies outside the
reported window.

## Support and numerical gates

Gaussian priors provide full-support tail control. Every graph and identity
full/LOGO fit is repeated on the broad box `mu in [-3,3]` and
`eta=log(tau) in [log(0.001),log(3)]`. The mode must remain in the local
quadrature grid, mass outside it must be below `1e-4`, broad edge mass below
`1e-10`, and minimum boundary-profile log drop above 30.

Every claim-relevant posterior, row log score, exact predictive quantile,
coverage decision, and aggregate ELPD is recomputed on an adjacent grid. Full-
run log scores, predictive log-ratio quantiles, and aggregate ELPD must agree
within `1e-10`; coverage decisions must be identical; and posterior log-scale
quantiles must agree within `0.001`.

The physical-density audit rescales both input and response and requires graph
and identity GPa log density to shift by exactly `-log(scale)`.

## Declared robustness analyses

The sensitivity surface is fixed rather than selected from the result:

- Student-`t4` versus Gaussian log-ratio discrepancy;
- `mu` prior SD `0.20`, `0.35` (primary), and `0.60`;
- direct-GPa normal and Student-`t4` likelihoods with
  `sigma=sqrt(3^2+(0.055 B_OPT)^2)`;
- a wider direct-GPa normal likelihood with
  `sigma=sqrt(5^2+(0.08 B_OPT)^2)`; and
- physical-GPa identity, scale, and affine prediction.

These diagnose robustness and one-scale adequacy; they do not replace the
registered Student-`t4` log-ratio primary after outcomes are viewed.

## Outputs and provenance

Every run writes strict `summary.json`, CSV evidence under `tables/`, and PNG
figures under `figures/`. The scientific surface contains:

- `posterior_targets.csv`, `grouped_predictions.csv`, and
  `predictive_scores.csv` for the sole graph posterior and matched held-out
  graph/identity comparison;
- `predictive_group_elpd_contributions.csv` and
  `predictive_formula_cluster_bootstrap.csv` for formula-level contributions
  and the locked interval;
- `logo_role_manifest.csv`, `logo_role_validation.csv`, and
  `heldout_outcome_leakage_audit.csv` for split and leakage evidence;
- `primary_grid_support_audit.csv`,
  `predictive_grid_refinement_{gates,details}.csv`, and
  `physical_density_jacobian_audit.csv` for support, convergence, and units;
- `direct_model_ladder_{predictions,scores}.csv`,
  `influence_and_residuals.csv`, `sensitivity.csv`, and
  `pairing_permutation_distribution.csv` for adequacy and robustness; and
- `geometry_route_contract.csv`, `geometry_finite_pushforward_audit.csv`,
  `geometry_hard_limit_audit.csv`, `geometry_global_support.csv`, and
  `detcorr_api_audit.csv` for the independent measure audit.

`claim_evaluations.csv` is the compact `JB-PRED`/`JB-GEOM` index. The three
figures are `grouped_predictions.png`, `pairing_permutation_control.png`, and
`posterior_targets.png`; the last figure depicts geometry target/support
routes, not a second scientific posterior. `summary.json`
rejects non-finite JSON and records every gate, data/configuration/source hash,
resampling setting, limitation, and claim-to-artifact hash. CSV `NaN` is used
only where a quantity is structurally inapplicable. The runner refuses to
overwrite an existing output directory unless `--force` requests an auditable
whole-directory replacement.

The immutable scientific choices are in
[`../../PROTOCOL_LOCK.json`](../../PROTOCOL_LOCK.json). Code, data, the fixture
manifest, protocol lock, environment constraints, and generated evidence are
hash-bound; `README.md` and `TECHNICAL_DETAILS.md` summarize that evidence.

## External data contract

Custom data are accepted only through explicit `--data` and `--data-manifest`
arguments. The CSV must contain positive finite `bulk_modulus_opt_gpa` and
`bulk_modulus_expt_gpa`, a unique `jvasp_id`, and a nonempty `formula`.
Formula is the outer-validation group, so all polymorphs sharing a formula are
held out together. Source-level deduplication must be completed before the run.

Copy [`data/FIXTURE_MANIFEST.json`](data/FIXTURE_MANIFEST.json) as the manifest
template and replace its source, retrieval, raw/processed identity, hashes,
bytes, row/group counts, selection, deduplication, dependence, transformation,
rights, and redistribution fields with truthful records for the selected
table. The required property declarations are exactly:

```json
{
  "dependent_observation_columns": ["bulk_modulus_expt_gpa"],
  "reference_input_columns": ["bulk_modulus_opt_gpa"],
  "units": {
    "bulk_modulus_opt_gpa": "GPa",
    "bulk_modulus_expt_gpa": "GPa"
  }
}
```

At runtime the runner reconciles the manifest with the selected file's SHA256,
name, byte count, row count, formula-group count, column roles, and units. It
rejects incomplete, blank, or inconsistent records and does not download or
join data. A replacement table must state its own rights basis; the NIST notice
for the bundled fixture does not license it.

For scientific reuse, also retain original record identifiers, experimental
references, phase, temperature, measurement method, reported uncertainty, and
calculation settings where available; record absence explicitly. A compatible
custom run is exploratory and does not inherit the fixture's claim status.

```bash
python run_analysis.py \
  --data expanded.csv \
  --data-manifest expanded.manifest.json \
  --full
```

## Supported scope and extension requirements

The results support fixture-conditional predictive calibration and
coordinate/measure consistency under the declared models. Broader materials
use requires source-level experimental uncertainty, temperature, phase,
measurement protocol, calculation settings, and independent formula/database
coverage. A positive-property model should replace the Gaussian adequacy
ladder before its nonphysical negative tails become claim-bearing, and the
pairing permutation should remain a negative-control diagnostic unless a
defensible exchangeability model is supplied. Those additions are required
before estimating a transferable functional-wide bulk-modulus correction.
