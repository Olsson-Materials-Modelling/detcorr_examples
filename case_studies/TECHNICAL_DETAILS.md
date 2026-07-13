# Scientific case studies: technical details

[Scientific overview](README.md) ·
[Data rights](THIRD_PARTY_NOTICES.md)

This document gathers the protocol, measure, provenance, and verification
details behind the public scientific overview. The generated JSON and CSV
artifacts remain authoritative for numerical values and terminal claim status.

## Repository role

The complete distribution has two independently verified tiers:

- a tutorial tier contains ten worked demonstrators of target-measure
  correction; and
- `case_studies/` tests three application-facing conclusions using grouped or
  temporal holdout, explicit discrepancy models, negative controls, influence
  checks, and coordinate/measure audits.

The Matbench and screened-lattice cases remain demonstrators. They are not
registered empirical studies and receive no scientific claim IDs.

## Measure contract

For a constraint `c(theta,u)=0`, a finite tube around the constraint can induce
a parameter measure different from the declared graph measure. If the state
base measure has density `q` in the displayed coordinate and the residual
Jacobian contributes `J`, the required hard-limit multiplier relative to that
base is `J/q`, not `J` in isolation.

A genuine state-coordinate pushforward transforms the state base density,
residual Jacobian, likelihood, residual metric, and integration element
together. The ratio `J/q` and the complete finite tube are then invariant.
Declaring Lebesgue measure independently in two coordinates instead defines
two different finite tubes; each is required to recover the graph target only
after its own determinant correction.

For a genuine state reparameterisation `v=h_theta(u)`, the state density is
pushed as

```text
q_v(v|theta) = q_u(h_theta^-1(v)|theta)
               |det D_v h_theta^-1(v)|.
```

The residual, residual metric, and state-integration element are transformed
with it. For `c'=A(theta)c`, equivalence requires
`R'=A^-T R A^-1`; changing `c` while leaving `R` fixed defines a different
finite tube. A parameter transformation `phi=T(theta)` similarly requires the
usual density Jacobian with respect to `dphi`. Direct, log, and reciprocal
state routes are compared through their induced probability measures, using
centred log densities when coordinates introduce parameter-independent
constants. Physical quantities are nondimensionalised before entering an
exponential or logarithm, and unit changes are checked through the response-
density Jacobian.

Target-measure correctness and predictive performance are deliberately
separate. Prediction is never used to prove the determinant identity, and the
determinant identity is never used to claim that a scientific model is
adequate. An omitted-volume model may occasionally score well by compensating
for misspecification; that does not make it the requested target.

## Locked scientific designs

The repository-level analysis choices, timing class, evidence ceiling,
geometry rules, active studies, and claim IDs are fixed in the machine-readable
[`PROTOCOL_LOCK.json`](PROTOCOL_LOCK.json). The active entry points are fixed
in [`STUDY_REGISTRY.json`](STUDY_REGISTRY.json). Fulton additionally has a
study-level [`REGISTERED_PROTOCOL.json`](studies/fulton_market/REGISTERED_PROTOCOL.json)
because its temporal roles and dependence model require a more detailed lock.
Human-readable implementations are recorded beside each study:

- [bulk-modulus technical details](studies/jarvis_bulk_modulus/TECHNICAL_DETAILS.md);
- [positive-gap technical details](studies/jarvis_band_gaps/TECHNICAL_DETAILS.md);
- [Fulton technical details](studies/fulton_market/TECHNICAL_DETAILS.md).

### JARVIS studies

The two JARVIS primaries compare exactly two predictive routes:

- `graph_primary`: a deterministic trapezoidal-grid Student-`t4` common scale
  in log property ratio; and
- `identity`: the same discrepancy model with the scale fixed to one.

Leave-one-formula-group-out evaluation keeps all polymorphs of one formula on
the same side of a fold. The full run uses 2,000 formula-cluster bootstrap
resamples and 2,000 formula-pairing permutations with seed `20260711`.
Predictive medians and intervals come from exact inversion of the finite
posterior-mixture CDF; no paired Monte Carlo quantile sample is used.

A separate direct identity/scale/affine ladder screens one-scale adequacy. Its
Gaussian support extends into nonphysical negative properties, so it is a
diagnostic rather than the positive-property generative model. It qualifies
the common-scale interpretation but is not substituted for the registered
primary.

The positive-gap study also repeats the complete held-out comparison at fixed
calculated-gap thresholds `0`, `0.25`, `0.50`, and `1.00 eV`. These are
retrospectively specified nested subsets: all four are reported, the full
fixture remains primary, and no threshold selects or upgrades the primary
claim. `JG-THRESHOLD` depends on threshold-specific predictive comparisons and
locked bootstrap intervals, not posterior movement alone.

### Fulton study

The Fulton primary supplies 33 rolling one-step conditional predictions. At
each origin:

```text
rows before t-16       fit the reduced log-price reference
rows t-16 through t-1  calibrate eta = log(lambda)
row t                  remains untouched and is scored once
```

The primary covariance is a plug-in stationary AR(1) estimated only from that
origin's reference-training residuals. Its declared coefficient bound is
`[-0.90,0.90]`. Matched identity, iid, coefficient-covariance, lag-four
negative-control, and last-price persistence comparisons are reported.
Paired score differences use a 90% circular moving-block interval with block
length four and seed `20260711`.

A post-outcome, non-selecting sensitivity repeats only the AR-coefficient bound
at `0.50`, `0.75`, `0.90`, and `0.99`. Because outcomes were already known,
this sensitivity can qualify but cannot upgrade the primary result. The
registered `0.90` bound remains primary regardless of the alternatives.

## Claim and evidence interpretation

The locked full reference run reports:

| Claim | Terminal status | Meaning |
|---|---|---|
| `JB-PRED` | `NOT_SUPPORTED` | The bulk common scale did not improve aggregate formula-LOGO log predictive density over identity. |
| `JB-GEOM` | `SUPPORTED` | The bulk coordinate and measure routes recover the declared target. |
| `JG-PRED` | `MIXED` | Positive-gap grouped prediction improves over identity, but the affine adequacy screen qualifies a common-scale interpretation. |
| `JG-THRESHOLD` | `SUPPORTED` | Four retrospective nested threshold subsets preserve the full-fixture predictive direction with bootstrap support. |
| `JG-GEOM` | `SUPPORTED` | The active coercive positive-gap routes satisfy the measure contract. |
| `FU-PRED` | `NOT_SUPPORTED` | The primary AR(1) rolling scale effect does not pass the registered predictive gate. |
| `FU-DEP` | `SUPPORTED / DEPENDENCE_SENSITIVE` | The matched iid and AR(1) conclusions differ materially. |
| `FU-GEOM` | `SUPPORTED / QUANTIFIED` | Omitting the declared state-volume factor measurably changes the rolling target and predictions. |

The evidence labels used by the generated claim records mean:

- **implementation**: the code, measure identity, and numerical route pass
  their declared checks;
- **E1 fixture-conditional**: an inference is supported on the checksum-pinned
  retrospective fixture under the declared model;
- **E2 conditional**: held-out temporal scoring is available, but the result
  remains conditional on the registered plug-in reference and dependence
  treatment; and
- **prospective/domain-general**: not claimed by this release; these require
  outcome-unseen data acquisition and independent population coverage.

Every terminal status is produced from a fixed conjunction of predictive,
uncertainty, leakage, support, numerical, influence, and negative-control gates
described in the relevant study technical document. A favourable point metric
cannot override a failed registered gate, and null, adverse, or mixed results
remain visible. The bundled JARVIS outputs are capped at E1 because the fixture
and implementation outputs were viewed before protocol 0.2 was narrowed.
Fulton prediction may reach only the conditional E2 level stated in
`PROTOCOL_LOCK.json`.

## Evidence scope and protocol timing

These are registered retrospective reanalyses rather than blinded
preregistrations: outcomes were available when the final protocols were fixed.
`PROTOCOL_LOCK.json` records protocol version `0.2-retrospective`, the
exposure-aware timing class, and each evidence ceiling. Post-outcome checks are
non-selecting: they may qualify a conclusion but cannot upgrade its status or
replace the registered primary.

The active claim-bearing JARVIS routes use the globally coercive log-property
residual fixed in `PROTOCOL_LOCK.json` and detailed in the two JARVIS technical
documents. The rejected reciprocal finite-penalty construction remains a
diagnostic because its full-state integral is improper; it is not used for
prediction.

## Numerical and leakage audits

The full scientific verifier checks, as applicable:

- checksum-pinned fixtures and explicit row-role manifests;
- no training/held-out overlap within formula or temporal folds;
- a held-out-outcome perturbation sentinel;
- analytic global-tail certificates and broad-grid support checks;
- adjacent-grid agreement for posterior, score, quantile, coverage, and ELPD
  endpoints;
- physical-density unit-change Jacobians;
- complete coordinate and residual-metric pushforwards;
- independently evaluated corrected hard limits;
- deliberate double-counting controls that must fail; and
- hashes connecting each claim to its tables, figures, protocol, and source.

For claim-bearing JARVIS runs, log scores, predictive log-ratio quantiles, and
aggregate ELPD must agree across adjacent grids within `1e-10`; coverage
decisions must be identical, and posterior log-scale quantiles must agree
within `0.001`. The support boxes, thresholds, gate values, and route contracts
are stated in the two JARVIS technical documents and emitted again in the
generated support, refinement, role, and geometry audit tables.

The Fulton geometry audit independently evaluates the normalized corrected
finite tube on the complete positive state domain using scrambled Sobol points.
It also verifies parameter-density and residual-metric pushforwards and the
public `detcorr>=0.10` callable Jacobian.

## Computational provenance

The hash-bound inputs are `PROTOCOL_LOCK.json`, `STUDY_REGISTRY.json`, Fulton’s
`REGISTERED_PROTOCOL.json`, fixture data and manifests, executable analysis
code, environment constraints, and shared numerical code. Generated
`summary.json`, CSV tables, figures, and claim evaluations are the authoritative
result surface. `README.md` and `TECHNICAL_DETAILS.md` summarize that evidence
and are excluded from the computational hashes. Verification fails if a
hash-bound input or generated artifact differs from the recorded manifest.

## Data provenance and reuse

[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) records attribution,
redistribution rights, source links, and change descriptions. The JARVIS gap
table is attributed under CC BY 4.0; the JARVIS bulk table is distributed under
NIST's public-information permission and disclaimer; and the Fulton fixture is
conveyed separately under GPL-3 to the extent the `wooldridge` package licence
applies, with the underlying Rdatasets rights caveat retained.

The bundled fixture manifests are the reusable data contract. They record the
source and citation, retrieval method and time, raw and processed identities,
SHA256 and byte count, row/group keys, observation and reference-input columns,
units, selection and deduplication rules, known dependence, transformation,
protocol version, rights basis, terms-check date, redistribution decision, and
third-party notice. The JARVIS runners reconcile the selected file's hash,
name, bytes, rows, groups, declared columns, and units against the supplied
manifest and reject mismatches. No study downloads, joins, sorts, imputes, or
drops external records silently.

Source-level deduplication must precede any expanded-data run: a larger table
is not independent evidence if it repeats the same experimental records. A
prospective extension must freeze its adapter, manifest, roles, protocol, and
hashes before the evaluation outcomes are obtained. Otherwise it is reported
as exploratory regardless of sample size.

## Outputs and provenance

Each study writes outside its source directory and produces:

- a strict `summary.json` with terminal claim evaluations;
- row-level predictions and score tables;
- role, leakage, support, geometry, sensitivity, and negative-control audits;
- evidence figures; and
- hashes connecting every claim to its source and artifacts.

The repository-level `run_manifest.json` binds the execution mode, dependency
versions, protocol hashes, shared computational inputs, study inputs, and every
generated output. Output directories are immutable by default; intentional
replacement requires `--force`.

The release includes a checksum-bound full run at `reference_results/`.
Ad-hoc and custom-data runs should use a different output root.

## Verification routes

Use a separate environment for the scientific subproject. The exact-artifact
constraints define the reference reproduction stack; broader `pyproject.toml`
ranges are supported for ordinary development. The public package source is
[Olsson-Materials-Modelling/detcorr](https://github.com/Olsson-Materials-Modelling/detcorr).

```bash
cd case_studies
python -m venv .venv
. .venv/bin/activate
python -m pip install \
  "detcorr @ git+https://github.com/Olsson-Materials-Modelling/detcorr.git@a414f063515359f70952e2717a506b75e69e4b0b"
python -m pip install -c constraints-publisher.txt -e '.[test]'
python scripts/environment_report.py --output environment-report.json
```

The pinned Git install is required for a claim-bearing full run and is recorded
with the installed source-file hashes in `run_manifest.json`. A compatible
adjacent checkout may be used for non-claim quick verification and tests.

The released verification routes are:

```bash
# Verify tests and the committed locked full payload
make test
make verify-reference

# Run and verify the CI-sized non-claim implementation path
make verify-quick

# Recompute and verify the full pinned-fixture analysis
make verify
```

The quick mode changes numerical resolution and resampling counts but not the
model, split, endpoint, or claim rule. It is an implementation check and is
never claim-bearing. Only the full mode on the checksum-pinned bundled fixture
with its locked resampling configuration carries the registered claim status.

On the reference one-CPU environment, `make test` took about 11 minutes and
`make full` took about 30 minutes. `make verify` performs both stages, validates
the committed and fresh run manifests, and byte-compares every scientific
artifact in the two result trees. The run manifests are checked independently
because their runtime inventories intentionally describe different executing
machines. The full runner processes bulk modulus, positive gaps, and Fulton
sequentially. JARVIS resampling can run for several minutes with little console
output; a completion message is printed after each study. These timings are
planning guidance rather than acceptance thresholds and will vary with hardware
and numerical libraries.

For a shared full result, retain `environment-report.json`, the console log,
`run_manifest.json`, every study summary, and the generated evidence files.
Output roots are immutable unless an intentional replacement uses `--force`.

Each study accepts an explicit compatible external table under the schema in
its `TECHNICAL_DETAILS.md`. Preserve the raw file, checksum, provenance, row
identities, rights decision, and completed manifest. A new dataset remains
exploratory unless its protocol, adapter, data identity, and evaluation roles
were frozen before outcomes were observed. A larger table is not independent
validation until source-level deduplication excludes shared experimental
records.

The standalone
[`full-analysis` workflow](.github/workflows/full-analysis.yml) retains the
environment report and full output as a GitHub Actions artifact. In a hosted
copy, open **Actions**, select **Full registered analysis**, choose **Run
workflow**, and download the `registered-analysis-<commit>` artifact after the
job completes. The workflow has a 120-minute timeout and downloads no mutable
scientific data at run time.

The root worked-example gate verifies the sibling tutorials,
`make verify-reference` verifies the committed scientific payload, and
`make verify` additionally proves that a fresh full run reproduces its
scientific artifact surface.

## Supported scope and extensions

The implemented designs support precise statements about declared constrained
measures, fixture-conditional calibration, grouped predictive comparisons,
temporal predictive sensitivity, model adequacy, and numerical consequences.
Extending those results to a universal materials correction, constitutive law,
causal market effect, elasticity, market-power coefficient, or an unobserved
population requires a new domain model, independently governed data, a
prospective protocol, and the relevant physical or causal identification
strategy.
