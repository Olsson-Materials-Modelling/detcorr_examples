# Technical details

This document is the repository-wide mathematical, numerical, and
reproducibility contract behind the public [README](README.md). Individual
projects provide the same two-document split: their README states the problem,
result, and route to run it; their `TECHNICAL_DETAILS.md` records the complete
application-specific assumptions and evidence.

## 1. Target-measure contract

Let `theta` denote parameters and `u` a state constrained by

```math
c(\theta,u)=0.
```

Assume locally that the state is unique, `u=G(theta)`, and that the square
state Jacobian `D_u c(theta,G(theta))` is nonsingular. If the intended graph
target is

```math
\pi_G(\theta) \propto \pi_0(\theta)
L\!\left(y\mid\theta,G(\theta)\right),
```

then the apparently natural full-space residual tube

```math
\pi_\rho(\theta,u) \propto \pi_0(\theta)L(y\mid\theta,u)
\exp\!\left[-\frac{\rho}{2}c(\theta,u)^T R(\theta)c(\theta,u)\right]
```

does not generally marginalise to that target. Laplace/coarea reduction in
the state direction gives

```math
\int \pi_\rho(\theta,u)\,du
\sim
\left(\frac{2\pi}{\rho}\right)^{m/2}
\frac{\pi_0(\theta)L(y\mid\theta,G(\theta))}
{|\det D_u c(\theta,G(\theta))|\sqrt{\det R(\theta)}}.
```

Constants independent of `theta` disappear on normalisation; the displayed
state-volume factor does not. Relative to Lebesgue state measure, recovery of
the graph target therefore requires

```math
J_c(\theta)\sqrt{\det R(\theta)},
\qquad
J_c(\theta)=|\det D_u c(\theta,G(\theta))|.
```

Equivalently, the corrected finite-penalty log density contains

```math
\log J_c + \tfrac12\log\det R - \tfrac{\rho}{2}c^T R c.
```

For rectangular constraints, nonunique solution sets, rank changes, or a
manifold target not representable as a local graph, the relevant coarea/Gram
factor and base measure must be derived for that geometry. The square formula
must not be transplanted by analogy.

### Base measures and coordinates

The correction is relative to a declared state base measure. If that measure
has density `q_theta(u)` in the displayed coordinate, its density participates
in the reduction; the multiplier required relative to that base is
`J_c/q_theta(G(theta))`, together with the residual-metric term. A state
coordinate change must transform the state base measure, likelihood, residual,
and metric together. A parameter coordinate change likewise transforms the
parameter density. Numerical equality of density ordinates across coordinates
is not the invariance criterion; equality of the induced probability measure
is.

All quantities entering an exponential or logarithm must be dimensionless, or
be nondimensionalised against a declared reference scale. Changing physical
units requires the corresponding change in residual metric or penalty units.
An unexplained determinant of a dimensionful operator is not a complete model
declaration.

## 2. When correction applies

The determinant/metric factor applies when the intended probability density is
declared on the solved graph and a residual tube is a numerical representation
of that target. It also exposes the target implicitly induced by leaving the
factor out.

It is not an automatic adjustment for every residual:

- If `c` is itself a normalised stochastic observation or process-noise model,
  that residual likelihood may be the intended model rather than an
  approximation to a graph target.
- If a different state or manifold base measure is intended, its density must
  enter the derivation.
- A parameter-independent Jacobian and metric contribute only a normalising
  constant; the calorimetry example is the explicit zero-effect control.
- Correction restores the declared measure. It does not establish that the
  forward model, likelihood, prior, discrepancy model, or data are adequate.

The modelling decision therefore comes first: state the target, state the base
measure, and only then derive the correction.

## 3. Residual scaling and finite penalties

For a nonsingular residual transformation

```math
c'(\theta,u)=A(\theta)c(\theta,u),
```

the equivalent finite tube uses

```math
R'(\theta)=A(\theta)^{-T}R(\theta)A(\theta)^{-1}.
```

On the graph, `D_u c'=A D_u c`, so the change in the Jacobian determinant is
cancelled by the change in `sqrt(det R)`. This preserves both the finite
quadratic penalty and the corrected hard-limit measure. Changing the residual
but keeping `R` fixed defines a different finite-penalty model; adding the
hard-limit determinant alone does not make those tubes identical.

The determinant formula is asymptotic in the narrow-tube limit. At finite
`rho`, every application must establish that:

- the state integral is proper over its complete support;
- the posterior grid or integration box contains the relevant mass;
- the result is stable to tighter penalties and wider state/support windows;
- quadrature, linear-solver, and discretisation errors are below the claimed
  effect; and
- any stiffness introduced by large `rho` is handled numerically rather than
  hidden by optimiser or sampler failure.

A locally plausible hard-limit curve cannot repair a finite tube whose tails
are globally improper.

## 4. Validation and non-circular evidence

Every worked example declares the parameter, state, residual, metric, units or
nondimensionalisation, target measure, prior, observation/discrepancy model,
and data role. Nonconstant cases then expose three distinct targets:

1. the reduced graph target obtained after solving the state relation;
2. the naive state-tube marginal, including its induced volume factor; and
3. the pointwise corrected state-tube marginal.

The load-bearing comparison is computed by an independent route: analytic
Gaussian integration, deterministic state quadrature, manufactured-solution
checks, or a separate exact matrix calculation. Defining
`corrected = naive + correction` and recovering the graph is useful algebraic
consistency, but it is not independent validation. The machine-readable
manifests distinguish those roles.

Focused tests additionally cover constraint satisfaction, determinant or
metric evaluation, global support, finite-penalty convergence, units,
identification where available, negative controls, and strict finite JSON.
Synthetic truth recovery validates the declared construction, not the named
physical domain. External-data examples validate the executable conditional
model, not transportability or causality.

The executable verification contract is encoded in the tests and verifier,
recorded in the generated manifests and each example's
`outputs/summary.json`, and checked by `make verify`.

## 5. Determinants, samplers, and estimators

### Exact and structured determinants

Use the most transparent stable route available: scalar or diagonal sums,
triangular structure, Cholesky/sparse LU, separability, or an independently
audited exact matrix calculation. Report determinant sign or positive-
definiteness assumptions explicitly. A dense automatic differentiation route
is not stronger evidence when the operator structure supplies an exact result.

### Matrix-free log determinants and SLQ

Stochastic Lanczos quadrature estimates a log determinant; it does not certify
one. A claim-bearing use needs fixed and reported probe distributions, probe
count, Lanczos degree, seeds, numerical linear-operator checks, and convergence
over posterior support. Common random numbers help expose the centred
log-determinant variation that changes posterior shape, while multiple seeds
test probe sensitivity. Where an exact subproblem exists, the stochastic route
must be compared against it.

The screened-lattice tutorial intentionally shows estimator risk: its committed
Gram/SLQ settings leave centred errors of several log units. The exact
separable/sparse determinant defines the inference result; SLQ is retained as a
diagnostic of why those settings are insufficient for inference.

### MCMC, SMC, and optimisation

A sampler checks whether an algorithm can explore a specified target; it does
not validate the target's derivation. The sampler examples therefore compare
reduced graph and deliberately naive targets with deterministic grid
references. There is no duplicate reduced-space target labelled “corrected”:
after correct hard-limit reduction, that target is the graph target.

MCMC acceptance and a single-chain autocorrelation ESS are limited diagnostics
and do not substitute for multi-chain convergence checks. SMC reports weight
ESS, resampling, particle diversity, and move acceptance; autocorrelation ESS
is not computed from arbitrarily ordered resampled particles. Constrained MAP
or augmented-Lagrangian traces are optimisation/feasibility evidence, not
posterior samples.

## 6. Data, likelihood, and scientific identification

External inputs and dependent observations are assigned distinct declared
roles. A held-out row, material group, or future time point must not contribute
to preprocessing, predictor fitting, plug-in scale estimation, or model
selection if it is later described as untouched evaluation data. A likelihood
scale that uses the observed response must be justified as part of a
normalised generative model rather than introduced to shape the answer.

Correction magnitude is not evidence for scientific importance. Identification
still depends on data variation, parameterisation, priors, likelihood and
discrepancy assumptions, dependence, discretisation, and model adequacy. The
scientific case studies therefore test prediction and robustness after the
geometry audit; predictive performance is never used to validate the
mathematical correction.

For domain reuse, keep the verified geometry audit separate from the new
scientific claim. Before examining new outcomes, declare the question and
claim ceiling, acquire independently governed data, freeze input/observation
roles and held-out groups or times, add domain-specific likelihood and
dependence structure, and predeclare adequacy, sensitivity and negative-control
gates. The property-specific starting points are summarised in the
[domain-reuse table](case_studies/README.md#building-on-the-results);
the nested case-study technical contracts provide executable templates.

## 7. Provenance and licensing map

Repository-authored software and documentation are licensed under
[`LICENSE`](LICENSE). That licence does not replace the terms attached to
third-party data.

- [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) maps the worked tutorials
  and scientific fixtures to their sources, attribution, operative terms, and
  redistribution boundaries.
- Each external-data directory contains a hash-pinned fixture manifest; the
  corresponding project `TECHNICAL_DETAILS.md` states the accepted custom-data
  schema and provenance requirements.
- [`case_studies/THIRD_PARTY_NOTICES.md`](case_studies/THIRD_PARTY_NOTICES.md)
  and
  [`case_studies/third_party_licenses/`](case_studies/third_party_licenses/)
  record the operative notices for the scientific payload.
- A SHA256 hash establishes the bytes used; it is not itself permission to
  redistribute them. Generated figures and tables do not erase upstream
  provenance.

Reuse must retain the applicable notices and re-evaluate data rights when a
fixture, source, or jurisdiction changes.

## 8. Reproduction contracts

### Portable worked-example verification

Install the supported public `detcorr` 0.10 source, then the root requirements.
The canonical source is
[Olsson-Materials-Modelling/detcorr](https://github.com/Olsson-Materials-Modelling/detcorr).
CI installs the validated public commit directly, so it cannot pass by
accidentally importing an adjacent source checkout. An adjacent checkout may
instead be installed explicitly with `python -m pip install ../detcorr`; the
requirements command below then retains the compatible installed version.

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install \
  "detcorr @ git+https://github.com/Olsson-Materials-Modelling/detcorr.git@a414f063515359f70952e2717a506b75e69e4b0b"
python -m pip install -r requirements.txt
make verify
```

`make verify` regenerates every registered tutorial below a clean temporary
root, requires the same output surface and figure geometry, compares declared
authoritative fields within quantity-specific tolerances, and runs the tests
with a compatible `detcorr` required.

### Exact artifact reproduction

The committed tutorial artifacts were produced with this reference environment:

| Component | Reference artifact value |
|---|---:|
| Python | CPython 3.12.13, Clang 21.1.4 build |
| Platform | `Linux-6.12.47-x86_64-with-glibc2.39` |
| detcorr | 0.10.0 |
| NumPy / pandas / SciPy | 2.3.5 / 2.2.3 / 1.17.0 |
| Matplotlib / pytest | 3.10.8 / 8.4.2 |

The exact Python distributions are in
[`constraints-publisher.txt`](constraints-publisher.txt). The validated
`detcorr` source is public commit
`a414f063515359f70952e2717a506b75e69e4b0b`.
The stricter gate is:

```bash
python -m pip install -r requirements.txt -c constraints-publisher.txt
make verify-release
```

It compares every generated byte, including PNG encoding and diagnostic text.
BLAS, fonts, image libraries, compiler, kernel, and C-library differences can
change bytes without changing the scientific result; byte identity is an exact
artifact check, not independent scientific validation. The reference
environment additionally used `Ruff 0.14.10`. The
constraints file locks Python distributions, not operating-system libraries,
so a portable semantic pass is the appropriate acceptance route on another
platform.

### Scientific case studies

The nested scientific tier has deliberately independent dependencies and must
be installed in its own environment:

```bash
cd case_studies
python -m venv .venv
. .venv/bin/activate
python -m pip install \
  "detcorr @ git+https://github.com/Olsson-Materials-Modelling/detcorr.git@a414f063515359f70952e2717a506b75e69e4b0b"
python -m pip install -c constraints-publisher.txt -e '.[test]'
make verify-quick
make test
make verify-reference
```

`verify-quick` is a fresh CI-sized implementation check.
`verify-reference` validates the committed locked full result surface without
recomputing the expensive analyses. `make verify` performs a fresh full run,
tests it, validates both run manifests, and compares every generated scientific
artifact with the committed reference. The run manifests themselves record the
executing platform and are therefore validated independently rather than
cross-machine byte-compared. Full/external execution, custom-data requirements,
and environment handoff are consolidated in
[`case_studies/TECHNICAL_DETAILS.md`](case_studies/TECHNICAL_DETAILS.md).

On the reference one-CPU environment, the case-study test suite took about
11 minutes and `make full` took about 30 minutes. `make verify` performs both
stages before final verification. The runner processes bulk modulus, positive
gaps, and Fulton sequentially; JARVIS resampling can remain quiet for several
minutes before a study-completion message appears. These times are planning
guidance, not acceptance thresholds.

The root convenience targets `make verify-all` and `make verify-all-release`
are available when one interpreter satisfies both dependency sets. Separate
reference environments remain the authoritative exact-artifact route.

## 9. Machine-readable evidence

- [`examples/WORKED_EXAMPLES_MANIFEST.json`](examples/WORKED_EXAMPLES_MANIFEST.json)
  binds the active tutorial registry to targets, drivers, summaries, validation
  routes, and figures.
- [`examples/FIGURE_MANIFEST.json`](examples/FIGURE_MANIFEST.json) records image
  paths, dimensions, sizes, and SHA256 hashes.
- [`examples/outputs_validation_metrics.json`](examples/outputs_validation_metrics.json)
  records the common comparison fields and identifies the load-bearing
  independent metric.
- [`case_studies/STUDY_REGISTRY.json`](case_studies/STUDY_REGISTRY.json) fixes the
  active scientific studies.
- [`case_studies/reference_results/run_manifest.json`](case_studies/reference_results/run_manifest.json)
  binds the locked full results to protocol, code, input, dependency, and
  output hashes.
Generated JSON and CSV are authoritative for exact values. Figures and README
tables are interpretations of that recorded surface.

## 10. Verification contract

A clean worked-example verification must reproduce the registered file surface
and figure geometry, match the declared scientific fields within their
quantity-specific tolerances, pass focused tests, preserve strict JSON, resolve
all public links, and exercise the supported `detcorr` API.
`make verify-release` additionally requires byte identity in the reference
artifact runtime.

For the scientific tier, `make -C case_studies test` checks the protocol,
manifests, data separation, numerical routines, and documentation contract;
`make -C case_studies verify-reference` verifies every committed full-result
hash and terminal claim; and `make -C case_studies verify` recomputes that full
surface before requiring an exact match of the scientific artifacts. Each
environment-specific run manifest is validated on its own. The manifests named
above are the durable reproducibility record.
