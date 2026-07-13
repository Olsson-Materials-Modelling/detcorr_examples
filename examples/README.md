# Worked detcorr tutorials

[Repository overview](../README.md) ·
[Scientific case studies](../case_studies/README.md) ·
[Technical contract](../TECHNICAL_DETAILS.md)

These ten independently runnable tutorials show how constraint geometry enters
an inferential target. Five transparent synthetic problems isolate one mechanism
at a time; five external-data demonstrators exercise held-out, sampling,
sparse-operator, and matrix-free workflows.

In this index, **graph target** means the posterior obtained after solving the
constraint for its state. It is unrelated to graph neural networks.

## Start here

From the repository root:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install \
  "detcorr @ git+https://github.com/Olsson-Materials-Modelling/detcorr.git@a414f063515359f70952e2717a506b75e69e4b0b"
python -m pip install -r requirements.txt
python examples/adsorption_langmuir_equilibrium/src/adsorption_langmuir_equilibrium_demo.py \
  --outputs /tmp/detcorr-langmuir
```

The public package source is
[Olsson-Materials-Modelling/detcorr](https://github.com/Olsson-Materials-Modelling/detcorr).

## Synthetic tutorials

- [Constant-determinant calorimetry](calorimetry_constant_determinant_control/README.md)
  is the zero-effect control.
- [First-order decay](kinetics_decay_time_stepping/README.md) shows the volume
  induced by an implicit time-stepping scheme.
- [Langmuir equilibrium](adsorption_langmuir_equilibrium/README.md) isolates a
  nonlinear bounded-state Jacobian.
- [Weighted sensor constraint](sensor_weighted_constraint/README.md) isolates
  the residual-metric contribution.
- [Elliptic PDE](elliptic_pde_1d/README.md) uses a structured discrete operator.

## External-data demonstrators

- [JARVIS bulk-modulus scale](materials_bulk_modulus_samplers_jarvis/README.md)
  exercises repeated latent-state volume and sampler diagnostics.
- [JARVIS positive-gap scale](materials_bandgap_samplers_jarvis/README.md)
  exercises a globally proper log-energy residual.
- [Fulton price scale](econometrics_fulton_market/README.md) uses a chronological
  held-out calibration block.
- [Matbench nonlinear coordinate](materials_bandgap_matbench/README.md) is a
  dimensional and held-out stress test.
- [Screened lattice](materials_screened_lattice_2d3d_jarvis/README.md) exercises
  exact sparse determinants and stochastic Lanczos quadrature diagnostics.

Each project README gives the result and shortest run route. Its paired
`TECHNICAL_DETAILS.md` records assumptions, validation, provenance, and exact
output contracts. Run the complete portable verification from the repository
root with `make verify`.
