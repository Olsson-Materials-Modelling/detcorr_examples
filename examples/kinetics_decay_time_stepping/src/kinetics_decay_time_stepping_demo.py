#!/usr/bin/env python3
"""First-order decay time-stepping demonstrator for detcorr.

The discrete state is an implicit-Euler trajectory for ``du/dt = -k u``:

    c_i(eta, u) = u_i - u_{i-1} + dt * exp(eta) * u_i,

where the numerical value of ``k`` is in s^-1 and
``eta = log(k / (1 s^-1))`` is dimensionless. The initial state is fixed, so
the state Jacobian is lower bidiagonal with diagonal ``1 + dt*k`` and exact
log determinant ``n * log(1 + dt*k)``.

The observations are transparently manufactured from a declared discrete truth
plus fixed, predeclared offsets. This makes the example a reproducible inference
and measure-correction control, not an empirical validation of first-order
kinetics. The full-space Gaussian residual tube is integrated over every state
analytically and independently of the reduced graph construction.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
COMMON = ROOT.parent / "detcorr_demo_common" / "src"
if str(COMMON) not in sys.path:
    sys.path.insert(0, str(COMMON))

from detcorr_demo_common import (  # noqa: E402
    centered_max_abs_diff,
    reproducibility_provenance,
    summarize_targets,
    write_json,
    write_table,
)

N_STEPS = 8
DT_SECONDS = 0.5
INITIAL_CONCENTRATION_MM = 2.0
RATE_REFERENCE_PER_S = 1.0
TRUE_RATE_PER_S = 0.65
TRUE_ETA = math.log(TRUE_RATE_PER_S / RATE_REFERENCE_PER_S)
OBSERVATION_SIGMA_MM = 0.08
STANDARDIZED_OFFSETS = np.array(
    [0.30, -0.55, 0.80, -0.25, 0.45, -0.70, 0.20, 0.55],
    dtype=float,
)
PRIOR_MEAN_ETA = 0.0
PRIOR_SD_ETA = 1.0
DEFAULT_RHO_PER_MM2 = 1.0e8
DEFAULT_ETA_MIN = -1.8
DEFAULT_ETA_MAX = 0.8
DEFAULT_GRID_SIZE = 1601


def rate_from_eta(eta: float) -> float:
    """Return the positive decay-rate numerical value in s^-1."""

    if not math.isfinite(eta):
        raise ValueError("eta must be finite")
    return RATE_REFERENCE_PER_S * math.exp(float(eta))


def time_step_factor(eta: float, *, dt_seconds: float = DT_SECONDS) -> float:
    """Return the dimensionless implicit-Euler diagonal factor."""

    if dt_seconds <= 0.0 or not math.isfinite(dt_seconds):
        raise ValueError("dt_seconds must be positive and finite")
    return 1.0 + float(dt_seconds) * rate_from_eta(float(eta))


def constraint_matrix(
    eta: float,
    *,
    n_steps: int = N_STEPS,
    dt_seconds: float = DT_SECONDS,
) -> np.ndarray:
    """Return the lower-bidiagonal state Jacobian ``D_u c``."""

    if int(n_steps) < 1:
        raise ValueError("n_steps must be positive")
    diagonal = np.full(
        int(n_steps),
        time_step_factor(float(eta), dt_seconds=float(dt_seconds)),
    )
    matrix = np.diag(diagonal)
    if int(n_steps) > 1:
        matrix += np.diag(-np.ones(int(n_steps) - 1), k=-1)
    return matrix


def constraint_rhs(
    *,
    n_steps: int = N_STEPS,
    initial_concentration_mm: float = INITIAL_CONCENTRATION_MM,
) -> np.ndarray:
    """Return ``b`` in the affine residual ``c(eta,u)=A(eta)u-b``."""

    if int(n_steps) < 1:
        raise ValueError("n_steps must be positive")
    if initial_concentration_mm <= 0.0 or not math.isfinite(initial_concentration_mm):
        raise ValueError("initial_concentration_mm must be positive and finite")
    rhs = np.zeros(int(n_steps), dtype=float)
    rhs[0] = float(initial_concentration_mm)
    return rhs


def solve_state(
    eta: float,
    *,
    n_steps: int = N_STEPS,
    dt_seconds: float = DT_SECONDS,
    initial_concentration_mm: float = INITIAL_CONCENTRATION_MM,
) -> np.ndarray:
    """Solve the recurrence exactly in reduced coordinates."""

    factor = time_step_factor(float(eta), dt_seconds=float(dt_seconds))
    powers = np.arange(1, int(n_steps) + 1, dtype=float)
    return float(initial_concentration_mm) / factor**powers


def residual(
    eta: float,
    state_mm: np.ndarray,
    *,
    dt_seconds: float = DT_SECONDS,
    initial_concentration_mm: float = INITIAL_CONCENTRATION_MM,
) -> np.ndarray:
    """Evaluate the implicit-Euler residual in mM."""

    state = np.asarray(state_mm, dtype=float)
    if state.ndim != 1 or state.size < 1:
        raise ValueError("state_mm must be a non-empty one-dimensional array")
    matrix = constraint_matrix(
        float(eta),
        n_steps=int(state.size),
        dt_seconds=float(dt_seconds),
    )
    rhs = constraint_rhs(
        n_steps=int(state.size),
        initial_concentration_mm=float(initial_concentration_mm),
    )
    return matrix @ state - rhs


def exact_logdet(
    eta: float,
    *,
    n_steps: int = N_STEPS,
    dt_seconds: float = DT_SECONDS,
) -> float:
    """Return ``log|det D_u c|`` from the triangular diagonal."""

    return float(
        int(n_steps) * math.log(time_step_factor(float(eta), dt_seconds=float(dt_seconds)))
    )


def synthetic_observation_table() -> pd.DataFrame:
    """Return the fully declared manufactured trajectory and observations."""

    truth = solve_state(TRUE_ETA)
    offsets_mm = OBSERVATION_SIGMA_MM * STANDARDIZED_OFFSETS
    observations = truth + offsets_mm
    return pd.DataFrame(
        {
            "step": np.arange(1, N_STEPS + 1, dtype=int),
            "time_s": DT_SECONDS * np.arange(1, N_STEPS + 1, dtype=float),
            "truth_implicit_euler_mm": truth,
            "standardized_offset": STANDARDIZED_OFFSETS,
            "offset_mm": offsets_mm,
            "observed_mm": observations,
            "sigma_mm": np.full(N_STEPS, OBSERVATION_SIGMA_MM),
        }
    )


def observed_trajectory_mm() -> np.ndarray:
    return synthetic_observation_table()["observed_mm"].to_numpy(float)


def log_prior_eta(eta: float) -> float:
    """Normal log prior with respect to ``d eta``, up to a constant."""

    if not math.isfinite(eta):
        return float("-inf")
    return -0.5 * ((float(eta) - PRIOR_MEAN_ETA) / PRIOR_SD_ETA) ** 2


def reduced_graph_logdensity(eta: float) -> float:
    """Evaluate the graph posterior after eliminating the state recurrence."""

    prediction = solve_state(float(eta))
    misfit = (prediction - observed_trajectory_mm()) / OBSERVATION_SIGMA_MM
    return log_prior_eta(float(eta)) - 0.5 * float(misfit @ misfit)


def gaussian_tube_log_integral(
    eta: float,
    observations_mm: np.ndarray,
    *,
    sigma_mm: float = OBSERVATION_SIGMA_MM,
    rho_per_mm2: float = DEFAULT_RHO_PER_MM2,
    dt_seconds: float = DT_SECONDS,
    initial_concentration_mm: float = INITIAL_CONCENTRATION_MM,
) -> float:
    """Integrate the observation likelihood and residual tube over ``u``.

    The returned value includes ``n/2*log(2*pi)`` but omits Gaussian
    normalization constants that do not depend on ``eta``. The quadratic is
    evaluated at its minimizer rather than by subtracting two large completed-
    square terms, which keeps the exact formula stable for large ``rho``.
    """

    observations = np.asarray(observations_mm, dtype=float)
    if observations.ndim != 1 or observations.size < 1:
        raise ValueError("observations_mm must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(observations)):
        raise ValueError("observations_mm must be finite")
    if sigma_mm <= 0.0 or not math.isfinite(sigma_mm):
        raise ValueError("sigma_mm must be positive and finite")
    if rho_per_mm2 <= 0.0 or not math.isfinite(rho_per_mm2):
        raise ValueError("rho_per_mm2 must be positive and finite")

    n_steps = int(observations.size)
    matrix = constraint_matrix(
        float(eta),
        n_steps=n_steps,
        dt_seconds=float(dt_seconds),
    )
    rhs = constraint_rhs(
        n_steps=n_steps,
        initial_concentration_mm=float(initial_concentration_mm),
    )
    observation_precision = 1.0 / float(sigma_mm) ** 2
    precision = observation_precision * np.eye(n_steps) + float(rho_per_mm2) * (matrix.T @ matrix)
    linear = observation_precision * observations + float(rho_per_mm2) * (matrix.T @ rhs)
    mode = np.linalg.solve(precision, linear)
    observation_term = observation_precision * float((mode - observations) @ (mode - observations))
    constraint = matrix @ mode - rhs
    constraint_term = float(rho_per_mm2) * float(constraint @ constraint)
    chol = np.linalg.cholesky(precision)
    logdet_precision = 2.0 * float(np.sum(np.log(np.diag(chol))))
    return (
        0.5 * n_steps * math.log(2.0 * math.pi)
        - 0.5 * logdet_precision
        - 0.5 * (observation_term + constraint_term)
    )


def integrated_tube_logdensity(
    eta: float,
    *,
    corrected: bool,
    rho_per_mm2: float = DEFAULT_RHO_PER_MM2,
) -> float:
    """Return the analytically integrated finite-penalty marginal over eta."""

    base = log_prior_eta(float(eta)) + gaussian_tube_log_integral(
        float(eta),
        observed_trajectory_mm(),
        rho_per_mm2=float(rho_per_mm2),
    )
    return base + (exact_logdet(float(eta)) if corrected else 0.0)


def build_posterior_grid(
    eta_grid: np.ndarray,
    *,
    rho_per_mm2: float = DEFAULT_RHO_PER_MM2,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    """Evaluate and normalize all three posterior routes on one grid."""

    grid = np.asarray(eta_grid, dtype=float)
    if grid.ndim != 1 or grid.size < 3:
        raise ValueError("eta_grid must contain at least three values")
    if not np.all(np.isfinite(grid)) or np.any(np.diff(grid) <= 0.0):
        raise ValueError("eta_grid must be finite and strictly increasing")

    logdensities = {
        "graph_reduced": np.asarray(
            [reduced_graph_logdensity(float(eta)) for eta in grid],
            dtype=float,
        ),
        "naive_finite_gaussian_tube": np.asarray(
            [
                integrated_tube_logdensity(
                    float(eta),
                    corrected=False,
                    rho_per_mm2=float(rho_per_mm2),
                )
                for eta in grid
            ],
            dtype=float,
        ),
        "detcorr_finite_gaussian_tube": np.asarray(
            [
                integrated_tube_logdensity(
                    float(eta),
                    corrected=True,
                    rho_per_mm2=float(rho_per_mm2),
                )
                for eta in grid
            ],
            dtype=float,
        ),
    }
    summary_table, densities, _ = summarize_targets(grid, logdensities)
    grid_table = pd.DataFrame(
        {
            "eta": grid,
            "rate_per_s": np.exp(grid) * RATE_REFERENCE_PER_S,
            "logdet_state_jacobian": np.asarray(
                [exact_logdet(float(eta)) for eta in grid],
                dtype=float,
            ),
        }
    )
    for name, values in logdensities.items():
        grid_table[f"logdensity_{name}"] = values
        grid_table[f"density_{name}"] = densities[name]
    return summary_table, grid_table, densities


def determinant_audit_table(eta_values: np.ndarray) -> pd.DataFrame:
    """Cross-check the triangular formula against dense ``slogdet``."""

    rows: list[dict[str, float]] = []
    for eta in np.asarray(eta_values, dtype=float):
        matrix = constraint_matrix(float(eta))
        sign, dense_logdet = np.linalg.slogdet(matrix)
        analytic = exact_logdet(float(eta))
        rows.append(
            {
                "eta": float(eta),
                "rate_per_s": rate_from_eta(float(eta)),
                "determinant_sign": float(sign),
                "analytic_logdet": analytic,
                "dense_slogdet": float(dense_logdet),
                "absolute_error": abs(analytic - float(dense_logdet)),
            }
        )
    return pd.DataFrame(rows)


def detcorr_api_audit(eta: float) -> dict[str, Any]:
    """Evaluate the full triangular Jacobian through the public detcorr API."""

    analytic = exact_logdet(float(eta))
    try:
        import detcorr
    except Exception as exc:  # pragma: no cover - installation dependent
        return {
            "detcorr_available": False,
            "reason": repr(exc),
            "eta": float(eta),
            "analytic_logdet": analytic,
        }
    result = detcorr.log_correction(
        jacobian=constraint_matrix(float(eta)),
        structure="triangular",
        singular_policy="raise",
    )
    return {
        "detcorr_available": True,
        "reason": None,
        "eta": float(eta),
        "analytic_logdet": analytic,
        "detcorr_logdet": float(result.value),
        "absolute_error": abs(float(result.value) - analytic),
        "result": result.to_dict(),
    }


def _grid_integral(grid: np.ndarray, values: np.ndarray) -> float:
    return float(
        np.sum(
            0.5
            * (np.asarray(values, dtype=float)[1:] + np.asarray(values, dtype=float)[:-1])
            * np.diff(np.asarray(grid, dtype=float))
        )
    )


def make_plots(
    eta_grid: np.ndarray,
    densities: dict[str, np.ndarray],
    outputs: Path,
) -> None:
    outputs.mkdir(parents=True, exist_ok=True)

    observations = synthetic_observation_table()
    times = np.arange(0, N_STEPS + 1, dtype=float) * DT_SECONDS
    discrete_truth = np.concatenate([[INITIAL_CONCENTRATION_MM], solve_state(TRUE_ETA)])
    continuous_reference = INITIAL_CONCENTRATION_MM * np.exp(-TRUE_RATE_PER_S * times)
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    ax.plot(
        times,
        continuous_reference,
        linestyle="--",
        linewidth=1.2,
        label="continuous ODE reference (not fitted)",
    )
    ax.plot(
        times,
        discrete_truth,
        marker="o",
        linewidth=1.5,
        label="implicit-Euler truth used",
    )
    ax.errorbar(
        observations["time_s"],
        observations["observed_mm"],
        yerr=observations["sigma_mm"],
        fmt="s",
        capsize=3,
        label="fixed synthetic observations",
    )
    ax.set_xlabel("time / s")
    ax.set_ylabel("concentration / mM")
    ax.set_title("Declared first-order decay demonstrator")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outputs / "synthetic_decay_trajectory.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    eta_values = np.asarray(eta_grid, dtype=float)
    ax.plot(
        eta_values,
        densities["naive_finite_gaussian_tube"],
        color="tab:orange",
        linewidth=1.8,
        label="naive integrated tube",
    )
    ax.plot(
        eta_values,
        densities["detcorr_finite_gaussian_tube"],
        color="tab:green",
        linewidth=2.2,
        label="detcorr integrated tube",
    )
    # Draw the graph reference last with a distinct style so exact overlap with
    # the corrected route remains visible rather than being hidden by it.
    ax.plot(
        eta_values,
        densities["graph_reduced"],
        color="tab:blue",
        linestyle="--",
        linewidth=1.6,
        label="reduced graph",
    )
    ax.axvline(TRUE_ETA, color="black", linestyle=":", label="declared truth")
    ax.set_xlabel(r"log-rate $\eta=\log(k/(1\ \mathrm{s}^{-1}))$")
    ax.set_ylabel("posterior density")
    ax.set_title("Graph target and analytically integrated residual tubes")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outputs / "eta_posterior_comparison.png", dpi=180)
    plt.close(fig)


def run_demo(args: argparse.Namespace) -> dict[str, Any]:
    eta_grid = np.linspace(float(args.eta_min), float(args.eta_max), int(args.n_grid))
    posterior_table, grid_table, densities = build_posterior_grid(
        eta_grid,
        rho_per_mm2=float(args.rho),
    )
    observations = synthetic_observation_table()
    determinant_table = determinant_audit_table(
        np.array([args.eta_min, TRUE_ETA, 0.0, args.eta_max], dtype=float)
    )

    rows = {row["target"]: row for row in posterior_table.to_dict(orient="records")}
    graph = rows["graph_reduced"]
    naive = rows["naive_finite_gaussian_tube"]
    corrected = rows["detcorr_finite_gaussian_tube"]
    correction_values = grid_table["logdet_state_jacobian"].to_numpy(float)
    correction_log_range = float(np.max(correction_values) - np.min(correction_values))
    naive_shift_in_sd = float((naive["mean"] - graph["mean"]) / graph["sd"])
    corrected_shift_in_sd = float((corrected["mean"] - graph["mean"]) / graph["sd"])
    naive_minus_corrected_shift_in_sd = float((naive["mean"] - corrected["mean"]) / graph["sd"])
    corrected_l1 = float(corrected["density_l1_vs_graph"])
    naive_l1 = float(naive["density_l1_vs_graph"])

    for target, density in densities.items():
        mean_rate = _grid_integral(
            eta_grid,
            grid_table["rate_per_s"].to_numpy(float) * density,
        )
        posterior_table.loc[
            posterior_table["target"] == target,
            "mean_rate_per_s",
        ] = mean_rate

    graph_mode = float(graph["mode"])
    api_audit = detcorr_api_audit(graph_mode)
    outputs = Path(args.outputs)
    table_dir = outputs / "tables"
    write_table(observations, table_dir / "synthetic_observations.csv")
    write_table(posterior_table, table_dir / "posterior_summary.csv")
    write_table(grid_table, table_dir / "posterior_grid.csv")
    write_table(determinant_table, table_dir / "determinant_audit.csv")
    make_plots(eta_grid, densities, outputs)

    summary: dict[str, Any] = {
        "demo": "kinetics_decay_time_stepping",
        "classification": "synthetic time-stepping verification problem",
        "provenance": reproducibility_provenance(example_root=ROOT),
        "target_measure": "graph posterior over eta with respect to d eta",
        "correction_driver": ("accumulated lower-triangular time-step Jacobian n*log(1+dt*k)"),
        "correction_log_range_on_grid": correction_log_range,
        "naive_mean_shift_in_graph_sd": naive_shift_in_sd,
        "corrected_density_l1_vs_graph": corrected_l1,
        "naive_density_l1_vs_graph": naive_l1,
        "model": {
            "continuous_equation": "du/dt=-k*u",
            "discrete_residual": "c_i=u_i-u_{i-1}+dt*k*u_i",
            "rate_parameterization": "k/(1 s^-1)=exp(eta)",
            "state_jacobian": "lower bidiagonal with diagonal 1+dt*k",
            "exact_logdet": "n*log(1+dt*k)",
            "n_steps": N_STEPS,
            "dt_seconds": DT_SECONDS,
            "initial_concentration_mm": INITIAL_CONCENTRATION_MM,
        },
        "synthetic_data_contract": {
            "status": "transparent manufactured discrete-model control",
            "true_eta": TRUE_ETA,
            "true_rate_per_s": TRUE_RATE_PER_S,
            "observation_sigma_mm": OBSERVATION_SIGMA_MM,
            "standardized_offsets": STANDARDIZED_OFFSETS.tolist(),
            "random_number_generation": "none",
            "generated_from_same_implicit_euler_model": True,
        },
        "configuration": {
            "eta_min": float(args.eta_min),
            "eta_max": float(args.eta_max),
            "n_grid": int(args.n_grid),
            "rho_per_mm2": float(args.rho),
            "prior_mean_eta": PRIOR_MEAN_ETA,
            "prior_sd_eta": PRIOR_SD_ETA,
        },
        "independent_validation": {
            "route": "stable exact multivariate Gaussian integration over u",
            "not_graph_minus_logdet_construction": True,
            "manufactured_residual_norm": float(
                np.linalg.norm(residual(TRUE_ETA, solve_state(TRUE_ETA)))
            ),
            "determinant_dense_max_abs_error": float(determinant_table["absolute_error"].max()),
            "corrected_centered_logdensity_max_abs_diff_vs_graph": (
                centered_max_abs_diff(
                    grid_table["logdensity_detcorr_finite_gaussian_tube"].to_numpy(float),
                    grid_table["logdensity_graph_reduced"].to_numpy(float),
                )
            ),
        },
        "outcomes": {
            "graph_mean_eta": float(graph["mean"]),
            "graph_sd_eta": float(graph["sd"]),
            "naive_mean_eta": float(naive["mean"]),
            "corrected_mean_eta": float(corrected["mean"]),
            "naive_mean_shift_eta": float(naive["mean"] - graph["mean"]),
            "corrected_mean_abs_error_eta": abs(float(corrected["mean"] - graph["mean"])),
            "declared_true_eta": TRUE_ETA,
        },
        "outcome": {
            "graph_mean": float(graph["mean"]),
            "graph_sd": float(graph["sd"]),
            "correction_log_range_on_grid": correction_log_range,
            "naive_mean_shift_in_graph_sd": naive_shift_in_sd,
            "corrected_mean_shift_in_graph_sd": corrected_shift_in_sd,
            "naive_minus_corrected_mean_in_graph_sd": (naive_minus_corrected_shift_in_sd),
            "naive_density_l1_vs_graph": naive_l1,
            "corrected_density_l1_vs_graph": corrected_l1,
        },
        "detcorr_api_audit": api_audit,
        "interpretation_boundary": (
            "The data are generated from the same discrete model. The example "
            "validates recurrence, Gaussian integration, and measure correction, "
            "not the empirical adequacy of first-order kinetics."
        ),
        "tables": {
            "posterior_summary": posterior_table.to_dict(orient="records"),
            "determinant_audit": determinant_table.to_dict(orient="records"),
        },
    }
    write_json(outputs / "summary.json", summary)

    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs", default=str(ROOT / "outputs"))
    parser.add_argument("--eta-min", type=float, default=DEFAULT_ETA_MIN)
    parser.add_argument("--eta-max", type=float, default=DEFAULT_ETA_MAX)
    parser.add_argument("--n-grid", type=int, default=DEFAULT_GRID_SIZE)
    parser.add_argument("--rho", type=float, default=DEFAULT_RHO_PER_MM2)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    summary = run_demo(args)
    print(f"Wrote kinetics decay outputs to {args.outputs}")
    print(
        json.dumps(
            {
                "demo": summary["demo"],
                "naive_density_l1_vs_graph": summary["naive_density_l1_vs_graph"],
                "corrected_density_l1_vs_graph": summary["corrected_density_l1_vs_graph"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
