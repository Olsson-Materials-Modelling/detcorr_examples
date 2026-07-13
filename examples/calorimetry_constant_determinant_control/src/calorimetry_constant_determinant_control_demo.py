#!/usr/bin/env python3
"""Synthetic calorimetry control with an exactly constant state determinant.

The state is the temperature rise ``delta_T`` in kelvin and the constrained
model is ``delta_T = Q/C``.  Because the residual derivative with respect to
``delta_T`` is one, the log determinant correction is exactly zero.  Exact
Gaussian integration over ``delta_T`` shows that naive and corrected tubes are
identical and converge to the reduced graph target as the penalty increases.
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

import detcorr_demo_common as common  # noqa: E402

HEAT_INPUT_J = 420.0
SYNTHETIC_TRUE_HEAT_CAPACITY_J_PER_K = 105.0
OBSERVATION_SD_K = 0.08
FIXED_NOISE_OFFSET_SD = 0.30
NOISE_FREE_TEMPERATURE_RISE_K = HEAT_INPUT_J / SYNTHETIC_TRUE_HEAT_CAPACITY_J_PER_K
OBSERVED_TEMPERATURE_RISE_K = (
    NOISE_FREE_TEMPERATURE_RISE_K + FIXED_NOISE_OFFSET_SD * OBSERVATION_SD_K
)

PRIOR_MEAN_HEAT_CAPACITY_J_PER_K = 110.0
PRIOR_SD_HEAT_CAPACITY_J_PER_K = 22.0
HEAT_CAPACITY_MIN_J_PER_K = 75.0
HEAT_CAPACITY_MAX_J_PER_K = 145.0
GRID_SIZE = 1001
PENALTY_RHO_PER_K2 = 1.0e5
CONVERGENCE_RHO_PER_K2 = (25.0, 100.0, 500.0, 2.0e3, 2.0e4, 1.0e5)


def solved_temperature_rise_k(heat_capacity_j_per_k: float) -> float:
    """Return ``delta_T=Q/C`` with Q in J and C in J/K."""

    heat_capacity = float(heat_capacity_j_per_k)
    if heat_capacity <= 0.0:
        raise ValueError("heat capacity must be positive")
    return HEAT_INPUT_J / heat_capacity


def constraint_residual_k(
    heat_capacity_j_per_k: float,
    temperature_rise_k: float,
) -> float:
    """Return ``c=delta_T-Q/C`` in kelvin."""

    return float(temperature_rise_k) - solved_temperature_rise_k(heat_capacity_j_per_k)


def state_jacobian() -> float:
    """Return ``D_deltaT c = 1``."""

    return 1.0


def state_logdet_correction() -> float:
    """Return the exact constant correction ``log|1|=0``."""

    return 0.0


def log_prior_heat_capacity(values: float | np.ndarray) -> np.ndarray:
    heat_capacity = np.asarray(values, dtype=float)
    result = (
        -0.5
        * ((heat_capacity - PRIOR_MEAN_HEAT_CAPACITY_J_PER_K) / PRIOR_SD_HEAT_CAPACITY_J_PER_K) ** 2
    )
    return np.where(heat_capacity > 0.0, result, -np.inf)


def observation_loglik_temperature(temperature_rise_k: float) -> float:
    standardized = (OBSERVED_TEMPERATURE_RISE_K - float(temperature_rise_k)) / OBSERVATION_SD_K
    return -0.5 * standardized**2


def graph_logdensity(heat_capacity_j_per_k: np.ndarray) -> np.ndarray:
    values = np.asarray(heat_capacity_j_per_k, dtype=float)
    predicted = HEAT_INPUT_J / values
    standardized = (OBSERVED_TEMPERATURE_RISE_K - predicted) / OBSERVATION_SD_K
    return log_prior_heat_capacity(values) - 0.5 * standardized**2


def exact_integrated_tube_logdensity(
    heat_capacity_j_per_k: np.ndarray,
    *,
    rho_per_k2: float,
    corrected: bool,
) -> np.ndarray:
    """Integrate the Gaussian full-space tube over ``delta_T`` exactly.

    ``rho`` has units K^-2 because the residual is in kelvin. The correction
    branch is retained explicitly as a control, but adds exactly ``log|1|=0``.
    Terms independent of heat capacity are omitted.
    """

    if rho_per_k2 <= 0.0:
        raise ValueError("rho_per_k2 must be positive")
    values = np.asarray(heat_capacity_j_per_k, dtype=float)
    observation_precision = 1.0 / OBSERVATION_SD_K**2
    precision = observation_precision + rho_per_k2
    log_precision = math.log(precision)
    result = np.empty_like(values)
    for index, heat_capacity in enumerate(values):
        constrained_mean = solved_temperature_rise_k(float(heat_capacity))
        linear = observation_precision * OBSERVED_TEMPERATURE_RISE_K + rho_per_k2 * constrained_mean
        constant = (
            observation_precision * OBSERVED_TEMPERATURE_RISE_K**2
            + rho_per_k2 * constrained_mean**2
        )
        integrated = 0.5 * linear**2 / precision
        integrated -= 0.5 * constant
        integrated -= 0.5 * log_precision
        if corrected:
            integrated += state_logdet_correction()
        result[index] = float(log_prior_heat_capacity(float(heat_capacity))) + integrated
    return result


def detcorr_api_audit(
    heat_capacity_j_per_k: float = SYNTHETIC_TRUE_HEAT_CAPACITY_J_PER_K,
) -> dict[str, Any]:
    """Check the zero correction through the public finite-penalty API."""

    try:
        from detcorr import evaluate_corrected_finite_penalty_target
    except Exception as exc:  # pragma: no cover - depends on installation
        return {"detcorr_available": False, "reason": repr(exc)}

    projected = solved_temperature_rise_k(heat_capacity_j_per_k)
    state = projected + 0.04

    def residual_from_state(theta: np.ndarray, state_vector: np.ndarray) -> np.ndarray:
        return np.array(
            [constraint_residual_k(float(theta[0]), float(state_vector[0]))],
            dtype=float,
        )

    target = evaluate_corrected_finite_penalty_target(
        residual_from_state,
        np.array([heat_capacity_j_per_k]),
        np.array([state]),
        rho=PENALTY_RHO_PER_K2,
        target_measure="graph",
        base_logdensity=float(log_prior_heat_capacity(heat_capacity_j_per_k))
        + observation_loglik_temperature(state),
        projected_state=np.array([projected]),
        log_correction_kwargs={
            "backend": "scipy",
            "finite_difference_scheme": "central",
            "singular_policy": "raise",
        },
        metadata={
            "case": "synthetic_calorimetry_constant_determinant_control",
            "residual_unit": "K",
            "rho_unit": "K^-2",
        },
    )
    return {
        "detcorr_available": True,
        "target_measure": target.target_measure,
        "primary_state_correction": float(target.primary_correction),
        "secondary_weight_correction": float(target.secondary_correction),
        "total_correction": float(target.correction),
        "expected_total_correction": 0.0,
        "correction_abs_error": abs(float(target.correction)),
        "state_jacobian_route": "callable residual with central finite differences",
        "diagnostics": target.sampler_diagnostics(),
    }


def convergence_table(
    heat_capacity_grid: np.ndarray,
    rho_values: tuple[float, ...] = CONVERGENCE_RHO_PER_K2,
) -> pd.DataFrame:
    graph = graph_logdensity(heat_capacity_grid)
    graph_density, _, graph_summary = common.normalize_on_grid(
        heat_capacity_grid,
        graph,
    )
    rows: list[dict[str, float]] = []
    for rho in rho_values:
        finite = exact_integrated_tube_logdensity(
            heat_capacity_grid,
            rho_per_k2=float(rho),
            corrected=False,
        )
        finite_density, _, finite_summary = common.normalize_on_grid(
            heat_capacity_grid,
            finite,
        )
        rows.append(
            {
                "rho_per_K2": float(rho),
                "constraint_tube_sd_K": 1.0 / math.sqrt(float(rho)),
                "centered_logdensity_max_abs_diff_vs_graph": (
                    common.centered_max_abs_diff(finite, graph)
                ),
                "density_l1_vs_graph": common.density_l1(
                    heat_capacity_grid,
                    finite_density,
                    graph_density,
                ),
                "posterior_mean_C_J_per_K": finite_summary.mean,
                "mean_shift_vs_graph_J_per_K": finite_summary.mean - graph_summary.mean,
            }
        )
    return pd.DataFrame(rows)


def _plot_posteriors(
    path: Path,
    heat_capacity: np.ndarray,
    graph_density: np.ndarray,
    finite_density: np.ndarray,
) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.plot(heat_capacity, graph_density, label="Reduced graph", linewidth=2.0)
    ax.plot(
        heat_capacity,
        finite_density,
        "--",
        label="Finite tube: naive = corrected",
    )
    ax.axvline(
        SYNTHETIC_TRUE_HEAT_CAPACITY_J_PER_K,
        color="black",
        linestyle=":",
        label="Synthetic truth",
    )
    ax.set_xlabel("heat capacity C (J/K)")
    ax.set_ylabel("posterior density")
    ax.set_title("Constant-determinant calorimetry control")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_convergence(path: Path, table: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    ax.loglog(
        table["rho_per_K2"],
        table["density_l1_vs_graph"],
        marker="o",
    )
    ax.set_xlabel("penalty rho (K^-2)")
    ax.set_ylabel("density L1 distance to graph")
    ax.set_title("Exact finite-tube convergence")
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_demo(
    outputs: Path,
    *,
    n_grid: int = GRID_SIZE,
    rho_per_k2: float = PENALTY_RHO_PER_K2,
) -> dict[str, Any]:
    if n_grid < 101:
        raise ValueError("n_grid must be at least 101")
    outputs.mkdir(parents=True, exist_ok=True)
    tables = outputs / "tables"
    heat_capacity = np.linspace(
        HEAT_CAPACITY_MIN_J_PER_K,
        HEAT_CAPACITY_MAX_J_PER_K,
        int(n_grid),
    )
    graph = graph_logdensity(heat_capacity)
    naive = exact_integrated_tube_logdensity(
        heat_capacity,
        rho_per_k2=rho_per_k2,
        corrected=False,
    )
    corrected = exact_integrated_tube_logdensity(
        heat_capacity,
        rho_per_k2=rho_per_k2,
        corrected=True,
    )
    targets = {
        "graph_reduced": graph,
        "finite_naive": naive,
        "finite_corrected": corrected,
    }
    posterior_table, densities, _ = common.summarize_targets(
        heat_capacity,
        targets,
    )
    common.write_table(posterior_table, tables / "posterior_summaries.csv")
    rho_table = convergence_table(heat_capacity)
    common.write_table(rho_table, tables / "rho_convergence.csv")

    graph_density = densities["graph_reduced"]
    naive_density = densities["finite_naive"]
    corrected_density = densities["finite_corrected"]
    rows = posterior_table.set_index("target")
    graph_mean = float(rows.loc["graph_reduced", "mean"])
    graph_sd = float(rows.loc["graph_reduced", "sd"])
    naive_mean = float(rows.loc["finite_naive", "mean"])
    corrected_mean = float(rows.loc["finite_corrected", "mean"])
    naive_graph_l1 = common.density_l1(
        heat_capacity,
        naive_density,
        graph_density,
    )
    corrected_graph_l1 = common.density_l1(
        heat_capacity,
        corrected_density,
        graph_density,
    )
    naive_corrected_l1 = common.density_l1(
        heat_capacity,
        naive_density,
        corrected_density,
    )
    audit = detcorr_api_audit()
    checks = {
        "synthetic_data_declared": True,
        "dimensional_equation_is_consistent": True,
        "state_jacobian_is_one": state_jacobian() == 1.0,
        "state_logdet_is_exactly_zero": state_logdet_correction() == 0.0,
        "finite_naive_equals_finite_corrected": bool(np.array_equal(naive, corrected)),
        "naive_corrected_density_l1_is_zero": naive_corrected_l1 == 0.0,
        "finite_tube_is_close_to_graph": corrected_graph_l1 < 2.0e-3,
        "convergence_improves_with_rho": bool(
            rho_table["density_l1_vs_graph"].iloc[-1] < rho_table["density_l1_vs_graph"].iloc[0]
        ),
        "detcorr_api_correction_is_zero": bool(
            audit.get("detcorr_available") and audit.get("correction_abs_error", 1.0) < 1.0e-8
        ),
    }
    checks["all_passed"] = all(checks.values())

    _plot_posteriors(
        outputs / "calorimetry_constant_determinant_posteriors.png",
        heat_capacity,
        graph_density,
        finite_density=corrected_density,
    )
    _plot_convergence(outputs / "finite_penalty_convergence.png", rho_table)

    summary: dict[str, Any] = {
        "example": "calorimetry_constant_determinant_control",
        "demo": "calorimetry_constant_determinant_control",
        "classification": "transparent_synthetic_constant_determinant_control",
        "status": "synthetic_worked_control_not_experimental_calorimetry",
        "target_measure": "graph posterior over C with respect to dC",
        "correction_driver": (
            "constant solved-state derivative D_deltaT c=1; no "
            "parameter-dependent determinant correction"
        ),
        "correction_log_range_on_grid": 0.0,
        "model": {
            "heat_input_J": HEAT_INPUT_J,
            "heat_capacity_unit": "J/K",
            "state_temperature_rise_unit": "K",
            "constraint": "c(C,delta_T)=delta_T-Q/C",
            "state_jacobian": 1.0,
            "state_logdet": 0.0,
            "penalty_rho": rho_per_k2,
            "penalty_rho_unit": "K^-2",
        },
        "synthetic_data": {
            "true_heat_capacity_J_per_K": SYNTHETIC_TRUE_HEAT_CAPACITY_J_PER_K,
            "noise_free_temperature_rise_K": NOISE_FREE_TEMPERATURE_RISE_K,
            "observation_sd_K": OBSERVATION_SD_K,
            "fixed_noise_offset_sd": FIXED_NOISE_OFFSET_SD,
            "observed_temperature_rise_K": OBSERVED_TEMPERATURE_RISE_K,
            "random_sampling_used": False,
        },
        "correction": {
            "state_logdet": 0.0,
            "correction_min_on_grid": 0.0,
            "correction_max_on_grid": 0.0,
            "correction_peak_to_peak": 0.0,
        },
        "posterior": {
            "graph_mean_C_J_per_K": graph_mean,
            "graph_sd_C_J_per_K": graph_sd,
            "finite_naive_mean_C_J_per_K": naive_mean,
            "finite_corrected_mean_C_J_per_K": corrected_mean,
            "finite_minus_graph_mean_C_J_per_K": naive_mean - graph_mean,
        },
        "outcomes": {
            "correction_log_range_on_grid": 0.0,
            "naive_mean_shift_in_graph_sd": (naive_mean - graph_mean) / graph_sd,
            "corrected_density_l1_vs_graph": corrected_graph_l1,
            "naive_density_l1_vs_graph": naive_graph_l1,
            "naive_corrected_density_l1": naive_corrected_l1,
            "naive_corrected_mean_difference": naive_mean - corrected_mean,
        },
        "outcome": {
            "correction_log_range_on_grid": 0.0,
            "naive_mean_shift_in_graph_sd": (naive_mean - graph_mean) / graph_sd,
            "corrected_mean_shift_in_graph_sd": (corrected_mean - graph_mean) / graph_sd,
            "naive_minus_corrected_mean_in_graph_sd": (naive_mean - corrected_mean) / graph_sd,
            "corrected_density_l1_vs_graph": corrected_graph_l1,
            "naive_density_l1_vs_graph": naive_graph_l1,
        },
        "independent_validation": {
            "route": "exact scalar Gaussian integration over temperature rise",
            "finite_tube_convergence_table": "tables/rho_convergence.csv",
        },
        "detcorr_api_audit": audit,
        "checks": checks,
        "limitations": [
            "The heat capacity, observation and fixed noise offset are synthetic.",
            "Heat input is treated as exact and heat loss is omitted.",
            "The heat capacity is constant over the small temperature rise.",
            "Agreement is a constant-determinant control, not material characterization.",
        ],
        "interpretation_boundary": (
            "Confirms that a constant state derivative introduces no relative "
            "correction; it is not experimental calorimetry."
        ),
        "outputs": {
            "posterior_figure": "calorimetry_constant_determinant_posteriors.png",
            "convergence_figure": "finite_penalty_convergence.png",
            "posterior_table": "tables/posterior_summaries.csv",
            "convergence_table": "tables/rho_convergence.csv",
        },
        "provenance": common.reproducibility_provenance(example_root=ROOT),
    }
    common.write_json(outputs / "summary.json", summary)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs", type=Path, default=ROOT / "outputs")
    parser.add_argument("--n-grid", type=int, default=GRID_SIZE)
    parser.add_argument("--rho", type=float, default=PENALTY_RHO_PER_K2)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    summary = run_demo(
        args.outputs,
        n_grid=args.n_grid,
        rho_per_k2=args.rho,
    )
    print(json.dumps({"example": summary["example"], "checks": summary["checks"]}, indent=2))


if __name__ == "__main__":
    main()
