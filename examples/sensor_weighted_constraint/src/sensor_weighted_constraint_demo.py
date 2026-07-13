#!/usr/bin/env python3
"""Synthetic multi-channel sensor example with an algorithmic constraint metric.

The physical observation model and the algorithmic constraint metric are kept
separate.  Observations are noisy measurements of the latent channel states,
while ``R(gamma)`` only determines the geometry of the finite-penalty tube
around ``u = gamma*x``.  Exact Gaussian integration over ``u`` independently
checks the required ``0.5*logdet(R)`` weight-volume correction.
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

CHANNEL_NAMES = ("A", "B", "C", "D")
N_CHANNEL = len(CHANNEL_NAMES)
STATE_UNIT = "mV"
GAIN_UNIT = "dimensionless"
REFERENCE_STATE_UNIT_MV = 1.0

BASE_SIGNAL_MV = np.array([1.0, 1.4, 1.8, 2.2], dtype=float)
OBSERVATION_SD_MV = np.array([0.25, 0.28, 0.32, 0.35], dtype=float)
SYNTHETIC_TRUE_GAMMA = 1.18
FIXED_NOISE_OFFSETS_SD = np.array([0.35, -0.60, 0.20, 0.50], dtype=float)
OBSERVED_SIGNAL_MV = (
    SYNTHETIC_TRUE_GAMMA * BASE_SIGNAL_MV + FIXED_NOISE_OFFSETS_SD * OBSERVATION_SD_MV
)

PRIOR_MEAN_GAMMA = 1.0
PRIOR_SD_GAMMA = 0.35
GAMMA_MIN = 0.20
GAMMA_MAX = 2.50
GRID_SIZE = 2301
# This penalty makes the expected O(1/rho) finite-penalty bias smaller than the
# validation guard over the deliberately widened support grid while keeping the
# exact 4x4 Gaussian integrations well conditioned.
PENALTY_RHO = 3.0e7
FINITE_PENALTY_ERROR_LIMIT = 5.0e-4

# These exponents define a deliberately parameter-dependent numerical
# preconditioner. They are not fitted observation precisions.
METRIC_LOG_SLOPES = np.array([2.4, 1.8, 1.2, 0.6], dtype=float)
METRIC_MIXING = np.array(
    [
        [1.0, 0.00, 0.00, 0.00],
        [0.18, 1.0, 0.00, 0.00],
        [0.00, -0.14, 1.0, 0.00],
        [0.08, 0.00, 0.12, 1.0],
    ],
    dtype=float,
)


def solved_state_mv(gamma: float) -> np.ndarray:
    """Return the constrained channel state ``u = gamma*x`` in mV."""

    return float(gamma) * BASE_SIGNAL_MV


def constraint_residual_mv(gamma: float, state_mv: np.ndarray) -> np.ndarray:
    """Return ``c(gamma,u)=u-gamma*x`` in the declared mV coordinates."""

    return np.asarray(state_mv, dtype=float) - solved_state_mv(gamma)


def state_jacobian() -> np.ndarray:
    """Return ``D_u c = I`` in the same mV input and residual coordinates."""

    return np.eye(N_CHANNEL)


def algorithmic_whitener(gamma: float) -> np.ndarray:
    """Return the numerical constraint whitener in inverse mV.

    The fixed triangular mixing represents channel coupling in a solver
    preconditioner. Exponential row scales keep the metric positive while
    making its parameter-dependent volume visible in a small example.
    """

    row_scales = np.exp(METRIC_LOG_SLOPES * (float(gamma) - 1.0))
    return np.diag(row_scales) @ METRIC_MIXING / REFERENCE_STATE_UNIT_MV


def algorithmic_metric(gamma: float) -> np.ndarray:
    """Return the SPD algorithmic metric ``R=W.T@W`` in mV^-2."""

    whitener = algorithmic_whitener(gamma)
    return whitener.T @ whitener


def metric_log_sqrt_det(gamma: float) -> float:
    """Return ``0.5*logdet(R)`` relative to the declared mV state volume."""

    sign, value = np.linalg.slogdet(algorithmic_metric(gamma))
    if sign <= 0.0:
        raise ValueError("algorithmic constraint metric is not SPD")
    return 0.5 * float(value)


def log_prior_gamma(gamma: float | np.ndarray) -> np.ndarray:
    values = np.asarray(gamma, dtype=float)
    result = np.full_like(values, -np.inf, dtype=float)
    positive = np.isfinite(values) & (values > 0.0)
    result[positive] = -0.5 * ((values[positive] - PRIOR_MEAN_GAMMA) / PRIOR_SD_GAMMA) ** 2
    return result


def grid_support_audit(
    gamma: np.ndarray,
    targets: dict[str, np.ndarray],
    cdfs: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Report endpoint separation and two-percent boundary-strip mass."""

    edge = max(2, int(math.ceil(0.02 * len(gamma))))
    rows: list[dict[str, Any]] = []
    for name, logdensity in targets.items():
        peak = float(np.max(logdensity))
        cdf = cdfs[name]
        rows.append(
            {
                "target": name,
                "mode_is_interior": bool(0 < int(np.argmax(logdensity)) < len(gamma) - 1),
                "lower_endpoint_log_drop": peak - float(logdensity[0]),
                "upper_endpoint_log_drop": peak - float(logdensity[-1]),
                "lower_edge_mass_2pct": float(cdf[edge - 1]),
                "upper_edge_mass_2pct": float(1.0 - cdf[-edge]),
            }
        )
    return pd.DataFrame(rows)


def observation_loglik_state(state_mv: np.ndarray) -> float:
    """Physical synthetic observation likelihood; unrelated to ``R(gamma)``."""

    standardized = (OBSERVED_SIGNAL_MV - np.asarray(state_mv, dtype=float)) / OBSERVATION_SD_MV
    return -0.5 * float(standardized @ standardized)


def graph_logdensity(gamma: np.ndarray) -> np.ndarray:
    """Reduced graph target with ``u`` eliminated exactly."""

    values = np.asarray(gamma, dtype=float)
    state = values[:, None] * BASE_SIGNAL_MV[None, :]
    standardized = (OBSERVED_SIGNAL_MV[None, :] - state) / OBSERVATION_SD_MV[None, :]
    return log_prior_gamma(values) - 0.5 * np.sum(standardized**2, axis=1)


def exact_integrated_tube_logdensity(
    gamma: np.ndarray,
    *,
    rho: float = PENALTY_RHO,
    include_weight_volume: bool,
) -> np.ndarray:
    """Integrate the finite-penalty full-space target over ``u`` exactly.

    The integral is evaluated from the combined Gaussian precision. It is not
    constructed by subtracting a determinant from the graph target.
    Terms independent of ``gamma`` are omitted.
    """

    if rho <= 0.0:
        raise ValueError("rho must be positive")
    values = np.asarray(gamma, dtype=float)
    observation_precision = np.diag(1.0 / OBSERVATION_SD_MV**2)
    observation_linear = observation_precision @ OBSERVED_SIGNAL_MV
    observation_constant = float(OBSERVED_SIGNAL_MV @ observation_precision @ OBSERVED_SIGNAL_MV)
    result = np.empty_like(values)
    for index, gain in enumerate(values):
        metric = algorithmic_metric(float(gain))
        constrained_mean = solved_state_mv(float(gain))
        precision = observation_precision + rho * metric
        linear = observation_linear + rho * metric @ constrained_mean
        constant = observation_constant + rho * float(constrained_mean @ metric @ constrained_mean)
        sign, logdet_precision = np.linalg.slogdet(precision)
        if sign <= 0.0:
            raise ValueError("combined Gaussian precision is not SPD")
        integrated = 0.5 * float(linear @ np.linalg.solve(precision, linear))
        integrated -= 0.5 * constant
        integrated -= 0.5 * float(logdet_precision)
        if include_weight_volume:
            integrated += metric_log_sqrt_det(float(gain))
        result[index] = float(log_prior_gamma(float(gain))) + integrated
    return result


def hard_tube_logdensity(gamma: np.ndarray, *, corrected: bool) -> np.ndarray:
    """Return the hard-limit tube target up to a gamma-independent constant."""

    values = np.asarray(gamma, dtype=float)
    graph = graph_logdensity(values)
    if corrected:
        return graph
    correction = np.asarray([metric_log_sqrt_det(float(g)) for g in values])
    return graph - correction


def detcorr_weighted_api_audit(gamma: float = SYNTHETIC_TRUE_GAMMA) -> dict[str, Any]:
    """Cross-check the weighted target through the public detcorr API."""

    try:
        from detcorr import evaluate_corrected_finite_penalty_target
    except Exception as exc:  # pragma: no cover - depends on installation
        return {"detcorr_available": False, "reason": repr(exc)}

    projected = solved_state_mv(gamma)
    state = projected + np.array([0.04, -0.03, 0.02, -0.01])
    metric = algorithmic_metric(gamma)
    target = evaluate_corrected_finite_penalty_target(
        constraint_residual_mv,
        gamma,
        state,
        rho=PENALTY_RHO,
        target_measure="graph",
        R=metric,
        base_logdensity=float(log_prior_gamma(gamma)) + observation_loglik_state(state),
        projected_state=projected,
        log_correction_kwargs={
            "backend": "scipy",
            "finite_difference_scheme": "central",
            "singular_policy": "raise",
        },
        metadata={
            "metric_role": "algorithmic_constraint_metric_not_observation_noise",
            "residual_unit": STATE_UNIT,
            "state_jacobian": "identity",
        },
    )
    expected_weight = metric_log_sqrt_det(gamma)
    residual = constraint_residual_mv(gamma, state)
    expected_norm = float(residual @ metric @ residual)
    return {
        "detcorr_available": True,
        "gamma": gamma,
        "primary_state_correction": float(target.primary_correction),
        "primary_abs_error": abs(float(target.primary_correction)),
        "state_jacobian_route": "callable residual with central finite differences",
        "secondary_weight_correction": float(target.secondary_correction),
        "expected_secondary_weight_correction": expected_weight,
        "secondary_abs_error": abs(float(target.secondary_correction) - expected_weight),
        "weighted_norm_squared": float(target.residual_norm_squared),
        "weighted_norm_abs_error": abs(float(target.residual_norm_squared) - expected_norm),
        "target_measure": target.target_measure,
        "diagnostics": target.sampler_diagnostics(),
    }


def metric_audit_table(gamma_values: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for gamma in np.asarray(gamma_values, dtype=float):
        metric = algorithmic_metric(float(gamma))
        eigenvalues = np.linalg.eigvalsh(metric)
        rows.append(
            {
                "gamma": gamma,
                "weight_volume_half_logdet": metric_log_sqrt_det(float(gamma)),
                "metric_min_eigenvalue_per_mV2": float(eigenvalues[0]),
                "metric_max_eigenvalue_per_mV2": float(eigenvalues[-1]),
                "metric_condition_number": float(eigenvalues[-1] / eigenvalues[0]),
            }
        )
    return pd.DataFrame(rows)


def _plot_posteriors(
    path: Path,
    gamma: np.ndarray,
    densities: dict[str, np.ndarray],
) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.plot(
        gamma,
        densities["finite_naive"],
        color="tab:orange",
        linewidth=1.8,
        label="Exact finite tube, missing weight volume",
    )
    ax.plot(
        gamma,
        densities["finite_corrected"],
        color="tab:green",
        linewidth=2.2,
        label="Exact finite tube + 0.5 logdet R",
    )
    # Draw the coincident graph reference last so the agreement is visible.
    ax.plot(
        gamma,
        densities["graph_reduced"],
        color="tab:blue",
        linestyle="--",
        linewidth=1.6,
        label="Reduced graph",
    )
    ax.axvline(SYNTHETIC_TRUE_GAMMA, color="black", linestyle=":", label="Synthetic truth")
    ax.set_xlabel("gain gamma (dimensionless)")
    ax.set_ylabel("posterior density")
    ax.set_title("Parameter-dependent algorithmic constraint metric")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_metric(path: Path, gamma: np.ndarray) -> None:
    corrections = np.asarray([metric_log_sqrt_det(float(g)) for g in gamma])
    min_eigenvalue = np.asarray(
        [np.linalg.eigvalsh(algorithmic_metric(float(g)))[0] for g in gamma]
    )
    max_eigenvalue = np.asarray(
        [np.linalg.eigvalsh(algorithmic_metric(float(g)))[-1] for g in gamma]
    )
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6))
    axes[0].plot(gamma, corrections, color="tab:purple")
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_xlabel("gain gamma")
    axes[0].set_ylabel("0.5 logdet R")
    axes[0].set_title("Weight-volume correction")
    axes[1].semilogy(gamma, min_eigenvalue, label="minimum eigenvalue")
    axes[1].semilogy(gamma, max_eigenvalue, label="maximum eigenvalue")
    axes[1].set_xlabel("gain gamma")
    axes[1].set_ylabel("metric eigenvalue (mV^-2)")
    axes[1].set_title("SPD metric range")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_demo(outputs: Path, *, n_grid: int = GRID_SIZE, rho: float = PENALTY_RHO) -> dict[str, Any]:
    if n_grid < 101:
        raise ValueError("n_grid must be at least 101")
    outputs.mkdir(parents=True, exist_ok=True)
    tables = outputs / "tables"
    gamma = np.linspace(GAMMA_MIN, GAMMA_MAX, int(n_grid))
    graph = graph_logdensity(gamma)
    finite_naive = exact_integrated_tube_logdensity(
        gamma,
        rho=rho,
        include_weight_volume=False,
    )
    finite_corrected = exact_integrated_tube_logdensity(
        gamma,
        rho=rho,
        include_weight_volume=True,
    )
    targets = {
        "graph_reduced": graph,
        "finite_naive": finite_naive,
        "finite_corrected": finite_corrected,
    }
    posterior_table, densities, cdfs = common.summarize_targets(gamma, targets)
    common.write_table(posterior_table, tables / "posterior_summaries.csv")
    support_table = grid_support_audit(gamma, targets, cdfs)
    common.write_table(support_table, tables / "grid_support_audit.csv")

    selected = common.select_grid_points(gamma, 11)
    validation_table, validation_stats = common.centered_validation_table(
        "gamma",
        selected,
        exact_integrated_tube_logdensity(
            selected,
            rho=rho,
            include_weight_volume=False,
        ),
        hard_tube_logdensity(selected, corrected=False),
        exact_integrated_tube_logdensity(
            selected,
            rho=rho,
            include_weight_volume=True,
        ),
        graph_logdensity(selected),
    )
    validation_stats["integral_route"] = "exact_multivariate_gaussian_state_integral"
    coarse_rho = rho / 10.0
    coarse_corrected = exact_integrated_tube_logdensity(
        selected,
        rho=coarse_rho,
        include_weight_volume=True,
    )
    coarse_corrected_error = common.centered_max_abs_diff(
        coarse_corrected,
        graph_logdensity(selected),
    )
    validation_stats.update(
        {
            "rho": rho,
            "coarse_rho_for_convergence_check": coarse_rho,
            "coarse_rho_corrected_vs_graph_centered_max_abs_diff": coarse_corrected_error,
            "error_ratio_vs_tenfold_coarser_rho": (
                validation_stats["finite_corrected_vs_graph_centered_max_abs_diff"]
                / coarse_corrected_error
            ),
            "finite_penalty_error_limit": FINITE_PENALTY_ERROR_LIMIT,
        }
    )
    common.write_table(validation_table, tables / "finite_penalty_validation.csv")

    metric_table = metric_audit_table(common.select_grid_points(gamma, 9))
    common.write_table(metric_table, tables / "algorithmic_metric_audit.csv")
    audit = detcorr_weighted_api_audit()

    rows = posterior_table.set_index("target")
    graph_mean = float(rows.loc["graph_reduced", "mean"])
    graph_sd = float(rows.loc["graph_reduced", "sd"])
    naive_mean = float(rows.loc["finite_naive", "mean"])
    corrected_mean = float(rows.loc["finite_corrected", "mean"])
    correction_grid = np.asarray([metric_log_sqrt_det(float(g)) for g in gamma])
    metric_min_eigenvalue = min(
        float(np.linalg.eigvalsh(algorithmic_metric(float(g)))[0])
        for g in common.select_grid_points(gamma, 31)
    )
    corrected_error = common.centered_max_abs_diff(finite_corrected, graph)
    naive_l1 = common.density_l1(
        gamma,
        densities["finite_naive"],
        densities["graph_reduced"],
    )
    corrected_l1 = common.density_l1(
        gamma,
        densities["finite_corrected"],
        densities["graph_reduced"],
    )
    minimum_endpoint_drop = float(
        support_table[["lower_endpoint_log_drop", "upper_endpoint_log_drop"]].to_numpy().min()
    )
    maximum_edge_mass = float(
        support_table[["lower_edge_mass_2pct", "upper_edge_mass_2pct"]].to_numpy().max()
    )
    checks = {
        "synthetic_data_declared": True,
        "state_jacobian_is_identity": bool(np.array_equal(state_jacobian(), np.eye(N_CHANNEL))),
        "state_logdet_is_zero": True,
        "algorithmic_metric_is_spd": metric_min_eigenvalue > 0.0,
        "metric_is_not_observation_noise": True,
        "exact_finite_corrected_matches_graph": (corrected_error < FINITE_PENALTY_ERROR_LIMIT),
        "finite_penalty_bias_decreases_with_rho": bool(
            validation_stats["finite_corrected_vs_graph_centered_max_abs_diff"]
            < 0.2 * coarse_corrected_error
        ),
        "missing_weight_volume_changes_posterior": naive_l1 > 0.05,
        "posterior_grid_support_is_interior": bool(
            support_table["mode_is_interior"].all()
            and minimum_endpoint_drop > 25.0
            and maximum_edge_mass < 1.0e-8
        ),
        "detcorr_weight_correction_matches": bool(
            audit.get("detcorr_available") and audit.get("secondary_abs_error", 1.0) < 1.0e-10
        ),
        "detcorr_primary_correction_matches": bool(
            audit.get("detcorr_available") and audit.get("primary_abs_error", 1.0) < 1.0e-8
        ),
        "detcorr_weighted_norm_matches": bool(
            audit.get("detcorr_available") and audit.get("weighted_norm_abs_error", 1.0) < 1.0e-12
        ),
    }
    checks["all_passed"] = all(checks.values())

    _plot_posteriors(outputs / "sensor_gain_posteriors.png", gamma, densities)
    _plot_metric(outputs / "algorithmic_metric_correction.png", gamma)

    summary: dict[str, Any] = {
        "example": "sensor_weighted_constraint",
        "demo": "sensor_weighted_constraint",
        "classification": "transparent_synthetic_measure_geometry_demonstrator",
        "status": "synthetic_worked_demonstrator_not_empirical_calibration",
        "target_measure": "graph posterior over gamma with respect to d gamma",
        "correction_driver": (
            "parameter-dependent 0.5*logdet(R) volume term from an SPD "
            "algorithmic constraint metric; D_u c is the identity"
        ),
        "correction_log_range_on_grid": float(np.ptp(correction_grid)),
        "model": {
            "constraint": "c(gamma,u)=u-gamma*x",
            "state_jacobian": "identity",
            "state_logdet": 0.0,
            "state_unit": STATE_UNIT,
            "gain_unit": GAIN_UNIT,
            "reference_state_volume": "one mV per channel",
            "penalty_rho": rho,
            "penalty_rho_unit": "dimensionless because R is in mV^-2",
            "metric_role": "algorithmic_constraint_metric_not_observation_noise",
            "prior": "Normal(1.0,0.35^2) truncated to gamma>0, with respect to d gamma",
            "grid_bounds": [GAMMA_MIN, GAMMA_MAX],
        },
        "synthetic_data": {
            "true_gamma": SYNTHETIC_TRUE_GAMMA,
            "base_signal_mV": BASE_SIGNAL_MV.tolist(),
            "observation_sd_mV": OBSERVATION_SD_MV.tolist(),
            "fixed_noise_offsets_sd": FIXED_NOISE_OFFSETS_SD.tolist(),
            "observed_signal_mV": OBSERVED_SIGNAL_MV.tolist(),
            "random_sampling_used": False,
        },
        "correction": {
            "primary_state_logdet": 0.0,
            "weight_half_logdet_at_graph_mean": metric_log_sqrt_det(graph_mean),
            "weight_half_logdet_min": float(np.min(correction_grid)),
            "weight_half_logdet_max": float(np.max(correction_grid)),
            "weight_half_logdet_peak_to_peak": float(np.ptp(correction_grid)),
        },
        "posterior": {
            "graph_mean": graph_mean,
            "naive_finite_tube_mean": naive_mean,
            "corrected_finite_tube_mean": corrected_mean,
            "naive_minus_graph_mean": naive_mean - graph_mean,
            "corrected_minus_graph_mean": corrected_mean - graph_mean,
            "naive_density_l1_vs_graph": naive_l1,
            "corrected_centered_logdensity_max_abs_diff": corrected_error,
        },
        "outcomes": {
            "correction_log_range_on_grid": float(np.ptp(correction_grid)),
            "naive_mean_shift_in_graph_sd": (naive_mean - graph_mean) / graph_sd,
            "corrected_density_l1_vs_graph": corrected_l1,
            "naive_density_l1_vs_graph": naive_l1,
            "naive_corrected_density_l1": common.density_l1(
                gamma,
                densities["finite_naive"],
                densities["finite_corrected"],
            ),
        },
        "outcome": {
            "correction_log_range_on_grid": float(np.ptp(correction_grid)),
            "naive_mean_shift_in_graph_sd": (naive_mean - graph_mean) / graph_sd,
            "corrected_mean_shift_in_graph_sd": (corrected_mean - graph_mean) / graph_sd,
            "naive_minus_corrected_mean_in_graph_sd": (naive_mean - corrected_mean) / graph_sd,
            "corrected_density_l1_vs_graph": corrected_l1,
            "naive_density_l1_vs_graph": naive_l1,
        },
        "independent_validation": {
            "route": "exact multivariate Gaussian integration over the latent channels",
            **validation_stats,
        },
        "finite_penalty_validation": validation_stats,
        "grid_support": {
            "minimum_endpoint_log_drop": minimum_endpoint_drop,
            "maximum_two_percent_edge_mass": maximum_edge_mass,
            "table": "tables/grid_support_audit.csv",
        },
        "detcorr_weighted_api_audit": audit,
        "checks": checks,
        "limitations": [
            "The four-channel data are synthetic and use fixed declared offsets.",
            "The metric variation is deliberately strong enough to expose the volume bias.",
            "R(gamma) is a numerical constraint metric, not a sensor covariance model.",
            "The scalar-gain model omits channel-specific calibration and model discrepancy.",
            "Log determinants are reported relative to the declared mV state volume.",
        ],
        "interpretation_boundary": (
            "Validates the weight-volume correction for a declared algorithmic "
            "metric; it is not empirical sensor calibration or a noise model."
        ),
        "outputs": {
            "posterior_figure": "sensor_gain_posteriors.png",
            "metric_figure": "algorithmic_metric_correction.png",
            "posterior_table": "tables/posterior_summaries.csv",
            "validation_table": "tables/finite_penalty_validation.csv",
            "metric_table": "tables/algorithmic_metric_audit.csv",
            "grid_support_table": "tables/grid_support_audit.csv",
        },
        "provenance": common.reproducibility_provenance(example_root=ROOT),
    }
    common.write_json(outputs / "summary.json", summary)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs", type=Path, default=ROOT / "outputs")
    parser.add_argument("--n-grid", type=int, default=GRID_SIZE)
    parser.add_argument("--rho", type=float, default=PENALTY_RHO)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    summary = run_demo(args.outputs, n_grid=args.n_grid, rho=args.rho)
    print(json.dumps({"example": summary["example"], "checks": summary["checks"]}, indent=2))


if __name__ == "__main__":
    main()
