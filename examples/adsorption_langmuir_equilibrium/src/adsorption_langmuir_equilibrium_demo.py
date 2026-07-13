#!/usr/bin/env python3
"""Synthetic Langmuir-equilibrium worked example for detcorr.

The coverage q is dimensionless and lies in (0, 1), pressure P is measured in
bar, and K=exp(eta)/bar. The implicit equilibrium constraint is

    c(eta, q; P) = logit(q) - log(K P) = 0,

with D_q c = 1 / [q(1-q)]. The example compares a reduced graph posterior with
naive and pointwise determinant-corrected finite-penalty state integrals.
Those state integrals are evaluated directly in q using stable, local
Gauss-Legendre quadrature; they are not reconstructed from graph-minus-logdet
identities.

All observations are synthetic. The truth, pressures, measurement uncertainty,
and deterministic offsets are fixed in data/SYNTHETIC_DESIGN.json.
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
    density_l1,
    normalize_on_grid,
    reproducibility_provenance,
    summarize_targets,
    try_detcorr_logdet_diag,
    write_json,
    write_table,
)

DEFAULT_DESIGN = ROOT / "data" / "SYNTHETIC_DESIGN.json"
DEFAULT_OUTPUTS = ROOT / "outputs"
DEFAULT_ETA_MIN = -1.5
DEFAULT_ETA_MAX = 1.1
DEFAULT_N_GRID = 801
DEFAULT_RHO = 1.0e5
DEFAULT_QUADRATURE_NODES = 48
DEFAULT_REFERENCE_NODES = 64
DEFAULT_WINDOW_SD = 10.0
PRIOR_ETA_SD = 1.0


def sigmoid(value: np.ndarray | float) -> np.ndarray:
    """Stable logistic function."""

    x = np.asarray(value, dtype=float)
    out = np.empty_like(x)
    positive = x >= 0.0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out


def logit(q: np.ndarray | float) -> np.ndarray:
    """Return log(q/(1-q)) for coverages strictly inside (0, 1)."""

    values = np.asarray(q, dtype=float)
    if np.any((values <= 0.0) | (values >= 1.0)):
        raise ValueError("coverage q must lie strictly inside (0, 1)")
    return np.log(values) - np.log1p(-values)


def solved_coverage(eta: np.ndarray | float, pressure_bar: np.ndarray) -> np.ndarray:
    """Solve q=KP/(1+KP) with K=exp(eta)/bar."""

    pressure = np.asarray(pressure_bar, dtype=float)
    if pressure.ndim != 1 or np.any(pressure <= 0.0):
        raise ValueError("pressures must be a positive one-dimensional array in bar")
    eta_values = np.asarray(eta, dtype=float)
    return sigmoid(eta_values[..., None] + np.log(pressure))


def constraint(
    eta: float,
    q: np.ndarray,
    pressure_bar: np.ndarray,
) -> np.ndarray:
    """Evaluate c=logit(q)-log(KP), with numeric pressure expressed in bar."""

    pressure = np.asarray(pressure_bar, dtype=float)
    if np.any(pressure <= 0.0):
        raise ValueError("pressure must be positive")
    return logit(q) - (float(eta) + np.log(pressure))


def state_jacobian_diagonal(q: np.ndarray) -> np.ndarray:
    """Return the exact diagonal D_q c=1/[q(1-q)]."""

    values = np.asarray(q, dtype=float)
    if np.any((values <= 0.0) | (values >= 1.0)):
        raise ValueError("coverage q must lie strictly inside (0, 1)")
    return 1.0 / (values * (1.0 - values))


def load_synthetic_design(path: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    """Load and validate the predeclared deterministic synthetic design."""

    design = json.loads(path.read_text(encoding="utf-8"))
    if design.get("status") != "synthetic_validation_only":
        raise ValueError("the design must declare synthetic_validation_only status")
    pressure = np.asarray(design["pressure_bar"], dtype=float)
    offsets = np.asarray(design["coverage_offsets"], dtype=float)
    if pressure.ndim != 1 or offsets.shape != pressure.shape or pressure.size < 3:
        raise ValueError("pressures and fixed offsets must be equal-length vectors")
    if np.any(pressure <= 0.0):
        raise ValueError("all pressures must be positive")
    sigma = float(design["coverage_sigma"])
    eta_true = float(design["eta_true"])
    k_true = float(design["K_true_per_bar"])
    if sigma <= 0.0 or k_true <= 0.0:
        raise ValueError("coverage_sigma and K_true_per_bar must be positive")
    if not math.isclose(math.log(k_true), eta_true, rel_tol=0.0, abs_tol=1.0e-14):
        raise ValueError("eta_true must equal log(K_true_per_bar * bar)")

    q_truth = solved_coverage(eta_true, pressure)
    q_observed = q_truth + offsets
    if np.any((q_observed <= 0.0) | (q_observed >= 1.0)):
        raise ValueError("fixed offsets must leave every observed coverage in (0, 1)")
    table = pd.DataFrame(
        {
            "pressure_bar": pressure,
            "q_truth": q_truth,
            "fixed_offset": offsets,
            "q_observed": q_observed,
            "sigma_q": sigma,
            "observation_status": "declared deterministic synthetic observation",
        }
    )
    return design, table


def log_prior_eta(eta: np.ndarray, *, sd: float = PRIOR_ETA_SD) -> np.ndarray:
    values = np.asarray(eta, dtype=float)
    return -0.5 * (values / float(sd)) ** 2


def reduced_graph_logdensity(eta_grid: np.ndarray, data: pd.DataFrame) -> np.ndarray:
    """Reduced posterior log density with q eliminated analytically."""

    eta = np.asarray(eta_grid, dtype=float)
    predicted = solved_coverage(eta, data["pressure_bar"].to_numpy(float))
    observed = data["q_observed"].to_numpy(float)
    sigma = data["sigma_q"].to_numpy(float)
    return log_prior_eta(eta) - 0.5 * np.sum(
        ((observed[None, :] - predicted) / sigma[None, :]) ** 2,
        axis=1,
    )


def graph_logdet(eta_grid: np.ndarray, pressure_bar: np.ndarray) -> np.ndarray:
    q_star = solved_coverage(np.asarray(eta_grid, dtype=float), pressure_bar)
    return np.sum(np.log(state_jacobian_diagonal(q_star)), axis=1)


def _row_logsumexp(log_values: np.ndarray) -> np.ndarray:
    maximum = np.max(log_values, axis=1)
    return maximum + np.log(np.sum(np.exp(log_values - maximum[:, None]), axis=1))


def finite_penalty_state_integrals(
    eta_grid: np.ndarray,
    data: pd.DataFrame,
    *,
    rho: float,
    n_nodes: int,
    window_sd: float = DEFAULT_WINDOW_SD,
) -> dict[str, Any]:
    """Integrate the full q-state tube by local Gauss-Legendre quadrature.

    For each eta and pressure, the q interval corresponds to
    |c(eta,q;P)| <= window_sd/sqrt(rho). The omitted two-sided Gaussian tail is
    bounded by erfc(window_sd/sqrt(2)). The integration itself is performed in
    q, with the pointwise determinant 1/[q(1-q)] included only in the corrected
    integrand.
    """

    eta = np.asarray(eta_grid, dtype=float)
    if eta.ndim != 1 or eta.size < 2 or np.any(np.diff(eta) <= 0.0):
        raise ValueError("eta_grid must be a strictly increasing vector")
    if rho <= 0.0 or int(n_nodes) < 2 or window_sd <= 0.0:
        raise ValueError("rho, n_nodes, and window_sd must be positive")

    base_nodes, base_weights = np.polynomial.legendre.leggauss(int(n_nodes))
    half_residual_width = float(window_sd / math.sqrt(rho))
    log_naive = log_prior_eta(eta)
    log_corrected = log_prior_eta(eta)

    for row in data.itertuples(index=False):
        center = eta + math.log(float(row.pressure_bar))
        q_low = sigmoid(center - half_residual_width)
        q_high = sigmoid(center + half_residual_width)
        midpoint = 0.5 * (q_high + q_low)
        half_width = 0.5 * (q_high - q_low)
        q_nodes = midpoint[:, None] + half_width[:, None] * base_nodes[None, :]
        log_weights = np.log(base_weights)[None, :] + np.log(half_width)[:, None]
        residual = logit(q_nodes) - center[:, None]
        log_kernel = (
            -0.5 * ((float(row.q_observed) - q_nodes) / float(row.sigma_q)) ** 2
            - 0.5 * float(rho) * residual**2
            + log_weights
        )
        log_naive += _row_logsumexp(log_kernel)
        pointwise_logdet = -np.log(q_nodes) - np.log1p(-q_nodes)
        log_corrected += _row_logsumexp(log_kernel + pointwise_logdet)

    return {
        "log_naive": log_naive,
        "log_corrected": log_corrected,
        "n_nodes": int(n_nodes),
        "rho": float(rho),
        "window_sd": float(window_sd),
        "two_sided_gaussian_tail_bound": float(math.erfc(float(window_sd) / math.sqrt(2.0))),
        "route": "direct_local_gauss_legendre_in_q",
    }


def quadrature_convergence_table(
    eta_grid: np.ndarray,
    data: pd.DataFrame,
    *,
    rho: float,
    node_counts: list[int],
    reference_nodes: int,
    window_sd: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Compare each quadrature order with a higher-order q-space reference."""

    reference = finite_penalty_state_integrals(
        eta_grid,
        data,
        rho=rho,
        n_nodes=reference_nodes,
        window_sd=window_sd,
    )
    ref_naive_density, _, ref_naive_summary = normalize_on_grid(eta_grid, reference["log_naive"])
    ref_corrected_density, _, ref_corrected_summary = normalize_on_grid(
        eta_grid, reference["log_corrected"]
    )
    rows: list[dict[str, float | int]] = []
    for n_nodes in node_counts:
        result = finite_penalty_state_integrals(
            eta_grid,
            data,
            rho=rho,
            n_nodes=int(n_nodes),
            window_sd=window_sd,
        )
        naive_density, _, naive_summary = normalize_on_grid(eta_grid, result["log_naive"])
        corrected_density, _, corrected_summary = normalize_on_grid(
            eta_grid, result["log_corrected"]
        )
        rows.append(
            {
                "n_nodes": int(n_nodes),
                "naive_centered_log_error_vs_reference": centered_max_abs_diff(
                    result["log_naive"], reference["log_naive"]
                ),
                "corrected_centered_log_error_vs_reference": centered_max_abs_diff(
                    result["log_corrected"], reference["log_corrected"]
                ),
                "naive_density_l1_vs_reference": density_l1(
                    eta_grid, naive_density, ref_naive_density
                ),
                "corrected_density_l1_vs_reference": density_l1(
                    eta_grid, corrected_density, ref_corrected_density
                ),
                "naive_mean": naive_summary.mean,
                "corrected_mean": corrected_summary.mean,
            }
        )
    table = pd.DataFrame(rows)
    stats = {
        "reference_nodes": int(reference_nodes),
        "reference_naive_mean": ref_naive_summary.mean,
        "reference_corrected_mean": ref_corrected_summary.mean,
        "final_corrected_centered_log_error": float(
            table["corrected_centered_log_error_vs_reference"].iloc[-1]
        ),
        "final_corrected_density_l1_error": float(
            table["corrected_density_l1_vs_reference"].iloc[-1]
        ),
    }
    return table, stats


def detcorr_diagonal_audit(
    eta: float,
    pressure_bar: np.ndarray,
) -> dict[str, Any]:
    q_star = solved_coverage(float(eta), pressure_bar)
    diagonal = state_jacobian_diagonal(q_star)
    analytic = float(np.sum(np.log(diagonal)))
    audit = try_detcorr_logdet_diag(diagonal)
    audit.update(
        {
            "eta": float(eta),
            "analytic_logdet": analytic,
            "q_min": float(np.min(q_star)),
            "q_max": float(np.max(q_star)),
            "derivative": "D_q c_i = 1/[q_i(1-q_i)]",
            "target_measure": "graph",
        }
    )
    if audit.get("detcorr_available"):
        audit["absolute_error"] = abs(float(audit["value"]) - analytic)
    return audit


def make_plots(
    outputs: Path,
    eta_grid: np.ndarray,
    densities: dict[str, np.ndarray],
    data: pd.DataFrame,
    *,
    eta_true: float,
    eta_posterior_mean: float,
) -> None:
    outputs.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    pressure_plot = np.geomspace(0.02, 3.2, 300)
    ax.plot(
        pressure_plot,
        solved_coverage(eta_true, pressure_plot),
        label="declared synthetic truth",
        linewidth=2.0,
    )
    ax.plot(
        pressure_plot,
        solved_coverage(eta_posterior_mean, pressure_plot),
        linestyle="--",
        label="graph-posterior mean",
    )
    ax.errorbar(
        data["pressure_bar"],
        data["q_observed"],
        yerr=data["sigma_q"],
        fmt="o",
        capsize=3,
        label="synthetic observations",
    )
    ax.set_xscale("log")
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("pressure P (bar)")
    ax.set_ylabel("dimensionless coverage q")
    ax.set_title("Declared synthetic Langmuir isotherm")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outputs / "synthetic_langmuir_isotherm.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    labels = {
        "graph_reduced": "reduced graph",
        "naive_finite_penalty": "naive state integral",
        "detcorr_finite_penalty": "pointwise-corrected state integral",
    }
    styles = {
        "graph_reduced": {"linewidth": 2.6},
        "naive_finite_penalty": {"linewidth": 2.0},
        "detcorr_finite_penalty": {"linewidth": 2.0, "linestyle": "--"},
    }
    for name, density in densities.items():
        ax.plot(eta_grid, density, label=labels[name], **styles[name])
    ax.axvline(eta_true, color="black", linestyle=":", label="declared truth")
    ax.set_xlabel("dimensionless log-equilibrium parameter eta")
    ax.set_ylabel("posterior density")
    ax.set_title("Finite-penalty state-volume correction")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outputs / "posterior_eta_targets.png", dpi=180)
    plt.close(fig)


def _parse_node_counts(text: str) -> list[int]:
    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not values or any(value < 2 for value in values):
        raise argparse.ArgumentTypeError("quadrature node counts must be integers >= 2")
    if values != sorted(set(values)):
        raise argparse.ArgumentTypeError("quadrature node counts must be unique and increasing")
    return values


def run_demo(args: argparse.Namespace) -> dict[str, Any]:
    design_path = Path(args.design)
    outputs = Path(args.outputs)
    design, data = load_synthetic_design(design_path)
    eta_grid = np.linspace(args.eta_min, args.eta_max, args.n_grid)
    graph_log = reduced_graph_logdensity(eta_grid, data)
    integrals = finite_penalty_state_integrals(
        eta_grid,
        data,
        rho=args.rho,
        n_nodes=args.quadrature_nodes,
        window_sd=args.quadrature_window_sd,
    )
    target_table, densities, _ = summarize_targets(
        eta_grid,
        {
            "graph_reduced": graph_log,
            "naive_finite_penalty": integrals["log_naive"],
            "detcorr_finite_penalty": integrals["log_corrected"],
        },
    )
    convergence_table, convergence_stats = quadrature_convergence_table(
        eta_grid,
        data,
        rho=args.rho,
        node_counts=args.quadrature_convergence_nodes,
        reference_nodes=args.quadrature_reference_nodes,
        window_sd=args.quadrature_window_sd,
    )

    rows = target_table.set_index("target")
    graph_row = rows.loc["graph_reduced"]
    naive_row = rows.loc["naive_finite_penalty"]
    corrected_row = rows.loc["detcorr_finite_penalty"]
    pressure = data["pressure_bar"].to_numpy(float)
    correction = graph_logdet(eta_grid, pressure)
    correction_range = float(np.max(correction) - np.min(correction))
    naive_shift_sd = float((naive_row["mean"] - graph_row["mean"]) / graph_row["sd"])
    corrected_shift_sd = float((corrected_row["mean"] - graph_row["mean"]) / graph_row["sd"])
    naive_minus_corrected_shift_sd = float(
        (naive_row["mean"] - corrected_row["mean"]) / graph_row["sd"]
    )
    corrected_l1 = float(corrected_row["density_l1_vs_graph"])
    naive_l1 = float(naive_row["density_l1_vs_graph"])
    quadrature_error = float(convergence_stats["final_corrected_centered_log_error"])
    eta_true = float(design["eta_true"])
    audit = detcorr_diagonal_audit(float(graph_row["mode"]), pressure)

    table_dir = outputs / "tables"
    write_table(data, table_dir / "synthetic_observations.csv")
    write_table(target_table, table_dir / "posterior_targets.csv")
    write_table(convergence_table, table_dir / "quadrature_convergence.csv")
    correction_table = pd.DataFrame(
        {
            "eta": eta_grid,
            "graph_logdet_Dq_c": correction,
        }
    )
    write_table(correction_table, table_dir / "graph_logdet_on_grid.csv")
    make_plots(
        outputs,
        eta_grid,
        densities,
        data,
        eta_true=eta_true,
        eta_posterior_mean=float(graph_row["mean"]),
    )

    summary: dict[str, Any] = {
        "demo": "adsorption_langmuir_equilibrium",
        "classification": "synthetic nonlinear-equilibrium verification problem",
        "synthetic_status": "declared deterministic synthetic validation only",
        "target_measure": "graph posterior over eta with respect to d eta",
        "correction_driver": ("nonlinear equilibrium-state derivative sum_i log(1/[q_i(1-q_i)])"),
        "correction_log_range_on_grid": correction_range,
        "naive_mean_shift_in_graph_sd": naive_shift_sd,
        "corrected_density_l1_vs_graph": corrected_l1,
        "naive_density_l1_vs_graph": naive_l1,
        "quadrature_convergence_error": quadrature_error,
        "design": design,
        "model": {
            "K_definition": "K=exp(eta)/bar",
            "constraint": "c=logit(q)-log(KP)=0",
            "solved_state": "q=KP/(1+KP)",
            "state_jacobian": "D_q c=1/[q(1-q)]",
            "parameter": "eta, dimensionless",
            "state": "q in (0,1), dimensionless",
            "pressure": "P in bar",
            "target_measure": "graph posterior over eta",
            "observation_model": "fixed-sigma Gaussian coverage errors",
        },
        "configuration": {
            "eta_grid": {
                "minimum": float(args.eta_min),
                "maximum": float(args.eta_max),
                "n_points": int(args.n_grid),
            },
            "prior_eta_sd": PRIOR_ETA_SD,
            "rho": float(args.rho),
            "rho_units": "dimensionless",
            "quadrature_nodes": int(args.quadrature_nodes),
            "quadrature_reference_nodes": int(args.quadrature_reference_nodes),
            "quadrature_convergence_nodes": args.quadrature_convergence_nodes,
            "quadrature_window_sd": float(args.quadrature_window_sd),
            "quadrature_tail_bound": integrals["two_sided_gaussian_tail_bound"],
        },
        "outcomes": {
            "eta_true": eta_true,
            "graph_mean": float(graph_row["mean"]),
            "graph_sd": float(graph_row["sd"]),
            "naive_mean": float(naive_row["mean"]),
            "corrected_mean": float(corrected_row["mean"]),
            "truth_error_in_graph_sd": float((graph_row["mean"] - eta_true) / graph_row["sd"]),
            "correction_log_range_on_grid": correction_range,
            "naive_mean_shift_in_graph_sd": naive_shift_sd,
            "corrected_density_l1_vs_graph": corrected_l1,
            "naive_density_l1_vs_graph": naive_l1,
            "quadrature_convergence_error": quadrature_error,
        },
        "outcome": {
            "graph_mean": float(graph_row["mean"]),
            "graph_sd": float(graph_row["sd"]),
            "correction_log_range_on_grid": correction_range,
            "naive_mean_shift_in_graph_sd": naive_shift_sd,
            "corrected_mean_shift_in_graph_sd": corrected_shift_sd,
            "naive_minus_corrected_mean_in_graph_sd": (naive_minus_corrected_shift_sd),
            "naive_density_l1_vs_graph": naive_l1,
            "corrected_density_l1_vs_graph": corrected_l1,
        },
        "quadrature": {
            "route": integrals["route"],
            "pointwise_corrected_integrand": ("likelihood(q)*exp(-rho*c(q)^2/2)*abs(D_q c(q))"),
            "convergence": convergence_stats,
        },
        "detcorr_diagonal_audit": audit,
        "independent_validation": {
            "route": "direct local Gauss-Legendre state integration in q",
            "not_graph_minus_logdet_construction": True,
            "quadrature_reference_nodes": int(args.quadrature_reference_nodes),
            "corrected_centered_logdensity_convergence_error": quadrature_error,
            "detcorr_diagonal_abs_error": float(audit["absolute_error"]),
        },
        "interpretation_boundary": (
            "This synthetic example verifies a nonlinear equilibrium-state "
            "Jacobian and its target-measure correction. It is not evidence for "
            "the empirical adequacy of a one-site Langmuir model."
        ),
        "tables": {
            "posterior_targets": target_table.to_dict(orient="records"),
            "quadrature_convergence": convergence_table.to_dict(orient="records"),
        },
        "provenance": reproducibility_provenance(example_root=ROOT),
    }
    write_json(outputs / "summary.json", summary)

    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", default=str(DEFAULT_DESIGN))
    parser.add_argument("--outputs", default=str(DEFAULT_OUTPUTS))
    parser.add_argument("--eta-min", type=float, default=DEFAULT_ETA_MIN)
    parser.add_argument("--eta-max", type=float, default=DEFAULT_ETA_MAX)
    parser.add_argument("--n-grid", type=int, default=DEFAULT_N_GRID)
    parser.add_argument("--rho", type=float, default=DEFAULT_RHO)
    parser.add_argument(
        "--quadrature-nodes",
        type=int,
        default=DEFAULT_QUADRATURE_NODES,
    )
    parser.add_argument(
        "--quadrature-reference-nodes",
        type=int,
        default=DEFAULT_REFERENCE_NODES,
    )
    parser.add_argument(
        "--quadrature-convergence-nodes",
        type=_parse_node_counts,
        default=[8, 12, 16, 24, 32, 48],
    )
    parser.add_argument(
        "--quadrature-window-sd",
        type=float,
        default=DEFAULT_WINDOW_SD,
    )
    return parser


def main() -> None:
    run_demo(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
