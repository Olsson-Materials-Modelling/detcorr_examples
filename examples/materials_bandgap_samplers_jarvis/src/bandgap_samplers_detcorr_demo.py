#!/usr/bin/env python3
"""JARVIS tabulated-band-gap logarithmic-energy-residual demonstrator.

The data are externally tabulated JARVIS OPT and experimental band gaps; the
source does not classify measurement type row by row. A common dimensionless
scale ``gamma`` maps the OPT gap to a
latent dimensionless energy state ``e_i=E_i/E0``.  The active residual is

    c_i(gamma, e_i) = log(e_i / (gamma E_i^OPT/E0)) = 0.

On the graph, ``e_i=gamma E_i^OPT/E0`` and ``|D_e c_i|=1/e_i``.
The observation likelihood is evaluated in energy space.  Gauss--Hermite
quadrature in ``t_i=log(e_i/(gamma E_i^OPT/E0))`` independently checks the
globally proper finite-penalty tube.

The reciprocal residual ``1/u-gamma*x`` under Lebesgue ``du`` is included only
as a rejected-construction diagnostic: its finite-penalty state integral
diverges at ``u -> infinity``.  Its hard-limit target is audited globally but
does not rescue the improper finite tube.  It is not sampled or truncated.

MCMC and SMC sample only the reduced graph and deliberately naive hard-tube
targets.  The corrected target is validated separately by deterministic
hard-limit algebra and direct latent-state quadrature; it is not aliased to a
third reduced sampler target.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from functools import cache
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.polynomial.hermite import hermgauss
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parents[1]
COMMON = ROOT.parent / "detcorr_demo_common" / "src"
if str(COMMON) not in sys.path:
    sys.path.insert(0, str(COMMON))

from detcorr_demo_common import (  # noqa: E402
    centered_validation_table,
    normalize_on_grid,
    relative_display_path,
    reproducibility_provenance,
    select_grid_points,
    sha256_file,
    summarize_targets,
    try_detcorr_logdet_diag,
    write_json,
    write_table,
)

JARVIS_DB_DOCS_URL = "https://jarvis-materials-design.github.io/dbdocs/jarvisdft/"
JARVIS_NIST_URL = "https://www.nist.gov/programs-projects/jarvis-dft"
FIXTURE_SHA256 = "4b7a5f437b1a41030bb65c31d56eaa9524cc91598bd09ff2825c02df60528631"
ENERGY_REFERENCE_EV = 1.0
PENALTY_RHO_UNITS = "dimensionless"
REQUIRED_COLUMNS = [
    "material_label",
    "formula",
    "jvasp_id",
    "bandgap_opt_ev",
    "bandgap_expt_ev",
]


@dataclass(frozen=True)
class SampleSummary:
    name: str
    n_samples: int
    mean: float
    sd: float
    q05: float
    q50: float
    q95: float
    mcmc_autocorr_ess: float | None = None
    accept_rate: float | None = None
    final_weight_ess: float | None = None
    final_unique_particles: int | None = None
    final_unique_fraction: float | None = None
    n_resamples: int | None = None


def load_bandgap_fixture(data_path: Path) -> pd.DataFrame:
    """Load every row with positive, finite OPT and experimental gaps."""

    df = pd.read_csv(data_path)
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"band-gap fixture is missing columns: {missing}")
    digest = sha256_file(data_path)
    if digest != FIXTURE_SHA256:
        raise ValueError(f"fixture SHA256 mismatch: expected {FIXTURE_SHA256}, got {digest}")
    df = df.dropna(subset=REQUIRED_COLUMNS).copy()
    for column in ["bandgap_opt_ev", "bandgap_expt_ev"]:
        df[column] = pd.to_numeric(df[column], errors="raise")
    finite = np.isfinite(df["bandgap_opt_ev"]) & np.isfinite(df["bandgap_expt_ev"])
    positive = (df["bandgap_opt_ev"] > 0.0) & (df["bandgap_expt_ev"] > 0.0)
    df = df.loc[finite & positive].copy()
    if len(df) < 2:
        raise ValueError("band-gap fixture must contain at least two positive complete rows")
    if df["jvasp_id"].duplicated().any():
        raise ValueError("band-gap fixture contains duplicate jvasp_id values")

    # A fixed, outcome-independent demonstrator likelihood, not an estimated
    # experimental error law.  Its scale is fixed by the OPT input before the
    # experimental response is evaluated, avoiding response-dependent weights.
    predictor = df["bandgap_opt_ev"].to_numpy(float)
    df["sigma_gap_ev"] = np.sqrt((0.15 * predictor) ** 2 + 0.12**2)
    df["opt_gap_over_e0"] = df["bandgap_opt_ev"] / ENERGY_REFERENCE_EV
    df["expt_gap_over_e0"] = df["bandgap_expt_ev"] / ENERGY_REFERENCE_EV
    df["sigma_gap_over_e0"] = df["sigma_gap_ev"] / ENERGY_REFERENCE_EV
    df["expt_over_opt_ratio"] = df["bandgap_expt_ev"] / df["bandgap_opt_ev"]
    df["observation_origin"] = (
        "external JARVIS-DB documentation OPT and tabulated experimental-gap "
        "table; measurement type unresolved"
    )
    return df.reset_index(drop=True)


def reciprocal_state(
    gamma: np.ndarray | float,
    opt_gap_ev: np.ndarray,
    *,
    energy_reference_ev: float = ENERGY_REFERENCE_EV,
) -> np.ndarray:
    """Return the rejected-construction reciprocal state for diagnostics only."""

    return float(energy_reference_ev) / (
        np.asarray(gamma, dtype=float)[..., None] * np.asarray(opt_gap_ev, dtype=float)
    )


def energy_from_reciprocal_state(
    state_u: np.ndarray | float,
    *,
    energy_reference_ev: float = ENERGY_REFERENCE_EV,
) -> np.ndarray:
    state = np.asarray(state_u, dtype=float)
    return float(energy_reference_ev) / state


def dimensionless_energy_state(
    gamma: np.ndarray | float,
    opt_gap_ev: np.ndarray,
    *,
    energy_reference_ev: float = ENERGY_REFERENCE_EV,
) -> np.ndarray:
    """Return the active graph state e=gamma E_OPT/E0."""

    return (
        np.asarray(gamma, dtype=float)[..., None]
        * np.asarray(opt_gap_ev, dtype=float)
        / float(energy_reference_ev)
    )


def log_energy_residual(
    gamma: float,
    state_e: np.ndarray,
    opt_gap_ev: np.ndarray,
    *,
    energy_reference_ev: float = ENERGY_REFERENCE_EV,
) -> np.ndarray:
    """Evaluate c=log(e/(gamma E_OPT/E0)) on positive inputs."""

    e = np.asarray(state_e, dtype=float)
    x = np.asarray(opt_gap_ev, dtype=float) / float(energy_reference_ev)
    if gamma <= 0.0 or np.any(e <= 0.0) or np.any(x <= 0.0):
        raise ValueError("gamma, state_e, and scaled OPT gaps must be positive")
    return np.log(e / (float(gamma) * x))


def log_prior_gamma(gamma: float, *, log_sd: float = 0.62) -> float:
    """Lognormal prior density with respect to d gamma."""

    if gamma <= 0.0 or not math.isfinite(gamma):
        return float("-inf")
    z = math.log(gamma)
    return -0.5 * (z / log_sd) ** 2 - math.log(gamma)


def log_likelihood_gamma(gamma: float, df: pd.DataFrame) -> float:
    """Fixed Gaussian likelihood in energy, not in reciprocal energy."""

    if gamma <= 0.0 or not math.isfinite(gamma):
        return float("-inf")
    prediction_ev = float(gamma) * df["bandgap_opt_ev"].to_numpy(float)
    observed_ev = df["bandgap_expt_ev"].to_numpy(float)
    sigma_ev = df["sigma_gap_ev"].to_numpy(float)
    return -0.5 * float(np.sum(((observed_ev - prediction_ev) / sigma_ev) ** 2))


def state_logdet(
    gamma: float,
    df: pd.DataFrame,
    *,
    solved_energy_residual: bool = False,
) -> float:
    """Return log|det D_e c| for c=log(e/(gamma E_OPT/E0))."""

    if solved_energy_residual:
        return 0.0
    if gamma <= 0.0 or not math.isfinite(gamma):
        return float("-inf")
    x = df["opt_gap_over_e0"].to_numpy(float)
    diagonal_abs = 1.0 / (float(gamma) * x)
    return float(np.sum(np.log(diagonal_abs)))


def hard_limit_log_components(
    gamma_grid: np.ndarray,
    df: pd.DataFrame,
    *,
    residual_scale_power: int = 0,
    solved_energy_residual: bool = False,
) -> dict[str, np.ndarray]:
    """Construct graph, naive hard-tube, and corrected log targets."""

    gamma_grid = np.asarray(gamma_grid, dtype=float)
    n = int(len(df))
    log_graph = np.full_like(gamma_grid, -np.inf, dtype=float)
    logdet = np.full_like(gamma_grid, -np.inf, dtype=float)
    scale_logdet = np.full_like(gamma_grid, -np.inf, dtype=float)
    residual_rms_ev = np.full_like(gamma_grid, np.nan, dtype=float)
    observed = df["bandgap_expt_ev"].to_numpy(float)
    opt = df["bandgap_opt_ev"].to_numpy(float)
    for index, gamma in enumerate(gamma_grid):
        if gamma <= 0.0:
            continue
        prediction = float(gamma) * opt
        residual_rms_ev[index] = float(np.sqrt(np.mean((observed - prediction) ** 2)))
        log_graph[index] = log_likelihood_gamma(float(gamma), df) + log_prior_gamma(float(gamma))
        logdet[index] = state_logdet(
            float(gamma),
            df,
            solved_energy_residual=solved_energy_residual,
        )
        scale_logdet[index] = float(n * int(residual_scale_power) * math.log(float(gamma)))
    total = logdet + scale_logdet
    naive = log_graph - total
    corrected = naive + total
    return {
        "log_graph": log_graph,
        "state_logdet_correction": logdet,
        "residual_scale_logdet_correction": scale_logdet,
        "total_logdet_correction": total,
        "log_naive": naive,
        "log_fully_corrected": corrected,
        "residual_rms_ev": residual_rms_ev,
    }


def build_main_table(
    gamma_grid: np.ndarray,
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    log_energy = hard_limit_log_components(gamma_grid, df)
    solved = hard_limit_log_components(
        gamma_grid,
        df,
        solved_energy_residual=True,
    )
    table, densities, _ = summarize_targets(
        gamma_grid,
        {
            "graph_reduced": log_energy["log_graph"],
            "naive_log_energy_residual_tube": log_energy["log_naive"],
            "detcorr_log_energy_residual_tube": log_energy["log_fully_corrected"],
            "constant_det_energy_coordinate_tube": solved["log_naive"],
        },
    )
    return table, densities


def build_domain_sensitivity_table(
    gamma_grid: np.ndarray,
    df: pd.DataFrame,
    thresholds_ev: Iterable[float],
) -> pd.DataFrame:
    """Report graph-target sensitivity to declared OPT-gap domain thresholds."""

    rows: list[dict[str, Any]] = []
    for threshold in thresholds_ev:
        value = float(threshold)
        if value < 0.0 or not math.isfinite(value):
            raise ValueError("OPT-gap thresholds must be finite and nonnegative")
        subset = df.loc[df["bandgap_opt_ev"] >= value].copy()
        if len(subset) < 2:
            raise ValueError(f"threshold {value} eV leaves fewer than two rows")
        density, _, posterior = normalize_on_grid(
            gamma_grid,
            hard_limit_log_components(gamma_grid, subset)["log_graph"],
        )
        _ = density
        rows.append(
            {
                "minimum_opt_gap_ev": value,
                "n_materials": int(len(subset)),
                "graph_mean": posterior.mean,
                "graph_sd": posterior.sd,
                "graph_mode": posterior.mode,
                "graph_q05": posterior.q05,
                "graph_q95": posterior.q95,
            }
        )
    return pd.DataFrame(rows)


@cache
def _hermite_rule(n_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = hermgauss(int(n_nodes))
    nodes.setflags(write=False)
    weights.setflags(write=False)
    return nodes, weights


def finite_penalty_log_integral_one(
    gamma: float,
    opt_gap_ev: float,
    observed_gap_ev: float,
    sigma_gap_ev: float,
    rho: float,
    *,
    n_nodes: int = 128,
    include_state_correction: bool = False,
) -> float:
    """Direct Gauss--Hermite integral over one positive energy state.

    The dimensionless integral is

        integral_0^inf exp[-0.5((E_obs/E0-e)/(sigma/E0))^2
                             -0.5 rho log(e/(gamma*x))^2] de.

    It is evaluated directly and never formed as graph minus log determinant.
    """

    if gamma <= 0.0 or opt_gap_ev <= 0.0 or sigma_gap_ev <= 0.0 or rho <= 0.0:
        raise ValueError("gamma, gaps, sigma, and rho must be positive")
    if int(n_nodes) < 16:
        raise ValueError("n_nodes must be at least 16")
    x = float(opt_gap_ev) / ENERGY_REFERENCE_EV
    y = float(observed_gap_ev) / ENERGY_REFERENCE_EV
    sigma = float(sigma_gap_ev) / ENERGY_REFERENCE_EV
    nodes, weights = _hermite_rule(int(n_nodes))
    t_state = math.sqrt(2.0 / float(rho)) * nodes
    energy_over_e0 = float(gamma) * x * np.exp(t_state)
    # de=e dt and t=sqrt(2/rho)z.  For the corrected finite tube,
    # |D_e c|=1/e cancels the e from de.  Thus naive and corrected
    # integrals are independently assembled before quadrature.
    terms = (
        np.log(weights)
        + math.log(math.sqrt(2.0 / float(rho)))
        - 0.5 * ((y - energy_over_e0) / sigma) ** 2
    )
    if not include_state_correction:
        terms = terms + np.log(energy_over_e0)
    maximum = float(np.max(terms))
    return float(maximum + math.log(float(np.sum(np.exp(terms - maximum)))))


def finite_penalty_validation_table(
    gamma_grid: np.ndarray,
    df: pd.DataFrame,
    *,
    n_points: int = 9,
    rho: float = 1.0e8,
    n_nodes: int = 128,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Validate the hard-tube limit by direct product quadrature over all states."""

    grid = np.asarray(gamma_grid, dtype=float)
    if rho <= 0.0 or not math.isfinite(rho):
        raise ValueError("rho must be positive and finite")
    if int(n_points) < 2:
        raise ValueError("n_points must be at least two")
    if grid.size < 3 or not np.all(np.isfinite(grid)) or np.any(np.diff(grid) <= 0.0):
        raise ValueError("gamma_grid must contain at least three increasing finite values")
    lower_index = max(0, int(0.12 * (grid.size - 1)))
    upper_index = min(grid.size, int(0.88 * (grid.size - 1)) + 1)
    values = select_grid_points(grid[lower_index:upper_index], n_points=n_points)
    components = hard_limit_log_components(values, df)
    opt = df["bandgap_opt_ev"].to_numpy(float)
    observed = df["bandgap_expt_ev"].to_numpy(float)
    sigma = df["sigma_gap_ev"].to_numpy(float)
    finite_logs: list[float] = []
    finite_corrected_logs: list[float] = []
    for gamma in values:
        log_integral = log_prior_gamma(float(gamma))
        log_corrected_integral = log_prior_gamma(float(gamma))
        for opt_i, observed_i, sigma_i in zip(opt, observed, sigma, strict=True):
            log_integral += finite_penalty_log_integral_one(
                float(gamma),
                float(opt_i),
                float(observed_i),
                float(sigma_i),
                float(rho),
                n_nodes=int(n_nodes),
            )
            log_corrected_integral += finite_penalty_log_integral_one(
                float(gamma),
                float(opt_i),
                float(observed_i),
                float(sigma_i),
                float(rho),
                n_nodes=int(n_nodes),
                include_state_correction=True,
            )
        finite_logs.append(log_integral)
        finite_corrected_logs.append(log_corrected_integral)
    finite = np.asarray(finite_logs, dtype=float)
    finite_corrected = np.asarray(finite_corrected_logs, dtype=float)
    table, stats = centered_validation_table(
        "gamma",
        values,
        finite,
        components["log_naive"],
        finite_corrected,
        components["log_graph"],
    )
    table = table.rename(columns={"finite_penalty_plus_logdet": "finite_corrected_tube_logdensity"})
    table["rho"] = float(rho)
    table["rho_units"] = PENALTY_RHO_UNITS
    table["n_materials"] = int(len(df))
    table["n_quadrature_nodes_per_state"] = int(n_nodes)
    stats.update(
        {
            "rho": float(rho),
            "rho_units": PENALTY_RHO_UNITS,
            "n_materials": int(len(df)),
            "n_quadrature_nodes_per_state": int(n_nodes),
            "integral_route": (
                "global_gauss_hermite_in_log_energy_residual_over_positive_energy_states"
            ),
            "load_bearing_metric": "finite_vs_hard_tube_centered_max_abs_diff",
            "load_bearing_metrics": [
                "finite_vs_hard_tube_centered_max_abs_diff",
                "finite_corrected_vs_graph_centered_max_abs_diff",
            ],
            "finite_corrected_vs_graph_role": (
                "independent corrected finite-tube integration: the state-dependent "
                "1/e correction is included inside each state integral"
            ),
        }
    )
    return table, stats


def detcorr_api_audit(gamma: float, df: pd.DataFrame) -> dict[str, Any]:
    x = df["opt_gap_over_e0"].to_numpy(float)
    diagonal = 1.0 / (float(gamma) * x)
    analytic = float(np.sum(np.log(np.abs(diagonal))))
    audit = try_detcorr_logdet_diag(diagonal)
    result: dict[str, Any] = {
        "target_measure": "graph",
        "gamma": float(gamma),
        "analytic_state_logdet": analytic,
        "constant_energy_coordinate_logdet": 0.0,
        "detcorr_available": bool(audit.get("detcorr_available")),
        "reason": audit.get("reason"),
        "detcorr_state_logdet": None,
        "state_abs_error": None,
    }
    if audit.get("detcorr_available"):
        result["detcorr_state_logdet"] = float(audit["value"])
        result["state_abs_error"] = abs(float(audit["value"]) - analytic)
        result["state_logdet_result"] = audit["result"]
    return result


def rejected_reciprocal_construction_diagnostic(
    df: pd.DataFrame,
    *,
    display_bounds: tuple[float, float] = (0.75, 2.85),
    z_min: float = -120.0,
    z_max: float = 20.0,
    n_points: int = 28001,
) -> dict[str, Any]:
    """Document why the reciprocal Lebesgue finite tube is not admissible.

    For u -> infinity, both 1/u terms vanish.  The likelihood and penalty
    converge to finite positive constants, so the integrand does not tend to
    zero and its integral over du diverges.
    """

    n = int(len(df))
    z = np.linspace(float(z_min), float(z_max), int(n_points))
    gamma = np.exp(z)
    x = df["opt_gap_over_e0"].to_numpy(float)
    constant = 2.0 * float(np.sum(np.log(x)))
    log_density_z = np.array(
        [
            log_likelihood_gamma(float(value), df)
            + log_prior_gamma(float(value))
            - (2.0 * n * float(log_value) + constant)
            + float(log_value)
            for value, log_value in zip(gamma, z, strict=True)
        ],
        dtype=float,
    )
    mode_index = int(np.argmax(log_density_z))
    lower_gamma, upper_gamma = display_bounds
    local = (gamma >= lower_gamma) & (gamma <= upper_gamma)
    remote = ~local

    def stable_logsum(values: np.ndarray) -> float:
        maximum = float(np.max(values))
        return maximum + math.log(float(np.sum(np.exp(values - maximum))))

    local_log_mass = stable_logsum(log_density_z[local])
    outside_log_mass = stable_logsum(log_density_z[remote])
    global_log_mass = stable_logsum(log_density_z)
    local_mass_fraction = math.exp(local_log_mass - global_log_mass)
    maximum = float(log_density_z[mode_index])
    edge = max(2, int(math.ceil(0.01 * len(z))))
    weights = np.exp(log_density_z - maximum)
    weights /= float(np.sum(weights))
    return {
        "construction": "u=E0/E; c=1/u-gamma*E_OPT/E0 under Lebesgue du",
        "status": "rejected_not_an_active_target",
        "finite_penalty_state_integral": "diverges for every finite positive gamma and rho",
        "divergence_reason": (
            "as u tends to infinity the integrand tends to a strictly positive "
            "gamma-dependent constant, hence its du integral is infinite"
        ),
        "hard_limit_naive_global_support": (
            "broad-grid audit places essentially all normalized hard-limit mass in the "
            "display range; this does not make the finite tube integrable"
        ),
        "verified_log_gamma_grid": {
            "minimum": float(z_min),
            "maximum": float(z_max),
            "n_points": int(n_points),
            "density_measure": "dz after adding +z to the dgamma log density",
        },
        "verified_global_mode_log_gamma": float(z[mode_index]),
        "verified_global_mode_gamma": float(gamma[mode_index]),
        "lower_endpoint_log_drop": maximum - float(log_density_z[0]),
        "upper_endpoint_log_drop": maximum - float(log_density_z[-1]),
        "lower_edge_mass_1pct": float(np.sum(weights[:edge])),
        "upper_edge_mass_1pct": float(np.sum(weights[-edge:])),
        "local_display_gamma_bounds": [float(lower_gamma), float(upper_gamma)],
        "local_display_mass_fraction": local_mass_fraction,
        "mass_outside_display_grid": 1.0 - local_mass_fraction,
        "log_mass_outside_minus_inside_display": outside_log_mass - local_log_mass,
    }


def autocorr_ess(samples: np.ndarray, max_lag: int | None = None) -> float:
    values = np.asarray(samples, dtype=float)
    n_values = int(values.size)
    if n_values < 4:
        return float(n_values)
    centered = values - float(np.mean(values))
    variance = float(np.dot(centered, centered) / n_values)
    if variance <= 0.0 or not math.isfinite(variance):
        return float(n_values)
    if max_lag is None:
        max_lag = min(1000, n_values // 2)
    correlation_sum = 0.0
    for lag in range(1, int(max_lag) + 1):
        correlation = float(np.dot(centered[:-lag], centered[lag:]) / ((n_values - lag) * variance))
        if correlation <= 0.0:
            break
        correlation_sum += correlation
    return float(max(1.0, min(n_values, n_values / (1.0 + 2.0 * correlation_sum))))


def summarize_samples(
    name: str,
    samples: np.ndarray,
    *,
    calculate_mcmc_ess: bool = False,
    accept_rate: float | None = None,
    final_weight_ess: float | None = None,
    final_unique_particles: int | None = None,
    n_resamples: int | None = None,
) -> SampleSummary:
    values = np.asarray(samples, dtype=float)
    q05, q50, q95 = np.quantile(values, [0.05, 0.5, 0.95])
    unique_fraction = None
    if final_unique_particles is not None:
        unique_fraction = float(final_unique_particles / max(int(values.size), 1))
    return SampleSummary(
        name=name,
        n_samples=int(values.size),
        mean=float(np.mean(values)),
        sd=float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        q05=float(q05),
        q50=float(q50),
        q95=float(q95),
        mcmc_autocorr_ess=autocorr_ess(values) if calculate_mcmc_ess else None,
        accept_rate=accept_rate,
        final_weight_ess=final_weight_ess,
        final_unique_particles=final_unique_particles,
        final_unique_fraction=unique_fraction,
        n_resamples=n_resamples,
    )


def log_density_gamma(gamma: float, df: pd.DataFrame, *, target: str) -> float:
    if gamma <= 0.0:
        return float("-inf")
    graph = log_likelihood_gamma(gamma, df) + log_prior_gamma(gamma)
    if target == "graph":
        return graph
    if target == "naive":
        return graph - state_logdet(gamma, df)
    raise ValueError(f"unknown target {target!r}")


def global_log_gamma_support_audit(
    df: pd.DataFrame,
    *,
    display_bounds: tuple[float, float],
    z_min: float = -120.0,
    z_max: float = 20.0,
    n_points: int = 28001,
) -> pd.DataFrame:
    """Normalize active targets on a deliberately broad log-gamma support.

    The target functions are densities with respect to d gamma.  Therefore the
    log-density in z=log(gamma) includes the +z change-of-variable term.  This
    audit justifies use of the compact plotting grid; it is not a second model.
    """

    lower_gamma, upper_gamma = display_bounds
    if lower_gamma <= 0.0 or upper_gamma <= lower_gamma:
        raise ValueError("display_bounds must be positive and ordered")
    if z_max <= z_min or int(n_points) < 1001:
        raise ValueError("global z bounds or resolution are invalid")
    z = np.linspace(float(z_min), float(z_max), int(n_points))
    gamma = np.exp(z)
    display = (gamma >= lower_gamma) & (gamma <= upper_gamma)
    edge = max(2, int(math.ceil(0.01 * len(z))))
    rows: list[dict[str, Any]] = []
    for target in ("graph", "naive"):
        log_density_z = np.array(
            [
                log_density_gamma(float(value), df, target=target) + float(log_value)
                for value, log_value in zip(gamma, z, strict=True)
            ],
            dtype=float,
        )
        maximum = float(np.max(log_density_z))
        weights = np.exp(log_density_z - maximum)
        weights /= float(np.sum(weights))
        mode_index = int(np.argmax(log_density_z))
        rows.append(
            {
                "target": target,
                "z_min": float(z_min),
                "z_max": float(z_max),
                "n_points": int(n_points),
                "mode_log_gamma": float(z[mode_index]),
                "mode_gamma": float(gamma[mode_index]),
                "lower_endpoint_log_drop": maximum - float(log_density_z[0]),
                "upper_endpoint_log_drop": maximum - float(log_density_z[-1]),
                "lower_edge_mass_1pct": float(np.sum(weights[:edge])),
                "upper_edge_mass_1pct": float(np.sum(weights[-edge:])),
                "mass_outside_display_grid": float(np.sum(weights[~display])),
                "display_gamma_min": float(lower_gamma),
                "display_gamma_max": float(upper_gamma),
                "density_measure_audited": "dz after adding +z to the dgamma log density",
            }
        )
    return pd.DataFrame(rows)


def run_mcmc(
    df: pd.DataFrame,
    *,
    target: str,
    n_steps: int,
    burn: int,
    proposal_sd: float,
    seed: int,
) -> tuple[np.ndarray, SampleSummary]:
    if int(n_steps) <= 0:
        raise ValueError("n_steps must be positive")
    if int(burn) < 0 or int(burn) >= int(n_steps):
        raise ValueError("burn must satisfy 0 <= burn < n_steps")
    if proposal_sd <= 0.0 or not math.isfinite(proposal_sd):
        raise ValueError("proposal_sd must be positive and finite")
    rng = np.random.default_rng(seed)
    z = math.log(1.55)

    def log_density_z(value: float) -> float:
        gamma = math.exp(float(value))
        return log_density_gamma(gamma, df, target=target) + float(value)

    current = log_density_z(z)
    kept: list[float] = []
    accepts = 0
    for step in range(int(n_steps)):
        proposal = float(z + rng.normal(0.0, proposal_sd))
        proposed = log_density_z(proposal)
        if math.log(rng.random()) < proposed - current:
            z = proposal
            current = proposed
            accepts += 1
        if step >= int(burn):
            kept.append(math.exp(z))
    samples = np.asarray(kept, dtype=float)
    summary = summarize_samples(
        f"mcmc_{target}",
        samples,
        calculate_mcmc_ess=True,
        accept_rate=float(accepts / max(int(n_steps), 1)),
    )
    return samples, summary


def smc_tempering(
    df: pd.DataFrame,
    *,
    target: str,
    n_particles: int,
    n_temper: int,
    move_steps: int,
    proposal_sd: float,
    seed: int,
) -> tuple[np.ndarray, SampleSummary, pd.DataFrame]:
    if int(n_particles) < 2:
        raise ValueError("n_particles must be at least two")
    if int(n_temper) < 1 or int(move_steps) < 1:
        raise ValueError("n_temper and move_steps must be positive")
    if proposal_sd <= 0.0 or not math.isfinite(proposal_sd):
        raise ValueError("proposal_sd must be positive and finite")
    rng = np.random.default_rng(seed)
    prior_sd = 0.62
    z = rng.normal(0.0, prior_sd, size=int(n_particles))
    log_weights = np.zeros(int(n_particles), dtype=float)
    n_resamples = 0
    rows: list[dict[str, Any]] = []

    def potential(z_values: np.ndarray) -> np.ndarray:
        gammas = np.exp(z_values)
        likelihood = np.array(
            [log_likelihood_gamma(float(gamma), df) for gamma in gammas],
            dtype=float,
        )
        if target == "graph":
            return likelihood
        if target == "naive":
            return likelihood - np.array(
                [state_logdet(float(gamma), df) for gamma in gammas],
                dtype=float,
            )
        raise ValueError(f"unknown SMC target {target!r}")

    def log_prior_z(z_values: np.ndarray) -> np.ndarray:
        return -0.5 * (z_values / prior_sd) ** 2

    def weight_ess(values: np.ndarray) -> float:
        shifted = values - float(np.max(values))
        weights = np.exp(shifted)
        return float(np.sum(weights) ** 2 / np.sum(weights * weights))

    temperatures = np.linspace(0.0, 1.0, int(n_temper) + 1)
    target_potential = potential(z)
    for temperature_index in range(1, len(temperatures)):
        previous_temperature = float(temperatures[temperature_index - 1])
        temperature = float(temperatures[temperature_index])
        log_weights += (temperature - previous_temperature) * target_potential
        ess_before = weight_ess(log_weights)
        resampled = ess_before < 0.55 * n_particles
        if resampled:
            probabilities = np.exp(log_weights - float(np.max(log_weights)))
            probabilities /= float(np.sum(probabilities))
            indices = rng.choice(
                np.arange(n_particles),
                size=n_particles,
                replace=True,
                p=probabilities,
            )
            z = z[indices]
            log_weights = np.zeros_like(log_weights)
            target_potential = potential(z)
            n_resamples += 1
        ess_after_resample = weight_ess(log_weights)
        accepts = 0
        for _ in range(int(move_steps)):
            proposal = z + rng.normal(0.0, proposal_sd, size=z.shape)
            proposal_potential = potential(proposal)
            log_current = log_prior_z(z) + temperature * target_potential
            log_proposal = log_prior_z(proposal) + temperature * proposal_potential
            accepted = np.log(rng.random(size=z.shape)) < (log_proposal - log_current)
            z[accepted] = proposal[accepted]
            target_potential[accepted] = proposal_potential[accepted]
            accepts += int(np.sum(accepted))
        rows.append(
            {
                "target": target,
                "temperature_index": int(temperature_index),
                "beta": temperature,
                "weight_ess_before_optional_resample": ess_before,
                "resampled_this_step": bool(resampled),
                "weight_ess_after_optional_resample": ess_after_resample,
                "move_accept_rate": accepts / max(int(move_steps) * int(n_particles), 1),
                "n_resamples_so_far": int(n_resamples),
            }
        )
    probabilities = np.exp(log_weights - float(np.max(log_weights)))
    probabilities /= float(np.sum(probabilities))
    final_weight_ess = weight_ess(log_weights)
    indices = rng.choice(
        np.arange(n_particles),
        size=n_particles,
        replace=True,
        p=probabilities,
    )
    samples = np.exp(z[indices])
    final_unique_particles = int(np.unique(samples).size)
    summary = summarize_samples(
        f"smc_{target}",
        samples,
        final_weight_ess=final_weight_ess,
        final_unique_particles=final_unique_particles,
        n_resamples=n_resamples,
    )
    return samples, summary, pd.DataFrame(rows)


def augmented_lagrangian_gamma_objective(
    gamma: float,
    energy_state: np.ndarray,
    scaled_dual: np.ndarray,
    opt_over_e0: np.ndarray,
    rho: float,
) -> float:
    """Gamma subproblem in the constant-determinant energy coordinate."""

    if gamma <= 0.0:
        return float("inf")
    residual = energy_state - float(gamma) * opt_over_e0 + scaled_dual
    return float(-log_prior_gamma(float(gamma)) + 0.5 * rho * np.sum(residual**2))


def run_alternating_augmented_lagrangian_map(
    df: pd.DataFrame,
    *,
    gamma_bounds: tuple[float, float],
    rho: float,
    n_iter: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run a fixed-iteration constrained-MAP feasibility comparison.

    The optimizer uses the constant-determinant state ``v=E/E0`` and constraint
    ``v-gamma E_OPT/E0=0``.  It is not a sampler and is not evidence for the
    log-energy-residual determinant correction.
    """

    gamma_min, gamma_max = gamma_bounds
    if gamma_min <= 0.0 or gamma_max <= gamma_min:
        raise ValueError("gamma_bounds must be positive and strictly ordered")
    if rho <= 0.0 or not math.isfinite(rho) or int(n_iter) < 1:
        raise ValueError("rho and n_iter must be positive")
    x = df["opt_gap_over_e0"].to_numpy(float)
    y = df["expt_gap_over_e0"].to_numpy(float)
    sigma = df["sigma_gap_over_e0"].to_numpy(float)
    gamma = 1.55
    energy_state = gamma * x
    scaled_dual = np.zeros_like(x)
    rows: list[dict[str, Any]] = []
    for iteration in range(int(n_iter)):
        observation_precision = 1.0 / (sigma * sigma)
        energy_state = (observation_precision * y + rho * (gamma * x - scaled_dual)) / (
            observation_precision + rho
        )
        previous_gamma = float(gamma)
        result = minimize_scalar(
            augmented_lagrangian_gamma_objective,
            args=(energy_state, scaled_dual, x, rho),
            bounds=gamma_bounds,
            method="bounded",
            options={"xatol": 1.0e-12},
        )
        if not bool(result.success):
            raise RuntimeError(
                f"bounded gamma subproblem failed at iteration {iteration}: {result.message}"
            )
        gamma = float(result.x)
        primal = energy_state - gamma * x
        scaled_dual = scaled_dual + primal
        rows.append(
            {
                "iteration": int(iteration),
                "gamma": gamma,
                "objective_augmented": float(result.fun),
                "feasibility_rms_dimensionless": float(np.sqrt(np.mean(primal**2))),
                "observation_rms_ev": float(
                    ENERGY_REFERENCE_EV * np.sqrt(np.mean((energy_state - y) ** 2))
                ),
                "gamma_step_abs": abs(gamma - previous_gamma),
                "gamma_subproblem_success": bool(result.success),
            }
        )
    table = pd.DataFrame(rows)
    summary = {
        "gamma_augmented_lagrangian": float(gamma),
        "feasibility_rms_dimensionless": float(table["feasibility_rms_dimensionless"].iloc[-1]),
        "observation_rms_ev": float(table["observation_rms_ev"].iloc[-1]),
        "final_gamma_step_abs": float(table["gamma_step_abs"].iloc[-1]),
        "all_gamma_subproblems_succeeded": bool(table["gamma_subproblem_success"].all()),
        "n_iter": int(n_iter),
        "rho": float(rho),
        "rho_units": PENALTY_RHO_UNITS,
        "method": "fixed-iteration alternating augmented-Lagrangian heuristic",
        "coordinate": "constant-determinant energy state v=E/E0",
        "target": "constrained graph MAP; optimization only, not posterior sampling",
    }
    return table, summary


def make_plots(
    gamma_grid: np.ndarray,
    densities: dict[str, np.ndarray],
    df: pd.DataFrame,
    domain_sensitivity: pd.DataFrame,
    mcmc_samples: dict[str, np.ndarray],
    smc_samples: dict[str, np.ndarray],
    optimizer_trace: pd.DataFrame,
    outputs: Path,
) -> None:
    outputs.mkdir(parents=True, exist_ok=True)

    fig, axis = plt.subplots(figsize=(7.3, 4.4))
    axis.plot(
        gamma_grid,
        densities["naive_log_energy_residual_tube"],
        linewidth=1.5,
        label="naive log-energy-residual tube",
    )
    axis.plot(
        gamma_grid,
        densities["detcorr_log_energy_residual_tube"],
        linewidth=2.2,
        alpha=0.75,
        label="detcorr tube",
    )
    axis.plot(
        gamma_grid,
        densities["constant_det_energy_coordinate_tube"],
        linestyle=":",
        linewidth=2.0,
        label="constant-det energy coordinate",
    )
    axis.plot(
        gamma_grid,
        densities["graph_reduced"],
        color="black",
        linestyle="--",
        linewidth=1.6,
        label="reduced graph target",
    )
    axis.set_xlabel("common OPT tabulated-gap scale gamma (dimensionless)")
    axis.set_ylabel("posterior density")
    axis.set_title("Log-energy residual changes the naive tube")
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outputs / "posterior_gamma_log_energy_residual.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(6.0, 5.2))
    axis.scatter(df["bandgap_opt_ev"], df["bandgap_expt_ev"], s=26, alpha=0.85)
    upper = 1.05 * max(
        float(df["bandgap_opt_ev"].max()),
        float(df["bandgap_expt_ev"].max()),
    )
    axis.plot([0.0, upper], [0.0, upper], linestyle="--", linewidth=1.0)
    axis.set_xlabel("JARVIS OPT gap / eV")
    axis.set_ylabel("tabulated experimental gap / eV")
    axis.set_title("External JARVIS comparison-table fixture")
    fig.tight_layout()
    fig.savefig(outputs / "bandgap_fixture_opt_vs_expt.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7.0, 4.2))
    axis.errorbar(
        domain_sensitivity["minimum_opt_gap_ev"],
        domain_sensitivity["graph_mean"],
        yerr=domain_sensitivity["graph_sd"],
        marker="o",
        capsize=3,
    )
    for row in domain_sensitivity.itertuples(index=False):
        axis.annotate(
            f"n={row.n_materials}",
            (row.minimum_opt_gap_ev, row.graph_mean),
            xytext=(4, 6),
            textcoords="offset points",
            fontsize=8,
        )
    axis.set_xlabel("minimum included OPT gap / eV")
    axis.set_ylabel("graph-target mean gamma (error bar: 1 SD)")
    axis.set_title("Declared domain-threshold sensitivity")
    fig.tight_layout()
    fig.savefig(outputs / "domain_threshold_sensitivity.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7.2, 4.3))
    axis.hist(
        mcmc_samples["graph"],
        bins=36,
        density=True,
        alpha=0.42,
        label="MCMC graph",
    )
    axis.hist(
        smc_samples["graph"],
        bins=36,
        density=True,
        alpha=0.42,
        label="SMC graph",
    )
    axis.plot(
        gamma_grid,
        densities["graph_reduced"],
        color="black",
        linewidth=1.5,
        label="grid reference",
    )
    axis.set_xlabel("gamma")
    axis.set_ylabel("density")
    axis.set_title("Reduced-target sampler check")
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outputs / "mcmc_smc_gamma_check.png", dpi=180)
    plt.close(fig)

    if not optimizer_trace.empty:
        fig, axis = plt.subplots(figsize=(7.0, 4.2))
        axis.plot(
            optimizer_trace["iteration"],
            optimizer_trace["gamma"],
            marker="o",
            markersize=2,
        )
        axis.set_xlabel("alternating augmented-Lagrangian iteration")
        axis.set_ylabel("gamma")
        axis.set_title("Constrained-MAP feasibility trace")
        fig.tight_layout()
        fig.savefig(outputs / "alternating_augmented_lagrangian_gamma_trace.png", dpi=180)
        plt.close(fig)


def run_demo(args: argparse.Namespace) -> dict[str, Any]:
    if args.gamma_min <= 0.0 or args.gamma_max <= args.gamma_min:
        raise ValueError("gamma grid bounds must be positive and strictly ordered")
    if int(args.n_grid) < 3:
        raise ValueError("n_grid must be at least three")
    data_path = Path(args.data)
    df = load_bandgap_fixture(data_path)
    gamma_grid = np.linspace(args.gamma_min, args.gamma_max, args.n_grid)
    main_table, densities = build_main_table(gamma_grid, df)
    global_support = global_log_gamma_support_audit(
        df,
        display_bounds=(args.gamma_min, args.gamma_max),
    )
    if (
        float(global_support[["lower_edge_mass_1pct", "upper_edge_mass_1pct"]].to_numpy().max())
        > 1.0e-10
        or float(global_support["mass_outside_display_grid"].max()) > 1.0e-8
        or float(
            global_support[["lower_endpoint_log_drop", "upper_endpoint_log_drop"]].to_numpy().min()
        )
        < 100.0
    ):
        raise RuntimeError("global log-gamma support audit failed")
    domain_sensitivity = build_domain_sensitivity_table(
        gamma_grid,
        df,
        thresholds_ev=[0.0, 0.2, 0.5, 1.0],
    )
    validation_table, validation_stats = finite_penalty_validation_table(
        gamma_grid,
        df,
        n_points=args.validation_points,
        rho=args.validation_rho,
        n_nodes=args.validation_nodes,
    )
    graph_mode = float(main_table.loc[main_table["target"] == "graph_reduced", "mode"].iloc[0])
    audit = detcorr_api_audit(graph_mode, df)
    rejected_reciprocal = rejected_reciprocal_construction_diagnostic(
        df,
        display_bounds=(args.gamma_min, args.gamma_max),
    )

    mcmc_samples: dict[str, np.ndarray] = {}
    mcmc_summaries: list[dict[str, Any]] = []
    for index, target in enumerate(["graph", "naive"]):
        samples, sampler_summary = run_mcmc(
            df,
            target=target,
            n_steps=args.mcmc_steps,
            burn=args.mcmc_burn,
            proposal_sd=args.mcmc_proposal_sd,
            seed=args.seed + index,
        )
        mcmc_samples[target] = samples
        mcmc_summaries.append(asdict(sampler_summary))

    smc_samples: dict[str, np.ndarray] = {}
    smc_summaries: list[dict[str, Any]] = []
    smc_diagnostics: list[pd.DataFrame] = []
    for index, target in enumerate(["graph", "naive"]):
        samples, sampler_summary, diagnostics = smc_tempering(
            df,
            target=target,
            n_particles=args.smc_particles,
            n_temper=args.smc_temper_steps,
            move_steps=args.smc_move_steps,
            proposal_sd=args.smc_proposal_sd,
            seed=args.seed + 100 + index,
        )
        smc_samples[target] = samples
        smc_summaries.append(asdict(sampler_summary))
        smc_diagnostics.append(diagnostics)
    smc_diagnostic_table = pd.concat(smc_diagnostics, ignore_index=True)

    optimizer_trace, optimizer_summary = run_alternating_augmented_lagrangian_map(
        df,
        gamma_bounds=(args.gamma_min, args.gamma_max),
        rho=args.augmented_lagrangian_rho,
        n_iter=args.augmented_lagrangian_iters,
    )
    optimizer_summary["grid_graph_mode"] = graph_mode
    optimizer_summary["abs_mode_error"] = abs(
        float(optimizer_summary["gamma_augmented_lagrangian"]) - graph_mode
    )

    grid_rows = {row["target"]: row for row in main_table.to_dict(orient="records")}
    target_map = {
        "mcmc_graph": "graph_reduced",
        "mcmc_naive": "naive_log_energy_residual_tube",
        "smc_graph": "graph_reduced",
        "smc_naive": "naive_log_energy_residual_tube",
    }
    sampler_rows: list[dict[str, Any]] = []
    for row in [*mcmc_summaries, *smc_summaries]:
        output = dict(row)
        reference_name = target_map[str(row["name"])]
        output["grid_reference_target"] = reference_name
        output["grid_reference_mean"] = float(grid_rows[reference_name]["mean"])
        output["mean_abs_error_vs_grid"] = abs(
            float(row["mean"]) - float(grid_rows[reference_name]["mean"])
        )
        sampler_rows.append(output)
    sampler_table = pd.DataFrame(sampler_rows)
    for column in ["final_unique_particles", "n_resamples"]:
        sampler_table[column] = pd.array(sampler_table[column], dtype="Int64")

    outputs = Path(args.outputs)
    table_directory = outputs / "tables"
    write_table(main_table, table_directory / "main_hard_limit_posteriors.csv")
    write_table(
        domain_sensitivity,
        table_directory / "opt_gap_domain_threshold_sensitivity.csv",
    )
    write_table(df, table_directory / "external_bandgap_fixture_positive.csv")
    write_table(
        validation_table,
        table_directory / "finite_penalty_tube_validation.csv",
    )
    write_table(
        pd.DataFrame([rejected_reciprocal]),
        table_directory / "rejected_reciprocal_construction.csv",
    )
    write_table(
        global_support,
        table_directory / "global_log_gamma_support_audit.csv",
    )
    write_table(sampler_table, table_directory / "mcmc_smc_sampler_summaries.csv")
    write_table(
        smc_diagnostic_table,
        table_directory / "smc_tempering_diagnostics.csv",
    )
    write_table(
        optimizer_trace,
        table_directory / "alternating_augmented_lagrangian_map_trace.csv",
    )
    make_plots(
        gamma_grid,
        densities,
        df,
        domain_sensitivity,
        mcmc_samples,
        smc_samples,
        optimizer_trace,
        outputs,
    )

    graph_summary = grid_rows["graph_reduced"]
    naive_summary = grid_rows["naive_log_energy_residual_tube"]
    corrected_summary = grid_rows["detcorr_log_energy_residual_tube"]
    graph_sd = float(graph_summary["sd"])
    correction_values = np.array(
        [state_logdet(float(gamma), df) for gamma in gamma_grid],
        dtype=float,
    )
    summary = {
        "demo": "materials_bandgap_samplers_jarvis",
        "classification": "external_data_simplified_calibration_demonstrator",
        "provenance": reproducibility_provenance(example_root=ROOT),
        "target_measure": "graph posterior over gamma with respect to d gamma",
        "correction_driver": (
            "log-energy residual derivative |D_e c_i|="
            "1/(gamma E_i^OPT/E0) repeated across 54 external table rows"
        ),
        "correction_log_range_on_grid": float(np.ptp(correction_values)),
        "dataset": {
            "fixture_path": relative_display_path(data_path, root=ROOT),
            "fixture_sha256": sha256_file(data_path),
            "source_record": JARVIS_DB_DOCS_URL,
            "source_nist": JARVIS_NIST_URL,
            "n_materials": int(len(df)),
            "observation_status": (
                "external_repository_tabulated_gap_values; experimental measurement type unresolved"
            ),
            "fixture_preparation": "pinned transcription of the public comparison table",
            "active_filter": "finite positive OPT and experimental gaps only",
            "source_verification_date": "2026-07-11",
            "licence_status": (
                "CC BY 4.0 applies to the source Scientific Data Table 3; attribution, "
                "change, third-party-material, and non-endorsement notices are retained "
                "in case_studies/THIRD_PARTY_NOTICES.md"
            ),
        },
        "configuration": {
            "seed": int(args.seed),
            "gamma_grid": {
                "minimum": float(args.gamma_min),
                "maximum": float(args.gamma_max),
                "n_points": int(args.n_grid),
            },
            "energy_reference_ev": ENERGY_REFERENCE_EV,
            "fixed_likelihood_sigma": (
                "sqrt((0.15 E_OPT)^2 + (0.12 eV)^2); fixed before observing E_expt"
            ),
            "finite_penalty_validation": {
                "rho": float(args.validation_rho),
                "rho_units": PENALTY_RHO_UNITS,
                "n_points": int(args.validation_points),
                "n_nodes_per_state": int(args.validation_nodes),
            },
            "mcmc": {
                "n_steps": int(args.mcmc_steps),
                "burn": int(args.mcmc_burn),
                "proposal_sd_log_gamma": float(args.mcmc_proposal_sd),
            },
            "smc": {
                "n_particles": int(args.smc_particles),
                "n_temper_steps": int(args.smc_temper_steps),
                "move_steps": int(args.smc_move_steps),
                "proposal_sd_log_gamma": float(args.smc_proposal_sd),
            },
            "alternating_augmented_lagrangian": {
                "rho": float(args.augmented_lagrangian_rho),
                "rho_units": PENALTY_RHO_UNITS,
                "n_iterations": int(args.augmented_lagrangian_iters),
            },
        },
        "model": {
            "full_space_base_measure": (
                "d gamma times the product of de_i over all positive dimensionless "
                "energy state coordinates e_i=E_i/E0"
            ),
            "prior": {
                "distribution": "lognormal",
                "density_measure": "d gamma",
                "log_parameter_distribution": "log(gamma) ~ Normal(0, 0.62^2)",
                "log_location": 0.0,
                "log_scale": 0.62,
                "support": "gamma > 0",
            },
            "target_measure": "graph",
            "state_base_density": "q(e)=1 with respect to product_i de_i",
            "correction_measure_contract": (
                "graph multiplier is J/q; here J=product_i(1/e_i) and q=1"
            ),
            "parameter_units": {"gamma": "dimensionless"},
            "state_units": {"e_i=E_i/E0": "dimensionless"},
            "residual_units": {"c_i": "dimensionless"},
            "energy_reference_ev": ENERGY_REFERENCE_EV,
            "constraint": "c_i(gamma,e_i)=log(e_i/(gamma E_i^OPT/E0))=0",
            "state_jacobian": (
                "D_e c_i=1/e_i=1/(gamma E_i^OPT/E0); "
                "constant-det energy residual c'=E_i/E0-gamma E_i^OPT/E0 has D c'=1"
            ),
            "likelihood": (
                "Gaussian in energy against external tabulated experimental gaps; "
                "measurement type unresolved"
            ),
            "dependent_observation": (
                "bandgap_expt_ev from JARVIS-DB documentation comparison table"
            ),
            "reference_input": ("bandgap_opt_ev from JARVIS-DB documentation comparison table"),
            "no_fit_to_answer": True,
            "sampling_targets": ["reduced_graph", "naive_hard_limit_tube"],
            "samplers": ["random_walk_metropolis", "smc_tempering"],
            "optimizer": "fixed_iteration_alternating_augmented_lagrangian_map",
            "corrected_tube_validation": [
                "deterministic_hard_limit",
                "global_finite_penalty_positive_energy_state_quadrature",
            ],
        },
        "detcorr_api_audit": audit,
        "rejected_reciprocal_construction": rejected_reciprocal,
        "global_log_gamma_support_audit": global_support.to_dict(orient="records"),
        "anti_circular_finite_penalty_validation": validation_stats,
        "independent_validation": {
            "route": (
                "global Gauss-Hermite integration in each log-energy residual over every "
                "positive energy state; no graph-minus-logdet construction"
            ),
            **validation_stats,
        },
        "outcome": {
            "correction_log_range_on_grid": float(np.ptp(correction_values)),
            "corrected_density_l1_vs_graph": float(corrected_summary["density_l1_vs_graph"]),
            "corrected_density_l1_role": (
                "algebraic hard-limit round-trip consistency; not independent validation"
            ),
            "naive_density_l1_vs_graph": float(naive_summary["density_l1_vs_graph"]),
            "corrected_mean_shift_in_graph_sd": (
                float(corrected_summary["mean"]) - float(graph_summary["mean"])
            )
            / graph_sd,
            "naive_mean_shift_in_graph_sd": (
                float(naive_summary["mean"]) - float(graph_summary["mean"])
            )
            / graph_sd,
            "naive_minus_corrected_mean_in_graph_sd": (
                float(naive_summary["mean"]) - float(corrected_summary["mean"])
            )
            / graph_sd,
            "graph_mean": float(graph_summary["mean"]),
            "graph_sd": graph_sd,
        },
        "optimizer_summary": optimizer_summary,
        "interpretation_boundary": (
            "gamma is a fixture-level conditional scale under a deliberately simple fixed "
            "likelihood; it is not a transferable material correction and does not support "
            "inference beyond this fixture and model"
        ),
        "tables": {
            "main": main_table.to_dict(orient="records"),
            "domain_threshold_sensitivity": domain_sensitivity.to_dict(orient="records"),
            "sampler_summaries": sampler_rows,
        },
    }
    write_json(outputs / "summary.json", summary)

    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        default=str(ROOT / "data" / "jarvis_bandgap_benchmark_fixture.csv"),
    )
    parser.add_argument("--outputs", default=str(ROOT / "outputs"))
    parser.add_argument("--gamma-min", type=float, default=0.75)
    parser.add_argument("--gamma-max", type=float, default=2.85)
    parser.add_argument("--n-grid", type=int, default=1600)
    parser.add_argument("--validation-points", type=int, default=9)
    parser.add_argument("--validation-rho", type=float, default=3.0e9)
    parser.add_argument("--validation-nodes", type=int, default=128)
    parser.add_argument("--seed", type=int, default=202612)
    parser.add_argument("--mcmc-steps", type=int, default=10000)
    parser.add_argument("--mcmc-burn", type=int, default=1800)
    parser.add_argument("--mcmc-proposal-sd", type=float, default=0.026)
    parser.add_argument("--smc-particles", type=int, default=3000)
    parser.add_argument("--smc-temper-steps", type=int, default=22)
    parser.add_argument("--smc-move-steps", type=int, default=2)
    parser.add_argument("--smc-proposal-sd", type=float, default=0.026)
    parser.add_argument("--augmented-lagrangian-rho", type=float, default=25.0)
    parser.add_argument("--augmented-lagrangian-iters", type=int, default=120)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    summary = run_demo(args)
    print(f"Wrote JARVIS band-gap log-energy-residual outputs to {args.outputs}")
    print(
        json.dumps(
            {
                "demo": summary["demo"],
                "n_materials": summary["dataset"]["n_materials"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
