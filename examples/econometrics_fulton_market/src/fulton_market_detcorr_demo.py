#!/usr/bin/env python3
"""Held-out Fulton fish-price calibration example for determinant correction.

The first 64 chronologically ordered observations fit a reduced-form price
reference.  Only the final 33 observations enter the calibration likelihood.
This separation avoids an in-sample OLS identity that would force the target
towards a scale of one.  Exact Gaussian integration over the held-out latent
price states independently checks the graph-measure correction.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
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

DATA_PATH = ROOT / "data" / "fish.csv"
MANIFEST_PATH = ROOT / "data" / "FIXTURE_MANIFEST.json"
DATA_URL = (
    "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/wooldridge/fish.csv"
)
DATA_SHA256 = "fa51db236cb3f9e7040cdc50b002542bcbb3d51077132848d1960a996ec9b3c9"

TRAIN_SIZE = 64
GRID_MIN = 0.25
GRID_MAX = 2.0
GRID_SIZE = 2401
PRIOR_LOG_SD = 0.5
PENALTY_RHO = 5.0e7

DESIGN_COLUMNS = ("wave2", "speed2", "mon", "tues", "wed", "thurs", "t")
REQUIRED_COLUMNS = ("rownames", "avgprc", *DESIGN_COLUMNS)


@dataclass(frozen=True)
class ReferenceFit:
    columns: tuple[str, ...]
    coefficients: tuple[float, ...]
    residual_sd_native_price: float
    training_price_scale: float
    training_r2: float
    n_train: int
    n_parameters: int


@dataclass(frozen=True)
class HeldoutProblem:
    frame: pd.DataFrame
    train: pd.DataFrame
    heldout: pd.DataFrame
    train_fitted_native: np.ndarray
    heldout_reference_native: np.ndarray
    observed_nondimensional: np.ndarray
    reference_nondimensional: np.ndarray
    sigma_nondimensional: float
    fit: ReferenceFit

    @property
    def n_state(self) -> int:
        return int(self.observed_nondimensional.size)


def load_fish(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load, validate, and chronologically order the pinned Fulton CSV."""

    frame = pd.read_csv(path)
    missing = [name for name in REQUIRED_COLUMNS if name not in frame.columns]
    if missing:
        raise ValueError(f"fish.csv is missing required columns: {missing}")
    frame = frame.dropna(subset=list(REQUIRED_COLUMNS)).copy()
    frame = frame.sort_values("t", kind="stable").reset_index(drop=True)
    if len(frame) != 97:
        raise ValueError(f"expected 97 complete observations, found {len(frame)}")
    if np.any(frame["avgprc"].to_numpy(dtype=float) <= 0.0):
        raise ValueError("avgprc must be strictly positive")
    times = frame["t"].to_numpy(dtype=float)
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("t must be strictly increasing after sorting")
    return frame


def design_matrix(frame: pd.DataFrame) -> np.ndarray:
    values = frame.loc[:, DESIGN_COLUMNS].to_numpy(dtype=float)
    return np.column_stack([np.ones(len(frame), dtype=float), values])


def prepare_heldout_problem(
    frame: pd.DataFrame,
    *,
    train_size: int = TRAIN_SIZE,
) -> HeldoutProblem:
    """Fit on the chronological training block and construct held-out inputs.

    Held-out prices are never used to estimate the reference coefficients,
    residual scale, or nondimensionalizing price scale.
    """

    ordered = frame.sort_values("t", kind="stable").reset_index(drop=True).copy()
    n_parameters = len(DESIGN_COLUMNS) + 1
    if train_size <= n_parameters + 2 or train_size >= len(ordered) - 5:
        raise ValueError("train_size leaves too little training or held-out data")
    train = ordered.iloc[:train_size].copy()
    heldout = ordered.iloc[train_size:].copy()

    x_train = design_matrix(train)
    x_heldout = design_matrix(heldout)
    y_train = train["avgprc"].to_numpy(dtype=float)
    coefficients, *_ = np.linalg.lstsq(x_train, y_train, rcond=None)
    train_fitted = x_train @ coefficients
    heldout_reference = x_heldout @ coefficients
    if np.any(heldout_reference <= 0.0):
        raise ValueError("held-out price references must be strictly positive")

    residual = y_train - train_fitted
    dof = len(train) - x_train.shape[1]
    sigma_native = math.sqrt(float(residual @ residual) / dof)
    price_scale = float(np.mean(y_train))
    if sigma_native <= 0.0 or price_scale <= 0.0:
        raise ValueError("training residual SD and training price scale must be positive")
    centered = y_train - float(np.mean(y_train))
    r2 = 1.0 - float(residual @ residual) / float(centered @ centered)

    fit = ReferenceFit(
        columns=("const", *DESIGN_COLUMNS),
        coefficients=tuple(float(value) for value in coefficients),
        residual_sd_native_price=float(sigma_native),
        training_price_scale=price_scale,
        training_r2=float(r2),
        n_train=int(len(train)),
        n_parameters=int(x_train.shape[1]),
    )
    return HeldoutProblem(
        frame=ordered,
        train=train,
        heldout=heldout,
        train_fitted_native=np.asarray(train_fitted, dtype=float),
        heldout_reference_native=np.asarray(heldout_reference, dtype=float),
        observed_nondimensional=(heldout["avgprc"].to_numpy(dtype=float) / price_scale),
        reference_nondimensional=np.asarray(heldout_reference / price_scale, dtype=float),
        sigma_nondimensional=float(sigma_native / price_scale),
        fit=fit,
    )


def lognormal_prior(lambda_scale: float | np.ndarray, *, log_sd: float) -> np.ndarray:
    values = np.asarray(lambda_scale, dtype=float)
    result = np.full_like(values, -np.inf, dtype=float)
    valid = values > 0.0
    log_values = np.log(values[valid])
    result[valid] = (
        -np.log(values[valid] * log_sd * math.sqrt(2.0 * math.pi))
        - 0.5 * (log_values / log_sd) ** 2
    )
    return result


def solved_state(lambda_scale: float, reference: np.ndarray) -> np.ndarray:
    """Return the nondimensional constrained price state ``u=a/lambda``."""

    if lambda_scale <= 0.0:
        raise ValueError("lambda_scale must be positive")
    return np.asarray(reference, dtype=float) / float(lambda_scale)


def constraint_residual(
    lambda_scale: float,
    state: np.ndarray,
    reference: np.ndarray,
) -> np.ndarray:
    """Return ``c(lambda,u)=a-lambda*u`` in nondimensional price units."""

    return np.asarray(reference, dtype=float) - float(lambda_scale) * np.asarray(state, dtype=float)


def observation_loglikelihood(state: np.ndarray, problem: HeldoutProblem) -> float:
    standardized = (
        problem.observed_nondimensional - np.asarray(state, dtype=float)
    ) / problem.sigma_nondimensional
    return float(
        -0.5 * (standardized @ standardized)
        - problem.n_state * math.log(problem.sigma_nondimensional * math.sqrt(2.0 * math.pi))
    )


def graph_logdensity(
    lambda_grid: np.ndarray,
    problem: HeldoutProblem,
    *,
    prior_log_sd: float = PRIOR_LOG_SD,
) -> np.ndarray:
    """Evaluate the reduced graph target after solving the state exactly."""

    values = np.asarray(lambda_grid, dtype=float)
    states = problem.reference_nondimensional[None, :] / values[:, None]
    standardized = (
        problem.observed_nondimensional[None, :] - states
    ) / problem.sigma_nondimensional
    return (
        lognormal_prior(values, log_sd=prior_log_sd)
        - 0.5 * np.sum(standardized**2, axis=1)
        - problem.n_state * math.log(problem.sigma_nondimensional * math.sqrt(2.0 * math.pi))
    )


def state_logdet(lambda_grid: np.ndarray, n_state: int) -> np.ndarray:
    """Return ``log|det D_u c| = n_state*log(lambda)``."""

    values = np.asarray(lambda_grid, dtype=float)
    if np.any(values <= 0.0):
        raise ValueError("lambda values must be positive")
    return int(n_state) * np.log(values)


def hard_tube_logdensity(
    lambda_grid: np.ndarray,
    problem: HeldoutProblem,
    *,
    prior_log_sd: float = PRIOR_LOG_SD,
    corrected: bool,
) -> np.ndarray:
    graph = graph_logdensity(lambda_grid, problem, prior_log_sd=prior_log_sd)
    if corrected:
        return graph
    return graph - state_logdet(lambda_grid, problem.n_state)


def exact_integrated_tube_logdensity(
    lambda_grid: np.ndarray,
    problem: HeldoutProblem,
    *,
    rho: float = PENALTY_RHO,
    prior_log_sd: float = PRIOR_LOG_SD,
    corrected: bool,
) -> np.ndarray:
    """Integrate every latent held-out price state analytically.

    For one state, the integral is

    ``int N(y|u,sigma^2) exp[-rho(a-lambda*u)^2/2] du``.

    The stable convolution formula below is evaluated directly.  It is not
    assembled from the reduced graph target or from the hard-limit identity.
    Terms constant in ``lambda`` may be omitted.
    """

    if rho <= 0.0:
        raise ValueError("rho must be positive")
    values = np.asarray(lambda_grid, dtype=float)
    if np.any(values <= 0.0):
        raise ValueError("lambda values must be positive")
    sigma2 = problem.sigma_nondimensional**2
    tube_variance = 1.0 / (rho * values**2)
    total_variance = sigma2 + tube_variance
    difference = (
        problem.observed_nondimensional[None, :]
        - problem.reference_nondimensional[None, :] / values[:, None]
    )
    result = lognormal_prior(values, log_sd=prior_log_sd)
    result -= 0.5 * problem.n_state * np.log1p(rho * values**2 * sigma2)
    result -= 0.5 * np.sum(difference**2 / total_variance[:, None], axis=1)
    if corrected:
        result += state_logdet(values, problem.n_state)
    return result


def detcorr_callable_audit(
    lambda_scale: float,
    problem: HeldoutProblem,
    *,
    rho: float = PENALTY_RHO,
) -> dict[str, Any]:
    """Cross-check the state Jacobian through detcorr's callable route."""

    try:
        from detcorr import evaluate_corrected_finite_penalty_target
    except Exception as exc:  # pragma: no cover - depends on installation
        return {"detcorr_available": False, "reason": repr(exc)}

    reference = problem.reference_nondimensional

    def residual_callable(theta: float, state: np.ndarray) -> np.ndarray:
        return constraint_residual(theta, state, reference)

    projected = solved_state(lambda_scale, reference)
    direction = np.sin(np.linspace(0.2, 2.6, problem.n_state))
    state = projected + 2.0e-5 * direction
    base = float(lognormal_prior(lambda_scale, log_sd=PRIOR_LOG_SD))
    base += observation_loglikelihood(state, problem)
    evaluation = evaluate_corrected_finite_penalty_target(
        residual_callable,
        float(lambda_scale),
        state,
        rho=rho,
        target_measure="graph",
        base_logdensity=base,
        projected_state=projected,
        log_correction_kwargs={
            "backend": "scipy",
            "finite_difference_scheme": "central",
            "singular_policy": "raise",
        },
        metadata={
            "state_coordinate": "heldout_price_divided_by_training_mean_price",
            "residual_unit": "dimensionless",
            "state_jacobian": "-lambda*I",
        },
    )
    expected = problem.n_state * math.log(lambda_scale)
    return {
        "detcorr_available": True,
        "lambda_scale": float(lambda_scale),
        "state_dimension": problem.n_state,
        "route": "callable residual with scipy central finite differences",
        "expected_state_logdet": expected,
        "detcorr_state_logdet": float(evaluation.primary_correction),
        "state_logdet_abs_error": abs(float(evaluation.primary_correction) - expected),
        "target_measure": evaluation.target_measure,
        "diagnostics": evaluation.sampler_diagnostics(),
    }


def _integral(values: np.ndarray, grid: np.ndarray) -> float:
    return float(np.sum(0.5 * (values[1:] + values[:-1]) * np.diff(grid)))


def grid_tail_checks(
    grid: np.ndarray,
    logdensities: dict[str, np.ndarray],
    densities: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Quantify endpoint separation and small boundary-strip mass."""

    n_tail = max(3, int(math.ceil(0.02 * grid.size)))
    rows: dict[str, Any] = {}
    for name, logdensity in logdensities.items():
        peak = float(np.max(logdensity))
        density = densities[name]
        rows[name] = {
            "left_endpoint_log_drop": peak - float(logdensity[0]),
            "right_endpoint_log_drop": peak - float(logdensity[-1]),
            "left_two_percent_grid_mass": _integral(density[:n_tail], grid[:n_tail]),
            "right_two_percent_grid_mass": _integral(density[-n_tail:], grid[-n_tail:]),
        }
    minimum_drop = min(
        min(row["left_endpoint_log_drop"], row["right_endpoint_log_drop"]) for row in rows.values()
    )
    maximum_tail_mass = max(
        max(row["left_two_percent_grid_mass"], row["right_two_percent_grid_mass"])
        for row in rows.values()
    )
    return {
        "targets": rows,
        "minimum_endpoint_log_drop": float(minimum_drop),
        "maximum_two_percent_grid_tail_mass": float(maximum_tail_mass),
        "passed": bool(minimum_drop > 25.0 and maximum_tail_mass < 1.0e-8),
    }


def sensitivity_table(
    frame: pd.DataFrame,
    grid: np.ndarray,
    *,
    rho: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    main_problem = prepare_heldout_problem(frame, train_size=TRAIN_SIZE)
    for prior_sd in (0.35, 0.50, 0.75):
        rows.append(
            _sensitivity_row(
                "prior_log_sd",
                prior_sd,
                main_problem,
                grid,
                rho=rho,
                prior_log_sd=prior_sd,
            )
        )
    for train_size in (60, 64, 70):
        problem = prepare_heldout_problem(frame, train_size=train_size)
        rows.append(
            _sensitivity_row(
                "chronological_train_size",
                float(train_size),
                problem,
                grid,
                rho=rho,
                prior_log_sd=PRIOR_LOG_SD,
            )
        )
    return pd.DataFrame(rows)


def _sensitivity_row(
    sensitivity: str,
    value: float,
    problem: HeldoutProblem,
    grid: np.ndarray,
    *,
    rho: float,
    prior_log_sd: float,
) -> dict[str, Any]:
    graph = graph_logdensity(grid, problem, prior_log_sd=prior_log_sd)
    naive = exact_integrated_tube_logdensity(
        grid,
        problem,
        rho=rho,
        prior_log_sd=prior_log_sd,
        corrected=False,
    )
    corrected = exact_integrated_tube_logdensity(
        grid,
        problem,
        rho=rho,
        prior_log_sd=prior_log_sd,
        corrected=True,
    )
    table, _, _ = common.summarize_targets(
        grid,
        {"graph_reduced": graph, "finite_naive": naive, "finite_corrected": corrected},
    )
    indexed = table.set_index("target")
    graph_mean = float(indexed.loc["graph_reduced", "mean"])
    graph_sd = float(indexed.loc["graph_reduced", "sd"])
    naive_mean = float(indexed.loc["finite_naive", "mean"])
    corrected_mean = float(indexed.loc["finite_corrected", "mean"])
    return {
        "sensitivity": sensitivity,
        "value": value,
        "n_train": len(problem.train),
        "n_heldout": len(problem.heldout),
        "prior_log_sd": prior_log_sd,
        "graph_mean": graph_mean,
        "graph_sd": graph_sd,
        "naive_mean": naive_mean,
        "corrected_mean": corrected_mean,
        "naive_shift_in_graph_sd": (naive_mean - graph_mean) / graph_sd,
        "corrected_shift_in_graph_sd": (corrected_mean - graph_mean) / graph_sd,
    }


def finite_penalty_convergence_table(
    grid: np.ndarray,
    problem: HeldoutProblem,
) -> pd.DataFrame:
    graph = graph_logdensity(grid, problem)
    hard_naive = hard_tube_logdensity(grid, problem, corrected=False)
    graph_density, _, graph_summary = common.normalize_on_grid(grid, graph)
    hard_density, _, hard_summary = common.normalize_on_grid(grid, hard_naive)
    rows: list[dict[str, Any]] = []
    for rho in (
        1.0,
        3.0,
        10.0,
        30.0,
        100.0,
        300.0,
        1.0e3,
        1.0e4,
        1.0e5,
        1.0e6,
        1.0e7,
        PENALTY_RHO,
    ):
        naive = exact_integrated_tube_logdensity(grid, problem, rho=rho, corrected=False)
        corrected = exact_integrated_tube_logdensity(grid, problem, rho=rho, corrected=True)
        naive_density, _, naive_summary = common.normalize_on_grid(grid, naive)
        corrected_density, _, corrected_summary = common.normalize_on_grid(grid, corrected)
        rows.append(
            {
                "rho": rho,
                "rho_unit": "dimensionless",
                "naive_mean": naive_summary.mean,
                "hard_naive_mean": hard_summary.mean,
                "corrected_mean": corrected_summary.mean,
                "graph_mean": graph_summary.mean,
                "naive_density_l1_vs_hard_naive": common.density_l1(
                    grid, naive_density, hard_density
                ),
                "corrected_density_l1_vs_graph": common.density_l1(
                    grid, corrected_density, graph_density
                ),
            }
        )
    return pd.DataFrame(rows)


def _plot_reference(path: Path, problem: HeldoutProblem) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0))
    train_t = problem.train["t"].to_numpy(dtype=float)
    heldout_t = problem.heldout["t"].to_numpy(dtype=float)
    axes[0].plot(
        train_t,
        problem.train["avgprc"],
        color="0.55",
        linewidth=1.0,
        label="training observations",
    )
    axes[0].plot(
        train_t,
        problem.train_fitted_native,
        color="tab:blue",
        linewidth=1.4,
        label="training fitted reference",
    )
    axes[0].plot(
        heldout_t,
        problem.heldout["avgprc"],
        color="black",
        marker="o",
        markersize=3,
        linewidth=1.0,
        label="held-out observations",
    )
    axes[0].plot(
        heldout_t,
        problem.heldout_reference_native,
        color="tab:orange",
        linestyle="--",
        linewidth=1.4,
        label="held-out reference prediction",
    )
    axes[0].axvline(float(train_t[-1]) + 0.5, color="0.2", linestyle=":")
    axes[0].set_xlabel("chronological observation index t")
    axes[0].set_ylabel("dataset average price (native units)")
    axes[0].set_title("Predeclared chronological split")
    axes[0].legend(
        frameon=True,
        framealpha=0.95,
        facecolor="white",
        edgecolor="0.8",
        fontsize=7,
    )

    observed = problem.heldout["avgprc"].to_numpy(dtype=float)
    reference = problem.heldout_reference_native
    lo = min(float(np.min(observed)), float(np.min(reference)))
    hi = max(float(np.max(observed)), float(np.max(reference)))
    axes[1].scatter(reference, observed, color="tab:orange", s=22, alpha=0.8)
    axes[1].plot([lo, hi], [lo, hi], color="0.25", linestyle=":")
    axes[1].set_xlabel("held-out reference prediction (native price units)")
    axes[1].set_ylabel("held-out observed price (native price units)")
    axes[1].set_title("Calibration data are out of fit sample")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_posteriors(
    path: Path,
    grid: np.ndarray,
    densities: dict[str, np.ndarray],
) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.3))
    labels = {
        "graph_reduced": "held-out reduced graph",
        "finite_naive": "exact finite tube, no determinant",
        "finite_corrected": "exact finite tube + state logdet",
    }
    ax.plot(
        grid,
        densities["graph_reduced"],
        color="tab:blue",
        linestyle="--",
        linewidth=2.0,
        label=labels["graph_reduced"],
    )
    ax.plot(
        grid,
        densities["finite_naive"],
        color="tab:orange",
        linewidth=1.8,
        label=labels["finite_naive"],
    )
    ax.plot(
        grid,
        densities["finite_corrected"],
        color="tab:green",
        linewidth=1.2,
        label=labels["finite_corrected"],
    )
    ax.set_xlabel("reference-to-state scale lambda (dimensionless)")
    ax.set_ylabel("posterior density")
    ax.set_title("Held-out scale target: graph, naive tube, and correction")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_convergence(path: Path, table: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(table["rho"], table["naive_mean"], marker="o", label="finite naive mean")
    ax.axhline(
        float(table["hard_naive_mean"].iloc[0]),
        color="tab:orange",
        linestyle=":",
        label="hard naive mean",
    )
    ax.plot(
        table["rho"],
        table["corrected_mean"],
        marker="s",
        label="finite corrected mean",
    )
    ax.axhline(
        float(table["graph_mean"].iloc[0]),
        color="tab:green",
        linestyle="--",
        label="reduced graph mean",
    )
    ax.set_xscale("log")
    ax.set_xlabel("dimensionless penalty strength rho")
    ax.set_ylabel("posterior mean of lambda (dimensionless)")
    ax.set_title("Exact state integration approaches the hard limit")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_demo(
    outputs: Path,
    *,
    data_path: Path = DATA_PATH,
    train_size: int = TRAIN_SIZE,
    n_grid: int = GRID_SIZE,
    rho: float = PENALTY_RHO,
) -> dict[str, Any]:
    if n_grid < 501:
        raise ValueError("n_grid must be at least 501")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    observed_hash = common.sha256_file(data_path)
    if observed_hash != DATA_SHA256:
        raise ValueError("fish.csv SHA256 does not match the pinned release payload")
    if manifest.get("sha256") != observed_hash or manifest.get("raw_sha256") != observed_hash:
        raise ValueError("Fulton fixture manifest does not match the selected CSV")
    if manifest.get("bundled_file") != data_path.name or manifest.get("row_count") != 97:
        raise ValueError("Fulton fixture manifest has inconsistent file or row declarations")
    outputs.mkdir(parents=True, exist_ok=True)
    tables = outputs / "tables"
    frame = load_fish(data_path)
    problem = prepare_heldout_problem(frame, train_size=train_size)
    grid = np.linspace(GRID_MIN, GRID_MAX, int(n_grid))

    graph = graph_logdensity(grid, problem)
    finite_naive = exact_integrated_tube_logdensity(grid, problem, rho=rho, corrected=False)
    finite_corrected = exact_integrated_tube_logdensity(grid, problem, rho=rho, corrected=True)
    targets = {
        "graph_reduced": graph,
        "finite_naive": finite_naive,
        "finite_corrected": finite_corrected,
    }
    posterior_table, densities, _ = common.summarize_targets(grid, targets)
    common.write_table(posterior_table, tables / "posterior_summaries.csv")

    selected = common.select_grid_points(grid, 13)
    validation_table, validation_stats = common.centered_validation_table(
        "lambda_scale",
        selected,
        exact_integrated_tube_logdensity(selected, problem, rho=rho, corrected=False),
        hard_tube_logdensity(selected, problem, corrected=False),
        exact_integrated_tube_logdensity(selected, problem, rho=rho, corrected=True),
        graph_logdensity(selected, problem),
    )
    validation_stats["integral_route"] = "exact_product_of_scalar_gaussian_state_integrals"
    validation_stats["rho"] = rho
    validation_stats["rho_unit"] = "dimensionless"
    common.write_table(validation_table, tables / "finite_penalty_validation.csv")

    convergence = finite_penalty_convergence_table(grid, problem)
    common.write_table(convergence, tables / "finite_penalty_convergence.csv")
    sensitivity = sensitivity_table(frame, grid, rho=rho)
    common.write_table(sensitivity, tables / "prior_and_split_sensitivity.csv")

    predictions = pd.DataFrame(
        {
            "source_row": problem.heldout["rownames"].to_numpy(dtype=int),
            "t": problem.heldout["t"].to_numpy(dtype=int),
            "observed_avgprc_native": problem.heldout["avgprc"].to_numpy(dtype=float),
            "reference_avgprc_native": problem.heldout_reference_native,
            "observed_nondimensional": problem.observed_nondimensional,
            "reference_nondimensional": problem.reference_nondimensional,
        }
    )
    common.write_table(predictions, tables / "heldout_predictions.csv")

    indexed = posterior_table.set_index("target")
    graph_mean = float(indexed.loc["graph_reduced", "mean"])
    graph_sd = float(indexed.loc["graph_reduced", "sd"])
    naive_mean = float(indexed.loc["finite_naive", "mean"])
    corrected_mean = float(indexed.loc["finite_corrected", "mean"])
    naive_l1 = common.density_l1(grid, densities["finite_naive"], densities["graph_reduced"])
    corrected_l1 = common.density_l1(
        grid, densities["finite_corrected"], densities["graph_reduced"]
    )
    correction = state_logdet(grid, problem.n_state)
    tail_checks = grid_tail_checks(grid, targets, densities)
    api_audit = detcorr_callable_audit(graph_mean, problem, rho=rho)

    train_rows = set(problem.train["rownames"].astype(int))
    heldout_rows = set(problem.heldout["rownames"].astype(int))
    checks = {
        "data_checksum_matches": True,
        "fixture_manifest_matches": True,
        "chronological_split_is_disjoint": not bool(train_rows & heldout_rows),
        "predeclared_train_size_used": len(problem.train) == train_size,
        "heldout_reference_is_positive": bool(np.all(problem.heldout_reference_native > 0.0)),
        "nondimensional_scale_uses_training_only": problem.fit.training_price_scale
        == float(problem.train["avgprc"].mean()),
        "independent_finite_corrected_matches_graph": corrected_l1 < 5.0e-5,
        "naive_tube_changes_posterior": naive_l1 > 0.25,
        "grid_tails_are_negligible": bool(tail_checks["passed"]),
        "detcorr_callable_logdet_matches": bool(
            api_audit.get("detcorr_available")
            and api_audit.get("state_logdet_abs_error", 1.0) < 2.0e-6
        ),
    }
    checks["all_passed"] = all(checks.values())

    _plot_reference(outputs / "heldout_price_reference.png", problem)
    _plot_posteriors(outputs / "heldout_scale_posteriors.png", grid, densities)
    _plot_convergence(outputs / "finite_penalty_convergence.png", convergence)

    summary: dict[str, Any] = {
        "example": "econometrics_fulton_market",
        "demo": "econometrics_fulton_market",
        "classification": "external_open_data_heldout_calibration_demonstrator",
        "status": "worked_real_data_demonstrator_not_structural_economics",
        "target_measure": "graph posterior over lambda with respect to d lambda",
        "correction_driver": (
            "repeated diagonal state derivative D_u c=-lambda I across "
            f"{len(problem.heldout)} held-out nondimensional price states"
        ),
        "correction_log_range_on_grid": float(np.ptp(correction)),
        "data": {
            "dataset": "Wooldridge/Graddy Fulton fish-market fish data",
            "source_url": DATA_URL,
            "bundled_path": "data/fish.csv",
            "sha256": observed_hash,
            "manifest": "data/FIXTURE_MANIFEST.json",
            "n_complete_rows": len(frame),
            "upstream_package": "CRAN wooldridge",
            "upstream_package_software_license": "GPL-3",
            "license_boundary": (
                "The CSV is conveyed separately under GPL-3 to the extent the official "
                "wooldridge package licence applies; the Rdatasets underlying-rights "
                "caveat is retained. See case_studies/THIRD_PARTY_NOTICES.md."
            ),
            "citation": (
                "K. Graddy (1995), Testing for Imperfect Competition at the "
                "Fulton Fish Market, RAND Journal of Economics 26, 75-92."
            ),
        },
        "split": {
            "rule": (
                f"first {len(problem.train)} chronological rows train; final "
                f"{len(problem.heldout)} rows held out"
            ),
            "n_train": len(problem.train),
            "n_heldout": len(problem.heldout),
            "training_t_range": [
                int(problem.train["t"].min()),
                int(problem.train["t"].max()),
            ],
            "heldout_t_range": [
                int(problem.heldout["t"].min()),
                int(problem.heldout["t"].max()),
            ],
            "heldout_prices_used_in_reference_fit": False,
            "heldout_prices_used_in_likelihood": True,
        },
        "model": {
            "full_space_base_measure": (
                "d lambda times the product of du_t over all held-out dimensionless "
                "price-state coordinates"
            ),
            "prior": {
                "distribution": "lognormal",
                "density_measure": "d lambda",
                "log_parameter_distribution": "log(lambda) ~ Normal(0, 0.5^2)",
                "log_location": 0.0,
                "log_scale": PRIOR_LOG_SD,
                "support": "lambda > 0",
            },
            "parameter": "lambda, dimensionless reference-to-state scale",
            "constraint": "c_t(lambda,u_t)=a_t-lambda*u_t",
            "state_solution": "u_t(lambda)=a_t/lambda",
            "state_jacobian": "D_u c=-lambda*I",
            "state_logdet": "n_heldout*log(lambda)",
            "observation_likelihood": "iid Gaussian held-out price errors with plug-in training SD",
            "state_coordinate": "native price divided by mean training price",
            "state_and_residual_unit": "dimensionless",
            "training_price_scale_native": problem.fit.training_price_scale,
            "sigma_nondimensional": problem.sigma_nondimensional,
            "penalty_rho": rho,
            "penalty_rho_unit": "dimensionless",
            "reference_fit": asdict(problem.fit),
        },
        "outcome": {
            "graph_mean": graph_mean,
            "graph_sd": graph_sd,
            "naive_mean": naive_mean,
            "corrected_mean": corrected_mean,
            "correction_log_range_on_grid": float(np.ptp(correction)),
            "naive_mean_shift_in_graph_sd": (naive_mean - graph_mean) / graph_sd,
            "corrected_mean_shift_in_graph_sd": (corrected_mean - graph_mean) / graph_sd,
            "naive_minus_corrected_mean_in_graph_sd": (naive_mean - corrected_mean) / graph_sd,
            "naive_density_l1_vs_graph": naive_l1,
            "corrected_density_l1_vs_graph": corrected_l1,
        },
        "independent_validation": {
            "route": (
                "stable exact Gaussian integration over all "
                f"{len(problem.heldout)} held-out latent price states, evaluated "
                "independently of the reduced graph target"
            ),
            **validation_stats,
        },
        "detcorr_callable_api_audit": api_audit,
        "grid_tail_checks": tail_checks,
        "sensitivity": {
            "scope": "declared demonstrator sensitivity, not a model-selection analysis",
            "prior_log_sd_values": [0.35, 0.5, 0.75],
            "chronological_train_sizes": [60, 64, 70],
            "table": "tables/prior_and_split_sensitivity.csv",
        },
        "checks": checks,
        "limitations": [
            (
                "The iid Gaussian likelihood ignores serial correlation, "
                "heteroskedasticity, and time-series model discrepancy."
            ),
            (
                "Reduced-form coefficients and sigma are plug-in training estimates; "
                "their uncertainty is not propagated."
            ),
            (
                "The chronological split is a transparent demonstration choice, "
                "not an optimized forecasting design."
            ),
            (
                "Lambda is a conditional dimensionless scale parameter, not a supply, "
                "demand, market-power, or causal elasticity."
            ),
            "The model does not support publication-grade claims about the Fulton market.",
        ],
        "interpretation_boundary": (
            "The defensible result is conditional and numerical: for this predeclared "
            "held-out price-scale model, omitting the state-volume determinant changes "
            "the lambda posterior, while exact state integration verifies the corrected "
            "graph target. It is not a structural or causal estimate of the fish market."
        ),
        "outputs": {
            "reference_figure": "heldout_price_reference.png",
            "posterior_figure": "heldout_scale_posteriors.png",
            "convergence_figure": "finite_penalty_convergence.png",
            "posterior_table": "tables/posterior_summaries.csv",
            "validation_table": "tables/finite_penalty_validation.csv",
            "convergence_table": "tables/finite_penalty_convergence.csv",
            "sensitivity_table": "tables/prior_and_split_sensitivity.csv",
            "heldout_predictions": "tables/heldout_predictions.csv",
        },
        "provenance": common.reproducibility_provenance(example_root=ROOT),
    }
    common.write_json(outputs / "summary.json", summary)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--outputs", "--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--train-size", type=int, default=TRAIN_SIZE)
    parser.add_argument("--n-grid", type=int, default=GRID_SIZE)
    parser.add_argument("--rho", type=float, default=PENALTY_RHO)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    summary = run_demo(
        args.outputs,
        data_path=args.data,
        train_size=args.train_size,
        n_grid=args.n_grid,
        rho=args.rho,
    )
    print(
        json.dumps(
            {"example": summary["example"], "checks": summary["checks"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
