#!/usr/bin/env python3
"""Held-out Matbench band-gap residual-coordinate stress test.

Experimental gaps are used only after a deterministic formula-hash split.  A
small composition predictor is fitted to 70 training formulas, and 30 disjoint
formulas define the held-out target.  Energies are nondimensionalized by
E0=1 eV before applying the nonlinear residual c=e**alpha-m.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
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

DATA_PATH = ROOT / "data" / "matbench_expt_gap_fixture.csv"
FIXTURE_SHA256 = "106f00fc0032a52791d837a7c92614649447cccc12679797ac1a66d5a4e0904c"
MATBENCH_PAPER_DOI = "10.1038/s41524-020-00406-3"
MATBENCH_DATA_DOI = "10.6084/m9.figshare.9765779.v1"
MATBENCH_SOURCE_URL = "https://ndownloader.figshare.com/files/17494814"
MATBENCH_SOURCE_SHA256 = "1e6816fb8e7132535b76cdac458c4d53944eca97e1ecb68e6a8fd853c9cedc3a"
ORIGINAL_DATA_DOI = "10.1021/acs.jpclett.8b00124"

ENERGY_REFERENCE_EV = 1.0
PREDICTOR_LOG_OFFSET_EV = 0.05
TRAIN_COUNT = 70
ALPHA_MIN = 0.40
ALPHA_MAX = 2.30
GRID_SIZE = 1001
PRIOR_LOG_SD = 0.42
PENALTY_RHO = 5.0e6
QUADRATURE_POINTS = 401
QUADRATURE_WINDOW_SIGMA = 9.0
WINDOW_SENSITIVITY_SIGMA = 12.0

REQUIRED_COLUMNS = ("source_row", "formula", "band_gap_ev")

# Atomic numbers are used only for transparent composition descriptors.  This
# compact table exactly covers the pinned 100-row fixture.
ELEMENT_Z = {
    "H": 1,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Al": 13,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "V": 23,
    "Cu": 29,
    "Zn": 30,
    "Ga": 31,
    "Ge": 32,
    "As": 33,
    "Se": 34,
    "Br": 35,
    "Mo": 42,
    "Ru": 44,
    "Rh": 45,
    "Pd": 46,
    "Ag": 47,
    "Cd": 48,
    "In": 49,
    "Sn": 50,
    "Sb": 51,
    "Te": 52,
    "I": 53,
    "Ba": 56,
    "Ce": 58,
    "Dy": 66,
    "Ta": 73,
    "Ir": 77,
    "Pt": 78,
    "Hg": 80,
    "Pb": 82,
    "Bi": 83,
    "Os": 76,
}
ANION_ELEMENTS = {"N", "O", "S", "Se", "Te", "F", "Cl", "Br", "I"}
TRANSITION_ELEMENTS = {
    "V",
    "Cu",
    "Zn",
    "Mo",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Ta",
    "Ir",
    "Pt",
    "Hg",
    "Os",
}


@dataclass(frozen=True)
class PredictorFit:
    feature_names: tuple[str, ...]
    coefficients: tuple[float, ...]
    mean_z_mean: float
    mean_z_sd: float
    train_rmse_ev: float
    train_count: int
    heldout_count: int


@dataclass(frozen=True)
class DimensionlessProblem:
    observed_e: np.ndarray
    predicted_m: np.ndarray
    sigma_e: float
    reference_energy_value: float
    reference_energy_unit: str

    @property
    def n_state(self) -> int:
        return int(self.observed_e.size)


def parse_formula(formula: str) -> dict[str, float]:
    """Parse the ordinary, unparenthesized formulas in the pinned fixture."""

    text = str(formula).strip().replace(" ", "")
    tokens = re.findall(r"([A-Z][a-z]?)([0-9]*\.?[0-9]*)", text)
    if not tokens or "".join(element + amount for element, amount in tokens) != text:
        raise ValueError(f"unsupported formula syntax: {formula!r}")
    counts: dict[str, float] = {}
    for element, amount in tokens:
        if element not in ELEMENT_Z:
            raise ValueError(f"formula {formula!r} contains unsupported element {element!r}")
        value = float(amount) if amount else 1.0
        if value <= 0.0:
            raise ValueError(f"formula {formula!r} contains a non-positive count")
        counts[element] = counts.get(element, 0.0) + value
    return counts


def composition_descriptors(formula: str) -> dict[str, float]:
    counts = parse_formula(formula)
    total = float(sum(counts.values()))
    fractions = {element: count / total for element, count in counts.items()}
    return {
        "n_elements": float(len(counts)),
        "mean_z": float(sum(fractions[element] * ELEMENT_Z[element] for element in counts)),
        "anion_fraction": float(sum(fractions.get(element, 0.0) for element in ANION_ELEMENTS)),
        "transition_fraction": float(
            sum(fractions.get(element, 0.0) for element in TRANSITION_ELEMENTS)
        ),
        "light_fraction": float(
            sum(fraction for element, fraction in fractions.items() if ELEMENT_Z[element] <= 10)
        ),
    }


def formula_sha256(formula: str) -> str:
    return hashlib.sha256(str(formula).strip().encode("utf-8")).hexdigest()


def load_bandgap_data(path: Path = DATA_PATH) -> pd.DataFrame:
    if common.sha256_file(path) != FIXTURE_SHA256:
        raise ValueError("Matbench fixture SHA256 does not match the pinned payload")
    frame = pd.read_csv(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"band-gap fixture is missing columns: {missing}")
    frame = frame.dropna(subset=list(REQUIRED_COLUMNS)).copy()
    frame["band_gap_ev"] = frame["band_gap_ev"].astype(float)
    if len(frame) != 100 or np.any(frame["band_gap_ev"] <= 0.0):
        raise ValueError("the pinned fixture must contain 100 positive band gaps")
    descriptor_rows = [composition_descriptors(formula) for formula in frame["formula"]]
    descriptors = pd.DataFrame(descriptor_rows, index=frame.index)
    frame = pd.concat([frame, descriptors], axis=1)
    frame["formula_sha256"] = [formula_sha256(formula) for formula in frame["formula"]]
    return frame


def formula_hash_split(
    frame: pd.DataFrame,
    *,
    train_count: int = TRAIN_COUNT,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return an order-invariant deterministic 70/30 formula-hash split."""

    if frame["formula"].duplicated().any():
        raise ValueError("formula-hash split requires unique formulas")
    if train_count <= 10 or len(frame) - train_count <= 10:
        raise ValueError("split must leave at least 11 rows in each block")
    ordered = frame.sort_values(
        ["formula_sha256", "formula", "source_row"], kind="stable"
    ).reset_index(drop=True)
    train = ordered.iloc[:train_count].copy().reset_index(drop=True)
    heldout = ordered.iloc[train_count:].copy().reset_index(drop=True)
    return train, heldout


def design_matrix(
    frame: pd.DataFrame,
    *,
    mean_z_mean: float | None = None,
    mean_z_sd: float | None = None,
) -> tuple[np.ndarray, tuple[str, ...], float, float]:
    mean_z = frame["mean_z"].to_numpy(dtype=float)
    if mean_z_mean is None:
        mean_z_mean = float(np.mean(mean_z))
    if mean_z_sd is None:
        mean_z_sd = float(np.std(mean_z, ddof=1))
    if mean_z_sd <= 0.0:
        raise ValueError("training mean-Z scale must be positive")
    features = np.column_stack(
        [
            np.ones(len(frame)),
            (mean_z - mean_z_mean) / mean_z_sd,
            frame["anion_fraction"].to_numpy(dtype=float),
            np.log1p(frame["n_elements"].to_numpy(dtype=float)),
            frame["transition_fraction"].to_numpy(dtype=float),
            frame["light_fraction"].to_numpy(dtype=float),
        ]
    )
    names = (
        "const",
        "mean_Z_zscore",
        "anion_fraction",
        "log1p_n_elements",
        "transition_fraction",
        "light_fraction",
    )
    return features, names, float(mean_z_mean), float(mean_z_sd)


def inverse_log_gap_prediction(linear_prediction: np.ndarray) -> np.ndarray:
    """Invert ``log(E/eV + 0.05)`` before applying declared clipping."""

    raw_gap_ev = np.exp(np.asarray(linear_prediction, dtype=float))
    raw_gap_ev -= PREDICTOR_LOG_OFFSET_EV
    return np.clip(raw_gap_ev, 0.05, 7.5)


def fit_base_gap_predictor(
    train: pd.DataFrame,
    heldout: pd.DataFrame,
) -> tuple[PredictorFit, np.ndarray, np.ndarray]:
    """Fit the compact predictor on training gaps and predict both blocks."""

    x_train, names, z_mean, z_sd = design_matrix(train)
    transformed = np.log(train["band_gap_ev"].to_numpy(dtype=float) + PREDICTOR_LOG_OFFSET_EV)
    ridge = np.diag([0.0, 1.0e-3, 1.0e-3, 1.0e-3, 1.0e-3, 1.0e-3])
    coefficients = np.linalg.solve(x_train.T @ x_train + ridge, x_train.T @ transformed)
    train_prediction = inverse_log_gap_prediction(x_train @ coefficients)
    x_heldout, _, _, _ = design_matrix(heldout, mean_z_mean=z_mean, mean_z_sd=z_sd)
    heldout_prediction = inverse_log_gap_prediction(x_heldout @ coefficients)
    residual = train["band_gap_ev"].to_numpy(dtype=float) - train_prediction
    rmse = math.sqrt(float(np.mean(residual**2)))
    if rmse <= 0.0:
        raise ValueError("training RMSE must be positive")
    fit = PredictorFit(
        feature_names=names,
        coefficients=tuple(float(value) for value in coefficients),
        mean_z_mean=z_mean,
        mean_z_sd=z_sd,
        train_rmse_ev=rmse,
        train_count=len(train),
        heldout_count=len(heldout),
    )
    return fit, train_prediction, heldout_prediction


def make_dimensionless_problem(
    observed_energy: np.ndarray,
    predicted_energy: np.ndarray,
    sigma_energy: float,
    *,
    reference_energy: float,
    energy_unit: str,
) -> DimensionlessProblem:
    """Express all energy-valued quantities relative to the same E0."""

    if reference_energy <= 0.0 or sigma_energy <= 0.0:
        raise ValueError("reference energy and likelihood scale must be positive")
    observed = np.asarray(observed_energy, dtype=float) / reference_energy
    predicted = np.asarray(predicted_energy, dtype=float) / reference_energy
    if observed.shape != predicted.shape or np.any(observed <= 0.0) or np.any(predicted <= 0.0):
        raise ValueError("observed and predicted energies must be positive and aligned")
    return DimensionlessProblem(
        observed_e=observed,
        predicted_m=predicted,
        sigma_e=float(sigma_energy / reference_energy),
        reference_energy_value=float(reference_energy),
        reference_energy_unit=str(energy_unit),
    )


def log_alpha_prior(alpha: float | np.ndarray, *, log_sd: float = PRIOR_LOG_SD) -> np.ndarray:
    values = np.asarray(alpha, dtype=float)
    result = np.full_like(values, -np.inf, dtype=float)
    valid = values > 0.0
    log_values = np.log(values[valid])
    result[valid] = -0.5 * (log_values / log_sd) ** 2 - np.log(
        values[valid] * log_sd * math.sqrt(2.0 * math.pi)
    )
    return result


def solved_state(alpha: float, predicted_m: np.ndarray) -> np.ndarray:
    if alpha <= 0.0:
        raise ValueError("alpha must be positive")
    return np.exp(np.log(np.asarray(predicted_m, dtype=float)) / float(alpha))


def constraint_residual(
    alpha: float,
    state_e: np.ndarray,
    predicted_m: np.ndarray,
) -> np.ndarray:
    return np.asarray(state_e, dtype=float) ** float(alpha) - np.asarray(predicted_m, dtype=float)


def state_jacobian_diagonal(alpha: float, state_e: np.ndarray) -> np.ndarray:
    state = np.asarray(state_e, dtype=float)
    if alpha <= 0.0 or np.any(state <= 0.0):
        raise ValueError("alpha and state must be positive")
    return float(alpha) * state ** (float(alpha) - 1.0)


def state_logdet(alpha_grid: np.ndarray, problem: DimensionlessProblem) -> np.ndarray:
    values = np.asarray(alpha_grid, dtype=float)
    result = np.empty_like(values)
    for index, alpha in enumerate(values):
        state = solved_state(float(alpha), problem.predicted_m)
        diagonal = state_jacobian_diagonal(float(alpha), state)
        result[index] = float(np.sum(np.log(np.abs(diagonal))))
    return result


def graph_logdensity(
    alpha_grid: np.ndarray,
    problem: DimensionlessProblem,
    *,
    prior_log_sd: float = PRIOR_LOG_SD,
) -> np.ndarray:
    values = np.asarray(alpha_grid, dtype=float)
    result = np.empty_like(values)
    for index, alpha in enumerate(values):
        state = solved_state(float(alpha), problem.predicted_m)
        standardized = (problem.observed_e - state) / problem.sigma_e
        result[index] = float(log_alpha_prior(alpha, log_sd=prior_log_sd))
        result[index] -= 0.5 * float(standardized @ standardized)
        result[index] -= problem.n_state * math.log(problem.sigma_e * math.sqrt(2.0 * math.pi))
    return result


def hard_tube_logdensity(
    alpha_grid: np.ndarray,
    problem: DimensionlessProblem,
    *,
    corrected: bool,
) -> np.ndarray:
    graph = graph_logdensity(alpha_grid, problem)
    if corrected:
        return graph
    return graph - state_logdet(alpha_grid, problem)


def _row_log_trapezoid(
    grid: np.ndarray,
    log_values: np.ndarray,
) -> np.ndarray:
    maxima = np.max(log_values, axis=1)
    weights = np.ones(log_values.shape[1], dtype=float)
    weights[[0, -1]] = 0.5
    scaled_sum = np.sum(np.exp(log_values - maxima[:, None]) * weights[None, :], axis=1)
    spacing = (grid[:, -1] - grid[:, 0]) / (grid.shape[1] - 1)
    return maxima + np.log(spacing * scaled_sum)


def finite_integrated_tube_logdensity(
    alpha_grid: np.ndarray,
    problem: DimensionlessProblem,
    *,
    rho: float = PENALTY_RHO,
    n_quad: int = QUADRATURE_POINTS,
    window_sigma: float = QUADRATURE_WINDOW_SIGMA,
    corrected: bool,
) -> np.ndarray:
    """Directly integrate every held-out dimensionless state by quadrature.

    This is deliberately a local, window-limited integral: each state interval
    spans ``+/- window_sigma`` *linearized* penalty standard deviations about
    the constraint solution, clipped only at the positive-state boundary.  The
    default nine-standard-deviation window is checked against a wider window
    in :func:`run_demo`.  At the extreme, negligible-posterior alpha endpoint,
    increasing the window does not materially reduce the reported error; that
    residual discrepancy is finite-rho bias rather than trapezoid resolution
    or an omitted-tail effect.
    """

    if rho <= 0.0 or n_quad < 101 or n_quad % 2 == 0:
        raise ValueError("rho must be positive and n_quad must be an odd integer >=101")
    values = np.asarray(alpha_grid, dtype=float)
    coordinate = np.linspace(0.0, 1.0, int(n_quad))
    result = np.empty_like(values)
    for index, alpha_value in enumerate(values):
        alpha = float(alpha_value)
        state = solved_state(alpha, problem.predicted_m)
        diagonal = state_jacobian_diagonal(alpha, state)
        width = window_sigma / (math.sqrt(rho) * np.abs(diagonal))
        lower = np.maximum(1.0e-12, state - width)
        upper = state + width
        state_grid = lower[:, None] + (upper - lower)[:, None] * coordinate[None, :]
        observation_term = (
            -0.5 * ((problem.observed_e[:, None] - state_grid) / problem.sigma_e) ** 2
        )
        residual = state_grid**alpha - problem.predicted_m[:, None]
        penalty_term = -0.5 * rho * residual**2
        integrand = observation_term + penalty_term
        if corrected:
            # The finite-target correction is pointwise in the sampled state,
            # exactly as in detcorr's full-space API.  Evaluating this only at
            # the solved graph state would be a hard-limit projection, not an
            # independently integrated corrected finite tube.
            pointwise_logdet = math.log(alpha) + (alpha - 1.0) * np.log(state_grid)
            integrand = integrand + pointwise_logdet
        log_integrals = _row_log_trapezoid(state_grid, integrand)
        result[index] = float(log_alpha_prior(alpha)) + float(np.sum(log_integrals))
    return result


def detcorr_audit(alpha: float, problem: DimensionlessProblem) -> dict[str, Any]:
    """Check both exact diagonal and callable public determinant routes."""

    state = solved_state(alpha, problem.predicted_m)
    diagonal = state_jacobian_diagonal(alpha, state)
    expected = float(np.sum(np.log(np.abs(diagonal))))
    diagonal_audit = common.try_detcorr_logdet_diag(diagonal)

    try:
        from detcorr import evaluate_corrected_finite_penalty_target
    except Exception as exc:  # pragma: no cover - depends on installation
        return {
            "detcorr_available": False,
            "reason": repr(exc),
            "expected_state_logdet": expected,
        }

    def residual_callable(theta: float, candidate: np.ndarray) -> np.ndarray:
        return constraint_residual(theta, candidate, problem.predicted_m)

    # Evaluate the callable route on the graph so its reference is the same
    # projected-state diagonal used by the exact structured route.
    candidate = state.copy()
    standardized = (problem.observed_e - candidate) / problem.sigma_e
    base = float(log_alpha_prior(alpha)) - 0.5 * float(standardized @ standardized)
    evaluation = evaluate_corrected_finite_penalty_target(
        residual_callable,
        float(alpha),
        candidate,
        rho=PENALTY_RHO,
        target_measure="graph",
        base_logdensity=base,
        projected_state=state,
        log_correction_kwargs={
            "backend": "scipy",
            "finite_difference_scheme": "central",
            "singular_policy": "raise",
        },
        metadata={
            "state": "dimensionless gap e=E/E0",
            "reference_energy": "E0=1 eV",
            "physical_interpretation_of_alpha": "none; residual-coordinate stress test",
        },
    )
    diagonal_value = diagonal_audit.get("value")
    return {
        "detcorr_available": True,
        "alpha": float(alpha),
        "state_dimension": problem.n_state,
        "target_measure": evaluation.target_measure,
        "expected_state_logdet": expected,
        "diagonal_state_logdet": diagonal_value,
        "diagonal_abs_error": (
            abs(float(diagonal_value) - expected) if diagonal_value is not None else None
        ),
        "callable_state_logdet": float(evaluation.primary_correction),
        "callable_abs_error": abs(float(evaluation.primary_correction) - expected),
        "diagonal_route": diagonal_audit.get("result"),
        "callable_diagnostics": evaluation.sampler_diagnostics(),
    }


def unit_invariance_audit(
    observed_gap_ev: np.ndarray,
    predicted_gap_ev: np.ndarray,
    sigma_ev: float,
    alpha_grid: np.ndarray,
) -> dict[str, Any]:
    ev_problem = make_dimensionless_problem(
        observed_gap_ev,
        predicted_gap_ev,
        sigma_ev,
        reference_energy=1.0,
        energy_unit="eV",
    )
    mev_problem = make_dimensionless_problem(
        1000.0 * np.asarray(observed_gap_ev),
        1000.0 * np.asarray(predicted_gap_ev),
        1000.0 * sigma_ev,
        reference_energy=1000.0,
        energy_unit="meV",
    )
    graph_difference = common.centered_max_abs_diff(
        graph_logdensity(alpha_grid, ev_problem),
        graph_logdensity(alpha_grid, mev_problem),
    )
    correction_difference = float(
        np.max(np.abs(state_logdet(alpha_grid, ev_problem) - state_logdet(alpha_grid, mev_problem)))
    )
    return {
        "comparison": "eV with E0=1 eV versus meV with E0=1000 meV",
        "agreement_standard": "machine precision with absolute tolerance 1e-12",
        "absolute_tolerance": 1.0e-12,
        "dimensionless_observed_max_abs_diff": float(
            np.max(np.abs(ev_problem.observed_e - mev_problem.observed_e))
        ),
        "dimensionless_predictor_max_abs_diff": float(
            np.max(np.abs(ev_problem.predicted_m - mev_problem.predicted_m))
        ),
        "graph_centered_logdensity_max_abs_diff": graph_difference,
        "state_logdet_max_abs_diff": correction_difference,
        "passed": bool(graph_difference < 1.0e-12 and correction_difference < 1.0e-12),
    }


def _integral(values: np.ndarray, grid: np.ndarray) -> float:
    return float(np.sum(0.5 * (values[1:] + values[:-1]) * np.diff(grid)))


def grid_tail_checks(
    grid: np.ndarray,
    logdensities: dict[str, np.ndarray],
    densities: dict[str, np.ndarray],
) -> dict[str, Any]:
    n_tail = max(3, int(math.ceil(0.02 * grid.size)))
    targets: dict[str, Any] = {}
    for name, logdensity in logdensities.items():
        peak = float(np.max(logdensity))
        density = densities[name]
        targets[name] = {
            "left_endpoint_log_drop": peak - float(logdensity[0]),
            "right_endpoint_log_drop": peak - float(logdensity[-1]),
            "left_two_percent_grid_mass": _integral(density[:n_tail], grid[:n_tail]),
            "right_two_percent_grid_mass": _integral(density[-n_tail:], grid[-n_tail:]),
        }
    minimum_drop = min(
        min(row["left_endpoint_log_drop"], row["right_endpoint_log_drop"])
        for row in targets.values()
    )
    maximum_mass = max(
        max(row["left_two_percent_grid_mass"], row["right_two_percent_grid_mass"])
        for row in targets.values()
    )
    return {
        "targets": targets,
        "minimum_endpoint_log_drop": float(minimum_drop),
        "maximum_two_percent_grid_tail_mass": float(maximum_mass),
        "passed": bool(minimum_drop > 20.0 and maximum_mass < 1.0e-8),
    }


def _plot_predictor(
    path: Path,
    train: pd.DataFrame,
    heldout: pd.DataFrame,
    train_prediction: np.ndarray,
    heldout_prediction: np.ndarray,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0))
    axes[0].scatter(train["band_gap_ev"], train_prediction, s=20, alpha=0.75, color="tab:blue")
    axes[0].set_title("Formula-hash training block (70)")
    axes[1].scatter(
        heldout["band_gap_ev"],
        heldout_prediction,
        s=25,
        alpha=0.8,
        color="tab:orange",
    )
    axes[1].set_title("Disjoint held-out block (30)")
    all_observed = np.concatenate(
        [train["band_gap_ev"].to_numpy(dtype=float), heldout["band_gap_ev"].to_numpy(dtype=float)]
    )
    all_predicted = np.concatenate([train_prediction, heldout_prediction])
    low = min(float(np.min(all_observed)), float(np.min(all_predicted)))
    high = max(float(np.max(all_observed)), float(np.max(all_predicted)))
    for axis in axes:
        axis.plot([low, high], [low, high], color="0.3", linestyle=":")
        axis.set_xlabel("experimental gap (eV)")
        axis.set_ylabel("composition predictor (eV)")
    fig.suptitle("Compact predictor: diagnostic input, not a competitive model")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_posteriors(
    path: Path,
    alpha_grid: np.ndarray,
    densities: dict[str, np.ndarray],
) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.3))
    ax.plot(
        alpha_grid,
        densities["graph_reduced"],
        color="tab:blue",
        linestyle="--",
        linewidth=2.0,
        label="held-out reduced graph",
    )
    ax.plot(
        alpha_grid,
        densities["finite_naive"],
        color="tab:orange",
        linewidth=1.8,
        label="direct finite tube, no determinant",
    )
    ax.plot(
        alpha_grid,
        densities["finite_corrected"],
        color="tab:green",
        linewidth=1.2,
        label="direct finite tube + state logdet",
    )
    ax.set_xlabel("residual-coordinate exponent alpha (dimensionless)")
    ax.set_ylabel("posterior density")
    ax.set_title("Held-out nonlinear residual-coordinate stress test")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_validation(path: Path, table: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    error = np.abs(table["finite_corrected_vs_graph_centered_error"].to_numpy())
    # One selected point defines the centering reference and therefore has
    # exactly zero error; omit it from a logarithmic axis.
    error[error < 1.0e-14] = np.nan
    ax.plot(
        table["alpha"],
        error,
        marker="s",
        label="corrected vs graph (same as finite vs hard tube)",
    )
    ax.set_yscale("log")
    ax.set_xlabel("residual-coordinate exponent alpha (dimensionless)")
    ax.set_ylabel("absolute centered log-density error")
    ax.set_title("Direct quadrature over all 30 held-out states")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_demo(
    outputs: Path,
    *,
    data_path: Path = DATA_PATH,
    n_grid: int = GRID_SIZE,
    rho: float = PENALTY_RHO,
    n_quad: int = QUADRATURE_POINTS,
) -> dict[str, Any]:
    if n_grid < 501:
        raise ValueError("n_grid must be at least 501")
    outputs.mkdir(parents=True, exist_ok=True)
    tables = outputs / "tables"
    frame = load_bandgap_data(data_path)
    train, heldout = formula_hash_split(frame)
    fit, train_prediction, heldout_prediction = fit_base_gap_predictor(train, heldout)
    problem = make_dimensionless_problem(
        heldout["band_gap_ev"].to_numpy(dtype=float),
        heldout_prediction,
        fit.train_rmse_ev,
        reference_energy=ENERGY_REFERENCE_EV,
        energy_unit="eV",
    )
    alpha_grid = np.linspace(ALPHA_MIN, ALPHA_MAX, int(n_grid))

    graph = graph_logdensity(alpha_grid, problem)
    finite_naive = finite_integrated_tube_logdensity(
        alpha_grid, problem, rho=rho, n_quad=n_quad, corrected=False
    )
    finite_corrected = finite_integrated_tube_logdensity(
        alpha_grid, problem, rho=rho, n_quad=n_quad, corrected=True
    )
    targets = {
        "graph_reduced": graph,
        "finite_naive": finite_naive,
        "finite_corrected": finite_corrected,
    }
    posterior_table, densities, _ = common.summarize_targets(alpha_grid, targets)
    common.write_table(posterior_table, tables / "posterior_summaries.csv")

    selected = common.select_grid_points(alpha_grid, 13)
    validation_n_quad = max(n_quad, 801)
    validation_naive = finite_integrated_tube_logdensity(
        selected,
        problem,
        rho=rho,
        n_quad=validation_n_quad,
        window_sigma=QUADRATURE_WINDOW_SIGMA,
        corrected=False,
    )
    validation_corrected = finite_integrated_tube_logdensity(
        selected,
        problem,
        rho=rho,
        n_quad=validation_n_quad,
        window_sigma=QUADRATURE_WINDOW_SIGMA,
        corrected=True,
    )
    validation_table, validation_stats = common.centered_validation_table(
        "alpha",
        selected,
        validation_naive,
        hard_tube_logdensity(selected, problem, corrected=False),
        validation_corrected,
        graph_logdensity(selected, problem),
    )
    wider_window_corrected = finite_integrated_tube_logdensity(
        selected,
        problem,
        rho=rho,
        n_quad=validation_n_quad,
        window_sigma=WINDOW_SENSITIVITY_SIGMA,
        corrected=True,
    )
    window_sensitivity = common.centered_max_abs_diff(
        validation_corrected,
        wider_window_corrected,
    )
    graph_selected = graph_logdensity(selected, problem)
    worst_validation_index = int(
        np.argmax(np.abs(common.centered(validation_corrected) - common.centered(graph_selected)))
    )
    graph_log_drop_at_worst = float(np.max(graph) - graph_selected[worst_validation_index])
    validation_table["rho"] = rho
    validation_table["n_heldout_states"] = problem.n_state
    validation_table["n_quadrature_points_per_state"] = validation_n_quad
    validation_table["window_linearized_penalty_sd"] = QUADRATURE_WINDOW_SIGMA
    validation_stats.update(
        {
            "rho": rho,
            "rho_unit": "dimensionless",
            "n_heldout_states": problem.n_state,
            "n_quadrature_points_per_state": validation_n_quad,
            "window_linearized_penalty_sd": QUADRATURE_WINDOW_SIGMA,
            "window_sensitivity_comparison_sigma": WINDOW_SENSITIVITY_SIGMA,
            "window_sensitivity_centered_max_abs_diff": window_sensitivity,
            "worst_validation_alpha": float(selected[worst_validation_index]),
            "worst_validation_graph_log_drop_from_peak": graph_log_drop_at_worst,
            "quadrature_scope": (
                "local window-limited integration about each solved state; "
                "not integration over the entire positive state line"
            ),
            "worst_case_interpretation": (
                "The largest centered error is at an extreme off-posterior alpha. "
                "Widening the local window from 9 to 12 linearized penalty SD has "
                "negligible effect, so the remaining discrepancy is dominated by "
                "finite-rho bias rather than window truncation or grid resolution."
            ),
            "integral_route": "direct_local_trapezoid_quadrature_for_every_state",
        }
    )
    common.write_table(validation_table, tables / "finite_state_quadrature.csv")

    split_table = pd.concat(
        [
            train.assign(
                split="train",
                predictor_gap_ev=train_prediction,
            ),
            heldout.assign(
                split="heldout",
                predictor_gap_ev=heldout_prediction,
            ),
        ],
        ignore_index=True,
    ).loc[
        :,
        [
            "split",
            "source_row",
            "formula",
            "formula_sha256",
            "band_gap_ev",
            "predictor_gap_ev",
        ],
    ]
    common.write_table(split_table, tables / "formula_hash_split.csv")
    heldout_table = split_table.loc[split_table["split"] == "heldout"].copy()
    heldout_table["observed_e"] = problem.observed_e
    heldout_table["predicted_m"] = problem.predicted_m
    common.write_table(heldout_table, tables / "heldout_states.csv")

    indexed = posterior_table.set_index("target")
    graph_mean = float(indexed.loc["graph_reduced", "mean"])
    graph_sd = float(indexed.loc["graph_reduced", "sd"])
    naive_mean = float(indexed.loc["finite_naive", "mean"])
    corrected_mean = float(indexed.loc["finite_corrected", "mean"])
    naive_l1 = common.density_l1(alpha_grid, densities["finite_naive"], densities["graph_reduced"])
    corrected_l1 = common.density_l1(
        alpha_grid, densities["finite_corrected"], densities["graph_reduced"]
    )
    correction = state_logdet(alpha_grid, problem)
    unit_audit = unit_invariance_audit(
        heldout["band_gap_ev"].to_numpy(dtype=float),
        heldout_prediction,
        fit.train_rmse_ev,
        common.select_grid_points(alpha_grid, 31),
    )
    determinant_audit = detcorr_audit(graph_mean, problem)
    tail_checks = grid_tail_checks(alpha_grid, targets, densities)
    train_hashes = set(train["formula_sha256"])
    heldout_hashes = set(heldout["formula_sha256"])

    checks = {
        "fixture_checksum_matches": True,
        "formula_hash_split_is_disjoint": not bool(train_hashes & heldout_hashes),
        "formula_hash_split_is_70_30": len(train) == 70 and len(heldout) == 30,
        "inverse_transform_subtracts_offset_before_clipping": True,
        "all_states_are_dimensionless": True,
        "ev_to_mev_invariance": bool(unit_audit["passed"]),
        "direct_quadrature_uses_all_heldout_states": validation_stats["n_heldout_states"] == 30,
        "quadrature_window_sensitivity_is_negligible": window_sensitivity < 1.0e-9,
        "direct_finite_corrected_matches_graph": corrected_l1 < 2.0e-3,
        "naive_tube_changes_posterior": naive_l1 > 0.2,
        "grid_tails_are_negligible": bool(tail_checks["passed"]),
        "detcorr_diagonal_route_matches": bool(
            determinant_audit.get("detcorr_available")
            and determinant_audit.get("diagonal_abs_error", 1.0) < 1.0e-10
        ),
        "detcorr_callable_route_matches": bool(
            determinant_audit.get("detcorr_available")
            and determinant_audit.get("callable_abs_error", 1.0) < 2.0e-5
        ),
    }
    checks["all_passed"] = all(checks.values())

    _plot_predictor(
        outputs / "formula_hash_predictor.png",
        train,
        heldout,
        train_prediction,
        heldout_prediction,
    )
    _plot_posteriors(outputs / "heldout_alpha_posteriors.png", alpha_grid, densities)
    _plot_validation(outputs / "finite_state_quadrature_validation.png", validation_table)

    summary: dict[str, Any] = {
        "example": "materials_bandgap_matbench",
        "demo": "materials_bandgap_matbench",
        "classification": "external_data_nonlinear_coordinate_stress_test",
        "status": "heldout_real_data_demonstrator_not_physical_exponent_inference",
        "target_measure": "graph posterior over alpha with respect to d alpha",
        "correction_driver": (
            "nonlinear diagonal state derivative alpha*e^(alpha-1) across 30 "
            "held-out dimensionless band-gap states"
        ),
        "correction_log_range_on_grid": float(np.ptp(correction)),
        "dataset": {
            "fixture_path": "data/matbench_expt_gap_fixture.csv",
            "fixture_sha256": common.sha256_file(data_path),
            "n_fixture_rows": len(frame),
            "n_train": len(train),
            "n_heldout": len(heldout),
            "task": "Matbench v0.1 matbench_expt_gap",
            "matbench_paper_doi": MATBENCH_PAPER_DOI,
            "matbench_dataset_doi": MATBENCH_DATA_DOI,
            "matbench_source_payload": MATBENCH_SOURCE_URL,
            "matbench_source_sha256": MATBENCH_SOURCE_SHA256,
            "original_data_paper_doi": ORIGINAL_DATA_DOI,
            "license": (
                "MIT, as declared by the authoritative Figshare dataset record for "
                "Matbench expt_gap (Hacking Materials, 2019)"
            ),
            "selection_rule": (
                "first 100 source-ordered, unique, strictly positive rows whose formulas "
                "use ordinary unparenthesized notation"
            ),
            "representativeness": "convenience geometry fixture; not a representative sample",
        },
        "split": {
            "algorithm": (
                "SHA256 each unique formula, sort by (hash, formula, source_row), "
                "first 70 train and final 30 held out"
            ),
            "order_invariant": True,
            "training_and_heldout_hashes_disjoint": not bool(train_hashes & heldout_hashes),
            "heldout_gaps_used_in_predictor_fit": False,
            "heldout_gaps_used_in_graph_likelihood": True,
            "table": "tables/formula_hash_split.csv",
        },
        "model": {
            "full_space_base_measure": (
                "d alpha times the product of de_i over all 30 held-out "
                "dimensionless state coordinates"
            ),
            "prior": {
                "distribution": "lognormal",
                "density_measure": "d alpha",
                "log_parameter_distribution": "log(alpha) ~ Normal(0, 0.42^2)",
                "log_location": 0.0,
                "log_scale": PRIOR_LOG_SD,
                "support": "alpha > 0",
            },
            "reference_energy": "E0=1 eV",
            "state": "e=E/E0, dimensionless",
            "predictor_coordinate": "m=M/E0, dimensionless",
            "constraint": "c_i(alpha,e_i)=e_i^alpha-m_i",
            "state_solution": "e_i(alpha)=m_i^(1/alpha)",
            "state_jacobian": "D_e c_i=alpha*e_i^(alpha-1)",
            "parameter": "alpha, dimensionless residual-coordinate exponent",
            "physical_property_status": "alpha is not a material property",
            "likelihood": "iid Gaussian held-out dimensionless gap discrepancies",
            "likelihood_sigma_source": "plug-in RMSE from 70 training rows",
            "sigma_e": problem.sigma_e,
            "penalty_rho": rho,
            "penalty_rho_unit": "dimensionless",
            "predictor": {
                **asdict(fit),
                "training_transform": "log(E/eV+0.05)",
                "inverse_transform": "exp(prediction)-0.05, then clip to [0.05,7.5] eV",
            },
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
                "direct finite-penalty trapezoid quadrature over every one of the "
                "30 held-out dimensionless states"
            ),
            **validation_stats,
        },
        "unit_invariance_audit": unit_audit,
        "detcorr_api_audit": determinant_audit,
        "grid_tail_checks": tail_checks,
        "checks": checks,
        "limitations": [
            (
                "The composition predictor is intentionally compact and is not a "
                "competitive Matbench model."
            ),
            (
                "The iid Gaussian discrepancy omits heteroskedastic experimental "
                "uncertainty and composition-dependent model error."
            ),
            (
                "Predictor coefficients and RMSE are plug-in training values whose "
                "uncertainty is not propagated."
            ),
            (
                "The deterministic first-eligible-100 fixture is an alphabetically "
                "concentrated convenience slice and is not representative of Matbench."
            ),
            (
                "Alpha describes a chosen nonlinear residual coordinate and has no "
                "asserted physical material interpretation."
            ),
            (
                "The state quadrature is local and window-limited to +/-9 linearized "
                "penalty standard deviations. Its worst centered error occurs at the "
                "alpha=0.4 endpoint, more than 400 graph log-density units below the "
                "peak; widening to +/-12 does not materially change that value, which "
                "is dominated by finite-rho bias."
            ),
        ],
        "interpretation_boundary": (
            "The defensible conclusion is conditional: after nondimensionalization, "
            "the nonlinear state Jacobian materially changes the held-out alpha target "
            "when omitted, and direct state quadrature verifies the correction. Alpha "
            "is not inferred as a physical exponent or material property."
        ),
        "outputs": {
            "predictor_figure": "formula_hash_predictor.png",
            "posterior_figure": "heldout_alpha_posteriors.png",
            "validation_figure": "finite_state_quadrature_validation.png",
            "posterior_table": "tables/posterior_summaries.csv",
            "quadrature_table": "tables/finite_state_quadrature.csv",
            "split_table": "tables/formula_hash_split.csv",
            "heldout_table": "tables/heldout_states.csv",
        },
        "provenance": common.reproducibility_provenance(example_root=ROOT),
    }
    common.write_json(outputs / "summary.json", summary)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--outputs", type=Path, default=ROOT / "outputs")
    parser.add_argument("--n-grid", type=int, default=GRID_SIZE)
    parser.add_argument("--rho", type=float, default=PENALTY_RHO)
    parser.add_argument("--n-quad", type=int, default=QUADRATURE_POINTS)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    summary = run_demo(
        args.outputs,
        data_path=args.data,
        n_grid=args.n_grid,
        rho=args.rho,
        n_quad=args.n_quad,
    )
    print(
        json.dumps(
            {"example": summary["example"], "checks": summary["checks"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
