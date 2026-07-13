#!/usr/bin/env python3
"""Paired-JARVIS scalar screened-lattice determinant demonstrator.

This is deliberately a nondimensional scalar elliptic-network model, not an
elasticity solver. OPT bulk moduli set heterogeneous operator coefficients;
the distinct tabulated experimental bulk-modulus column sets a reciprocal response
proxy.  The physical data are real, while the spatial surrogate and its fixed
discrepancy model remain illustrative.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.integrate import trapezoid
from scipy.sparse import linalg as spla

ROOT = Path(__file__).resolve().parents[1]
COMMON = ROOT.parent / "detcorr_demo_common" / "src"
if str(COMMON) not in sys.path:
    sys.path.insert(0, str(COMMON))

from detcorr_demo_common import (  # noqa: E402
    centered_max_abs_diff,
    centered_validation_table,
    density_l1,
    normalize_on_grid,
    relative_display_path,
    reproducibility_provenance,
    select_grid_points,
    sha256_file,
    summarize_targets,
    write_json,
    write_table,
)

JARVIS_DB_DOCS_URL = "https://jarvis-materials-design.github.io/dbdocs/jarvisdft/"
FIXTURE_SHA256 = "6d9c37c52ef1d7eeb62d0cce3bc00f5b4c979b4b5d2b3bdcfcda8ab6a73cae86"
REQUIRED_COLUMNS = [
    "material",
    "formula",
    "jvasp_id",
    "bulk_modulus_opt_gpa",
    "bulk_modulus_expt_gpa",
]


@dataclass(frozen=True)
class SpectralBasis:
    """Separable eigenbasis for a nondimensional Dirichlet grid."""

    dimension: int
    grid_size: int
    axis_eigenvalues: np.ndarray
    response_weights: np.ndarray
    load_index: tuple[int, ...]
    observation_index: tuple[int, ...]

    @property
    def n_state_per_material(self) -> int:
        return int(self.grid_size**self.dimension)


def load_paired_fixture(data_path: Path) -> pd.DataFrame:
    """Load the pinned OPT/experimental JARVIS comparison table."""

    df = pd.read_csv(data_path)
    missing = [name for name in REQUIRED_COLUMNS if name not in df.columns]
    if missing:
        raise ValueError(f"paired JARVIS fixture is missing columns: {missing}")
    digest = sha256_file(data_path)
    if digest != FIXTURE_SHA256:
        raise ValueError(f"fixture SHA256 mismatch: expected {FIXTURE_SHA256}, got {digest}")
    if df["jvasp_id"].duplicated().any():
        raise ValueError("jvasp_id values must be unique")
    for name in ["bulk_modulus_opt_gpa", "bulk_modulus_expt_gpa"]:
        df[name] = pd.to_numeric(df[name], errors="raise")
        if (df[name] <= 0.0).any():
            raise ValueError(f"{name} values must be positive")
    return df.reset_index(drop=True)


def laplacian_1d(n: int) -> sparse.csr_matrix:
    """Unscaled graph Laplacian on a nondimensional Dirichlet path."""

    if n < 3:
        raise ValueError("grid size must be at least 3")
    return sparse.diags(
        [-np.ones(n - 1), 2.0 * np.ones(n), -np.ones(n - 1)],
        offsets=[-1, 0, 1],
        format="csr",
    )


def laplacian_1d_eigendecomposition(n: int) -> tuple[np.ndarray, np.ndarray]:
    modes = np.arange(1, n + 1, dtype=float)
    eigenvalues = 2.0 - 2.0 * np.cos(np.pi * modes / (n + 1.0))
    points = np.arange(1, n + 1, dtype=float)[:, None]
    eigenvectors = np.sqrt(2.0 / (n + 1.0)) * np.sin(points * modes[None, :] * np.pi / (n + 1.0))
    return eigenvalues, eigenvectors


def make_spectral_basis(
    dimension: int,
    grid_size: int,
    *,
    load_index: tuple[int, ...] | None = None,
    observation_index: tuple[int, ...] | None = None,
) -> SpectralBasis:
    if dimension not in (2, 3):
        raise ValueError("dimension must be 2 or 3")
    if grid_size < 3:
        raise ValueError("grid size must be at least 3")
    if load_index is None:
        load_index = tuple([grid_size // 2] * dimension)
    if observation_index is None:
        observation_index = tuple([max(1, grid_size - 2)] * dimension)
    if len(load_index) != dimension or len(observation_index) != dimension:
        raise ValueError("load and observation indices must match dimension")
    if any(index < 0 or index >= grid_size for index in (*load_index, *observation_index)):
        raise ValueError("load and observation indices must lie inside the grid")

    eig, vec = laplacian_1d_eigendecomposition(grid_size)
    axis_eigenvalues: list[np.ndarray] = []
    response_weights: list[float] = []
    for modes in itertools.product(range(grid_size), repeat=dimension):
        axis_eigenvalues.append(np.asarray([eig[mode] for mode in modes], dtype=float))
        weight = 1.0
        for axis, mode in enumerate(modes):
            weight *= float(vec[observation_index[axis], mode] * vec[load_index[axis], mode])
        response_weights.append(weight)
    return SpectralBasis(
        dimension=dimension,
        grid_size=grid_size,
        axis_eigenvalues=np.asarray(axis_eigenvalues),
        response_weights=np.asarray(response_weights),
        load_index=load_index,
        observation_index=observation_index,
    )


def operator_coefficients(df: pd.DataFrame, dimension: int) -> np.ndarray:
    """Map external OPT moduli to positive nondimensional coefficients."""

    if dimension not in (2, 3):
        raise ValueError("dimension must be 2 or 3")
    opt = df["bulk_modulus_opt_gpa"].to_numpy(float)
    relative = opt / float(np.median(opt))
    return np.repeat(relative[:, None], dimension, axis=1)


def lattice_eigenvalues(axis_coefficients: np.ndarray, basis: SpectralBasis) -> np.ndarray:
    coefficients = np.asarray(axis_coefficients, dtype=float)
    if coefficients.shape != (basis.dimension,):
        raise ValueError(f"axis coefficients must have shape ({basis.dimension},)")
    return basis.axis_eigenvalues @ coefficients


def response_from_spectrum(
    scale: float,
    axis_coefficients: np.ndarray,
    basis: SpectralBasis,
    *,
    pin: float,
) -> float:
    eigenvalues = lattice_eigenvalues(axis_coefficients, basis)
    return float(np.sum(basis.response_weights / (pin + float(scale) * eigenvalues)))


def logdet_from_spectrum(
    scale: float,
    coefficients: np.ndarray,
    basis: SpectralBasis,
    *,
    pin: float,
) -> float:
    total = 0.0
    for coefficient in np.asarray(coefficients, dtype=float):
        eigenvalues = lattice_eigenvalues(coefficient, basis)
        total += float(np.sum(np.log(pin + float(scale) * eigenvalues)))
    return total


def reference_response(basis: SpectralBasis, *, pin: float) -> float:
    """Response at scale=1 and unit coefficient, fixed before inference."""

    value = response_from_spectrum(1.0, np.ones(basis.dimension), basis, pin=pin)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("reference response must be finite and positive")
    return value


def build_paired_observations(
    df: pd.DataFrame,
    *,
    dimension: int,
    grid_size: int,
    pin: float,
    relative_discrepancy: float = 0.18,
    discrepancy_floor: float = 0.04,
) -> pd.DataFrame:
    """Build distinct operator-input and response-proxy columns from the paired table.

    OPT values determine operator coefficients. Experimental values determine
    a reciprocal response proxy. The Gaussian scales are fixed demonstrator
    discrepancy weights; they are not reported experimental uncertainties.
    """

    if relative_discrepancy <= 0.0 or discrepancy_floor <= 0.0:
        raise ValueError("discrepancy scales must be positive")
    basis = make_spectral_basis(dimension, grid_size)
    g_ref = reference_response(basis, pin=pin)
    coefficients = operator_coefficients(df, dimension)
    experimental = df["bulk_modulus_expt_gpa"].to_numpy(float)
    experimental_ratio = float(np.median(experimental)) / experimental
    normalized_sigma = np.sqrt(
        (relative_discrepancy * experimental_ratio) ** 2 + discrepancy_floor**2
    )

    rows: list[dict[str, Any]] = []
    for index, row in df.iterrows():
        item: dict[str, Any] = {
            "jvasp_id": row["jvasp_id"],
            "material": row["material"],
            "formula": row["formula"],
            "bulk_modulus_opt_gpa": float(row["bulk_modulus_opt_gpa"]),
            "bulk_modulus_expt_gpa": float(row["bulk_modulus_expt_gpa"]),
            "normalized_experimental_response_proxy": float(experimental_ratio[index]),
            "reference_green_response": float(g_ref),
            "observed_response": float(g_ref * experimental_ratio[index]),
            "sigma_response": float(g_ref * normalized_sigma[index]),
            "observation_origin": (
                "operator from external OPT modulus; response proxy from separately "
                "tabulated experimental modulus"
            ),
        }
        for axis in range(dimension):
            item[f"axis_coeff_{axis}"] = float(coefficients[index, axis])
        rows.append(item)
    return pd.DataFrame(rows)


def log_prior_scale(scale: float, *, log_sd: float = 0.32) -> float:
    if scale <= 0.0:
        return -math.inf
    return -0.5 * (math.log(scale) / log_sd) ** 2 - math.log(scale)


def graph_logdensity(
    scale_grid: np.ndarray,
    observations: pd.DataFrame,
    *,
    dimension: int,
    grid_size: int,
    pin: float,
) -> tuple[np.ndarray, np.ndarray]:
    coefficient_columns = [f"axis_coeff_{axis}" for axis in range(dimension)]
    coefficients = observations[coefficient_columns].to_numpy(float)
    response = observations["observed_response"].to_numpy(float)
    sigma = observations["sigma_response"].to_numpy(float)
    basis = make_spectral_basis(dimension, grid_size)
    log_graph = np.full_like(scale_grid, -np.inf, dtype=float)
    prediction_rms = np.full_like(scale_grid, np.nan, dtype=float)
    for index, scale in enumerate(scale_grid):
        if scale <= 0.0:
            continue
        prediction = np.asarray(
            [
                response_from_spectrum(float(scale), coefficient, basis, pin=pin)
                for coefficient in coefficients
            ]
        )
        residual = response - prediction
        prediction_rms[index] = float(np.sqrt(np.mean(residual**2)))
        log_graph[index] = -0.5 * float(np.sum((residual / sigma) ** 2)) + log_prior_scale(
            float(scale)
        )
    return log_graph, prediction_rms


def hard_limit_log_components(
    scale_grid: np.ndarray,
    observations: pd.DataFrame,
    *,
    dimension: int,
    grid_size: int,
    pin: float,
    residual_scale_power: int = 0,
) -> dict[str, np.ndarray]:
    coefficient_columns = [f"axis_coeff_{axis}" for axis in range(dimension)]
    coefficients = observations[coefficient_columns].to_numpy(float)
    basis = make_spectral_basis(dimension, grid_size)
    log_graph, prediction_rms = graph_logdensity(
        scale_grid,
        observations,
        dimension=dimension,
        grid_size=grid_size,
        pin=pin,
    )
    state_logdet = np.asarray(
        [logdet_from_spectrum(float(scale), coefficients, basis, pin=pin) for scale in scale_grid]
    )
    state_dimension = int(len(observations) * basis.n_state_per_material)
    residual_scale_logdet = state_dimension * int(residual_scale_power) * np.log(scale_grid)
    correction = state_logdet + residual_scale_logdet
    naive = log_graph - correction
    return {
        "log_graph": log_graph,
        "state_logdet_correction": state_logdet,
        "residual_scale_logdet_correction": residual_scale_logdet,
        "total_logdet_correction": correction,
        "log_naive": naive,
        "log_fully_corrected": naive + correction,
        "prediction_rms": prediction_rms,
    }


def build_main_table(
    scale_grid: np.ndarray,
    observations: pd.DataFrame,
    *,
    dimension: int,
    grid_size: int,
    pin: float,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, np.ndarray]]:
    components = hard_limit_log_components(
        scale_grid,
        observations,
        dimension=dimension,
        grid_size=grid_size,
        pin=pin,
    )
    table, densities, _ = summarize_targets(
        scale_grid,
        {
            "graph_reduced": components["log_graph"],
            "naive_screened_lattice_tube": components["log_naive"],
            "detcorr_screened_lattice_tube": components["log_fully_corrected"],
        },
    )
    return table, densities, components


def build_support_table(scale_grid: np.ndarray, densities: dict[str, np.ndarray]) -> pd.DataFrame:
    """Report edge mass and mode location to detect grid-truncated posteriors."""

    n_edge = max(3, int(0.02 * len(scale_grid)))
    rows: list[dict[str, Any]] = []
    for target, density in densities.items():
        _, _, summary = normalize_on_grid(scale_grid, np.log(np.maximum(density, 1e-300)))
        left_mass = float(trapezoid(density[:n_edge], scale_grid[:n_edge]))
        right_mass = float(trapezoid(density[-n_edge:], scale_grid[-n_edge:]))
        mode_index = int(np.argmax(density))
        rows.append(
            {
                "target": target,
                "mode": summary.mode,
                "mode_index": mode_index,
                "mode_is_interior": bool(0 < mode_index < len(scale_grid) - 1),
                "left_two_percent_grid_mass": left_mass,
                "right_two_percent_grid_mass": right_mass,
                "maximum_edge_mass": max(left_mass, right_mass),
            }
        )
    return pd.DataFrame(rows)


def build_residual_scaling_table(
    scale_grid: np.ndarray,
    observations: pd.DataFrame,
    *,
    dimension: int,
    grid_size: int,
    pin: float,
    powers: Iterable[int],
) -> pd.DataFrame:
    base = hard_limit_log_components(
        scale_grid,
        observations,
        dimension=dimension,
        grid_size=grid_size,
        pin=pin,
    )
    graph_density, _, graph_summary = normalize_on_grid(scale_grid, base["log_graph"])
    rows: list[dict[str, Any]] = []
    for power in powers:
        components = hard_limit_log_components(
            scale_grid,
            observations,
            dimension=dimension,
            grid_size=grid_size,
            pin=pin,
            residual_scale_power=int(power),
        )
        naive_density, _, naive_summary = normalize_on_grid(scale_grid, components["log_naive"])
        corrected_density, _, corrected_summary = normalize_on_grid(
            scale_grid, components["log_fully_corrected"]
        )
        naive_mode_index = int(np.argmax(naive_density))
        naive_mode_is_interior = bool(0 < naive_mode_index < len(scale_grid) - 1)
        naive_l1 = density_l1(scale_grid, naive_density, graph_density)
        rows.append(
            {
                "scale_power_s": int(power),
                "naive_mean": naive_summary.mean if naive_mode_is_interior else None,
                "naive_mean_grid_conditional": naive_summary.mean,
                "naive_mode": naive_summary.mode,
                "naive_mode_is_interior": naive_mode_is_interior,
                "naive_status": (
                    "grid-supported"
                    if naive_mode_is_interior
                    else "lower-cutoff-limited; mean not reported"
                ),
                "corrected_mean": corrected_summary.mean,
                "graph_mean": graph_summary.mean,
                "naive_density_l1_vs_graph": (naive_l1 if naive_mode_is_interior else None),
                "naive_density_l1_grid_conditional": naive_l1,
                "corrected_density_l1_vs_graph": density_l1(
                    scale_grid, corrected_density, graph_density
                ),
                "corrected_centered_logdensity_max_abs_diff_vs_graph": (
                    centered_max_abs_diff(components["log_fully_corrected"], base["log_graph"])
                ),
            }
        )
    return pd.DataFrame(rows)


def _records_with_nulls(table: pd.DataFrame) -> list[dict[str, Any]]:
    """Return records with explicit JSON nulls for inapplicable table cells."""

    return table.astype(object).where(table.notna(), None).to_dict(orient="records")


def kronecker_laplacian(
    coefficients: np.ndarray, *, dimension: int, grid_size: int
) -> sparse.csr_matrix:
    coefficients = np.asarray(coefficients, dtype=float)
    if coefficients.shape != (dimension,):
        raise ValueError(f"coefficients must have shape ({dimension},)")
    one_dimensional = laplacian_1d(grid_size)
    identity = sparse.eye(grid_size, format="csr")
    total = sparse.csr_matrix((grid_size**dimension, grid_size**dimension))
    for axis, coefficient in enumerate(coefficients):
        factors = [identity] * dimension
        factors[axis] = one_dimensional
        term = factors[0]
        for factor in factors[1:]:
            term = sparse.kron(term, factor, format="csr")
        total = total + float(coefficient) * term
    return total.tocsr()


def material_matrix(
    scale: float,
    coefficients: np.ndarray,
    *,
    dimension: int,
    grid_size: int,
    pin: float,
) -> sparse.csr_matrix:
    state_size = grid_size**dimension
    return (
        pin * sparse.eye(state_size, format="csr")
        + float(scale) * kronecker_laplacian(coefficients, dimension=dimension, grid_size=grid_size)
    ).tocsr()


def block_sparse_matrix(
    scale: float,
    coefficients: np.ndarray,
    *,
    dimension: int,
    grid_size: int,
    pin: float,
) -> sparse.csc_matrix:
    return sparse.block_diag(
        [
            material_matrix(
                scale,
                coefficient,
                dimension=dimension,
                grid_size=grid_size,
                pin=pin,
            )
            for coefficient in coefficients
        ],
        format="csc",
    )


def block_linear_operator(
    scale: float,
    coefficients: np.ndarray,
    *,
    dimension: int,
    grid_size: int,
    pin: float,
) -> spla.LinearOperator:
    blocks = [
        material_matrix(
            scale,
            coefficient,
            dimension=dimension,
            grid_size=grid_size,
            pin=pin,
        )
        for coefficient in coefficients
    ]
    block_size = blocks[0].shape[0]
    total_size = block_size * len(blocks)

    def matvec(vector: np.ndarray) -> np.ndarray:
        vector = np.asarray(vector, dtype=float)
        output = np.empty_like(vector)
        for index, block in enumerate(blocks):
            lo = index * block_size
            hi = lo + block_size
            output[lo:hi] = block @ vector[lo:hi]
        return output

    return spla.LinearOperator((total_size, total_size), matvec=matvec, rmatvec=matvec, dtype=float)


def exact_sparse_lu_logdet(matrix: sparse.spmatrix) -> float:
    lu = spla.splu(matrix.tocsc())
    return float(np.sum(np.log(np.abs(lu.U.diagonal()))))


def _point_vector(basis: SpectralBasis, index: tuple[int, ...]) -> np.ndarray:
    output = np.zeros(basis.n_state_per_material)
    output[int(np.ravel_multi_index(index, (basis.grid_size,) * basis.dimension))] = 1.0
    return output


def _linear_gaussian_penalty_log_integral(
    matrix: sparse.spmatrix,
    force: np.ndarray,
    observation_vector: np.ndarray,
    observed: float,
    sigma: float,
    *,
    rho: float,
) -> float:
    """Evaluate the full-state finite-penalty Gaussian integral exactly."""

    matrix = matrix.tocsc()
    state_size = int(matrix.shape[0])
    precision = (rho * (matrix.T @ matrix)).tocsc() + sparse.csc_matrix(
        np.outer(observation_vector, observation_vector) / sigma**2
    )
    linear = rho * (matrix.T @ force) + observed * observation_vector / sigma**2
    lu = spla.splu(precision)
    logdet_precision = float(np.sum(np.log(np.abs(lu.U.diagonal()))))
    solved = lu.solve(np.asarray(linear))
    constant = rho * float(force @ force) + observed**2 / sigma**2
    return (
        0.5 * state_size * math.log(2.0 * math.pi)
        - 0.5 * logdet_precision
        + 0.5 * float(linear @ solved)
        - 0.5 * constant
    )


def finite_penalty_validation_table(
    scale_grid: np.ndarray,
    observations: pd.DataFrame,
    *,
    dimension: int,
    grid_size: int,
    pin: float,
    n_points: int = 13,
    rho: float = 5.0e10,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Independently integrate the finite-penalty target over all states."""

    if int(n_points) < 2:
        raise ValueError("n_points must be at least two so both posterior supports are checked")
    all_components = hard_limit_log_components(
        scale_grid,
        observations,
        dimension=dimension,
        grid_size=grid_size,
        pin=pin,
    )
    _, graph_cdf, _ = normalize_on_grid(scale_grid, all_components["log_graph"])
    _, naive_cdf, _ = normalize_on_grid(scale_grid, all_components["log_naive"])
    graph_central = scale_grid[(graph_cdf >= 0.01) & (graph_cdf <= 0.99)]
    naive_central = scale_grid[(naive_cdf >= 0.01) & (naive_cdf <= 0.99)]
    graph_count = (int(n_points) + 1) // 2
    naive_count = int(n_points) // 2
    values = np.unique(
        np.concatenate(
            [
                select_grid_points(graph_central, n_points=graph_count),
                select_grid_points(naive_central, n_points=naive_count),
            ]
        )
    )
    components = hard_limit_log_components(
        values,
        observations,
        dimension=dimension,
        grid_size=grid_size,
        pin=pin,
    )
    coefficient_columns = [f"axis_coeff_{axis}" for axis in range(dimension)]
    coefficients = observations[coefficient_columns].to_numpy(float)
    response = observations["observed_response"].to_numpy(float)
    sigma = observations["sigma_response"].to_numpy(float)
    basis = make_spectral_basis(dimension, grid_size)
    force = _point_vector(basis, basis.load_index)
    observation_vector = _point_vector(basis, basis.observation_index)
    finite_logs: list[float] = []
    for scale in values:
        log_integral = log_prior_scale(float(scale))
        for coefficient, observed, sigma_i in zip(coefficients, response, sigma, strict=True):
            matrix = material_matrix(
                float(scale),
                coefficient,
                dimension=dimension,
                grid_size=grid_size,
                pin=pin,
            )
            log_integral += _linear_gaussian_penalty_log_integral(
                matrix,
                force,
                observation_vector,
                float(observed),
                float(sigma_i),
                rho=rho,
            )
        finite_logs.append(log_integral)
    finite = np.asarray(finite_logs)
    # Re-adding the exact state determinant is a consistency display, not a
    # second independent validation.  The finite-vs-hard comparison below is
    # the load-bearing full-state integration check.
    finite_corrected = finite + components["state_logdet_correction"]
    table, stats = centered_validation_table(
        "scale",
        values,
        finite,
        components["log_naive"],
        finite_corrected,
        components["log_graph"],
    )
    state_dimension = int(len(observations) * basis.n_state_per_material)
    table["rho"] = float(rho)
    table["state_dimension"] = state_dimension
    stats.update(
        {
            "route": "exact_sparse_multivariate_Gaussian_state_integration",
            "rho": float(rho),
            "rho_units": "inverse squared nondimensional residual",
            "state_dimension": state_dimension,
            "n_materials": int(len(observations)),
            "n_points": int(len(values)),
            "support_scope": "union of central 98% graph and naive posterior support",
            "load_bearing_metric": "finite_vs_hard_tube_centered_max_abs_diff",
            "finite_corrected_vs_graph_role": (
                "derived from the independently integrated finite tube by re-adding "
                "the same exact determinant; a consistency view, not a separate check"
            ),
            "graph_central_98_interval": [
                float(graph_central[0]),
                float(graph_central[-1]),
            ],
            "naive_central_98_interval": [
                float(naive_central[0]),
                float(naive_central[-1]),
            ],
        }
    )
    return table, stats


def _slq_configurations(
    probe_counts: list[int], lanczos_degrees: list[int]
) -> list[tuple[int, int]]:
    probes = sorted({int(value) for value in probe_counts})
    degrees = sorted({int(value) for value in lanczos_degrees})
    configurations = [(probe, degrees[0]) for probe in probes]
    configurations.extend((probes[-1], degree) for degree in degrees[1:])
    return configurations


def build_determinant_audit(
    reference_scale: float,
    support_scales: Iterable[float],
    df: pd.DataFrame,
    *,
    pin: float,
    audit_grid_2d: int,
    audit_grid_3d: int,
    stress_grid_3d: int,
    slq_probe_counts: list[int],
    slq_degrees: list[int],
    slq_seeds: list[int],
) -> pd.DataFrame:
    """Audit exact logdets and centered SLQ variation over posterior support."""

    rows: list[dict[str, Any]] = []
    for dimension, grid_size in [(2, audit_grid_2d), (3, audit_grid_3d)]:
        coefficients = operator_coefficients(df, dimension)
        basis = make_spectral_basis(dimension, grid_size)
        analytic = logdet_from_spectrum(reference_scale, coefficients, basis, pin=pin)
        matrix = block_sparse_matrix(
            reference_scale,
            coefficients,
            dimension=dimension,
            grid_size=grid_size,
            pin=pin,
        )
        sparse_value = exact_sparse_lu_logdet(matrix)
        rows.append(
            {
                "scenario": f"{dimension}d_grid_{grid_size}",
                "route": "exact_sparse_lu",
                "dimension": dimension,
                "grid_size": grid_size,
                "state_dimension": int(matrix.shape[0]),
                "scale": float(reference_scale),
                "reference_scale": float(reference_scale),
                "value": sparse_value,
                "analytic_reference": analytic,
                "centered_value": 0.0,
                "centered_analytic_reference": 0.0,
                "centered_abs_error": abs(sparse_value - analytic),
                "n_probe": None,
                "lanczos_degree": None,
                "seed": None,
                "error_estimate": None,
                "interpretation": "exact cross-check",
            }
        )

    dimension = 3
    grid_size = stress_grid_3d
    coefficients = operator_coefficients(df, dimension)
    basis = make_spectral_basis(dimension, grid_size)
    scales = [float(reference_scale)]
    for value in support_scales:
        if not any(abs(float(value) - prior) < 1e-12 for prior in scales):
            scales.append(float(value))
    analytic = {
        scale: logdet_from_spectrum(scale, coefficients, basis, pin=pin) for scale in scales
    }
    try:
        import detcorr  # type: ignore
    except Exception as exc:  # pragma: no cover - environment dependent
        rows.append(
            {
                "scenario": f"3d_grid_{grid_size}",
                "route": "matrix_free_gram_slq_unavailable",
                "dimension": dimension,
                "grid_size": grid_size,
                "state_dimension": int(len(df) * grid_size**dimension),
                "scale": float(reference_scale),
                "reference_scale": float(reference_scale),
                "value": None,
                "analytic_reference": analytic[reference_scale],
                "centered_value": None,
                "centered_analytic_reference": 0.0,
                "centered_abs_error": None,
                "n_probe": None,
                "lanczos_degree": None,
                "seed": None,
                "error_estimate": None,
                "interpretation": repr(exc),
            }
        )
        return pd.DataFrame(rows)

    for seed in slq_seeds:
        for n_probe, degree in _slq_configurations(slq_probe_counts, slq_degrees):
            estimates: dict[float, tuple[float, float | None]] = {}
            for scale in scales:
                operator = block_linear_operator(
                    scale,
                    coefficients,
                    dimension=dimension,
                    grid_size=grid_size,
                    pin=pin,
                )
                result = detcorr.log_correction(
                    jacobian=operator,
                    method="gram_slq",
                    n_probe=n_probe,
                    lanczos_degree=min(degree, operator.shape[0]),
                    seed=seed,
                    reorthogonalize=True,
                    adjoint_check=True,
                    exact_check_max_dimension=0,
                    warn_relative_standard_error=None,
                )
                estimates[scale] = (
                    float(result.value),
                    result.error_estimate,
                )
            reference_estimate = estimates[reference_scale][0]
            for scale in scales:
                estimate, error_estimate = estimates[scale]
                centered_estimate = estimate - reference_estimate
                centered_exact = analytic[scale] - analytic[reference_scale]
                rows.append(
                    {
                        "scenario": f"3d_grid_{grid_size}",
                        "route": "matrix_free_centered_gram_slq",
                        "dimension": dimension,
                        "grid_size": grid_size,
                        "state_dimension": int(len(df) * grid_size**dimension),
                        "scale": scale,
                        "reference_scale": float(reference_scale),
                        "value": estimate,
                        "analytic_reference": analytic[scale],
                        "centered_value": centered_estimate,
                        "centered_analytic_reference": centered_exact,
                        "centered_abs_error": abs(centered_estimate - centered_exact),
                        "n_probe": n_probe,
                        "lanczos_degree": degree,
                        "seed": seed,
                        "error_estimate": error_estimate,
                        "interpretation": (
                            "same seed is reused at every scale; compare centered "
                            "error over posterior support, not absolute logdet offset"
                        ),
                    }
                )
    return pd.DataFrame(rows)


def make_plots(
    scale_grid: np.ndarray,
    densities: dict[str, np.ndarray],
    scaling_table: pd.DataFrame,
    observations: pd.DataFrame,
    audit_table: pd.DataFrame,
    outputs: Path,
) -> None:
    outputs.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(
        scale_grid,
        densities["naive_screened_lattice_tube"],
        color="tab:orange",
        label="naive residual tube",
    )
    ax.plot(
        scale_grid,
        densities["detcorr_screened_lattice_tube"],
        color="tab:green",
        linewidth=3.0,
        label="detcorr tube",
    )
    ax.plot(
        scale_grid,
        densities["graph_reduced"],
        color="tab:blue",
        linestyle="--",
        linewidth=1.8,
        zorder=4,
        label="reduced graph posterior",
    )
    ax.set_ylabel("posterior density")
    ax.legend(frameon=False)
    ax.set_xlabel("nondimensional operator scale s")
    ax.set_title("Paired-data screened-lattice scale posterior")
    fig.tight_layout()
    fig.savefig(outputs / "posterior_spatial_lattice_scale.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.scatter(
        observations["bulk_modulus_opt_gpa"],
        observations["bulk_modulus_expt_gpa"],
        s=28,
    )
    limit = float(
        max(
            observations["bulk_modulus_opt_gpa"].max(),
            observations["bulk_modulus_expt_gpa"].max(),
        )
    )
    ax.plot([0.0, limit], [0.0, limit], linestyle="--", color="0.45", linewidth=1)
    ax.set_xlabel("JARVIS OPT bulk modulus / GPa")
    ax.set_ylabel("tabulated experimental bulk modulus / GPa")
    ax.set_title("Distinct paired operator-input and response-proxy columns")
    fig.tight_layout()
    fig.savefig(outputs / "paired_bulk_modulus_inputs.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.3))
    ax.plot(
        scaling_table["scale_power_s"],
        scaling_table["naive_mean"],
        marker="o",
        label="naive (only where grid-supported)",
    )
    ax.plot(
        scaling_table["scale_power_s"],
        scaling_table["corrected_mean"],
        marker="o",
        label="detcorr",
    )
    ax.axhline(
        float(scaling_table["graph_mean"].iloc[0]),
        linestyle="--",
        color="0.4",
        linewidth=1,
        label="graph",
    )
    ax.set_xlabel("residual multiplier power q in s^q c")
    ax.set_ylabel("posterior mean s")
    ax.set_title("Hard-limit residual-coordinate check")
    if not scaling_table["naive_mode_is_interior"].all():
        ax.text(
            0.02,
            0.62,
            "missing naive points are lower-cutoff limited",
            transform=ax.transAxes,
            fontsize=8,
        )
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outputs / "residual_scaling_invariance.png", dpi=180)
    plt.close(fig)

    stochastic = audit_table[audit_table["route"].astype(str) == "matrix_free_centered_gram_slq"]
    stochastic = stochastic[stochastic["scale"] != stochastic["reference_scale"]]
    if not stochastic.empty:
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        grouped = stochastic.groupby(["n_probe", "lanczos_degree"])
        aggregate = grouped["centered_abs_error"].max().reset_index()
        for degree, group in aggregate.groupby("lanczos_degree"):
            ax.plot(
                group["n_probe"],
                group["centered_abs_error"],
                marker="o",
                label=f"degree {int(degree)}",
            )
        ax.set_yscale("log")
        ax.set_xlabel("SLQ probe count")
        ax.set_ylabel("max centered logdet error over support")
        ax.set_title("Matrix-free posterior-shape audit (worst seed)")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(outputs / "exact_vs_matrix_free_audit.png", dpi=180)
        plt.close(fig)


def _row(table: pd.DataFrame, target: str) -> dict[str, Any]:
    return table.loc[table["target"] == target].iloc[0].to_dict()


def run_demo(args: argparse.Namespace) -> dict[str, Any]:
    data_path = Path(args.data)
    df = load_paired_fixture(data_path)
    observations = build_paired_observations(
        df,
        dimension=args.dimension,
        grid_size=args.grid_size,
        pin=args.pin,
        relative_discrepancy=args.relative_discrepancy,
        discrepancy_floor=args.discrepancy_floor,
    )
    scale_grid = np.linspace(args.scale_min, args.scale_max, args.n_grid)
    main_table, densities, components = build_main_table(
        scale_grid,
        observations,
        dimension=args.dimension,
        grid_size=args.grid_size,
        pin=args.pin,
    )
    support_table = build_support_table(scale_grid, densities)
    if not support_table["mode_is_interior"].all():
        raise RuntimeError("scale grid truncates at least one posterior mode")
    if float(support_table["maximum_edge_mass"].max()) > 1.0e-3:
        raise RuntimeError("scale grid leaves excessive posterior mass at an edge")
    scaling_table = build_residual_scaling_table(
        scale_grid,
        observations,
        dimension=args.dimension,
        grid_size=args.grid_size,
        pin=args.pin,
        powers=[0, 1, 2, 3],
    )
    graph = _row(main_table, "graph_reduced")
    graph_mode = float(graph["mode"])
    support_scales = [float(graph["q05"]), graph_mode, float(graph["q95"])]
    audit_table = build_determinant_audit(
        graph_mode,
        support_scales,
        df,
        pin=args.pin,
        audit_grid_2d=args.audit_grid_2d,
        audit_grid_3d=args.audit_grid_3d,
        stress_grid_3d=args.stress_grid_3d,
        slq_probe_counts=[int(value) for value in args.slq_probes.split(",")],
        slq_degrees=[int(value) for value in args.slq_degrees.split(",")],
        slq_seeds=[int(value) for value in args.slq_seeds.split(",")],
    )
    validation_table, validation_stats = finite_penalty_validation_table(
        scale_grid,
        observations,
        dimension=args.dimension,
        grid_size=args.grid_size,
        pin=args.pin,
        rho=args.rho,
    )

    outputs = Path(args.outputs)
    table_dir = outputs / "tables"
    write_table(main_table, table_dir / "main_hard_limit_posteriors.csv")
    write_table(scaling_table, table_dir / "residual_scaling_sensitivity.csv")
    write_table(observations, table_dir / "paired_screened_lattice_observations.csv")
    write_table(audit_table, table_dir / "exact_vs_matrix_free_audit.csv")
    write_table(validation_table, table_dir / "finite_penalty_tube_validation.csv")
    write_table(support_table, table_dir / "grid_support_diagnostics.csv")
    make_plots(scale_grid, densities, scaling_table, observations, audit_table, outputs)

    naive = _row(main_table, "naive_screened_lattice_tube")
    corrected = _row(main_table, "detcorr_screened_lattice_tube")
    graph_sd = float(graph["sd"])
    correction_range = float(np.ptp(components["state_logdet_correction"]))
    stochastic = audit_table[audit_table["route"] == "matrix_free_centered_gram_slq"]
    stochastic = stochastic[stochastic["scale"] != stochastic["reference_scale"]]
    matrix_free_summary = {
        "route": "centered Gram/SLQ differences over q05/mode/q95 support",
        "common_random_numbers": True,
        "n_seeds": int(stochastic["seed"].nunique()) if not stochastic.empty else 0,
        "max_centered_abs_error": (
            float(stochastic["centered_abs_error"].max()) if not stochastic.empty else None
        ),
        "interpretation": (
            "Estimator diagnostics only; exact/separable correction remains the "
            "posterior route in this demonstrator."
        ),
    }
    summary = {
        "demo": "materials_screened_lattice_2d3d_jarvis",
        "classification": "paired_external_data_simplified_spatial_surrogate_demonstrator",
        "target_measure": "graph posterior over s with respect to Lebesgue measure ds",
        "correction_driver": (
            "parameter-dependent sparse screened-lattice state operator repeated "
            f"across {len(df)} paired materials"
        ),
        "correction_log_range_on_grid": correction_range,
        "correction_route": "separable spectrum and exact sparse LU; centered Gram/SLQ diagnostic",
        "independent_validation": validation_stats,
        "interpretation_boundary": (
            "Real OPT and experimental bulk-modulus columns condition a deliberately "
            "nondimensional scalar network. The inferred s is not an elastic constant, "
            "material property, or transferable physical calibration."
        ),
        "provenance": reproducibility_provenance(example_root=ROOT),
        "dataset": {
            "fixture_path": relative_display_path(data_path, root=ROOT),
            "fixture_sha256": sha256_file(data_path),
            "source_record": JARVIS_DB_DOCS_URL,
            "n_materials": int(len(df)),
            "operator_input": "bulk_modulus_opt_gpa",
            "dependent_response_input": "bulk_modulus_expt_gpa",
            "source_verification_date": "2026-07-11",
            "upstream_license_status": (
                "NIST public-information permission with attribution and disclaimer; "
                "the table is not relicensed as MIT or CC BY"
            ),
        },
        "model": {
            "full_space_base_measure": (
                "ds times product_j d^(grid_size^dimension)u_j over all "
                "dimensionless scalar state coordinates"
            ),
            "prior": {
                "distribution": "lognormal",
                "density_measure": "ds",
                "log_parameter_distribution": "log(s) ~ Normal(0, 0.32^2)",
                "log_location": 0.0,
                "log_scale": 0.32,
                "support": "s > 0",
            },
            "constraint": "c_j(s,u_j)=(pin I+s k_j L_d)u_j-f=0",
            "state": "nondimensional scalar grid field",
            "not_modelled": (
                "vector elasticity, shear, tensor constitutive laws, or physical mesh spacing"
            ),
            "dimension": int(args.dimension),
            "grid_size": int(args.grid_size),
            "state_dimension": int(len(df) * args.grid_size**args.dimension),
            "pin": float(args.pin),
            "normalization": (
                "response divided by the Green response at s=1 and k=1 on the fixed grid"
            ),
            "discrepancy_model": {
                "status": "fixed demonstrator Gaussian discrepancy, not experimental uncertainty",
                "relative": float(args.relative_discrepancy),
                "floor_in_normalized_response": float(args.discrepancy_floor),
            },
        },
        "outcome": {
            "graph_mean": float(graph["mean"]),
            "graph_sd": graph_sd,
            "naive_mean_shift_in_graph_sd": (float(naive["mean"] - graph["mean"]) / graph_sd),
            "corrected_mean_shift_in_graph_sd": (
                float(corrected["mean"] - graph["mean"]) / graph_sd
            ),
            "naive_minus_corrected_mean_in_graph_sd": (
                float(naive["mean"] - corrected["mean"]) / graph_sd
            ),
            "naive_density_l1_vs_graph": float(naive["density_l1_vs_graph"]),
            "corrected_density_l1_vs_graph": float(corrected["density_l1_vs_graph"]),
            "corrected_density_l1_role": (
                "algebraic hard-limit round-trip consistency; not independent validation"
            ),
            "correction_log_range_on_grid": correction_range,
            "maximum_grid_edge_mass": float(support_table["maximum_edge_mass"].max()),
        },
        "matrix_free_summary": matrix_free_summary,
        "tables": {
            "main": main_table.to_dict(orient="records"),
            "grid_support": support_table.to_dict(orient="records"),
            "residual_scaling": _records_with_nulls(scaling_table),
            "determinant_audit": _records_with_nulls(audit_table),
            "finite_penalty_validation": validation_table.to_dict(orient="records"),
        },
    }
    write_json(outputs / "summary.json", summary)

    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", default=str(ROOT / "data" / "jarvis_bulk_modulus_paired_fixture.csv")
    )
    parser.add_argument("--outputs", default=str(ROOT / "outputs"))
    parser.add_argument("--dimension", type=int, choices=[2, 3], default=3)
    parser.add_argument("--grid-size", type=int, default=5)
    parser.add_argument("--pin", type=float, default=0.06)
    parser.add_argument("--relative-discrepancy", type=float, default=0.18)
    parser.add_argument("--discrepancy-floor", type=float, default=0.04)
    parser.add_argument("--scale-min", type=float, default=0.04)
    parser.add_argument("--scale-max", type=float, default=3.0)
    parser.add_argument("--n-grid", type=int, default=900)
    parser.add_argument("--rho", type=float, default=5.0e10)
    parser.add_argument("--audit-grid-2d", type=int, default=8)
    parser.add_argument("--audit-grid-3d", type=int, default=4)
    parser.add_argument("--stress-grid-3d", type=int, default=5)
    parser.add_argument("--slq-probes", default="4,12")
    parser.add_argument("--slq-degrees", default="24,40")
    parser.add_argument("--slq-seeds", default="731,1741")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    summary = run_demo(args)
    print(f"Wrote paired-JARVIS screened-lattice outputs to {args.outputs}")
    print(
        json.dumps(
            {
                "demo": summary["demo"],
                "n_materials": summary["dataset"]["n_materials"],
                "state_dimension": summary["model"]["state_dimension"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
