"""Consistent finite-difference 1D elliptic inverse-problem benchmark.

The benchmark uses the nondimensional finite-difference inverse problem

    -exp(theta) u''(x) + MASS u(x) = 1,  0 < x < 1,
    u(0) = u(1) = 0.

The second derivative uses the standard centred ``1 / h**2`` stencil at the
interior grid points.  A noisy synthetic observation of the midpoint value is
generated from a declared reference parameter; it is a transparent numerical
benchmark, not an empirical PDE calibration.

The benchmark compares three deterministic posterior routes:

* a reduced-space target over theta with the state eliminated by a linear solve,
* a naive full-space finite-penalty target integrated over the state u, and
* a determinant-corrected full-space finite-penalty target integrated over u.

The full-space integration is analytic because the state equation and the
single observation model are linear-Gaussian conditional on theta.  This keeps
the benchmark CI-safe while still exposing the determinant route that a sampler
would use pointwise: ``log |det A(theta)|`` for
``c(theta, u) = A(theta)u - f``.  For large penalty strength rho, the corrected
full-space marginal converges to the reduced-space posterior up to a
normalization constant, whereas the naive full-space marginal retains the
Jacobian-volume bias.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from detcorr_demo_common import require_supported_detcorr

_detcorr = require_supported_detcorr(context="the elliptic PDE example")
corrected_finite_penalty_target = _detcorr.corrected_finite_penalty_target
logabsdet_matrix = _detcorr.logabsdet_matrix

N_STATE = 6
DOMAIN_LENGTH = 1.0
MASS = 0.5
FORCING_VALUE = 1.0
SYNTHETIC_TRUE_THETA = 0.25
OBSERVATION_NOISE = 0.015
SYNTHETIC_NOISE_OFFSET_SD = 0.25
PRIOR_SD = 1.15
RHO = 1.0e5
THETA_MIN = -1.4
THETA_MAX = 1.4
GRID_SIZE = 1001


def stiffness_matrix(n: int = N_STATE) -> np.ndarray:
    """Return the centred-FD matrix for ``-d2/dx2`` with Dirichlet data."""

    if n < 1:
        raise ValueError("n must be positive.")
    h = DOMAIN_LENGTH / (n + 1)
    main = np.full(n, 2.0 / h**2)
    off = np.full(n - 1, -1.0 / h**2)
    K = np.diag(main)
    K += np.diag(off, k=1)
    K += np.diag(off, k=-1)
    return K


GRID_SPACING = DOMAIN_LENGTH / (N_STATE + 1)
K_STIFFNESS = stiffness_matrix()
FORCING = np.full(N_STATE, FORCING_VALUE)
OBSERVATION_VECTOR = np.zeros(N_STATE)
# With an even number of interior points, linear interpolation at x=0.5 is the
# mean of the two central nodal values.
OBSERVATION_VECTOR[N_STATE // 2 - 1 : N_STATE // 2 + 1] = 0.5


def diffusion(theta: float) -> float:
    """Positive scalar PDE coefficient."""

    return math.exp(theta)


def operator_matrix(theta: float) -> np.ndarray:
    """Affine-in-state PDE operator A(theta)."""

    return diffusion(theta) * K_STIFFNESS + MASS * np.eye(N_STATE)


def residual(theta: float, u: np.ndarray) -> np.ndarray:
    """PDE residual c(theta, u) = A(theta)u - f."""

    return operator_matrix(theta) @ np.asarray(u, dtype=float) - FORCING


def solve_state(theta: float) -> np.ndarray:
    """Reduced-space state solve u = g(theta)."""

    return np.linalg.solve(operator_matrix(theta), FORCING)


def observation(u: np.ndarray) -> float:
    """Linearly interpolated midpoint-observation functional ``H u``."""

    return float(OBSERVATION_VECTOR @ np.asarray(u, dtype=float))


NOISE_FREE_OBSERVATION = observation(solve_state(SYNTHETIC_TRUE_THETA))
OBSERVED_VALUE = NOISE_FREE_OBSERVATION + SYNTHETIC_NOISE_OFFSET_SD * OBSERVATION_NOISE


def log_prior(theta: float) -> float:
    return -0.5 * (theta / PRIOR_SD) ** 2


def reduced_logposterior(theta: float) -> float:
    """Reduced-space target with the state equation solved exactly."""

    u = solve_state(theta)
    misfit = (observation(u) - OBSERVED_VALUE) / OBSERVATION_NOISE
    return log_prior(theta) - 0.5 * misfit**2


def _logdet_spd(matrix: np.ndarray) -> float:
    chol = np.linalg.cholesky(matrix)
    return 2.0 * float(np.sum(np.log(np.diag(chol))))


def integrated_fullspace_logposterior(theta: float, *, corrected: bool) -> float:
    """Analytically integrate the full-space finite-penalty target over u.

    Terms independent of ``theta``, including ``n/2 log(2 pi)``, are omitted
    because every returned route is normalized on the same theta grid.
    """

    A = operator_matrix(theta)
    h_vec = OBSERVATION_VECTOR
    inv_sigma2 = 1.0 / (OBSERVATION_NOISE**2)
    Q = RHO * (A.T @ A) + inv_sigma2 * np.outer(h_vec, h_vec)
    linear = RHO * (A.T @ FORCING) + inv_sigma2 * h_vec * OBSERVED_VALUE
    constant = RHO * float(FORCING @ FORCING) + inv_sigma2 * OBSERVED_VALUE**2
    gaussian_integral = 0.5 * float(linear @ np.linalg.solve(Q, linear))
    gaussian_integral -= 0.5 * constant
    gaussian_integral -= 0.5 * _logdet_spd(Q)
    correction = _logdet_spd(A) if corrected else 0.0
    return log_prior(theta) + correction + gaussian_integral


def audited_pointwise_corrected_target(theta: float, u: np.ndarray):
    """Evaluate the pointwise corrected full-space target with detcorr."""

    A = operator_matrix(theta)
    return corrected_finite_penalty_target(
        residual(theta, u),
        rho=RHO,
        logdet=logabsdet_matrix(A, method="spd", singular_policy="raise"),
        target_measure="graph",
        base_logdensity=log_prior(theta)
        - 0.5 * ((observation(u) - OBSERVED_VALUE) / OBSERVATION_NOISE) ** 2,
        state=u,
        projected_state=solve_state(theta),
        metadata={
            "case": "1d_elliptic_reaction_diffusion_fd",
            "determinant_route": "exact_spd_cholesky",
            "target_measure": "graph",
            "discretization": "centred_second_order_finite_difference",
        },
    )


@dataclass(frozen=True)
class GridPosterior:
    theta: np.ndarray
    logdensity: np.ndarray
    weights: np.ndarray
    mean: float
    variance: float
    q05: float
    q50: float
    q95: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean": self.mean,
            "variance": self.variance,
            "q05": self.q05,
            "q50": self.q50,
            "q95": self.q95,
            "logdensity_min": float(np.min(self.logdensity)),
            "logdensity_max": float(np.max(self.logdensity)),
        }


def _normalize_grid(theta: np.ndarray, logdensity: np.ndarray) -> GridPosterior:
    shifted = logdensity - float(np.max(logdensity))
    weights = np.exp(shifted)
    weights = weights / float(np.sum(weights))
    mean = float(np.sum(weights * theta))
    variance = float(np.sum(weights * (theta - mean) ** 2))
    cdf = np.cumsum(weights)
    q05 = float(np.interp(0.05, cdf, theta))
    q50 = float(np.interp(0.50, cdf, theta))
    q95 = float(np.interp(0.95, cdf, theta))
    return GridPosterior(
        theta=theta,
        logdensity=logdensity,
        weights=weights,
        mean=mean,
        variance=variance,
        q05=q05,
        q50=q50,
        q95=q95,
    )


def _route(theta: np.ndarray, name: str) -> tuple[GridPosterior, float]:
    start = time.perf_counter()
    if name == "reduced_space":
        logdensity = np.asarray([reduced_logposterior(float(t)) for t in theta], dtype=float)
    elif name == "corrected_full_space":
        logdensity = np.asarray(
            [integrated_fullspace_logposterior(float(t), corrected=True) for t in theta],
            dtype=float,
        )
    elif name == "naive_full_space":
        logdensity = np.asarray(
            [integrated_fullspace_logposterior(float(t), corrected=False) for t in theta],
            dtype=float,
        )
    else:  # pragma: no cover - defensive
        raise ValueError(f"unknown route {name!r}.")
    elapsed = time.perf_counter() - start
    return _normalize_grid(theta, logdensity), elapsed


def _audit_pointwise_targets() -> dict[str, Any]:
    theta_values = [-0.8, -0.2, 0.4, 0.9]
    max_abs_logdet_error = 0.0
    max_projection_error_norm = 0.0
    max_reduced_residual_norm = 0.0
    for theta in theta_values:
        projected_state = solve_state(theta)
        max_reduced_residual_norm = max(
            max_reduced_residual_norm,
            float(np.linalg.norm(residual(theta, projected_state))),
        )
        u = projected_state + 0.01 * np.sin(np.arange(N_STATE) + theta)
        target = audited_pointwise_corrected_target(theta, u)
        expected_logdet = _logdet_spd(operator_matrix(theta))
        max_abs_logdet_error = max(
            max_abs_logdet_error,
            abs(float(target.primary_correction) - expected_logdet),
        )
        feasibility = target.sampler_diagnostics()["feasibility"]
        max_projection_error_norm = max(
            max_projection_error_norm,
            float(feasibility["projection_error_norm"]),
        )
    return {
        "n_audit_points": len(theta_values),
        "max_abs_logdet_error": max_abs_logdet_error,
        "max_projection_error_norm": max_projection_error_norm,
        "max_reduced_residual_norm": max_reduced_residual_norm,
        "determinant_route": "logabsdet_matrix(..., method='spd')",
        "target_measure": "graph",
    }


def run_benchmark(*, verbose: bool = True) -> dict[str, Any]:
    """Run the deterministic 1D elliptic PDE benchmark."""

    theta = np.linspace(THETA_MIN, THETA_MAX, GRID_SIZE)
    reduced, reduced_seconds = _route(theta, "reduced_space")
    corrected, corrected_seconds = _route(theta, "corrected_full_space")
    naive, naive_seconds = _route(theta, "naive_full_space")

    mean_abs_diff_corrected = abs(corrected.mean - reduced.mean)
    var_abs_diff_corrected = abs(corrected.variance - reduced.variance)
    mean_abs_diff_naive = abs(naive.mean - reduced.mean)
    var_abs_diff_naive = abs(naive.variance - reduced.variance)
    audit = _audit_pointwise_targets()

    summary = {
        "phase": "post_stage9_stabilization",
        "case": "one_dimensional_elliptic_reaction_diffusion_fd",
        "n_state": N_STATE,
        "rho": RHO,
        "grid_size": GRID_SIZE,
        "model": {
            "equation": "-exp(theta) u'' + MASS*u = 1",
            "nondimensional": True,
            "domain": [0.0, DOMAIN_LENGTH],
            "boundary_conditions": "homogeneous_dirichlet",
            "discretization": "centred_second_order_finite_difference",
            "grid_spacing": GRID_SPACING,
            "mass_coefficient": MASS,
            "forcing_value": FORCING_VALUE,
        },
        "synthetic_data": {
            "true_theta": SYNTHETIC_TRUE_THETA,
            "observation_location": 0.5,
            "noise_free_observation": NOISE_FREE_OBSERVATION,
            "observation_noise_sd": OBSERVATION_NOISE,
            "fixed_noise_offset_sd": SYNTHETIC_NOISE_OFFSET_SD,
            "observed_value": OBSERVED_VALUE,
        },
        "routes": {
            "reduced_space": reduced.to_dict(),
            "corrected_full_space": corrected.to_dict(),
            "naive_full_space": naive.to_dict(),
        },
        "correctness": {
            "corrected_matches_reduced_mean": mean_abs_diff_corrected < 5.0e-4,
            "corrected_matches_reduced_variance": var_abs_diff_corrected < 5.0e-4,
            "naive_differs_from_reduced_mean": mean_abs_diff_naive > 1.0e-1,
            "naive_differs_from_reduced_variance": var_abs_diff_naive > 1.0e-3,
            "pointwise_logdet_matches_spd_formula": audit["max_abs_logdet_error"] < 1.0e-10,
            "reduced_state_solves_constraint": audit["max_reduced_residual_norm"] < 1.0e-10,
            "all_correct": bool(
                mean_abs_diff_corrected < 5.0e-4
                and var_abs_diff_corrected < 5.0e-4
                and mean_abs_diff_naive > 1.0e-1
                and var_abs_diff_naive > 1.0e-3
                and audit["max_abs_logdet_error"] < 1.0e-10
                and audit["max_reduced_residual_norm"] < 1.0e-10
            ),
        },
        "posterior_differences": {
            "corrected_minus_reduced_mean_abs": mean_abs_diff_corrected,
            "corrected_minus_reduced_variance_abs": var_abs_diff_corrected,
            "naive_minus_reduced_mean_abs": mean_abs_diff_naive,
            "naive_minus_reduced_variance_abs": var_abs_diff_naive,
        },
        "audit": audit,
        "timing_notes": {
            "reduced_space_seconds": reduced_seconds,
            "corrected_full_space_seconds": corrected_seconds,
            "naive_full_space_seconds": naive_seconds,
            "note": (
                "CI route uses deterministic grid quadrature and small dense SPD "
                "Cholesky solves; sampler-facing pointwise checks use detcorr's exact "
                "state-Jacobian correction."
            ),
        },
        "memory_notes": {
            "state_matrix_shape": (N_STATE, N_STATE),
            "grid_points": GRID_SIZE,
            "note": (
                "The benchmark stores only a one-dimensional theta grid and small "
                "dense state matrices.  Large PDE deployments should use sparse LU "
                "or matrix-free diagnostics instead of dense grids."
            ),
        },
        "interpretation": (
            "For the declared centred finite-difference discretization, "
            "c(theta,u)=A(theta)u-f and the graph-posterior correction is the exact "
            "state-Jacobian term log|det A(theta)|. The independently integrated "
            "corrected full-space finite-penalty marginal matches the reduced-space "
            "posterior at high rho, while the naive full-space marginal retains the "
            "Jacobian-volume bias. This validates the determinant construction on a "
            "synthetic discretized PDE; it is not an empirical calibration result."
        ),
    }
    if verbose:
        print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


if __name__ == "__main__":  # pragma: no cover - manual example entry point
    run_benchmark(verbose=True)
