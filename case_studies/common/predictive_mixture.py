"""Deterministic posterior-mixture quantiles in log-ratio coordinates.

The JARVIS predictive distributions are finite posterior mixtures whose
components are Normal or Student-t(4) distributions in a log ratio.  Drawing
one noise value per resampled posterior component does not compute a mixture
quantile and couples the answer to an arbitrary pairing.  This module instead
inverts the finite-mixture CDF directly.

Exact evaluation is the default and is never compressed implicitly.  A caller
with a very large grid may explicitly set ``component_limit``.  In that mode,
components are put in a canonical ``(location, scale, weight)`` order and the
discrete posterior measure is approximated by deterministic midpoint
systematic resampling.  Repeated selected components are merged.  The reported
canonical-mass CDF discrepancy describes the component measure along that
ordering; it is *not* a bound on predictive-CDF or quantile error.  Quantitative
quantile error must therefore be checked by increasing the limit or including
the exact result with :func:`audit_mixture_quantile_refinement`.

All locations, scales, and returned quantiles use the log-ratio coordinate.
Physical-unit quantiles should be formed afterwards with the exact monotone
map appropriate to the study, for example ``x * exp(log_ratio_quantile)``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr, ndtri, stdtr, stdtrit

MixtureFamily = Literal["normal", "student_t4"]


@dataclass(frozen=True)
class MixtureQuantileDiagnostics:
    """Numerical and component-measure diagnostics for one inversion."""

    family: MixtureFamily
    method: Literal["exact", "systematic_midpoint"]
    input_component_count: int
    positive_weight_component_count: int
    unique_component_count: int
    evaluated_component_count: int
    requested_component_limit: int | None
    systematic_mass_step: float | None
    canonical_mass_cdf_max_abs_error: float | None
    cdf_evaluation_count: int
    max_abs_cdf_root_residual: float
    max_initial_bracket_width: float
    root_xtol: float
    root_rtol: float


@dataclass(frozen=True)
class MixtureQuantileResult:
    """Requested probabilities, log-ratio quantiles, and inversion diagnostics."""

    probabilities: np.ndarray
    quantiles: np.ndarray
    diagnostics: MixtureQuantileDiagnostics


@dataclass(frozen=True)
class MixtureQuantileRefinement:
    """Quantitative comparison across explicit component limits.

    ``component_limits`` uses ``None`` for exact evaluation.  Changes compare
    each result with its predecessor.  If an exact result is present,
    ``max_abs_error_vs_exact`` contains one value per result; otherwise it is
    ``None``.
    """

    component_limits: tuple[int | None, ...]
    results: tuple[MixtureQuantileResult, ...]
    max_abs_change_from_previous: tuple[float, ...]
    exact_reference_index: int | None
    max_abs_error_vs_exact: tuple[float, ...] | None


def _readonly_copy(values: np.ndarray) -> np.ndarray:
    result = np.array(values, dtype=float, copy=True)
    result.setflags(write=False)
    return result


def _validate_probabilities(probabilities: np.ndarray) -> np.ndarray:
    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("probabilities must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(values)) or np.any((values <= 0.0) | (values >= 1.0)):
        raise ValueError("probabilities must be finite and strictly inside (0, 1)")
    return values


def _validate_components(
    locations: np.ndarray,
    scales: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    location = np.asarray(locations, dtype=float)
    scale = np.asarray(scales, dtype=float)
    weight = np.asarray(weights, dtype=float)
    if location.ndim != 1 or location.size == 0:
        raise ValueError("locations must be a non-empty one-dimensional array")
    if scale.shape != location.shape or weight.shape != location.shape:
        raise ValueError("locations, scales, and weights must have matching shapes")
    if not np.all(np.isfinite(location)):
        raise ValueError("locations must be finite")
    if not np.all(np.isfinite(scale)) or np.any(scale <= 0.0):
        raise ValueError("scales must be finite and strictly positive")
    if not np.all(np.isfinite(weight)) or np.any(weight < 0.0):
        raise ValueError("weights must be finite and nonnegative")

    input_count = int(location.size)
    positive = weight > 0.0
    positive_count = int(np.count_nonzero(positive))
    if positive_count == 0:
        raise ValueError("weights must have positive total")
    location = location[positive]
    scale = scale[positive]
    weight = weight[positive]

    # Scaling before summation avoids overflow for finite but very large input
    # weights.  Underflowed terms are below float64's representable mass.
    weight = weight / float(np.max(weight))
    total = float(np.sum(weight, dtype=np.float64))
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("weights cannot be normalized")
    weight = weight / total
    represented = weight > 0.0
    location = location[represented]
    scale = scale[represented]
    weight = weight[represented]

    # Canonical ordering makes exact summation and optional compression
    # independent of the caller's component order.  Exact duplicate
    # (location, scale) pairs can be merged without approximation.
    order = np.lexsort((weight, scale, location))
    location = location[order]
    scale = scale[order]
    weight = weight[order]
    starts = np.r_[
        True,
        (location[1:] != location[:-1]) | (scale[1:] != scale[:-1]),
    ]
    start_indices = np.flatnonzero(starts)
    location = location[start_indices]
    scale = scale[start_indices]
    weight = np.add.reduceat(weight, start_indices)
    weight = weight / float(np.sum(weight, dtype=np.float64))
    return location, scale, weight, input_count, positive_count


def _compress_components(
    locations: np.ndarray,
    scales: np.ndarray,
    weights: np.ndarray,
    component_limit: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, float | None, float | None]:
    if component_limit is not None:
        if isinstance(component_limit, bool) or not isinstance(component_limit, (int, np.integer)):
            raise TypeError("component_limit must be a positive integer or None")
        if component_limit <= 0:
            raise ValueError("component_limit must be a positive integer or None")
    if component_limit is None or locations.size <= component_limit:
        return locations, scales, weights, "exact", None, None

    count = int(component_limit)
    original_cdf = np.cumsum(weights, dtype=np.float64)
    original_cdf[-1] = 1.0
    points = (np.arange(count, dtype=float) + 0.5) / count
    selected = np.searchsorted(original_cdf, points, side="left")
    indices, multiplicities = np.unique(selected, return_counts=True)
    compressed_weight = multiplicities.astype(float) / count

    # This is the actual discrepancy between the original and resampled
    # component measures along the declared canonical order.  It diagnoses
    # mass resolution only; component CDFs need not be monotone in this order.
    compressed_cdf = np.zeros_like(original_cdf)
    compressed_cdf[indices] = compressed_weight
    compressed_cdf = np.cumsum(compressed_cdf, dtype=np.float64)
    compressed_cdf[-1] = 1.0
    mass_error = float(np.max(np.abs(original_cdf - compressed_cdf)))
    return (
        locations[indices],
        scales[indices],
        compressed_weight,
        "systematic_midpoint",
        1.0 / count,
        mass_error,
    )


def _distribution_functions(family: MixtureFamily):
    if family == "normal":
        return ndtr, ndtri
    if family == "student_t4":
        return lambda value: stdtr(4.0, value), lambda probability: stdtrit(4.0, probability)
    raise ValueError("family must be 'normal' or 'student_t4'")


def posterior_mixture_logratio_quantiles(
    locations: np.ndarray,
    scales: np.ndarray,
    weights: np.ndarray,
    probabilities: np.ndarray,
    *,
    family: MixtureFamily,
    component_limit: int | None = None,
    xtol: float = 1.0e-12,
    rtol: float = 8.0 * np.finfo(float).eps,
) -> MixtureQuantileResult:
    """Invert a finite posterior-mixture CDF in the log-ratio coordinate.

    Parameters
    ----------
    locations, scales, weights:
        One-dimensional posterior component arrays.  Weights may be
        unnormalised and zero, but must be finite and nonnegative.
    probabilities:
        Quantile probabilities strictly inside ``(0, 1)``.  Input order and
        duplicates are preserved in the result.
    family:
        ``"normal"`` or ``"student_t4"``.
    component_limit:
        ``None`` (the default) evaluates the exact finite mixture.  A positive
        integer explicitly enables deterministic component compression only
        when the number of unique positive-mass components exceeds the limit.
    xtol, rtol:
        Absolute and relative tolerances passed to Brent's root solver.

    Returns
    -------
    MixtureQuantileResult
        Quantiles are in the same log-ratio coordinate as ``locations``.
    """

    probability = _validate_probabilities(probabilities)
    if family not in ("normal", "student_t4"):
        raise ValueError("family must be 'normal' or 'student_t4'")
    if not math.isfinite(xtol) or xtol <= 0.0:
        raise ValueError("xtol must be finite and positive")
    minimum_rtol = 4.0 * np.finfo(float).eps
    if not math.isfinite(rtol) or rtol < minimum_rtol:
        raise ValueError(f"rtol must be finite and at least {minimum_rtol}")

    location, scale, weight, input_count, positive_count = _validate_components(
        locations, scales, weights
    )
    unique_count = int(location.size)
    (
        evaluated_location,
        evaluated_scale,
        evaluated_weight,
        method,
        mass_step,
        mass_error,
    ) = _compress_components(location, scale, weight, component_limit)
    cdf, ppf = _distribution_functions(family)

    cdf_evaluations = 0
    max_residual = 0.0
    max_bracket_width = 0.0
    solved: dict[float, float] = {}
    for requested in np.unique(probability):
        # Using component quantiles at probabilities strictly below/above the
        # target gives a bracket with a numerical sign margin: at the lower
        # endpoint every component CDF is <= p/2, and at the upper endpoint
        # every component CDF is >= (1+p)/2.
        lower_base = float(ppf(0.5 * requested))
        upper_base = float(ppf(0.5 * (1.0 + requested)))
        lower = float(np.min(evaluated_location + evaluated_scale * lower_base))
        upper = float(np.max(evaluated_location + evaluated_scale * upper_base))
        if not math.isfinite(lower) or not math.isfinite(upper) or not lower < upper:
            raise ValueError("could not construct a finite non-degenerate quantile bracket")
        max_bracket_width = max(max_bracket_width, upper - lower)

        def objective(value: float, target: float = float(requested)) -> float:
            nonlocal cdf_evaluations
            cdf_evaluations += 1
            standardized = (value - evaluated_location) / evaluated_scale
            mixture_cdf = float(np.sum(evaluated_weight * cdf(standardized), dtype=np.float64))
            return mixture_cdf - target

        root = float(brentq(objective, lower, upper, xtol=xtol, rtol=rtol))
        residual = abs(objective(root))
        max_residual = max(max_residual, residual)
        solved[float(requested)] = root

    quantiles = np.array([solved[float(value)] for value in probability], dtype=float)
    diagnostics = MixtureQuantileDiagnostics(
        family=family,
        method=method,
        input_component_count=input_count,
        positive_weight_component_count=positive_count,
        unique_component_count=unique_count,
        evaluated_component_count=int(evaluated_location.size),
        requested_component_limit=None if component_limit is None else int(component_limit),
        systematic_mass_step=mass_step,
        canonical_mass_cdf_max_abs_error=mass_error,
        cdf_evaluation_count=cdf_evaluations,
        max_abs_cdf_root_residual=max_residual,
        max_initial_bracket_width=max_bracket_width,
        root_xtol=float(xtol),
        root_rtol=float(rtol),
    )
    return MixtureQuantileResult(
        probabilities=_readonly_copy(probability),
        quantiles=_readonly_copy(quantiles),
        diagnostics=diagnostics,
    )


def audit_mixture_quantile_refinement(
    locations: np.ndarray,
    scales: np.ndarray,
    weights: np.ndarray,
    probabilities: np.ndarray,
    *,
    family: MixtureFamily,
    component_limits: Sequence[int | None],
    xtol: float = 1.0e-12,
    rtol: float = 8.0 * np.finfo(float).eps,
) -> MixtureQuantileRefinement:
    """Run explicit component-limit refinements and compare log-ratio quantiles.

    Include ``None`` in ``component_limits`` to quantify every approximation
    against exact finite-mixture inversion.  Limits are kept in caller order;
    duplicate entries are rejected because they do not constitute refinement.
    """

    limits = tuple(component_limits)
    if len(limits) < 2:
        raise ValueError("component_limits must contain at least two refinement levels")
    canonical_limits: list[int | None] = []
    for limit in limits:
        if limit is None:
            canonical_limits.append(None)
        elif isinstance(limit, bool) or not isinstance(limit, (int, np.integer)):
            raise TypeError("component limits must be positive integers or None")
        elif limit <= 0:
            raise ValueError("component limits must be positive integers or None")
        else:
            canonical_limits.append(int(limit))
    if len(set(canonical_limits)) != len(canonical_limits):
        raise ValueError("component_limits must not contain duplicates")

    results = tuple(
        posterior_mixture_logratio_quantiles(
            locations,
            scales,
            weights,
            probabilities,
            family=family,
            component_limit=limit,
            xtol=xtol,
            rtol=rtol,
        )
        for limit in canonical_limits
    )
    changes = tuple(
        float(np.max(np.abs(current.quantiles - previous.quantiles)))
        for previous, current in zip(results[:-1], results[1:], strict=True)
    )
    exact_indices = [index for index, limit in enumerate(canonical_limits) if limit is None]
    exact_index = exact_indices[0] if exact_indices else None
    errors = None
    if exact_index is not None:
        exact = results[exact_index].quantiles
        errors = tuple(float(np.max(np.abs(result.quantiles - exact))) for result in results)
    return MixtureQuantileRefinement(
        component_limits=tuple(canonical_limits),
        results=results,
        max_abs_change_from_previous=changes,
        exact_reference_index=exact_index,
        max_abs_error_vs_exact=errors,
    )


__all__ = [
    "MixtureFamily",
    "MixtureQuantileDiagnostics",
    "MixtureQuantileRefinement",
    "MixtureQuantileResult",
    "audit_mixture_quantile_refinement",
    "posterior_mixture_logratio_quantiles",
]
