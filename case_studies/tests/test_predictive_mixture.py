from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import brentq
from scipy.stats import norm
from scipy.stats import t as student_t

from common.predictive_mixture import (
    audit_mixture_quantile_refinement,
    posterior_mixture_logratio_quantiles,
)


def _reference_quantiles(
    locations: np.ndarray,
    scales: np.ndarray,
    weights: np.ndarray,
    probabilities: np.ndarray,
    family: str,
) -> np.ndarray:
    normalized = weights / weights.sum()
    distribution = norm if family == "normal" else student_t(df=4)

    def solve(probability: float) -> float:
        def objective(value: float) -> float:
            return float(
                np.sum(normalized * distribution.cdf((value - locations) / scales)) - probability
            )

        return float(brentq(objective, -20.0, 20.0, xtol=1.0e-13))

    return np.array([solve(float(probability)) for probability in probabilities])


@pytest.mark.parametrize("family", ["normal", "student_t4"])
def test_exact_mixture_matches_independent_full_cdf_inversion(family: str) -> None:
    locations = np.array([-1.3, -0.1, 0.4, 1.8])
    scales = np.array([0.25, 0.8, 0.35, 1.1])
    weights = np.array([1.0, 5.0, 2.0, 0.5])
    probabilities = np.array([0.95, 0.05, 0.5, 0.25, 0.5])

    result = posterior_mixture_logratio_quantiles(
        locations,
        scales,
        weights,
        probabilities,
        family=family,
    )
    reference = _reference_quantiles(locations, scales, weights, probabilities, family)

    assert np.allclose(result.quantiles, reference, rtol=0.0, atol=2.0e-12)
    assert result.diagnostics.method == "exact"
    assert result.diagnostics.evaluated_component_count == 4
    assert result.diagnostics.requested_component_limit is None
    assert result.diagnostics.max_abs_cdf_root_residual < 1.0e-11
    assert np.array_equal(result.probabilities, probabilities)
    assert not result.quantiles.flags.writeable


@pytest.mark.parametrize("family", ["normal", "student_t4"])
def test_single_component_reduces_to_declared_distribution(family: str) -> None:
    probabilities = np.array([0.01, 0.25, 0.5, 0.9, 0.99])
    location = 0.37
    scale = 0.61
    expected_base = norm.ppf(probabilities)
    if family == "student_t4":
        expected_base = student_t.ppf(probabilities, df=4)
    expected = location + scale * expected_base

    result = posterior_mixture_logratio_quantiles(
        np.array([location]),
        np.array([scale]),
        np.array([9.0]),
        probabilities,
        family=family,
    )

    assert np.allclose(result.quantiles, expected, rtol=0.0, atol=3.0e-12)


@pytest.mark.parametrize("family", ["normal", "student_t4"])
def test_explicit_compression_is_deterministic_and_refines_toward_exact(family: str) -> None:
    coordinate = np.linspace(-1.0, 1.0, 2001)
    locations = 1.1 * coordinate + 0.12 * np.sin(5.0 * coordinate)
    scales = 0.16 + 0.11 * (coordinate + 1.0) + 0.04 * np.cos(3.0 * coordinate) ** 2
    weights = np.exp(-1.7 * (coordinate - 0.15) ** 2) * (1.0 + 0.18 * np.cos(7.0 * coordinate))
    probabilities = np.array([0.05, 0.25, 0.5, 0.75, 0.95])

    audit = audit_mixture_quantile_refinement(
        locations,
        scales,
        weights,
        probabilities,
        family=family,
        component_limits=(32, 128, 512, None),
    )
    assert audit.exact_reference_index == 3
    assert audit.max_abs_error_vs_exact is not None
    coarse_error, medium_error, fine_error, exact_error = audit.max_abs_error_vs_exact
    assert coarse_error > medium_error > fine_error
    assert fine_error < 3.0e-5
    assert exact_error == 0.0
    assert audit.results[0].diagnostics.method == "systematic_midpoint"
    assert audit.results[-1].diagnostics.method == "exact"
    assert audit.results[0].diagnostics.canonical_mass_cdf_max_abs_error is not None
    assert (
        audit.results[0].diagnostics.canonical_mass_cdf_max_abs_error
        <= audit.results[0].diagnostics.systematic_mass_step
    )

    repeated = posterior_mixture_logratio_quantiles(
        locations,
        scales,
        weights,
        probabilities,
        family=family,
        component_limit=128,
    )
    permutation = np.random.default_rng(20260711).permutation(locations.size)
    reordered = posterior_mixture_logratio_quantiles(
        locations[permutation],
        scales[permutation],
        weights[permutation],
        probabilities,
        family=family,
        component_limit=128,
    )
    assert np.array_equal(repeated.quantiles, audit.results[1].quantiles)
    assert np.array_equal(reordered.quantiles, repeated.quantiles)
    assert reordered.diagnostics == repeated.diagnostics


def test_duplicate_components_and_zero_weights_are_handled_exactly() -> None:
    result = posterior_mixture_logratio_quantiles(
        np.array([0.2, 0.2, 10.0]),
        np.array([0.5, 0.5, 1.0]),
        np.array([2.0, 3.0, 0.0]),
        np.array([0.5]),
        family="normal",
        component_limit=1,
    )

    assert result.quantiles[0] == pytest.approx(0.2, abs=1.0e-13)
    assert result.diagnostics.input_component_count == 3
    assert result.diagnostics.positive_weight_component_count == 2
    assert result.diagnostics.unique_component_count == 1
    assert result.diagnostics.method == "exact"


@pytest.mark.parametrize(
    ("locations", "scales", "weights", "probabilities", "message"),
    [
        ([0.0], [0.0], [1.0], [0.5], "scales"),
        ([0.0], [1.0], [-1.0], [0.5], "weights"),
        ([0.0], [1.0], [0.0], [0.5], "positive total"),
        ([0.0], [1.0], [1.0], [0.0], "strictly inside"),
        ([0.0], [1.0, 2.0], [1.0], [0.5], "matching shapes"),
    ],
)
def test_invalid_mixture_inputs_are_rejected(
    locations: list[float],
    scales: list[float],
    weights: list[float],
    probabilities: list[float],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        posterior_mixture_logratio_quantiles(
            np.array(locations),
            np.array(scales),
            np.array(weights),
            np.array(probabilities),
            family="normal",
        )


def test_compression_and_refinement_options_are_explicitly_validated() -> None:
    values = np.array([0.0, 1.0])
    scales = np.ones(2)
    weights = np.ones(2)
    probabilities = np.array([0.5])
    with pytest.raises(ValueError, match="positive integer"):
        posterior_mixture_logratio_quantiles(
            values,
            scales,
            weights,
            probabilities,
            family="normal",
            component_limit=0,
        )
    with pytest.raises(ValueError, match="at least two"):
        audit_mixture_quantile_refinement(
            values,
            scales,
            weights,
            probabilities,
            family="normal",
            component_limits=(16,),
        )
    with pytest.raises(ValueError, match="duplicates"):
        audit_mixture_quantile_refinement(
            values,
            scales,
            weights,
            probabilities,
            family="normal",
            component_limits=(16, 16),
        )
