from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import quad

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
COMMON = ROOT.parent / "detcorr_demo_common" / "src"
for path in [str(SRC), str(COMMON)]:
    if path not in sys.path:
        sys.path.insert(0, path)

import kinetics_decay_time_stepping_demo as demo  # noqa: E402


def _skip_or_fail_if_detcorr_unavailable(audit: dict) -> None:
    if audit.get("detcorr_available"):
        return
    message = f"detcorr unavailable: {audit.get('reason')}"
    if os.environ.get("DETCORR_EXAMPLES_REQUIRE_DETCORR") == "1":
        pytest.fail(message)
    pytest.skip(message)


def test_manufactured_recurrence_and_offsets_are_exact() -> None:
    state = demo.solve_state(demo.TRUE_ETA)
    factor = 1.0 + demo.DT_SECONDS * demo.TRUE_RATE_PER_S
    closed_form = demo.INITIAL_CONCENTRATION_MM / factor ** np.arange(
        1,
        demo.N_STEPS + 1,
    )
    assert np.allclose(state, closed_form, rtol=0.0, atol=1.0e-15)
    assert np.linalg.norm(demo.residual(demo.TRUE_ETA, state)) < 1.0e-14

    table = demo.synthetic_observation_table()
    reconstructed = (
        table["truth_implicit_euler_mm"].to_numpy(float)
        + demo.OBSERVATION_SIGMA_MM * demo.STANDARDIZED_OFFSETS
    )
    assert np.array_equal(table["standardized_offset"], demo.STANDARDIZED_OFFSETS)
    assert np.allclose(table["observed_mm"], reconstructed, rtol=0.0, atol=1.0e-15)


@pytest.mark.parametrize("eta", [-1.4, demo.TRUE_ETA, 0.0, 0.6])
def test_triangular_determinant_matches_dense_slogdet(eta: float) -> None:
    matrix = demo.constraint_matrix(eta)
    assert np.allclose(np.diag(matrix, k=-1), -1.0)
    assert np.allclose(np.triu(matrix, k=1), 0.0)
    sign, dense_logdet = np.linalg.slogdet(matrix)
    assert sign == 1.0
    assert math.isclose(
        demo.exact_logdet(eta),
        float(dense_logdet),
        rel_tol=1.0e-14,
        abs_tol=1.0e-14,
    )


def test_custom_time_step_is_used_consistently_in_logdet() -> None:
    eta = -0.2
    dt_seconds = 0.125
    matrix = demo.constraint_matrix(eta, n_steps=5, dt_seconds=dt_seconds)
    _, dense_logdet = np.linalg.slogdet(matrix)
    assert demo.exact_logdet(
        eta,
        n_steps=5,
        dt_seconds=dt_seconds,
    ) == pytest.approx(dense_logdet, rel=0.0, abs=1.0e-12)


def test_exact_gaussian_integral_matches_scalar_quadrature() -> None:
    eta = -0.3
    observation = np.array([1.22])
    sigma = 0.11
    rho = 350.0
    analytic = demo.gaussian_tube_log_integral(
        eta,
        observation,
        sigma_mm=sigma,
        rho_per_mm2=rho,
    )
    factor = demo.time_step_factor(eta)
    initial = demo.INITIAL_CONCENTRATION_MM

    def integrand(state: float) -> float:
        observation_term = ((state - observation[0]) / sigma) ** 2
        residual_term = rho * (factor * state - initial) ** 2
        return math.exp(-0.5 * (observation_term + residual_term))

    numeric, error = quad(integrand, -np.inf, np.inf, epsabs=1.0e-13, epsrel=1.0e-12)
    assert error < 1.0e-10
    assert math.isclose(analytic, math.log(numeric), rel_tol=0.0, abs_tol=2.0e-12)


def test_gaussian_integral_is_stable_at_large_constraint_precision() -> None:
    eta = -0.3
    observation = np.array([1.22])
    sigma = 0.11
    rho = 1.0e12
    analytic = demo.gaussian_tube_log_integral(
        eta,
        observation,
        sigma_mm=sigma,
        rho_per_mm2=rho,
    )
    factor = demo.time_step_factor(eta)
    observation_precision = 1.0 / sigma**2
    constraint_precision = rho * factor**2
    combined = observation_precision + constraint_precision
    minimum_quadratic = (
        observation_precision
        * constraint_precision
        / combined
        * (observation[0] - demo.INITIAL_CONCENTRATION_MM / factor) ** 2
    )
    expected = 0.5 * math.log(2.0 * math.pi) - 0.5 * math.log(combined) - 0.5 * minimum_quadratic
    assert math.isclose(analytic, expected, rel_tol=0.0, abs_tol=2.0e-12)


def test_corrected_tube_recovers_graph_and_naive_is_negative_control() -> None:
    grid = np.linspace(demo.DEFAULT_ETA_MIN, demo.DEFAULT_ETA_MAX, 801)
    table, _, _ = demo.build_posterior_grid(grid)
    rows = {row["target"]: row for row in table.to_dict(orient="records")}
    graph = rows["graph_reduced"]
    naive = rows["naive_finite_gaussian_tube"]
    corrected = rows["detcorr_finite_gaussian_tube"]

    assert float(corrected["density_l1_vs_graph"]) < 5.0e-4
    assert abs(float(corrected["mean"]) - float(graph["mean"])) < 5.0e-6
    assert float(naive["density_l1_vs_graph"]) > 0.07
    shift_in_sd = (float(naive["mean"]) - float(graph["mean"])) / float(graph["sd"])
    assert shift_in_sd < -0.09


def test_detcorr_api_matches_exact_triangular_logdet_if_available() -> None:
    audit = demo.detcorr_api_audit(demo.TRUE_ETA)
    _skip_or_fail_if_detcorr_unavailable(audit)
    assert audit["result"]["method"] == "triangular"
    assert audit["absolute_error"] < 1.0e-12


def test_run_demo_writes_strict_outputs(tmp_path: Path) -> None:
    args = demo.build_arg_parser().parse_args(
        [
            "--outputs",
            str(tmp_path),
            "--n-grid",
            "501",
        ]
    )
    summary = demo.run_demo(args)
    assert summary["demo"] == "kinetics_decay_time_stepping"
    assert summary["correction_log_range_on_grid"] > 5.0
    assert summary["naive_mean_shift_in_graph_sd"] < -0.09
    assert summary["corrected_density_l1_vs_graph"] < 5.0e-4
    assert summary["naive_density_l1_vs_graph"] > 0.07

    expected = [
        "summary.json",
        "synthetic_decay_trajectory.png",
        "eta_posterior_comparison.png",
        "tables/synthetic_observations.csv",
        "tables/posterior_summary.csv",
        "tables/posterior_grid.csv",
        "tables/determinant_audit.csv",
    ]
    for relative in expected:
        assert (tmp_path / relative).is_file()
    assert not list(tmp_path.rglob("*.md"))

    serialized = (tmp_path / "summary.json").read_text(encoding="utf-8")
    assert "NaN" not in serialized
    assert "Infinity" not in serialized
    parsed = json.loads(
        serialized,
        parse_constant=lambda value: pytest.fail(f"non-finite JSON: {value}"),
    )
    assert parsed["synthetic_data_contract"]["random_number_generation"] == "none"
