from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
COMMON = ROOT.parent / "detcorr_demo_common" / "src"
for path in (SRC, COMMON):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import bandgap_matbench_detcorr_demo as demo  # noqa: E402

import detcorr_demo_common as common  # noqa: E402


@pytest.fixture(scope="module")
def fitted_problem():
    frame = demo.load_bandgap_data()
    train, heldout = demo.formula_hash_split(frame)
    fit, train_prediction, heldout_prediction = demo.fit_base_gap_predictor(train, heldout)
    problem = demo.make_dimensionless_problem(
        heldout["band_gap_ev"].to_numpy(dtype=float),
        heldout_prediction,
        fit.train_rmse_ev,
        reference_energy=1.0,
        energy_unit="eV",
    )
    return frame, train, heldout, fit, train_prediction, heldout_prediction, problem


def test_fixture_checksum_and_cli_defaults_are_pinned() -> None:
    assert common.sha256_file(demo.DATA_PATH) == demo.FIXTURE_SHA256
    args = demo.build_arg_parser().parse_args([])
    assert args.data == demo.DATA_PATH
    assert args.outputs == ROOT / "outputs"
    assert args.n_quad == demo.QUADRATURE_POINTS


def test_formula_parser_and_hash_split_are_deterministic_and_disjoint(
    fitted_problem,
) -> None:
    frame, train, heldout, *_ = fitted_problem
    counts = demo.parse_formula("Ag0.5Ge1Pb1.75S4")
    assert counts == {"Ag": 0.5, "Ge": 1.0, "Pb": 1.75, "S": 4.0}
    assert len(train) == 70 and len(heldout) == 30
    assert set(train["formula_sha256"]).isdisjoint(heldout["formula_sha256"])

    shuffled = frame.sample(frac=1.0, random_state=13).reset_index(drop=True)
    shuffled_train, shuffled_heldout = demo.formula_hash_split(shuffled)
    assert set(train["formula"]) == set(shuffled_train["formula"])
    assert set(heldout["formula"]) == set(shuffled_heldout["formula"])


def test_predictor_inverse_is_correct_and_heldout_gaps_do_not_enter_fit(
    fitted_problem,
) -> None:
    _, train, heldout, fit, _, heldout_prediction, _ = fitted_problem
    transformed = np.log(np.array([1.05, 0.07, 9.0]))
    inverted = demo.inverse_log_gap_prediction(transformed)
    np.testing.assert_allclose(inverted, [1.0, 0.05, 7.5])

    altered = heldout.copy()
    altered["band_gap_ev"] *= 2.0
    altered_fit, _, altered_prediction = demo.fit_base_gap_predictor(train, altered)
    assert altered_fit.coefficients == fit.coefficients
    assert altered_fit.train_rmse_ev == fit.train_rmse_ev
    np.testing.assert_allclose(altered_prediction, heldout_prediction, rtol=0.0, atol=0.0)


def test_dimensionless_constraint_derivative_and_ev_mev_invariance(
    fitted_problem,
) -> None:
    _, _, heldout, fit, _, heldout_prediction, problem = fitted_problem
    alpha = 0.79
    state = demo.solved_state(alpha, problem.predicted_m)
    np.testing.assert_allclose(
        demo.constraint_residual(alpha, state, problem.predicted_m),
        0.0,
        atol=3.0e-15,
    )
    direction = np.cos(np.linspace(0.1, 2.0, problem.n_state))
    epsilon = 1.0e-6
    plus = demo.constraint_residual(alpha, state + epsilon * direction, problem.predicted_m)
    minus = demo.constraint_residual(alpha, state - epsilon * direction, problem.predicted_m)
    finite_difference = (plus - minus) / (2.0 * epsilon)
    expected = demo.state_jacobian_diagonal(alpha, state) * direction
    np.testing.assert_allclose(finite_difference, expected, rtol=2.0e-9, atol=2.0e-10)

    grid = np.linspace(0.5, 1.6, 31)
    audit = demo.unit_invariance_audit(
        heldout["band_gap_ev"].to_numpy(dtype=float),
        heldout_prediction,
        fit.train_rmse_ev,
        grid,
    )
    assert audit["passed"] is True
    assert abs(audit["graph_centered_logdensity_max_abs_diff"]) <= 1.0e-12
    assert abs(audit["state_logdet_max_abs_diff"]) <= 1.0e-12
    assert audit["agreement_standard"].startswith("machine precision")


def test_local_quadrature_window_is_stable_and_finite_rho_limited(
    fitted_problem,
) -> None:
    *_, problem = fitted_problem
    alpha = np.linspace(demo.ALPHA_MIN, demo.ALPHA_MAX, 13)
    graph = demo.graph_logdensity(alpha, problem)
    default_window = demo.finite_integrated_tube_logdensity(
        alpha,
        problem,
        n_quad=801,
        window_sigma=demo.QUADRATURE_WINDOW_SIGMA,
        corrected=True,
    )
    wider_window = demo.finite_integrated_tube_logdensity(
        alpha,
        problem,
        n_quad=801,
        window_sigma=demo.WINDOW_SENSITIVITY_SIGMA,
        corrected=True,
    )
    higher_rho = demo.finite_integrated_tube_logdensity(
        alpha,
        problem,
        rho=2.0 * demo.PENALTY_RHO,
        n_quad=801,
        window_sigma=demo.QUADRATURE_WINDOW_SIGMA,
        corrected=True,
    )
    default_error = common.centered_max_abs_diff(default_window, graph)
    higher_rho_error = common.centered_max_abs_diff(higher_rho, graph)
    assert common.centered_max_abs_diff(default_window, wider_window) < 1.0e-9
    assert higher_rho_error < 0.51 * default_error


def test_corrected_finite_quadrature_uses_pointwise_state_jacobian(fitted_problem) -> None:
    *_, problem = fitted_problem
    alpha_values = np.array([0.67, 1.31])
    observed = demo.finite_integrated_tube_logdensity(
        alpha_values,
        problem,
        n_quad=501,
        corrected=True,
    )
    independently_assembled = []
    trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    coordinate = np.linspace(0.0, 1.0, 501)
    for alpha in alpha_values:
        state = demo.solved_state(float(alpha), problem.predicted_m)
        diagonal = demo.state_jacobian_diagonal(float(alpha), state)
        width = demo.QUADRATURE_WINDOW_SIGMA / (np.sqrt(demo.PENALTY_RHO) * np.abs(diagonal))
        lower = np.maximum(1.0e-12, state - width)
        upper = state + width
        state_grid = lower[:, None] + (upper - lower)[:, None] * coordinate[None, :]
        log_kernel = -0.5 * ((problem.observed_e[:, None] - state_grid) / problem.sigma_e) ** 2
        log_kernel -= (
            0.5
            * demo.PENALTY_RHO
            * (state_grid ** float(alpha) - problem.predicted_m[:, None]) ** 2
        )
        log_kernel += np.log(float(alpha)) + (float(alpha) - 1.0) * np.log(state_grid)
        row_max = np.max(log_kernel, axis=1)
        integrals = trapezoid(np.exp(log_kernel - row_max[:, None]), state_grid, axis=1)
        independently_assembled.append(
            float(demo.log_alpha_prior(float(alpha))) + float(np.sum(row_max + np.log(integrals)))
        )
    np.testing.assert_allclose(observed, independently_assembled, rtol=0.0, atol=2.0e-12)

    naive = demo.finite_integrated_tube_logdensity(
        alpha_values,
        problem,
        n_quad=501,
        corrected=False,
    )
    projected_graph_jacobian = naive + demo.state_logdet(alpha_values, problem)
    assert common.centered_max_abs_diff(observed, projected_graph_jacobian) > 1.0e-6


def test_direct_quadrature_uses_all_states_and_approaches_hard_targets(
    fitted_problem,
) -> None:
    *_, problem = fitted_problem
    alpha = np.linspace(0.50, 1.70, 9)
    finite_naive = demo.finite_integrated_tube_logdensity(
        alpha, problem, n_quad=801, corrected=False
    )
    finite_corrected = demo.finite_integrated_tube_logdensity(
        alpha, problem, n_quad=801, corrected=True
    )
    hard_naive = demo.hard_tube_logdensity(alpha, problem, corrected=False)
    graph = demo.graph_logdensity(alpha, problem)
    assert problem.n_state == 30
    assert common.centered_max_abs_diff(finite_naive, hard_naive) < 2.0e-2
    assert common.centered_max_abs_diff(finite_corrected, graph) < 2.0e-2


def test_direct_finite_posterior_recovers_graph_and_naive_differs(
    fitted_problem,
) -> None:
    *_, problem = fitted_problem
    alpha = np.linspace(demo.ALPHA_MIN, demo.ALPHA_MAX, 801)
    graph = demo.graph_logdensity(alpha, problem)
    naive = demo.finite_integrated_tube_logdensity(alpha, problem, n_quad=301, corrected=False)
    corrected = demo.finite_integrated_tube_logdensity(alpha, problem, n_quad=301, corrected=True)
    table, densities, _ = common.summarize_targets(
        alpha,
        {"graph_reduced": graph, "finite_naive": naive, "finite_corrected": corrected},
    )
    rows = table.set_index("target")
    graph_mean = float(rows.loc["graph_reduced", "mean"])
    graph_sd = float(rows.loc["graph_reduced", "sd"])
    naive_mean = float(rows.loc["finite_naive", "mean"])
    corrected_l1 = common.density_l1(
        alpha, densities["finite_corrected"], densities["graph_reduced"]
    )
    naive_l1 = common.density_l1(alpha, densities["finite_naive"], densities["graph_reduced"])
    assert (naive_mean - graph_mean) / graph_sd < -1.8
    assert naive_l1 > 1.0
    assert corrected_l1 < 2.0e-6


def test_diagonal_and_callable_detcorr_routes_match(fitted_problem) -> None:
    *_, problem = fitted_problem
    audit = demo.detcorr_audit(0.79, problem)
    if not audit.get("detcorr_available"):
        message = f"detcorr unavailable: {audit.get('reason')}"
        if os.environ.get("DETCORR_EXAMPLES_REQUIRE_DETCORR") == "1":
            pytest.fail(message)
        pytest.skip(message)
    assert audit["target_measure"] == "graph"
    assert audit["state_dimension"] == 30
    assert audit["diagonal_abs_error"] < 1.0e-10
    assert audit["callable_abs_error"] < 2.0e-5


def test_grid_tails_are_negligible(fitted_problem) -> None:
    *_, problem = fitted_problem
    alpha = np.linspace(demo.ALPHA_MIN, demo.ALPHA_MAX, 601)
    targets = {
        "graph_reduced": demo.graph_logdensity(alpha, problem),
        "finite_naive": demo.finite_integrated_tube_logdensity(
            alpha, problem, n_quad=201, corrected=False
        ),
        "finite_corrected": demo.finite_integrated_tube_logdensity(
            alpha, problem, n_quad=201, corrected=True
        ),
    }
    _, densities, _ = common.summarize_targets(alpha, targets)
    checks = demo.grid_tail_checks(alpha, targets, densities)
    assert checks["passed"] is True
    assert checks["minimum_endpoint_log_drop"] > 25.0
    assert checks["maximum_two_percent_grid_tail_mass"] < 1.0e-10


def test_run_demo_writes_only_current_strict_outputs(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    summary = demo.run_demo(outputs, n_grid=501, n_quad=201)
    assert summary["checks"]["all_passed"] is True
    assert summary["classification"] == "external_data_nonlinear_coordinate_stress_test"
    assert summary["split"]["training_and_heldout_hashes_disjoint"] is True
    assert summary["model"]["prior"]["density_measure"] == "d alpha"
    assert summary["model"]["prior"]["log_scale"] == pytest.approx(0.42)
    assert summary["model"]["full_space_base_measure"].startswith(
        "d alpha times the product of de_i"
    )
    assert summary["dataset"]["license"].startswith("MIT")
    assert summary["dataset"]["matbench_dataset_doi"] == demo.MATBENCH_DATA_DOI
    assert summary["outcome"]["naive_mean_shift_in_graph_sd"] < -1.8
    assert summary["outcome"]["corrected_density_l1_vs_graph"] < 2.0e-6
    assert len(list(outputs.glob("*.png"))) == 3
    assert len(list((outputs / "tables").glob("*.csv"))) == 4

    text = (outputs / "summary.json").read_text(encoding="utf-8")
    parsed = json.loads(text, parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
    assert parsed["checks"]["all_passed"] is True
    assert "NaN" not in text and "Infinity" not in text


def test_formula_hash_split_table_has_unique_rows(fitted_problem) -> None:
    frame, train, heldout, *_ = fitted_problem
    combined = pd.concat([train, heldout], ignore_index=True)
    assert len(combined) == len(frame)
    assert combined["formula"].is_unique
    assert combined["formula_sha256"].is_unique
