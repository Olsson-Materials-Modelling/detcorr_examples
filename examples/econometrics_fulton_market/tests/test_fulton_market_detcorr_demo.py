from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import quad

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
COMMON = ROOT.parent / "detcorr_demo_common" / "src"
for path in (SRC, COMMON):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import fulton_market_detcorr_demo as demo  # noqa: E402

import detcorr_demo_common as common  # noqa: E402


@pytest.fixture(scope="module")
def frame():
    return demo.load_fish(ROOT / "data" / "fish.csv")


@pytest.fixture(scope="module")
def problem(frame):
    return demo.prepare_heldout_problem(frame)


def test_data_checksum_and_cli_defaults_are_pinned() -> None:
    assert common.sha256_file(ROOT / "data" / "fish.csv") == demo.DATA_SHA256
    args = demo.build_arg_parser().parse_args([])
    assert args.data == ROOT / "data" / "fish.csv"
    assert args.outputs == ROOT / "outputs"
    assert args.train_size == 64


def test_chronological_split_is_disjoint_and_heldout_is_not_fit_input(frame) -> None:
    original = demo.prepare_heldout_problem(frame)
    changed = frame.copy()
    changed.loc[64:, "avgprc"] *= 1.7
    perturbed = demo.prepare_heldout_problem(changed)

    assert list(original.train["t"]) == list(range(1, 65))
    assert list(original.heldout["t"]) == list(range(65, 98))
    assert set(original.train["rownames"]).isdisjoint(original.heldout["rownames"])
    np.testing.assert_allclose(
        original.fit.coefficients, perturbed.fit.coefficients, rtol=0.0, atol=0.0
    )
    assert original.fit.residual_sd_native_price == perturbed.fit.residual_sd_native_price
    assert original.fit.training_price_scale == perturbed.fit.training_price_scale
    np.testing.assert_allclose(
        original.heldout_reference_native,
        perturbed.heldout_reference_native,
        rtol=0.0,
        atol=0.0,
    )
    assert not np.allclose(original.observed_nondimensional, perturbed.observed_nondimensional)


def test_constraint_units_solution_and_state_jacobian(problem) -> None:
    lambda_scale = 0.81
    state = demo.solved_state(lambda_scale, problem.reference_nondimensional)
    residual = demo.constraint_residual(lambda_scale, state, problem.reference_nondimensional)
    np.testing.assert_allclose(residual, 0.0, atol=2.0e-16)

    direction = np.cos(np.linspace(0.0, 2.0, problem.n_state))
    epsilon = 1.0e-6
    plus = demo.constraint_residual(
        lambda_scale,
        state + epsilon * direction,
        problem.reference_nondimensional,
    )
    minus = demo.constraint_residual(
        lambda_scale,
        state - epsilon * direction,
        problem.reference_nondimensional,
    )
    finite_difference = (plus - minus) / (2.0 * epsilon)
    np.testing.assert_allclose(
        finite_difference, -lambda_scale * direction, rtol=1.0e-9, atol=5.0e-11
    )
    assert np.isclose(
        demo.state_logdet(np.array([lambda_scale]), problem.n_state)[0],
        problem.n_state * np.log(lambda_scale),
    )


def test_scalar_gaussian_integral_formula_matches_direct_quadrature(problem) -> None:
    lambda_scale = 0.83
    rho = 2.5e4
    y = float(problem.observed_nondimensional[4])
    reference = float(problem.reference_nondimensional[4])
    sigma = problem.sigma_nondimensional

    def integrand(state: float) -> float:
        likelihood = np.exp(-0.5 * ((y - state) / sigma) ** 2) / (np.sqrt(2.0 * np.pi) * sigma)
        tube = np.exp(-0.5 * rho * (reference - lambda_scale * state) ** 2)
        return float(likelihood * tube)

    numeric, error = quad(integrand, -np.inf, np.inf, epsabs=1.0e-13, epsrel=1.0e-13)
    variance = sigma**2 + 1.0 / (rho * lambda_scale**2)
    analytic_log = -0.5 * np.log1p(rho * lambda_scale**2 * sigma**2)
    analytic_log -= 0.5 * (y - reference / lambda_scale) ** 2 / variance
    assert error < 1.0e-11
    assert np.isclose(np.log(numeric), analytic_log, rtol=0.0, atol=2.0e-11)


def test_exact_full_state_integral_recovers_graph_and_exposes_naive_bias(problem) -> None:
    grid = np.linspace(demo.GRID_MIN, demo.GRID_MAX, 2401)
    graph = demo.graph_logdensity(grid, problem)
    naive = demo.exact_integrated_tube_logdensity(grid, problem, corrected=False)
    corrected = demo.exact_integrated_tube_logdensity(grid, problem, corrected=True)
    table, densities, _ = common.summarize_targets(
        grid,
        {"graph_reduced": graph, "finite_naive": naive, "finite_corrected": corrected},
    )
    rows = table.set_index("target")
    graph_sd = float(rows.loc["graph_reduced", "sd"])
    naive_shift = (
        float(rows.loc["finite_naive", "mean"]) - float(rows.loc["graph_reduced", "mean"])
    ) / graph_sd
    corrected_l1 = common.density_l1(
        grid, densities["finite_corrected"], densities["graph_reduced"]
    )
    naive_l1 = common.density_l1(grid, densities["finite_naive"], densities["graph_reduced"])
    assert naive_shift < -1.5
    assert naive_l1 > 1.0
    assert corrected_l1 < 1.0e-6


def test_callable_detcorr_audit_matches_diagonal_jacobian(problem) -> None:
    audit = demo.detcorr_callable_audit(0.79, problem)
    if not audit.get("detcorr_available"):
        message = f"detcorr unavailable: {audit.get('reason')}"
        if os.environ.get("DETCORR_EXAMPLES_REQUIRE_DETCORR") == "1":
            pytest.fail(message)
        pytest.skip(message)
    assert audit["target_measure"] == "graph"
    assert audit["state_dimension"] == 33
    assert audit["state_logdet_abs_error"] < 2.0e-6


def test_grid_tails_and_sensitivity_are_reported(problem, frame) -> None:
    grid = np.linspace(demo.GRID_MIN, demo.GRID_MAX, 1201)
    targets = {
        "graph_reduced": demo.graph_logdensity(grid, problem),
        "finite_naive": demo.exact_integrated_tube_logdensity(grid, problem, corrected=False),
        "finite_corrected": demo.exact_integrated_tube_logdensity(grid, problem, corrected=True),
    }
    _, densities, _ = common.summarize_targets(grid, targets)
    tails = demo.grid_tail_checks(grid, targets, densities)
    sensitivity = demo.sensitivity_table(frame, grid, rho=demo.PENALTY_RHO)
    assert tails["passed"] is True
    assert set(sensitivity["sensitivity"]) == {
        "prior_log_sd",
        "chronological_train_size",
    }
    split_rows = sensitivity.loc[
        sensitivity["sensitivity"] == "chronological_train_size", "n_train"
    ]
    assert set(split_rows) == {
        60,
        64,
        70,
    }
    assert np.max(np.abs(sensitivity["corrected_shift_in_graph_sd"])) < 2.0e-5


def test_run_demo_writes_strict_current_outputs(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    summary = demo.run_demo(outputs, n_grid=1001)
    assert summary["checks"]["all_passed"] is True
    assert summary["classification"] == "external_open_data_heldout_calibration_demonstrator"
    assert summary["split"]["heldout_prices_used_in_reference_fit"] is False
    assert summary["model"]["prior"]["density_measure"] == "d lambda"
    assert summary["model"]["prior"]["log_scale"] == pytest.approx(0.5)
    assert summary["model"]["full_space_base_measure"].startswith(
        "d lambda times the product of du_t"
    )
    assert summary["outcome"]["naive_mean_shift_in_graph_sd"] < -1.5
    assert summary["outcome"]["corrected_density_l1_vs_graph"] < 1.0e-6
    assert len(list(outputs.glob("*.png"))) == 3
    assert len(list((outputs / "tables").glob("*.csv"))) == 5

    text = (outputs / "summary.json").read_text(encoding="utf-8")
    parsed = json.loads(text, parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
    assert parsed["checks"]["all_passed"] is True
    assert "NaN" not in text and "Infinity" not in text


def test_custom_split_is_described_dynamically(tmp_path: Path) -> None:
    summary = demo.run_demo(tmp_path / "custom", train_size=60, n_grid=501)
    assert summary["split"]["n_train"] == 60
    assert summary["split"]["n_heldout"] == 37
    assert "first 60 chronological rows" in summary["split"]["rule"]
    assert "37 held-out" in summary["correction_driver"]
    assert "all 37 held-out latent price states" in summary["independent_validation"]["route"]
