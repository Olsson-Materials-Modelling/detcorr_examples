from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
COMMON = ROOT.parent / "detcorr_demo_common" / "src"
for path in [str(SRC), str(COMMON)]:
    if path not in sys.path:
        sys.path.insert(0, path)

import adsorption_langmuir_equilibrium_demo as demo  # noqa: E402


def _design():
    return demo.load_synthetic_design(ROOT / "data" / "SYNTHETIC_DESIGN.json")


def _grid_and_targets(n_grid: int = 601):
    _, data = _design()
    eta_grid = np.linspace(demo.DEFAULT_ETA_MIN, demo.DEFAULT_ETA_MAX, n_grid)
    graph = demo.reduced_graph_logdensity(eta_grid, data)
    integrals = demo.finite_penalty_state_integrals(
        eta_grid,
        data,
        rho=demo.DEFAULT_RHO,
        n_nodes=demo.DEFAULT_QUADRATURE_NODES,
    )
    table, _, _ = demo.summarize_targets(
        eta_grid,
        {
            "graph_reduced": graph,
            "naive_finite_penalty": integrals["log_naive"],
            "detcorr_finite_penalty": integrals["log_corrected"],
        },
    )
    return data, eta_grid, table


def _skip_or_fail_if_detcorr_unavailable(audit: dict) -> None:
    if audit.get("detcorr_available"):
        return
    message = f"detcorr unavailable: {audit.get('reason')}"
    if os.environ.get("DETCORR_EXAMPLES_REQUIRE_DETCORR") == "1":
        pytest.fail(message)
    pytest.skip(message)


def test_synthetic_design_is_fixed_and_declared() -> None:
    design, data = _design()
    assert design["status"] == "synthetic_validation_only"
    np.testing.assert_allclose(
        design["coverage_offsets"],
        [0.006, -0.005, 0.004, -0.007, 0.006, -0.004, 0.002],
        rtol=0.0,
        atol=0.0,
    )
    assert np.log(design["K_true_per_bar"]) == pytest.approx(design["eta_true"])
    assert np.all((data["q_observed"] > 0.0) & (data["q_observed"] < 1.0))
    assert data["observation_status"].str.contains("synthetic").all()


def test_solved_constraint_is_zero() -> None:
    design, data = _design()
    pressure = data["pressure_bar"].to_numpy(float)
    for eta in [design["eta_true"], -0.9, 0.7]:
        q_star = demo.solved_coverage(float(eta), pressure)
        np.testing.assert_allclose(
            demo.constraint(float(eta), q_star, pressure),
            0.0,
            rtol=0.0,
            atol=2.0e-14,
        )


def test_state_derivative_matches_finite_difference() -> None:
    q = np.array([0.04, 0.18, 0.43, 0.72])
    analytic = demo.state_jacobian_diagonal(q)
    step = 1.0e-7
    finite_difference = (demo.logit(q + step) - demo.logit(q - step)) / (2.0 * step)
    np.testing.assert_allclose(analytic, finite_difference, rtol=2.0e-9, atol=1.0e-9)


def test_gauss_legendre_quadrature_converges() -> None:
    _, data = _design()
    eta_grid = np.linspace(demo.DEFAULT_ETA_MIN, demo.DEFAULT_ETA_MAX, 501)
    table, stats = demo.quadrature_convergence_table(
        eta_grid,
        data,
        rho=demo.DEFAULT_RHO,
        node_counts=[8, 16, 24, 32, 48],
        reference_nodes=64,
        window_sd=demo.DEFAULT_WINDOW_SD,
    )
    corrected_errors = table["corrected_centered_log_error_vs_reference"].to_numpy(float)
    assert np.all(np.diff(corrected_errors) < 0.0)
    assert stats["final_corrected_centered_log_error"] < 1.0e-9
    assert stats["final_corrected_density_l1_error"] < 1.0e-10


def test_pointwise_corrected_integral_recovers_graph_density() -> None:
    _, _, table = _grid_and_targets()
    rows = table.set_index("target")
    graph = rows.loc["graph_reduced"]
    corrected = rows.loc["detcorr_finite_penalty"]
    assert float(corrected["density_l1_vs_graph"]) < 1.0e-3
    assert abs(float(corrected["mean"] - graph["mean"])) < 0.002 * float(graph["sd"])


def test_naive_integral_is_a_real_negative_control() -> None:
    _, _, table = _grid_and_targets()
    rows = table.set_index("target")
    graph = rows.loc["graph_reduced"]
    naive = rows.loc["naive_finite_penalty"]
    shift_sd = float((naive["mean"] - graph["mean"]) / graph["sd"])
    assert float(naive["density_l1_vs_graph"]) > 0.1
    assert shift_sd > 0.15


def test_detcorr_diagonal_audit_matches_analytic_logdet() -> None:
    design, data = _design()
    audit = demo.detcorr_diagonal_audit(design["eta_true"], data["pressure_bar"].to_numpy(float))
    _skip_or_fail_if_detcorr_unavailable(audit)
    assert audit["target_measure"] == "graph"
    assert audit["absolute_error"] < 1.0e-12
    assert audit["result"]["exact"] is True


def test_run_demo_writes_strict_outputs(tmp_path: Path) -> None:
    args = demo.build_arg_parser().parse_args(
        [
            "--outputs",
            str(tmp_path),
            "--n-grid",
            "401",
            "--quadrature-nodes",
            "32",
            "--quadrature-reference-nodes",
            "48",
            "--quadrature-convergence-nodes",
            "8,16,24,32",
        ]
    )
    summary = demo.run_demo(args)
    assert summary["demo"] == "adsorption_langmuir_equilibrium"
    assert summary["synthetic_status"].startswith("declared deterministic")
    for field in [
        "correction_log_range_on_grid",
        "naive_mean_shift_in_graph_sd",
        "corrected_density_l1_vs_graph",
        "naive_density_l1_vs_graph",
        "quadrature_convergence_error",
    ]:
        assert isinstance(summary[field], float)

    summary_path = tmp_path / "summary.json"
    payload = json.loads(
        summary_path.read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )
    assert payload["design"]["status"] == "synthetic_validation_only"
    assert (tmp_path / "tables" / "posterior_targets.csv").is_file()
    assert (tmp_path / "tables" / "quadrature_convergence.csv").is_file()
    assert (tmp_path / "posterior_eta_targets.png").stat().st_size > 10_000
    assert (tmp_path / "synthetic_langmuir_isotherm.png").stat().st_size > 10_000
