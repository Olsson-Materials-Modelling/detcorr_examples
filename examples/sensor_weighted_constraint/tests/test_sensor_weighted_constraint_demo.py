from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
COMMON = ROOT.parent / "detcorr_demo_common" / "src"
for path in (SRC, COMMON):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import sensor_weighted_constraint_demo as demo  # noqa: E402


def test_constraint_units_and_identity_state_jacobian() -> None:
    gamma = 1.13
    state = demo.solved_state_mv(gamma)
    np.testing.assert_allclose(demo.constraint_residual_mv(gamma, state), 0.0)
    np.testing.assert_array_equal(demo.state_jacobian(), np.eye(demo.N_CHANNEL))
    assert demo.STATE_UNIT == "mV"
    assert demo.GAIN_UNIT == "dimensionless"
    assert np.isneginf(demo.log_prior_gamma(0.0))
    assert np.isneginf(demo.log_prior_gamma(-0.1))
    assert np.isfinite(demo.log_prior_gamma(1.0))


def test_algorithmic_metric_is_spd_and_parameter_dependent() -> None:
    metrics = [demo.algorithmic_metric(gamma) for gamma in (0.7, 1.0, 1.5)]
    assert all(np.all(np.linalg.eigvalsh(metric) > 0.0) for metric in metrics)
    assert not np.allclose(metrics[0], metrics[-1])
    correction_range = demo.metric_log_sqrt_det(1.5) - demo.metric_log_sqrt_det(0.7)
    assert correction_range > 4.0


def test_exact_state_integration_recovers_graph_only_with_weight_volume() -> None:
    grid = np.linspace(demo.GAMMA_MIN, demo.GAMMA_MAX, 501)
    graph = demo.graph_logdensity(grid)
    naive = demo.exact_integrated_tube_logdensity(
        grid,
        rho=demo.PENALTY_RHO,
        include_weight_volume=False,
    )
    corrected = demo.exact_integrated_tube_logdensity(
        grid,
        rho=demo.PENALTY_RHO,
        include_weight_volume=True,
    )
    corrected_error = demo.common.centered_max_abs_diff(corrected, graph)
    assert corrected_error < demo.FINITE_PENALTY_ERROR_LIMIT
    coarse_corrected = demo.exact_integrated_tube_logdensity(
        grid,
        rho=demo.PENALTY_RHO / 10.0,
        include_weight_volume=True,
    )
    coarse_error = demo.common.centered_max_abs_diff(coarse_corrected, graph)
    assert corrected_error < 0.2 * coarse_error
    naive_density, _, _ = demo.common.normalize_on_grid(grid, naive)
    graph_density, _, _ = demo.common.normalize_on_grid(grid, graph)
    assert demo.common.density_l1(grid, naive_density, graph_density) > 0.05


def test_detcorr_weighted_api_reports_secondary_not_primary_correction() -> None:
    audit = demo.detcorr_weighted_api_audit()
    assert audit["detcorr_available"] is True
    assert audit["target_measure"] == "graph"
    assert abs(audit["primary_state_correction"]) < 1.0e-8
    assert audit["primary_abs_error"] < 1.0e-8
    assert audit["state_jacobian_route"].startswith("callable residual")
    assert audit["secondary_abs_error"] < 1.0e-10
    assert audit["weighted_norm_abs_error"] < 1.0e-12


def test_run_demo_writes_strict_outputs(tmp_path: Path) -> None:
    summary = demo.run_demo(tmp_path, n_grid=401)
    assert summary["status"] == "synthetic_worked_demonstrator_not_empirical_calibration"
    assert summary["checks"]["all_passed"] is True
    assert summary["checks"]["finite_penalty_bias_decreases_with_rho"] is True
    assert summary["checks"]["posterior_grid_support_is_interior"] is True
    assert summary["grid_support"]["minimum_endpoint_log_drop"] > 25.0
    assert summary["grid_support"]["maximum_two_percent_edge_mass"] < 1.0e-8
    assert summary["model"]["metric_role"] == (
        "algorithmic_constraint_metric_not_observation_noise"
    )
    assert summary["outcomes"]["correction_log_range_on_grid"] > 4.0
    assert abs(summary["outcomes"]["naive_mean_shift_in_graph_sd"]) > 0.2
    assert summary["outcomes"]["corrected_density_l1_vs_graph"] < 1.0e-3
    assert summary["outcomes"]["naive_density_l1_vs_graph"] > 0.05
    assert (
        summary["independent_validation"]["finite_corrected_vs_graph_centered_max_abs_diff"]
        < demo.FINITE_PENALTY_ERROR_LIMIT
    )
    required = [
        "summary.json",
        "sensor_gain_posteriors.png",
        "algorithmic_metric_correction.png",
        "tables/posterior_summaries.csv",
        "tables/finite_penalty_validation.csv",
        "tables/algorithmic_metric_audit.csv",
        "tables/grid_support_audit.csv",
    ]
    assert all((tmp_path / name).is_file() for name in required)
    assert not list(tmp_path.rglob("*.md"))
    payload = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert payload["checks"]["all_passed"] is True
    assert "NaN" not in (tmp_path / "summary.json").read_text(encoding="utf-8")
