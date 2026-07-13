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

import calorimetry_constant_determinant_control_demo as demo  # noqa: E402


def test_dimensional_calorimetry_constraint_and_solved_state() -> None:
    heat_capacity = 120.0
    state = demo.solved_temperature_rise_k(heat_capacity)
    assert state == demo.HEAT_INPUT_J / heat_capacity
    assert demo.constraint_residual_k(heat_capacity, state) == 0.0
    assert demo.state_jacobian() == 1.0
    assert demo.state_logdet_correction() == 0.0


def test_exact_naive_and_corrected_integrals_are_identical() -> None:
    grid = np.linspace(
        demo.HEAT_CAPACITY_MIN_J_PER_K,
        demo.HEAT_CAPACITY_MAX_J_PER_K,
        401,
    )
    naive = demo.exact_integrated_tube_logdensity(
        grid,
        rho_per_k2=demo.PENALTY_RHO_PER_K2,
        corrected=False,
    )
    corrected = demo.exact_integrated_tube_logdensity(
        grid,
        rho_per_k2=demo.PENALTY_RHO_PER_K2,
        corrected=True,
    )
    np.testing.assert_array_equal(naive, corrected)


def test_exact_finite_integral_converges_to_reduced_graph() -> None:
    grid = np.linspace(
        demo.HEAT_CAPACITY_MIN_J_PER_K,
        demo.HEAT_CAPACITY_MAX_J_PER_K,
        501,
    )
    table = demo.convergence_table(grid)
    assert table["density_l1_vs_graph"].iloc[-1] < 2.0e-3
    assert table["density_l1_vs_graph"].iloc[-1] < table["density_l1_vs_graph"].iloc[0]
    assert table["constraint_tube_sd_K"].iloc[-1] < table["constraint_tube_sd_K"].iloc[0]


def test_detcorr_api_reports_exact_zero_correction() -> None:
    audit = demo.detcorr_api_audit()
    assert audit["detcorr_available"] is True
    assert audit["target_measure"] == "graph"
    assert abs(audit["primary_state_correction"]) < 1.0e-8
    assert audit["secondary_weight_correction"] == 0.0
    assert abs(audit["total_correction"]) < 1.0e-8
    assert audit["correction_abs_error"] < 1.0e-8
    assert audit["state_jacobian_route"].startswith("callable residual")


def test_run_demo_writes_strict_control_outputs(tmp_path: Path) -> None:
    summary = demo.run_demo(tmp_path, n_grid=401)
    assert summary["status"] == "synthetic_worked_control_not_experimental_calorimetry"
    assert summary["checks"]["all_passed"] is True
    outcomes = summary["outcomes"]
    assert outcomes["correction_log_range_on_grid"] == 0.0
    assert outcomes["naive_corrected_density_l1"] == 0.0
    assert outcomes["naive_corrected_mean_difference"] == 0.0
    assert outcomes["corrected_density_l1_vs_graph"] < 2.0e-3
    assert outcomes["naive_density_l1_vs_graph"] == outcomes["corrected_density_l1_vs_graph"]
    required = [
        "summary.json",
        "calorimetry_constant_determinant_posteriors.png",
        "finite_penalty_convergence.png",
        "tables/posterior_summaries.csv",
        "tables/rho_convergence.csv",
    ]
    assert all((tmp_path / name).is_file() for name in required)
    assert not list(tmp_path.rglob("*.md"))
    text = (tmp_path / "summary.json").read_text(encoding="utf-8")
    payload = json.loads(text)
    assert payload["checks"]["all_passed"] is True
    assert "NaN" not in text
