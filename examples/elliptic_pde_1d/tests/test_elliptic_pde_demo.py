from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
COMMON = ROOT.parent / "detcorr_demo_common" / "src"
for path in [str(SRC), str(COMMON)]:
    if path not in sys.path:
        sys.path.insert(0, path)

import elliptic_pde_demo as demo  # noqa: E402
import elliptic_pde_model as benchmark  # noqa: E402


def test_manufactured_second_derivative_and_solved_residual() -> None:
    grid = np.arange(1, benchmark.N_STATE + 1) * benchmark.GRID_SPACING
    manufactured = 0.5 * grid * (1.0 - grid)
    np.testing.assert_allclose(
        benchmark.K_STIFFNESS @ manufactured,
        np.ones(benchmark.N_STATE),
        rtol=0.0,
        atol=1.0e-13,
    )
    for theta in [-0.8, 0.0, 0.7]:
        assert np.linalg.norm(benchmark.residual(theta, benchmark.solve_state(theta))) < 1e-12


def test_routes_recover_graph_and_retain_negative_control() -> None:
    _, _, table = demo.build_routes()
    rows = table.set_index("target")
    assert rows.loc["detcorr_full_space", "density_l1_vs_graph"] < 1.0e-4
    assert abs(rows.loc["detcorr_full_space", "mean_shift_in_graph_sd"]) < 1.0e-3
    assert rows.loc["naive_full_space", "density_l1_vs_graph"] > 0.25
    assert abs(rows.loc["naive_full_space", "mean_shift_in_graph_sd"]) > 0.5


def test_run_demo_writes_accessible_artifacts(tmp_path: Path) -> None:
    summary = demo.run_demo(tmp_path)
    assert summary["target_measure"] == ("graph posterior over theta with respect to d theta")
    assert summary["independent_validation"]["detcorr_logdet_abs_error"] < 1.0e-10
    assert summary["correction_log_range_on_grid"] > 10.0
    assert (tmp_path / "summary.json").is_file()
    assert (tmp_path / "tables" / "posterior_routes.csv").is_file()
    assert (tmp_path / "elliptic_posterior_routes.png").is_file()
    assert (tmp_path / "elliptic_state_profiles.png").is_file()
