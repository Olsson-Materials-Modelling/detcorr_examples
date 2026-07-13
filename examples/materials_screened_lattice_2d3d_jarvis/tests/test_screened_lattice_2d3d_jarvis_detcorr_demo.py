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

import screened_lattice_2d3d_jarvis_detcorr_demo as demo  # noqa: E402

from detcorr_demo_common import centered_max_abs_diff  # noqa: E402


def _df():
    return demo.load_paired_fixture(ROOT / "data" / "jarvis_bulk_modulus_paired_fixture.csv")


def _observations(*, dimension: int = 3, grid_size: int = 4, n_grid: int = 300):
    df = _df()
    observations = demo.build_paired_observations(
        df, dimension=dimension, grid_size=grid_size, pin=0.06
    )
    return df, observations, np.linspace(0.04, 3.0, n_grid)


def test_paired_fixture_checksum_is_pinned() -> None:
    path = ROOT / "data" / "jarvis_bulk_modulus_paired_fixture.csv"
    assert demo.sha256_file(path) == demo.FIXTURE_SHA256


def test_checksum_is_enforced_after_fixture_rename(tmp_path: Path) -> None:
    source = ROOT / "data" / "jarvis_bulk_modulus_paired_fixture.csv"
    renamed = tmp_path / "renamed.csv"
    renamed.write_bytes(source.read_bytes())
    assert len(demo.load_paired_fixture(renamed)) == 36
    renamed.write_bytes(source.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        demo.load_paired_fixture(renamed)


def test_operator_and_response_use_distinct_external_columns() -> None:
    df, observations, _ = _observations()
    opt = df["bulk_modulus_opt_gpa"].to_numpy(float)
    expt = df["bulk_modulus_expt_gpa"].to_numpy(float)
    expected_coefficients = opt / np.median(opt)
    expected_response = np.median(expt) / expt
    assert np.allclose(observations["axis_coeff_0"], expected_coefficients)
    assert np.allclose(observations["normalized_experimental_response_proxy"], expected_response)
    assert not np.allclose(expected_coefficients * expected_response, 1.0)
    assert "experimental" in observations["observation_origin"].iloc[0]


def test_response_normalization_is_fixed_reference_green_value() -> None:
    _, observations, _ = _observations(dimension=3, grid_size=4)
    basis = demo.make_spectral_basis(3, 4)
    expected = demo.response_from_spectrum(1.0, np.ones(3), basis, pin=0.06)
    assert np.allclose(observations["reference_green_response"], expected)
    ratio = observations["observed_response"] / observations["reference_green_response"]
    assert np.allclose(ratio, observations["normalized_experimental_response_proxy"])


@pytest.mark.parametrize("dimension,grid_size", [(2, 5), (3, 4)])
def test_sparse_logdet_matches_separable_reference(dimension: int, grid_size: int) -> None:
    df = _df()
    coefficients = demo.operator_coefficients(df, dimension)
    basis = demo.make_spectral_basis(dimension, grid_size)
    matrix = demo.block_sparse_matrix(
        1.07,
        coefficients,
        dimension=dimension,
        grid_size=grid_size,
        pin=0.06,
    )
    sparse_value = demo.exact_sparse_lu_logdet(matrix)
    analytic = demo.logdet_from_spectrum(1.07, coefficients, basis, pin=0.06)
    assert abs(sparse_value - analytic) < 2.0e-10


def test_corrected_target_matches_graph_and_grid_support_is_interior() -> None:
    _, observations, grid = _observations(dimension=3, grid_size=4, n_grid=500)
    table, densities, components = demo.build_main_table(
        grid, observations, dimension=3, grid_size=4, pin=0.06
    )
    assert (
        centered_max_abs_diff(components["log_fully_corrected"], components["log_graph"]) < 1.0e-10
    )
    assert centered_max_abs_diff(components["log_naive"], components["log_graph"]) > 10.0
    support = demo.build_support_table(grid, densities)
    assert support["mode_is_interior"].all()
    assert float(support["maximum_edge_mass"].max()) < 1.0e-3
    corrected = table[table["target"] == "detcorr_screened_lattice_tube"].iloc[0]
    assert float(corrected["density_l1_vs_graph"]) < 1.0e-9


def test_exact_full_state_finite_penalty_validation() -> None:
    _, observations, grid = _observations(dimension=3, grid_size=4, n_grid=260)
    table, stats = demo.finite_penalty_validation_table(
        grid,
        observations,
        dimension=3,
        grid_size=4,
        pin=0.06,
        n_points=9,
    )
    # Coarse test grids may contain fewer distinct points than requested inside
    # a very narrow posterior interval; both disjoint supports must still appear.
    assert len(table) >= 7
    assert stats["route"] == "exact_sparse_multivariate_Gaussian_state_integration"
    assert stats["support_scope"] == "union of central 98% graph and naive posterior support"
    assert stats["load_bearing_metric"] == "finite_vs_hard_tube_centered_max_abs_diff"
    assert "not a separate check" in stats["finite_corrected_vs_graph_role"]
    assert stats["naive_central_98_interval"][1] < stats["graph_central_98_interval"][0]
    assert float(table["scale"].min()) < 0.5
    assert float(table["scale"].max()) > 0.9
    assert stats["finite_vs_hard_tube_centered_max_abs_diff"] < 1.0e-2
    assert stats["finite_corrected_vs_graph_centered_max_abs_diff"] < 1.0e-2


def test_residual_coordinate_correction_is_invariant() -> None:
    _, observations, grid = _observations(dimension=3, grid_size=5, n_grid=300)
    table = demo.build_residual_scaling_table(
        grid,
        observations,
        dimension=3,
        grid_size=5,
        pin=0.06,
        powers=[0, 1, 2],
    )
    assert float(table["corrected_density_l1_vs_graph"].max()) < 1.0e-8
    assert float(table["corrected_centered_logdensity_max_abs_diff_vs_graph"].max()) < 1.0e-9
    limited = table[~table["naive_mode_is_interior"]]
    assert not limited.empty
    assert limited["naive_mean"].isna().all()
    assert limited["naive_status"].str.contains("cutoff").all()
    records = demo._records_with_nulls(table)
    limited_records = [row for row in records if not row["naive_mode_is_interior"]]
    assert all(row["naive_mean"] is None for row in limited_records)
    assert all(row["naive_density_l1_vs_graph"] is None for row in limited_records)


def test_matrix_free_audit_uses_centered_support_errors_and_multiple_seeds() -> None:
    df = _df()
    audit = demo.build_determinant_audit(
        1.0,
        [0.8, 1.0, 1.2],
        df,
        pin=0.06,
        audit_grid_2d=5,
        audit_grid_3d=4,
        stress_grid_3d=4,
        slq_probe_counts=[2, 4],
        slq_degrees=[8, 12],
        slq_seeds=[31, 47],
    )
    exact = audit[audit["route"] == "exact_sparse_lu"]
    assert len(exact) == 2
    assert float(exact["centered_abs_error"].max()) < 2.0e-10
    stochastic = audit[audit["route"] == "matrix_free_centered_gram_slq"]
    if stochastic.empty:
        if os.environ.get("DETCORR_EXAMPLES_REQUIRE_DETCORR") == "1":
            pytest.fail("release mode requires real matrix-free Gram/SLQ rows")
        pytest.skip("detcorr Gram/SLQ unavailable")
    assert stochastic["value"].notna().all()
    assert stochastic["centered_abs_error"].notna().all()
    assert stochastic["seed"].nunique() == 2
    assert set(np.round(stochastic["scale"], 10)) == {0.8, 1.0, 1.2}
    reference = stochastic[np.isclose(stochastic["scale"], stochastic["reference_scale"])]
    assert np.allclose(reference["centered_value"], 0.0)


def test_run_demo_writes_strict_summary(tmp_path: Path) -> None:
    args = demo.build_arg_parser().parse_args(
        [
            "--outputs",
            str(tmp_path),
            "--grid-size",
            "4",
            "--n-grid",
            "300",
            "--audit-grid-2d",
            "5",
            "--audit-grid-3d",
            "4",
            "--stress-grid-3d",
            "4",
            "--slq-probes",
            "2",
            "--slq-degrees",
            "8",
            "--slq-seeds",
            "31,47",
        ]
    )
    summary = demo.run_demo(args)
    assert summary["demo"] == "materials_screened_lattice_2d3d_jarvis"
    assert "not an elastic constant" in summary["interpretation_boundary"]
    assert summary["dataset"]["operator_input"] == "bulk_modulus_opt_gpa"
    assert summary["dataset"]["dependent_response_input"] == "bulk_modulus_expt_gpa"
    assert summary["model"]["prior"]["density_measure"] == "ds"
    assert summary["model"]["prior"]["log_scale"] == pytest.approx(0.32)
    assert summary["model"]["full_space_base_measure"].startswith("ds times product_j")
    assert "NIST public-information permission" in summary["dataset"]["upstream_license_status"]
    text = (tmp_path / "summary.json").read_text(encoding="utf-8")
    payload = json.loads(
        text,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON token: {value}")
        ),
    )
    assert payload["outcome"]["maximum_grid_edge_mass"] < 1.0e-3
    exact_rows = [
        row for row in payload["tables"]["determinant_audit"] if row["route"] == "exact_sparse_lu"
    ]
    assert exact_rows
    assert all(row["n_probe"] is None for row in exact_rows)
    assert all(row["lanczos_degree"] is None for row in exact_rows)
    assert all(row["seed"] is None for row in exact_rows)
    cutoff_rows = [
        row for row in payload["tables"]["residual_scaling"] if not row["naive_mode_is_interior"]
    ]
    assert all(row["naive_mean"] is None for row in cutoff_rows)
    assert all(row["naive_density_l1_vs_graph"] is None for row in cutoff_rows)
    assert (tmp_path / "tables" / "grid_support_diagnostics.csv").exists()
