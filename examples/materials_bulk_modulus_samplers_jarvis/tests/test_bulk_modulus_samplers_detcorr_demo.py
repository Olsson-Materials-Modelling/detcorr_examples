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

import bulk_modulus_samplers_detcorr_demo as demo  # noqa: E402

from detcorr_demo_common import centered_max_abs_diff  # noqa: E402


def _skip_or_fail_if_detcorr_unavailable(audit: dict) -> None:
    if audit.get("detcorr_available"):
        return
    message = f"detcorr unavailable: {audit.get('reason')}"
    if os.environ.get("DETCORR_EXAMPLES_REQUIRE_DETCORR") == "1":
        pytest.fail(message)
    pytest.skip(message)


def _fixture(n_grid: int = 700):
    df = demo.load_bulk_modulus_fixture(ROOT / "data" / "jarvis_bulk_modulus_benchmark_fixture.csv")
    grid = np.linspace(0.72, 1.34, n_grid)
    return df, grid


def test_data_checksum_is_pinned() -> None:
    assert (
        demo.sha256_file(ROOT / "data" / "jarvis_bulk_modulus_benchmark_fixture.csv")
        == demo.FIXTURE_SHA256
    )


def test_checksum_is_enforced_after_fixture_rename(tmp_path: Path) -> None:
    source = ROOT / "data" / "jarvis_bulk_modulus_benchmark_fixture.csv"
    renamed = tmp_path / "renamed.csv"
    renamed.write_bytes(source.read_bytes())
    assert len(demo.load_bulk_modulus_fixture(renamed)) == 36
    renamed.write_bytes(source.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        demo.load_bulk_modulus_fixture(renamed)


def test_discrepancy_scale_uses_only_opt_reference_input() -> None:
    frame, _ = _fixture(n_grid=50)
    expected = np.sqrt((0.055 * frame["bulk_modulus_opt_gpa"].to_numpy(float)) ** 2 + 3.0**2)
    np.testing.assert_allclose(frame["sigma_gpa"], expected, rtol=0.0, atol=1.0e-14)


def test_invalid_numerical_configurations_are_rejected() -> None:
    df, grid = _fixture(n_grid=20)
    with pytest.raises(ValueError, match="burn"):
        demo.run_mcmc(
            df,
            target="graph",
            n_steps=10,
            burn=10,
            proposal_sd=0.03,
            seed=1,
        )
    with pytest.raises(ValueError, match=r"GPa\^-2"):
        demo.finite_penalty_validation_table(grid, df, rho=0.0)
    with pytest.raises(ValueError, match="n_iter"):
        demo.run_alternating_augmented_lagrangian_map(
            df,
            gamma_bounds=(0.72, 1.34),
            rho=0.1,
            n_iter=0,
        )


def test_centered_logdensity_comparison_rejects_support_mismatch() -> None:
    first = np.array([0.0, -np.inf])
    second = np.array([0.0, 0.0])
    assert np.isinf(centered_max_abs_diff(first, second))


def test_observations_are_external_values_not_generated() -> None:
    df, _ = _fixture(n_grid=50)
    assert "bulk_modulus_opt_gpa" in df.columns
    assert "bulk_modulus_expt_gpa" in df.columns
    assert "true_gamma" not in df.columns
    assert "generated_bulk_modulus" not in df.columns
    assert np.all(df["bulk_modulus_expt_gpa"].to_numpy(float) > 0.0)
    mae = float(
        np.mean(
            np.abs(
                df["bulk_modulus_opt_gpa"].to_numpy(float)
                - df["bulk_modulus_expt_gpa"].to_numpy(float)
            )
        )
    )
    assert round(mae, 2) == demo.SOURCE_TABLE_REPORTED_MAE_GPA


def test_solved_state_residual_has_constant_determinant() -> None:
    df, grid = _fixture(n_grid=400)
    comp = demo.hard_limit_log_components(grid, df)
    solved = demo.hard_limit_log_components(grid, df, solved_residual=True)
    assert centered_max_abs_diff(solved["log_naive"], comp["log_graph"]) < 1.0e-12
    assert centered_max_abs_diff(comp["log_fully_corrected"], comp["log_graph"]) < 1.0e-12
    assert centered_max_abs_diff(comp["log_naive"], comp["log_graph"]) > 1.0


def test_finite_penalty_validation_is_independent_gaussian_integral() -> None:
    df, grid = _fixture(n_grid=500)
    table, stats = demo.finite_penalty_validation_table(grid, df, n_points=7, rho=1.0e6)
    assert len(table) == 7
    assert stats["integral_route"] == "exact_scalar_gaussian_product"
    assert stats["load_bearing_metric"] == "finite_vs_hard_tube_centered_max_abs_diff"
    assert "not a separate check" in stats["finite_corrected_vs_graph_role"]
    assert stats["finite_vs_hard_tube_centered_max_abs_diff"] < 2.0e-3
    assert stats["finite_corrected_vs_graph_centered_max_abs_diff"] < 2.0e-3

    _, high_precision_stats = demo.finite_penalty_validation_table(
        grid,
        df,
        n_points=7,
        rho=1.0e8,
    )
    assert high_precision_stats["finite_vs_hard_tube_centered_max_abs_diff"] < 1.0e-6


def test_exact_gaussian_integral_matches_direct_quadrature() -> None:
    gamma, b_opt, observed, sigma, rho = 1.03, 100.0, 108.0, 7.0, 0.5
    constraint_mean = gamma * b_opt
    constraint_sd = gamma / math.sqrt(rho)
    half_width = 12.0 * max(sigma, constraint_sd)

    def integrand(state: float) -> float:
        observation_term = ((state - observed) / sigma) ** 2
        constraint_term = rho * (state / gamma - b_opt) ** 2
        return math.exp(-0.5 * (observation_term + constraint_term))

    numerical, _ = quad(
        integrand,
        min(observed, constraint_mean) - half_width,
        max(observed, constraint_mean) + half_width,
        points=[observed, constraint_mean],
        epsabs=1.0e-12,
        epsrel=1.0e-12,
        limit=200,
    )
    analytic = demo.gaussian_penalty_log_integral_one(
        gamma,
        b_opt,
        observed,
        sigma,
        rho,
    )
    assert math.log(numerical) == pytest.approx(analytic, rel=0.0, abs=1.0e-11)


def test_residual_scaling_equivalence() -> None:
    df, grid = _fixture(n_grid=500)
    table = demo.build_hard_limit_residual_scaling_table(
        grid,
        df,
        powers=[0, 1, 2, 3],
    )
    assert float(table["corrected_abs_diff"].max()) < 1e-10
    assert float(table["corrected_density_l1_vs_graph"].max()) < 1e-9
    assert float(table["naive_density_l1_vs_graph"].max()) > 0.15


def test_mcmc_and_smc_match_grid_reference_under_small_run() -> None:
    df, grid = _fixture(n_grid=700)
    main_table, _ = demo.build_main_table(grid, df)
    graph_mean = float(main_table.loc[main_table["target"] == "graph_reduced", "mean"].iloc[0])
    naive_mean = float(
        main_table.loc[
            main_table["target"] == "naive_bulk_modulus_tube",
            "mean",
        ].iloc[0]
    )
    samples, summary = demo.run_mcmc(
        df,
        target="graph",
        n_steps=1800,
        burn=300,
        proposal_sd=0.035,
        seed=11,
    )
    assert len(samples) == 1500
    assert 0.15 < float(summary.accept_rate) < 0.75
    assert summary.mcmc_autocorr_ess is not None
    assert summary.final_weight_ess is None
    assert abs(float(summary.mean) - graph_mean) < 0.004
    smc_samples, smc_summary, diag = demo.smc_tempering(
        df,
        target="naive",
        n_particles=700,
        n_temper=8,
        move_steps=1,
        proposal_sd=0.035,
        seed=12,
    )
    assert len(smc_samples) == 700
    assert not diag.empty
    assert smc_summary.mcmc_autocorr_ess is None
    assert smc_summary.final_weight_ess is not None
    assert 1.0 <= float(smc_summary.final_weight_ess) <= 700.0
    assert smc_summary.final_unique_particles is not None
    assert 1 <= int(smc_summary.final_unique_particles) <= 700
    assert smc_summary.final_unique_fraction is not None
    assert 0.0 < float(smc_summary.final_unique_fraction) <= 1.0
    assert abs(float(smc_summary.mean) - naive_mean) < 0.006


def test_corrected_is_not_aliased_as_a_sampler_target() -> None:
    df, _ = _fixture(n_grid=100)
    with pytest.raises(ValueError, match="unknown target"):
        demo.run_mcmc(
            df,
            target="corrected",
            n_steps=20,
            burn=5,
            proposal_sd=0.035,
            seed=1,
        )
    with pytest.raises(ValueError, match="unknown SMC target"):
        demo.smc_tempering(
            df,
            target="corrected",
            n_particles=20,
            n_temper=2,
            move_steps=1,
            proposal_sd=0.035,
            seed=2,
        )


def test_alternating_augmented_lagrangian_map_reaches_grid_mode() -> None:
    df, grid = _fixture(n_grid=900)
    main_table, _ = demo.build_main_table(grid, df)
    grid_mode = float(main_table.loc[main_table["target"] == "graph_reduced", "mode"].iloc[0])
    table, summary = demo.run_alternating_augmented_lagrangian_map(
        df,
        gamma_bounds=(0.72, 1.34),
        rho=0.1,
        n_iter=80,
    )
    assert "not posterior sampling" in summary["target"]
    assert summary["method"] == ("fixed-iteration alternating augmented-Lagrangian heuristic")
    assert summary["all_gamma_subproblems_succeeded"] is True
    assert bool(table["gamma_subproblem_success"].all())
    assert summary["feasibility_rms_gpa"] < 2.5e-4
    assert abs(summary["gamma_augmented_lagrangian"] - grid_mode) < 0.004


def test_detcorr_api_matches_diagonal_logdet_if_available() -> None:
    df, _ = _fixture(n_grid=100)
    audit = demo.detcorr_api_audit(1.02, len(df))
    _skip_or_fail_if_detcorr_unavailable(audit)
    assert audit["target_measure"] == "graph"
    assert audit["state_abs_error"] < 1.0e-10


def test_run_demo_writes_expected_outputs(tmp_path: Path) -> None:
    args = demo.build_arg_parser().parse_args(
        [
            "--outputs",
            str(tmp_path),
            "--n-grid",
            "500",
            "--mcmc-steps",
            "1800",
            "--mcmc-burn",
            "300",
            "--smc-particles",
            "600",
            "--smc-temper-steps",
            "8",
            "--smc-move-steps",
            "1",
            "--augmented-lagrangian-iters",
            "60",
        ]
    )
    summary = demo.run_demo(args)
    assert summary["demo"] == "materials_bulk_modulus_samplers_jarvis"
    assert summary["dataset"]["fixture_preparation"].startswith("pinned transcription")
    assert summary["dataset"]["source_verification_date"] == "2026-07-11"
    assert summary["dataset"]["fixture_mae_rounds_to_source_value"] is True
    assert summary["configuration"]["seed"] == 202611
    assert summary["configuration"]["finite_penalty_validation"]["rho_units"] == "GPa^-2"
    assert summary["configuration"]["alternating_augmented_lagrangian"]["rho_units"] == "GPa^-2"
    assert summary["model"]["parameter_units"]["gamma"] == "dimensionless"
    assert summary["model"]["residual_units"]["c_i"] == "GPa"
    assert summary["model"]["prior"]["density_measure"] == "d gamma"
    assert summary["model"]["prior"]["log_scale"] == pytest.approx(0.24)
    assert summary["model"]["full_space_base_measure"].endswith("dB_i in numerical GPa coordinates")
    assert "NIST public-information permission" in summary["dataset"]["upstream_license_status"]
    assert "inference beyond this fixture and model" in summary["interpretation_boundary"]
    assert len(summary["limitations"]) >= 5
    assert (tmp_path / "summary.json").exists()
    serialized_summary = (tmp_path / "summary.json").read_text(encoding="utf-8")
    assert "NaN" not in serialized_summary
    assert "Infinity" not in serialized_summary
    json.loads(serialized_summary, parse_constant=lambda value: pytest.fail(value))
    assert (tmp_path / "tables" / "mcmc_smc_sampler_summaries.csv").exists()
    assert (tmp_path / "tables" / "alternating_augmented_lagrangian_map_trace.csv").exists()
    sampler_table = summary["tables"]["sampler_summaries"]
    assert {row["name"] for row in sampler_table} == {
        "mcmc_graph",
        "mcmc_naive",
        "smc_graph",
        "smc_naive",
    }
    for row in sampler_table:
        if row["name"].startswith("smc_"):
            assert isinstance(row["final_unique_particles"], int)
            assert isinstance(row["n_resamples"], int)
