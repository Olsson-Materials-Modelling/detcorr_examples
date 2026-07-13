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

import bandgap_samplers_detcorr_demo as demo  # noqa: E402

from detcorr_demo_common import centered_max_abs_diff  # noqa: E402


def _skip_or_fail_if_detcorr_unavailable(audit: dict) -> None:
    if audit.get("detcorr_available"):
        return
    message = f"detcorr unavailable: {audit.get('reason')}"
    if os.environ.get("DETCORR_EXAMPLES_REQUIRE_DETCORR") == "1":
        pytest.fail(message)
    pytest.skip(message)


def _fixture(n_grid: int = 700):
    frame = demo.load_bandgap_fixture(ROOT / "data" / "jarvis_bandgap_benchmark_fixture.csv")
    grid = np.linspace(0.75, 2.85, n_grid)
    return frame, grid


def test_data_checksum_and_positive_row_contract() -> None:
    data_path = ROOT / "data" / "jarvis_bandgap_benchmark_fixture.csv"
    assert demo.sha256_file(data_path) == demo.FIXTURE_SHA256
    frame, _ = _fixture(n_grid=50)
    assert len(frame) == 54
    assert frame["jvasp_id"].is_unique
    assert np.all(frame["bandgap_opt_ev"].to_numpy(float) > 0.0)
    assert np.all(frame["bandgap_expt_ev"].to_numpy(float) > 0.0)
    assert "true_gamma" not in frame.columns
    assert "generated_bandgap" not in frame.columns


def test_checksum_is_enforced_after_fixture_rename(tmp_path: Path) -> None:
    source = ROOT / "data" / "jarvis_bandgap_benchmark_fixture.csv"
    renamed = tmp_path / "renamed.csv"
    renamed.write_bytes(source.read_bytes())
    assert len(demo.load_bandgap_fixture(renamed)) == 54
    renamed.write_bytes(source.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        demo.load_bandgap_fixture(renamed)


def test_likelihood_scale_is_fixed_by_opt_input_not_observed_response() -> None:
    frame, _ = _fixture(n_grid=50)
    expected = np.sqrt((0.15 * frame["bandgap_opt_ev"].to_numpy(float)) ** 2 + 0.12**2)
    response_based = np.sqrt((0.15 * frame["bandgap_expt_ev"].to_numpy(float)) ** 2 + 0.12**2)
    assert np.allclose(frame["sigma_gap_ev"].to_numpy(float), expected)
    assert not np.allclose(expected, response_based)


def test_log_energy_state_and_jacobian_are_dimensionless_and_unit_invariant() -> None:
    opt_ev = np.array([0.4, 1.5, 5.0])
    gamma = 1.7
    state_ev = demo.dimensionless_energy_state(gamma, opt_ev, energy_reference_ev=1.0)
    state_mev = demo.dimensionless_energy_state(
        gamma,
        1000.0 * opt_ev,
        energy_reference_ev=1000.0,
    )
    assert np.allclose(state_ev, state_mev)
    assert np.allclose(state_ev, gamma * opt_ev)
    assert np.allclose(demo.log_energy_residual(gamma, state_ev, opt_ev), 0.0)

    step = 1.0e-6
    numeric = (
        demo.log_energy_residual(gamma, state_ev + step, opt_ev)
        - demo.log_energy_residual(gamma, state_ev - step, opt_ev)
    ) / (2.0 * step)
    analytic = 1.0 / state_ev
    assert np.allclose(numeric, analytic, rtol=2.0e-8, atol=2.0e-8)


def test_energy_likelihood_is_evaluated_in_energy_space() -> None:
    frame, _ = _fixture(n_grid=50)
    gamma = 1.55
    expected = -0.5 * np.sum(
        (
            (
                frame["bandgap_expt_ev"].to_numpy(float)
                - gamma * frame["bandgap_opt_ev"].to_numpy(float)
            )
            / frame["sigma_gap_ev"].to_numpy(float)
        )
        ** 2
    )
    assert demo.log_likelihood_gamma(gamma, frame) == pytest.approx(expected)


def test_graph_correction_and_constant_determinant_control() -> None:
    frame, grid = _fixture(n_grid=800)
    log_energy = demo.hard_limit_log_components(grid, frame)
    solved = demo.hard_limit_log_components(
        grid,
        frame,
        solved_energy_residual=True,
    )
    assert (
        centered_max_abs_diff(log_energy["log_fully_corrected"], log_energy["log_graph"]) < 1.0e-10
    )
    assert centered_max_abs_diff(solved["log_naive"], log_energy["log_graph"]) < 1.0e-12
    assert centered_max_abs_diff(log_energy["log_naive"], log_energy["log_graph"]) > 30.0


def test_global_log_energy_state_quadrature_validates_finite_tube() -> None:
    frame, grid = _fixture(n_grid=500)
    table, stats = demo.finite_penalty_validation_table(
        grid,
        frame,
        n_points=7,
        rho=3.0e9,
        n_nodes=96,
    )
    assert len(table) == 7
    assert stats["n_materials"] == 54
    assert stats["integral_route"].startswith("global_gauss_hermite")
    assert stats["load_bearing_metric"] == "finite_vs_hard_tube_centered_max_abs_diff"
    assert "included inside each state integral" in stats["finite_corrected_vs_graph_role"]
    assert stats["finite_vs_hard_tube_centered_max_abs_diff"] < 1.0e-4
    assert stats["finite_corrected_vs_graph_centered_max_abs_diff"] < 1.0e-4
    row = frame.iloc[0]
    naive = demo.finite_penalty_log_integral_one(
        1.6,
        row["bandgap_opt_ev"],
        row["bandgap_expt_ev"],
        row["sigma_gap_ev"],
        1.0e9,
        n_nodes=96,
    )
    corrected = demo.finite_penalty_log_integral_one(
        1.6,
        row["bandgap_opt_ev"],
        row["bandgap_expt_ev"],
        row["sigma_gap_ev"],
        1.0e9,
        n_nodes=96,
        include_state_correction=True,
    )
    assert naive != pytest.approx(corrected)


def test_reciprocal_lebesgue_construction_is_rejected_globally() -> None:
    frame, _ = _fixture(n_grid=50)
    diagnostic = demo.rejected_reciprocal_construction_diagnostic(frame)
    assert diagnostic["status"] == "rejected_not_an_active_target"
    assert diagnostic["finite_penalty_state_integral"].startswith("diverges")
    assert 0.75 < diagnostic["verified_global_mode_gamma"] < 2.85
    assert diagnostic["mass_outside_display_grid"] < 1.0e-8
    assert diagnostic["log_mass_outside_minus_inside_display"] < -100.0
    assert (
        min(
            diagnostic["lower_endpoint_log_drop"],
            diagnostic["upper_endpoint_log_drop"],
        )
        > 100.0
    )


def test_active_targets_have_global_log_gamma_support() -> None:
    frame, _ = _fixture(n_grid=50)
    table = demo.global_log_gamma_support_audit(
        frame,
        display_bounds=(0.75, 2.85),
    )
    assert set(table["target"]) == {"graph", "naive"}
    assert float(table["mass_outside_display_grid"].max()) < 1.0e-8
    assert float(table[["lower_edge_mass_1pct", "upper_edge_mass_1pct"]].to_numpy().max()) < 1.0e-10
    assert (
        float(table[["lower_endpoint_log_drop", "upper_endpoint_log_drop"]].to_numpy().min())
        > 100.0
    )
    assert table["mode_gamma"].between(0.75, 2.85).all()


def test_domain_threshold_sensitivity_is_explicit() -> None:
    frame, grid = _fixture(n_grid=700)
    table = demo.build_domain_sensitivity_table(
        grid,
        frame,
        thresholds_ev=[0.0, 0.2, 0.5, 1.0],
    )
    assert table["n_materials"].tolist() == [54, 49, 46, 35]
    assert float(table["graph_mean"].iloc[0]) > float(table["graph_mean"].iloc[-1])
    assert float(table["graph_mean"].max() - table["graph_mean"].min()) > 0.04


def test_detcorr_api_matches_analytic_diagonal_logdet_if_available() -> None:
    frame, _ = _fixture(n_grid=100)
    audit = demo.detcorr_api_audit(1.56, frame)
    _skip_or_fail_if_detcorr_unavailable(audit)
    assert audit["target_measure"] == "graph"
    assert audit["state_abs_error"] < 1.0e-10


def test_mcmc_and_smc_match_grid_references_without_duplicate_corrected_target() -> None:
    frame, grid = _fixture(n_grid=900)
    table, _ = demo.build_main_table(grid, frame)
    graph_mean = float(table.loc[table["target"] == "graph_reduced", "mean"].iloc[0])
    naive_mean = float(
        table.loc[table["target"] == "naive_log_energy_residual_tube", "mean"].iloc[0]
    )
    mcmc_samples, mcmc_summary = demo.run_mcmc(
        frame,
        target="graph",
        n_steps=2200,
        burn=400,
        proposal_sd=0.026,
        seed=11,
    )
    assert len(mcmc_samples) == 1800
    assert 0.35 < float(mcmc_summary.accept_rate) < 0.85
    assert mcmc_summary.mcmc_autocorr_ess is not None
    assert abs(float(mcmc_summary.mean) - graph_mean) < 0.012

    smc_samples, smc_summary, diagnostics = demo.smc_tempering(
        frame,
        target="naive",
        n_particles=900,
        n_temper=10,
        move_steps=1,
        proposal_sd=0.026,
        seed=12,
    )
    assert len(smc_samples) == 900
    assert not diagnostics.empty
    assert smc_summary.mcmc_autocorr_ess is None
    assert smc_summary.final_weight_ess is not None
    assert smc_summary.final_unique_particles is not None
    assert abs(float(smc_summary.mean) - naive_mean) < 0.012
    with pytest.raises(ValueError, match="unknown"):
        demo.log_density_gamma(1.5, frame, target="corrected")


def test_alternating_augmented_lagrangian_is_labeled_and_feasible() -> None:
    frame, grid = _fixture(n_grid=1000)
    table, _ = demo.build_main_table(grid, frame)
    grid_mode = float(table.loc[table["target"] == "graph_reduced", "mode"].iloc[0])
    _, summary = demo.run_alternating_augmented_lagrangian_map(
        frame,
        gamma_bounds=(0.75, 2.85),
        rho=25.0,
        n_iter=120,
    )
    assert summary["method"].startswith("fixed-iteration alternating")
    assert summary["target"].endswith("not posterior sampling")
    assert summary["feasibility_rms_dimensionless"] < 1.0e-5
    assert abs(summary["gamma_augmented_lagrangian"] - grid_mode) < 0.003


def test_run_demo_writes_strict_json_current_outputs_and_figures(tmp_path: Path) -> None:
    args = demo.build_arg_parser().parse_args(
        [
            "--outputs",
            str(tmp_path),
            "--n-grid",
            "600",
            "--validation-points",
            "7",
            "--validation-nodes",
            "96",
            "--mcmc-steps",
            "2200",
            "--mcmc-burn",
            "400",
            "--smc-particles",
            "800",
            "--smc-temper-steps",
            "10",
            "--smc-move-steps",
            "1",
            "--augmented-lagrangian-iters",
            "100",
        ]
    )
    summary = demo.run_demo(args)
    assert summary["dataset"]["n_materials"] == 54
    assert summary["model"]["prior"]["density_measure"] == "d gamma"
    assert summary["model"]["prior"]["log_scale"] == pytest.approx(0.62)
    assert summary["model"]["full_space_base_measure"].startswith(
        "d gamma times the product of de_i"
    )
    assert "CC BY 4.0" in summary["dataset"]["licence_status"]
    assert summary["model"]["sampling_targets"] == [
        "reduced_graph",
        "naive_hard_limit_tube",
    ]
    assert summary["outcome"]["naive_density_l1_vs_graph"] > 0.5
    assert summary["rejected_reciprocal_construction"]["status"] == "rejected_not_an_active_target"
    summary_path = tmp_path / "summary.json"
    parsed = json.loads(
        summary_path.read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )
    assert parsed["tables"]["sampler_summaries"][2]["mcmc_autocorr_ess"] is None
    assert (tmp_path / "posterior_gamma_log_energy_residual.png").exists()
    assert (tmp_path / "bandgap_fixture_opt_vs_expt.png").exists()
    assert (tmp_path / "domain_threshold_sensitivity.png").exists()
    assert (tmp_path / "tables" / "alternating_augmented_lagrangian_map_trace.csv").exists()
    assert (tmp_path / "tables" / "rejected_reciprocal_construction.csv").exists()
    assert (tmp_path / "tables" / "global_log_gamma_support_audit.csv").exists()


def test_active_source_stays_within_declared_calibration_scope() -> None:
    source = (ROOT / "src" / "bandgap_samplers_detcorr_demo.py").read_text(encoding="utf-8")
    lowered = source.lower()
    assert "wavelength" not in lowered
    assert "absorption edge" not in lowered
    assert "admm" not in lowered
