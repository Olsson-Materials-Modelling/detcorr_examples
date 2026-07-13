from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

STUDY = Path(__file__).resolve().parents[1]
REPO = STUDY.parents[1]
sys.path.insert(0, str(STUDY / "src"))
sys.path.insert(0, str(REPO))

import bandgap_analysis as analysis  # noqa: E402

from common.jarvis_geometry import build_geometry_audit  # noqa: E402
from common.jarvis_protocol import RESAMPLING_SEED, logo_role_manifest  # noqa: E402


def fixture() -> pd.DataFrame:
    return analysis.load_fixture(STUDY / "data" / "jarvis_bandgap_benchmark_fixture.csv")


@pytest.fixture(scope="module")
def quick_output(tmp_path_factory: pytest.TempPathFactory) -> tuple[dict, Path]:
    output = tmp_path_factory.mktemp("gap-quick")
    data = STUDY / "data" / "jarvis_bandgap_benchmark_fixture.csv"
    summary = analysis.run(
        data,
        output,
        "quick",
        99,
        data_manifest_path=STUDY / "data" / "FIXTURE_MANIFEST.json",
    )
    return summary, output


def test_fixture_is_positive_and_formula_grouped() -> None:
    frame = fixture()
    assert len(frame) == 54
    assert frame["formula"].nunique() == 50
    assert (frame[["bandgap_opt_ev", "bandgap_expt_ev"]] > 0).all().all()
    repeated = set(frame.loc[frame["formula"].duplicated(False), "formula"])
    assert repeated == {"AlN", "CdS", "GaN", "TiO2"}


def test_sensitivity_axes_are_one_at_a_time_and_explicit() -> None:
    table = analysis.sensitivity_table(fixture(), analysis.make_grid("quick")).set_index("analysis")
    assert {
        "likelihood_family",
        "mu_prior_sd",
        "physical_sigma_floor",
        "physical_sigma_fraction",
    }.issubset(table.columns)
    assert table.loc["primary_robust_log_student_t4", "likelihood_family"] == "student_t4"
    assert table.loc["gaussian_log_sensitivity", "likelihood_family"] == "normal"
    assert table.loc["narrower_mu_prior", "likelihood_family"] == "student_t4"
    assert table.loc["wider_mu_prior", "likelihood_family"] == "student_t4"
    assert table.loc["narrower_mu_prior", "mu_prior_sd"] == 0.40
    assert table.loc["wider_mu_prior", "mu_prior_sd"] == 1.00


def test_custom_loader_normalizes_edges_and_rejects_ambiguous_keys(tmp_path: Path) -> None:
    frame = fixture()
    accepted = frame.copy()
    accepted.loc[0, "formula"] = f"  {accepted.loc[0, 'formula']}  "
    accepted.loc[0, "jvasp_id"] = f"  {accepted.loc[0, 'jvasp_id']}  "
    accepted.loc[0, "material_label"] = f"  {accepted.loc[0, 'material_label']}  "
    path = tmp_path / "accepted.csv"
    accepted.to_csv(path, index=False)
    loaded = analysis.load_fixture(path, require_pinned_checksum=False)
    assert loaded.loc[0, "formula"] == frame.loc[0, "formula"]
    assert loaded.loc[0, "jvasp_id"] == frame.loc[0, "jvasp_id"]

    for index, (column, value) in enumerate(
        (
            ("formula", "Ti O2"),
            ("formula", "   "),
            ("jvasp_id", "JVASP 1"),
            ("material_label", "   "),
        )
    ):
        invalid = frame.copy()
        invalid.loc[0, column] = value
        path = tmp_path / f"invalid-{index}.csv"
        invalid.to_csv(path, index=False)
        with pytest.raises(ValueError):
            analysis.load_fixture(path, require_pinned_checksum=False)


def test_formula_logo_roles_hold_all_polymorphs_together() -> None:
    frame = fixture()
    roles, validation = logo_role_manifest(
        frame, protocol_config={"protocol": "test", "seed": RESAMPLING_SEED}
    )
    assert validation["all_passed"]
    assert roles[roles["row_role"] == "heldout"].groupby("jvasp_id").size().eq(1).all()

    predictions = analysis.grouped_log_scale_predictions(frame, analysis.make_grid("quick"))
    for formula in ("AlN", "CdS", "GaN", "TiO2"):
        part = predictions[predictions["heldout_formula"] == formula]
        assert part["jvasp_id"].nunique() == 2
        assert set(part["n_train"]) == {52}
    assert predictions.groupby(["jvasp_id", "model"]).size().eq(1).all()
    assert {"coverage50", "coverage90"}.issubset(predictions)
    assert set(predictions["quantile_cdf_method"]) == {"exact"}


def test_physical_density_and_heldout_outcome_sentinels() -> None:
    frame = fixture()
    density = analysis.physical_density_jacobian_audit(frame, analysis.make_grid("quick"))
    leakage = analysis.heldout_outcome_leakage_sentinel(frame, analysis.make_grid("quick"))
    assert density["all_passed"]
    assert density["graph_absolute_error"] < 1e-10
    assert density["identity_absolute_error"] < 1e-10
    assert leakage["all_passed"]
    assert leakage["n_heldout_rows_perturbed"] == len(frame)


def test_geometry_pushforwards_and_rejected_old_reciprocal_route() -> None:
    frame = fixture()
    x = frame["bandgap_opt_ev"].to_numpy(float)
    y = frame["bandgap_expt_ev"].to_numpy(float)
    sigma = np.sqrt(0.12**2 + (0.15 * x) ** 2)
    audit = build_geometry_audit(x, y, sigma, mu_sd=0.75, local_bounds=(-1.6, 1.6))
    stats = audit["stats"]
    assert stats["all_passed"]
    assert stats["n_parameter_points"] == 25
    assert stats["direct_vs_log_centered_max_abs_error"] < 1e-8
    assert stats["reciprocal_vs_log_centered_max_abs_error"] < 1e-8
    assert stats["corrected_hard_vs_graph_max_abs_error"] < 1e-10
    assert stats["double_counting_failure_centered_max_abs_error"] > 1e-3

    rejected = analysis.rejected_reciprocal_diagnostic(frame)
    assert rejected["status"] == "rejected_not_an_active_target"
    assert rejected["finite_tube_proper"] is False
    assert rejected["all_reference_inputs_finite_positive"] is True
    assert rejected["formal_naive_hard_computed_globally"] is True
    assert rejected["formal_naive_hard_n_grid_points"] == 28001
    assert rejected["formal_naive_hard_mode_mu"] < -30.0
    assert rejected["formal_naive_hard_remote_minus_local_max_logdensity"] > 30.0
    assert rejected["formal_naive_hard_lower_endpoint_log_drop"] > 30.0
    assert rejected["formal_naive_hard_upper_endpoint_log_drop"] > 30.0


def test_detcorr_api_matches_locked_log_residual() -> None:
    x = fixture()["bandgap_opt_ev"].to_numpy(float)
    audit = analysis.detcorr_api_audit(1.6, x)
    assert audit["absolute_error"] < 1e-12


def test_manifest_has_governance_lineage_and_checksum_binding() -> None:
    data = STUDY / "data" / "jarvis_bandgap_benchmark_fixture.csv"
    manifest = analysis.load_data_manifest(
        STUDY / "data" / "FIXTURE_MANIFEST.json",
        analysis.sha256(data),
        data_path=data,
        frame=fixture(),
    )
    assert manifest["processed_sha256"] == analysis.sha256(data)
    assert manifest["protocol_version"] == "0.2-retrospective"
    assert manifest["evidence_timing_class"] == "registered_reanalysis"
    assert manifest["protocol_timing"] == "post_outcome_retrospective"


def test_manifest_declarations_are_reconciled_to_selected_input(tmp_path: Path) -> None:
    data = STUDY / "data" / "jarvis_bandgap_benchmark_fixture.csv"
    source = json.loads((STUDY / "data" / "FIXTURE_MANIFEST.json").read_text())
    invalid_values = {
        "processed_filename": "wrong.csv",
        "processed_bytes": source["processed_bytes"] + 1,
        "row_count": source["row_count"] + 1,
        "n_rows": source["n_rows"] + 1,
        "n_formula_groups": source["n_formula_groups"] + 1,
        "dependent_observation_columns": ["bandgap_opt_ev"],
        "reference_input_columns": ["bandgap_expt_ev"],
        "units": {"bandgap_opt_ev": "GPa", "bandgap_expt_ev": "GPa"},
    }
    for index, (field, value) in enumerate(invalid_values.items()):
        payload = {**source, field: value}
        path = tmp_path / f"invalid-manifest-{index}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match="declarations do not match"):
            analysis.load_data_manifest(
                path,
                analysis.sha256(data),
                data_path=data,
                frame=fixture(),
            )


def test_threshold_analysis_is_predictive_not_posterior_only() -> None:
    frame = fixture()
    grid = analysis.make_grid("quick")
    full = analysis.grouped_log_scale_predictions(frame, grid)
    result = analysis.domain_predictive_sensitivity(
        frame, grid, bootstrap_replicates=199, full_predictions=full
    )
    evidence = result["evidence"]
    scores = result["scores"]
    assert evidence["minimum_opt_gap_ev"].tolist() == [0.0, 0.25, 0.5, 1.0]
    assert evidence["bootstrap_seed"].eq(RESAMPLING_SEED).all()
    assert evidence["bootstrap_replicates"].eq(199).all()
    assert (evidence["aggregate_delta_elpd"] > 0.0).all()
    assert scores.groupby("minimum_opt_gap_ev")["model"].nunique().eq(2).all()
    assert np.isfinite(
        scores[
            [
                "rmse_point_prediction_ev",
                "mae_point_prediction_ev",
                "coverage50",
                "coverage90",
                "mean_interval50_width_ev",
                "mean_interval90_width_ev",
            ]
        ].to_numpy(float)
    ).all()


def test_quick_run_is_nonclaim_diagnostic_with_exact_terminal_results(
    quick_output: tuple[dict, Path],
) -> None:
    summary, output = quick_output
    assert summary["claim_bearing_full_run"] is False
    assert summary["implementation_checks"]["all_passed"] is True
    claims = {row["claim_id"]: row for row in summary["claim_evaluations"]}
    assert {row["status"] for row in claims.values()} == {"EXPLORATORY"}
    assert claims["JG-PRED"]["diagnostic_gate_status"] == "MIXED"
    assert claims["JG-GEOM"]["diagnostic_gate_status"] == "SUPPORTED"
    assert claims["JG-THRESHOLD"]["diagnostic_gate_status"] == "SUPPORTED"
    assert claims["JG-PRED"]["gates"]["affine_screen_does_not_beat_scale"] is False
    assert claims["JG-PRED"]["gates"]["direct_scale_screen_rmse_improves_identity"] is True
    assert claims["JG-PRED"]["gates"]["predictive_grid_refinement_passes"] is True
    assert claims["JG-PRED"]["gates"]["coverage50_90_and_widths_are_emitted"] is True
    concentration = claims["JG-PRED"]["primary_result"][
        "maximum_absolute_formula_contribution_fraction"
    ]
    assert 0.0 < concentration < 0.5
    assert (
        claims["JG-THRESHOLD"]["gates"]["every_threshold_graph_and_identity_support_audit_passes"]
        is True
    )
    assert claims["JG-THRESHOLD"]["gates"]["every_threshold_role_manifest_passes"] is True
    assert (
        claims["JG-THRESHOLD"]["gates"]["each_threshold_reports_rmse_mae_coverage50_90_and_widths"]
        is True
    )
    assert (
        claims["JG-THRESHOLD"]["gates"][
            "every_threshold_bootstrap_interval_supports_full_fixture_sign"
        ]
        is True
    )
    assert (
        claims["JG-THRESHOLD"]["gates"]["every_threshold_heldout_outcome_leakage_sentinel_passes"]
        is True
    )
    assert all(row["claim_bearing_full_run"] is False for row in claims.values())
    assert summary["direct_model_ladder_negative_q05_counts"] == {
        "affine_direct": 7,
        "identity": 42,
        "scale_direct": 17,
    }
    assert any("fundamental-versus-optical" in item for item in summary["limitations"])
    assert any("negative-gap tails" in item for item in summary["limitations"])
    assert (
        summary["protocol"]["configuration"]["claim_rules"][
            "threshold_bootstrap90_must_exclude_zero_in_full_fixture_direction"
        ]
        is True
    )
    assert {
        "tables/direct_model_ladder_predictions.csv",
        "tables/heldout_outcome_leakage_audit.csv",
        "tables/influence_and_residuals.csv",
        "tables/logo_role_validation.csv",
        "tables/pairing_permutation_distribution.csv",
    }.issubset({item["path"] for item in claims["JG-PRED"]["evidence_artifacts"]})
    assert "tables/detcorr_api_audit.csv" in {
        item["path"] for item in claims["JG-GEOM"]["evidence_artifacts"]
    }
    assert {
        "tables/physical_density_jacobian_audit.csv",
        "tables/predictive_grid_refinement_details.csv",
    }.issubset({item["path"] for item in claims["JG-THRESHOLD"]["evidence_artifacts"]})

    posterior = pd.read_csv(output / "tables" / "posterior_targets.csv")
    assert posterior["target"].tolist() == ["graph_primary"]
    assert "corrected" not in summary["posterior"]
    threshold = pd.read_csv(output / "tables" / "opt_gap_threshold_predictive_evidence.csv")
    assert len(threshold) == 4
    threshold_support = pd.read_csv(output / "tables" / "opt_gap_threshold_grid_support_audit.csv")
    assert set(threshold_support["model"]) == {"graph_primary", "identity"}
    assert set(threshold_support["minimum_opt_gap_ev"]) == {0.0, 0.25, 0.5, 1.0}
    refinement = pd.read_csv(output / "tables" / "predictive_grid_refinement_gates.csv")
    assert refinement["passed"].all()
    contributions = pd.read_csv(output / "tables" / "predictive_group_elpd_contributions.csv")
    assert np.isclose(contributions["fraction_total_absolute_delta_elpd"].sum(), 1.0)
    assert np.isclose(contributions["fraction_total_absolute_delta_elpd"].max(), concentration)
    permutation = pd.read_csv(output / "tables" / "pairing_permutation_distribution.csv")
    assert set(permutation["seed"]) == {RESAMPLING_SEED}
    assert permutation["is_at_or_below_observed"].dtype == bool
    for claim_id in ("JG-PRED", "JG-GEOM"):
        assert any(
            item["artifact_type"] == "diagnostic_figure"
            for item in claims[claim_id]["evidence_artifacts"]
        )
    assert all(
        item["artifact_type"] == "table" for item in claims["JG-THRESHOLD"]["evidence_artifacts"]
    )

    for claim in claims.values():
        assert set(claim["source_hashes"]) >= {
            "study_analysis",
            "common_geometry",
            "common_protocol",
            "common_evidence",
            "protocol_lock",
        }
        assert all(len(value) == 64 for value in claim["source_hashes"].values())
        for artifact in claim["evidence_artifacts"]:
            assert (output / artifact["path"]).is_file()
