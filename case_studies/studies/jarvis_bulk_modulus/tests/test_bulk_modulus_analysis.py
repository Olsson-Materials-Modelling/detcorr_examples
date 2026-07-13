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

import bulk_modulus_analysis as analysis  # noqa: E402

from common.jarvis_geometry import build_geometry_audit  # noqa: E402
from common.jarvis_protocol import RESAMPLING_SEED, logo_role_manifest  # noqa: E402


def fixture() -> pd.DataFrame:
    return analysis.load_fixture(STUDY / "data" / "jarvis_bulk_modulus_benchmark_fixture.csv")


@pytest.fixture(scope="module")
def quick_output(tmp_path_factory: pytest.TempPathFactory) -> tuple[dict, Path]:
    output = tmp_path_factory.mktemp("bulk-quick")
    data = STUDY / "data" / "jarvis_bulk_modulus_benchmark_fixture.csv"
    summary = analysis.run(
        data,
        output,
        "quick",
        99,
        data_manifest_path=STUDY / "data" / "FIXTURE_MANIFEST.json",
    )
    return summary, output


def test_fixture_identity_group_and_aggregate_check() -> None:
    frame = fixture()
    assert len(frame) == 36
    assert frame["formula"].nunique() == 35
    mae = np.mean(np.abs(frame["bulk_modulus_expt_gpa"] - frame["bulk_modulus_opt_gpa"]))
    assert np.isclose(mae, 8.505555555555555)


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
    assert table.loc["narrower_mu_prior", "mu_prior_sd"] == 0.20
    assert table.loc["wider_mu_prior", "mu_prior_sd"] == 0.60


def test_custom_loader_normalizes_edges_and_rejects_ambiguous_keys(tmp_path: Path) -> None:
    frame = fixture()
    accepted = frame.copy()
    accepted.loc[0, "formula"] = f"  {accepted.loc[0, 'formula']}  "
    accepted.loc[0, "jvasp_id"] = f"  {accepted.loc[0, 'jvasp_id']}  "
    accepted.loc[0, "material"] = f"  {accepted.loc[0, 'material']}  "
    path = tmp_path / "accepted.csv"
    accepted.to_csv(path, index=False)
    loaded = analysis.load_fixture(path, require_pinned_checksum=False)
    assert loaded.loc[0, "formula"] == frame.loc[0, "formula"]
    assert loaded.loc[0, "jvasp_id"] == frame.loc[0, "jvasp_id"]

    for column, value in (
        ("formula", "Ti O2"),
        ("formula", "   "),
        ("jvasp_id", "JVASP 1"),
        ("material", "   "),
    ):
        invalid = frame.copy()
        invalid.loc[0, column] = value
        path = tmp_path / f"invalid-{column}-{len(value)}.csv"
        invalid.to_csv(path, index=False)
        with pytest.raises(ValueError):
            analysis.load_fixture(path, require_pinned_checksum=False)


def test_formula_logo_roles_and_predictions_do_not_split_tio2() -> None:
    frame = fixture()
    config = {"protocol": "test", "seed": RESAMPLING_SEED}
    roles, validation = logo_role_manifest(frame, protocol_config=config)
    assert validation["all_passed"]
    assert roles["protocol_hash"].nunique() == 1
    assert roles[roles["row_role"] == "heldout"].groupby("jvasp_id").size().eq(1).all()

    predictions = analysis.grouped_log_scale_predictions(frame, analysis.make_grid("quick"))
    tio2 = predictions[predictions["heldout_formula"] == "TiO2"]
    assert tio2["jvasp_id"].nunique() == 2
    assert set(tio2["n_train"]) == {34}
    assert predictions.groupby(["jvasp_id", "model"]).size().eq(1).all()
    assert {"coverage50", "coverage90"}.issubset(predictions)
    assert set(predictions["quantile_cdf_method"]) == {"exact"}


def test_physical_density_contains_minus_log_y_jacobian() -> None:
    audit = analysis.physical_density_jacobian_audit(fixture(), analysis.make_grid("quick"))
    assert audit["all_passed"]
    assert audit["graph_absolute_error"] < 1e-10
    assert audit["identity_absolute_error"] < 1e-10


def test_heldout_outcome_perturbation_cannot_change_fit_or_prediction() -> None:
    audit = analysis.heldout_outcome_leakage_sentinel(fixture(), analysis.make_grid("quick"))
    assert audit["all_passed"]
    assert audit["n_heldout_rows_perturbed"] == len(fixture())


def test_geometry_routes_are_computed_not_algebraically_asserted() -> None:
    frame = fixture()
    x = frame["bulk_modulus_opt_gpa"].to_numpy(float)
    y = frame["bulk_modulus_expt_gpa"].to_numpy(float)
    sigma = np.sqrt(3.0**2 + (0.055 * x) ** 2)
    audit = build_geometry_audit(x, y, sigma, mu_sd=0.35, local_bounds=(-0.8, 0.8))
    stats = audit["stats"]
    assert stats["all_passed"]
    assert stats["n_parameter_points"] == 25
    assert stats["direct_vs_log_centered_max_abs_error"] < 1e-8
    assert stats["reciprocal_vs_log_centered_max_abs_error"] < 1e-8
    assert stats["double_counting_failure_centered_max_abs_error"] > 1e-3
    assert set(audit["contracts"]["route"]) == {
        "log_genuine",
        "direct_genuine",
        "reciprocal_genuine",
        "direct_independent_lebesgue",
        "reciprocal_independent_lebesgue",
        "log_residual_rescaled_metric",
    }


def test_detcorr_api_matches_locked_log_residual() -> None:
    x = fixture()["bulk_modulus_opt_gpa"].to_numpy(float)
    audit = analysis.detcorr_api_audit(1.07, x)
    assert audit["absolute_error"] < 1e-12


def test_manifest_has_governance_lineage_and_checksum_binding() -> None:
    data = STUDY / "data" / "jarvis_bulk_modulus_benchmark_fixture.csv"
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
    data = STUDY / "data" / "jarvis_bulk_modulus_benchmark_fixture.csv"
    source = json.loads((STUDY / "data" / "FIXTURE_MANIFEST.json").read_text())
    invalid_values = {
        "processed_filename": "wrong.csv",
        "processed_bytes": source["processed_bytes"] + 1,
        "row_count": source["row_count"] + 1,
        "n_rows": source["n_rows"] + 1,
        "n_formula_groups": source["n_formula_groups"] + 1,
        "dependent_observation_columns": ["bulk_modulus_opt_gpa"],
        "reference_input_columns": ["bulk_modulus_expt_gpa"],
        "units": {"bulk_modulus_opt_gpa": "eV", "bulk_modulus_expt_gpa": "eV"},
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


def test_quick_run_is_nonclaim_diagnostic_with_exact_terminal_results(
    quick_output: tuple[dict, Path],
) -> None:
    summary, output = quick_output
    assert summary["claim_bearing_full_run"] is False
    assert summary["implementation_checks"]["all_passed"] is True
    claims = {row["claim_id"]: row for row in summary["claim_evaluations"]}
    assert claims["JB-PRED"]["status"] == "EXPLORATORY"
    assert claims["JB-PRED"]["diagnostic_gate_status"] == "NOT_SUPPORTED"
    assert claims["JB-GEOM"]["diagnostic_gate_status"] == "SUPPORTED"
    assert claims["JB-PRED"]["gates"]["aggregate_delta_elpd_positive"] is False
    assert claims["JB-PRED"]["gates"]["predictive_grid_refinement_passes"] is True
    assert claims["JB-PRED"]["gates"]["coverage50_90_and_widths_are_emitted"] is True
    concentration = claims["JB-PRED"]["primary_result"][
        "maximum_absolute_formula_contribution_fraction"
    ]
    assert 0.0 < concentration < 0.5
    assert all(row["timing_class"] == "verification" for row in claims.values())
    assert all(row["claim_bearing_full_run"] is False for row in claims.values())
    assert (
        summary["protocol"]["configuration"]["claim_rules"][
            "substantial_case_concentration_fraction"
        ]
        == 0.5
    )
    assert summary["direct_model_ladder_negative_q05_counts"] == {
        "affine_direct": 7,
        "identity": 7,
        "scale_direct": 7,
    }
    assert any("negative-modulus tails" in item for item in summary["limitations"])
    assert {
        "tables/direct_model_ladder_predictions.csv",
        "tables/heldout_outcome_leakage_audit.csv",
        "tables/influence_and_residuals.csv",
        "tables/logo_role_validation.csv",
        "tables/pairing_permutation_distribution.csv",
    }.issubset({item["path"] for item in claims["JB-PRED"]["evidence_artifacts"]})
    assert "tables/detcorr_api_audit.csv" in {
        item["path"] for item in claims["JB-GEOM"]["evidence_artifacts"]
    }

    posterior = pd.read_csv(output / "tables" / "posterior_targets.csv")
    assert posterior["target"].tolist() == ["graph_primary"]
    assert "corrected" not in summary["posterior"]
    support = pd.read_csv(output / "tables" / "primary_grid_support_audit.csv")
    assert set(support["model"]) == {"graph_primary", "identity"}
    refinement = pd.read_csv(output / "tables" / "predictive_grid_refinement_gates.csv")
    assert refinement["passed"].all()
    contributions = pd.read_csv(output / "tables" / "predictive_group_elpd_contributions.csv")
    assert np.isclose(contributions["fraction_total_absolute_delta_elpd"].sum(), 1.0)
    assert np.isclose(contributions["fraction_total_absolute_delta_elpd"].max(), concentration)
    permutation = pd.read_csv(output / "tables" / "pairing_permutation_distribution.csv")
    assert set(permutation["seed"]) == {RESAMPLING_SEED}
    assert permutation["is_at_or_below_observed"].dtype == bool

    for claim in claims.values():
        assert set(claim["source_hashes"]) >= {
            "study_analysis",
            "common_geometry",
            "common_protocol",
            "common_evidence",
            "protocol_lock",
        }
        assert all(len(value) == 64 for value in claim["source_hashes"].values())
        assert any(
            item["artifact_type"] == "diagnostic_figure" for item in claim["evidence_artifacts"]
        )
        for artifact in claim["evidence_artifacts"]:
            assert (output / artifact["path"]).is_file()

    on_disk = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert on_disk["implementation_checks"]["all_passed"] is True
