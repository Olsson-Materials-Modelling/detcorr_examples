from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import integrate, stats

STUDY_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = STUDY_ROOT.parents[1]
for path in (STUDY_ROOT, REPOSITORY_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import fulton_analysis as analysis  # noqa: E402
import run_analysis as entrypoint  # noqa: E402


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return analysis.load_fish(STUDY_ROOT / "data" / "fish.csv")


def test_registered_fixture_checksum_schema_and_cli(frame: pd.DataFrame) -> None:
    assert analysis.sha256_file(STUDY_ROOT / "data" / "fish.csv") == analysis.DATA_SHA256
    assert len(frame) == 97
    assert list(frame["t"]) == list(range(1, 98))
    args = entrypoint.build_parser().parse_args([])
    assert args.data == STUDY_ROOT / "data" / "fish.csv"
    assert args.output_root == REPOSITORY_ROOT / "outputs" / "fulton_market"
    assert args.full is False and args.quick is False


def test_data_manifest_is_required_and_hash_bound(frame: pd.DataFrame, tmp_path: Path) -> None:
    manifest_path = STUDY_ROOT / "data" / "FIXTURE_MANIFEST.json"
    manifest = analysis.load_data_manifest(
        manifest_path,
        data_path=analysis.DATA_PATH,
        frame=frame,
    )
    assert manifest["sha256"] == analysis.DATA_SHA256
    assert manifest["row_count"] == len(frame)

    changed = dict(manifest)
    changed["sha256"] = "0" * 64
    invalid = tmp_path / "invalid-manifest.json"
    invalid.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA256"):
        analysis.load_data_manifest(invalid, data_path=analysis.DATA_PATH, frame=frame)

    copied_data = tmp_path / "fish-copy.csv"
    copied_data.write_bytes(analysis.DATA_PATH.read_bytes())
    with pytest.raises(SystemExit):
        entrypoint.main(["--data", str(copied_data)])
    protocol, protocol_sha = analysis.load_and_validate_protocol()
    assert protocol["version"] == 2
    assert protocol["dependence"]["phi_bound_sensitivity"]["absolute_limits"] == [
        0.5,
        0.75,
        0.9,
        0.99,
    ]
    assert protocol["score_uncertainty"]["seed"] == 20260711
    assert protocol["score_uncertainty"]["interval_quantiles"] == [0.05, 0.95]
    assert len(protocol_sha) == 64


@pytest.mark.parametrize(
    ("field_path", "changed_value"),
    [
        (("calibration_window",), 15),
        (("prior_parameters", "eta_sd"), 0.6),
        (("score_uncertainty", "full_replicates"), 4999),
        (("registered_claim_ids",), ["FU-PRED", "FU-DEP"]),
        (("dependence", "primary"), "iid"),
        (("dependence", "phi_bound_sensitivity", "absolute_limits"), [0.5, 0.9]),
        (("baselines",), ["last_price_persistence_baseline"]),
        (("geometry_audit", "rho"), 100000.0),
        (("matched_identity_models", "iid"), "wrong_identity"),
        (("test_information", "price_outcome_used_before_scoring"), True),
        (("sensitivities",), ["iid log-price errors"]),
    ],
)
def test_protocol_validation_rejects_all_claim_bearing_drift(
    field_path: tuple[str, ...], changed_value: object, tmp_path: Path
) -> None:
    protocol = json.loads(analysis.PROTOCOL_PATH.read_text(encoding="utf-8"))
    target = protocol
    for key in field_path[:-1]:
        target = target[key]
    target[field_path[-1]] = changed_value
    changed = tmp_path / "REGISTERED_PROTOCOL.json"
    changed.write_text(json.dumps(protocol), encoding="utf-8")
    with pytest.raises(ValueError, match="REGISTERED_PROTOCOL"):
        analysis.load_and_validate_protocol(changed)


def test_loader_refuses_silent_sorting_missing_values_and_nonpositive_prices(
    frame: pd.DataFrame, tmp_path: Path
) -> None:
    reversed_rows = frame.iloc[::-1]
    reverse_path = tmp_path / "reverse.csv"
    reversed_rows.to_csv(reverse_path, index=False)
    with pytest.raises(ValueError, match="strictly increasing"):
        analysis.load_fish(reverse_path)

    missing = frame.copy()
    missing.loc[3, "wave2"] = np.nan
    missing_path = tmp_path / "missing.csv"
    missing.to_csv(missing_path, index=False)
    with pytest.raises(ValueError, match="missing values"):
        analysis.load_fish(missing_path)

    nonpositive = frame.copy()
    nonpositive.loc[3, "avgprc"] = 0.0
    nonpositive_path = tmp_path / "nonpositive.csv"
    nonpositive.to_csv(nonpositive_path, index=False)
    with pytest.raises(ValueError, match="strictly positive"):
        analysis.load_fish(nonpositive_path)


def test_eta_posterior_matches_direct_grid_and_naive_tilt() -> None:
    reference = np.array([-0.2, 0.1, 0.35, 0.28])
    observed = np.array([-0.1, 0.04, 0.17, 0.19])
    covariance = analysis.ar1_covariance(4, sigma=0.22, phi=0.37)
    graph = analysis.eta_posterior(reference, observed, covariance)
    naive = analysis.eta_posterior(reference, observed, covariance, omit_state_volume=True)
    assert naive.mean == pytest.approx(graph.mean - 4 * graph.variance)
    assert naive.variance == pytest.approx(graph.variance)

    eta = np.linspace(graph.mean - 8 * graph.sd, graph.mean + 8 * graph.sd, 20001)
    precision = np.linalg.inv(covariance)
    q = reference - observed
    logdensity = -0.5 * eta**2 / analysis.PRIOR_ETA_SD**2
    residual = q[None, :] - eta[:, None]
    logdensity -= 0.5 * np.einsum("ni,ij,nj->n", residual, precision, residual)
    density = np.exp(logdensity - np.max(logdensity))
    density /= analysis.trapezoid(density, eta)
    numeric_mean = analysis.trapezoid(eta * density, eta)
    numeric_variance = analysis.trapezoid((eta - numeric_mean) ** 2 * density, eta)
    assert numeric_mean == pytest.approx(graph.mean, abs=2.0e-10)
    assert numeric_variance == pytest.approx(graph.variance, abs=2.0e-10)


def test_lognormal_crps_matches_direct_cdf_integral() -> None:
    prediction = analysis.LognormalPrediction(log_mean=-0.16, log_sd=0.31)
    observed = 0.93
    distribution = stats.lognorm(s=prediction.log_sd, scale=math.exp(prediction.log_mean))
    lower, _ = integrate.quad(lambda z: distribution.cdf(z) ** 2, 0.0, observed)
    upper, _ = integrate.quad(lambda z: distribution.sf(z) ** 2, observed, np.inf, epsabs=1.0e-11)
    assert analysis.lognormal_crps(observed, prediction) == pytest.approx(
        lower + upper, abs=2.0e-10
    )


def test_residual_and_parameter_pushforwards_are_equivalent(frame: pd.DataFrame) -> None:
    table, summary = analysis.residual_pushforward_audit(33)
    assert summary["max_abs_correction_error"] < 1.0e-14
    assert summary["max_abs_penalty_error"] < 1.0e-14
    assert np.any(np.abs(table["unpushed_metric_changes_log_correction_by"]) > 1.0)
    parameter = analysis.parameter_pushforward_audit(frame, n_grid=1601)
    assert parameter["lambda_to_eta_to_lambda_total_variation"] < 1.0e-12
    assert parameter["lambda_mean_direct"] == pytest.approx(
        parameter["lambda_mean_via_eta"], abs=2.0e-7
    )


def test_primary_model_positive_state_geometry_gate(frame: pd.DataFrame) -> None:
    table, audit = analysis.primary_model_geometry_audit(frame, full=False)
    assert len(table) == 25
    assert audit["claim_bearing_model"].startswith("rolling primary AR(1)")
    assert audit["exact_gaussian_verification_used_as_claim_evidence"] is False
    assert audit["max_abs_log_J_over_q_coordinate_error"] < 1.0e-12
    assert (
        audit["finite_vs_graph_error_fraction_of_determinant_variation"]
        < audit["registered_max_error_fraction"]
    )
    assert (
        audit["qmc_error_fraction_of_determinant_variation"]
        < audit["registered_max_error_fraction"]
    )
    assert audit["support_certificate"]["finite_tube_integrable_for_every_lambda_positive"]
    assert audit["support_certificate"]["global_eta_target_normalizable"]
    assert audit["passed"] is True
    np.testing.assert_allclose(
        table["direct_log_J_over_q"],
        table["log_state_log_J_over_q"],
        rtol=0.0,
        atol=1.0e-12,
    )


def test_exact_state_integration_verification(frame: pd.DataFrame) -> None:
    table, summary, _ = analysis.exact_verification_spine(frame, n_grid=2401)
    indexed = table.set_index("target")
    assert summary["n_train"] == 64
    assert summary["n_calibration_states"] == 33
    assert indexed.loc["graph_reduced", "mean"] == pytest.approx(0.755391, abs=2.0e-5)
    assert summary["naive_shift_graph_sd"] < -1.5
    assert summary["naive_total_variation_vs_graph"] > 0.5
    assert summary["corrected_total_variation_vs_graph"] < 1.0e-6


def test_rolling_roles_are_disjoint_and_test_outcome_is_untouched(
    frame: pd.DataFrame,
) -> None:
    original = analysis._model_prediction_rows(frame)
    changed = frame.copy()
    changed.loc[64, "avgprc"] *= 1.9
    perturbed = analysis._model_prediction_rows(changed)

    first_original = original.loc[original["test_t"] == 65].sort_values("model")
    first_perturbed = perturbed.loc[perturbed["test_t"] == 65].sort_values("model")
    predictive_columns = [
        "predictive_log_mean",
        "predictive_log_sd",
        "predictive_mean",
        "predictive_median",
        "interval_lower",
        "interval_upper",
        "eta_posterior_mean",
        "eta_posterior_sd",
    ]
    np.testing.assert_allclose(
        first_original[predictive_columns].to_numpy(dtype=float),
        first_perturbed[predictive_columns].to_numpy(dtype=float),
        equal_nan=True,
        rtol=0.0,
        atol=0.0,
    )
    assert not np.allclose(first_original["log_score"], first_perturbed["log_score"])
    assert list(first_original["prediction_sha256_before_test_score"]) == list(
        first_perturbed["prediction_sha256_before_test_score"]
    )
    assert np.all(original["reference_train_t_end"] < original["calibration_t_start"])
    assert np.all(original["calibration_t_end"] < original["test_t"])
    counts = original.groupby("model", observed=True)["test_t"].nunique()
    assert set(counts) == {33}
    assert set(analysis.MODEL_ORDER) == set(counts.index.astype(str))


def test_role_manifest_is_complete_and_disjoint(frame: pd.DataFrame) -> None:
    manifest, assertions = analysis.build_role_manifest(frame)
    assert all(value for value in assertions.values() if isinstance(value, bool))
    assert set(manifest["role"]) == {"train", "calibration", "test"}
    tests = manifest.loc[manifest["role"] == "test"]
    assert list(tests["source_t"]) == list(range(65, 98))
    calibration_counts = manifest.groupby("origin_test_t")["role"].apply(
        lambda values: (values == "calibration").sum()
    )
    assert set(calibration_counts) == {16}


def test_matched_identity_effects_use_locked_resamples(frame: pd.DataFrame) -> None:
    predictions = analysis._model_prediction_rows(frame)
    matched = analysis.matched_calibration_effects(predictions, bootstrap_replicates=500)
    assert set(matched["assumption_family"]) == {
        "ar1_plugin",
        "iid",
        "ar1_beta_uncertainty",
    }
    assert set(matched["bootstrap_seed"]) == {20260711}
    assert set(matched["interval_level"]) == {0.9}
    ar1 = matched.set_index("assumption_family").loc["ar1_plugin"]
    iid = matched.set_index("assumption_family").loc["iid"]
    assert ar1["moving_block_90pct_mean_delta_elpd_lower"] < 0.0
    assert iid["moving_block_90pct_mean_delta_elpd_lower"] > 0.0


def test_phi_clip_limit_validation_and_fixture_activity(frame: pd.DataFrame) -> None:
    for invalid in (-0.1, 0.0, 1.0, 1.1, np.inf, np.nan):
        with pytest.raises(ValueError, match="phi_limit"):
            analysis.fit_reference(frame.iloc[:48], response="log", phi_limit=invalid)

    primary = analysis.fit_reference(frame.iloc[:48], response="log")
    stressed = analysis.fit_reference(frame.iloc[:48], response="log", phi_limit=0.5)
    assert primary.phi_raw == pytest.approx(0.6387567747)
    assert primary.phi_used == primary.phi_raw
    assert stressed.phi_raw == primary.phi_raw
    assert stressed.phi_used == 0.5


def test_phi_clip_sensitivity_is_one_axis_nonselecting_and_stable(
    frame: pd.DataFrame,
) -> None:
    predictions = analysis._model_prediction_rows(frame)
    matched = analysis.matched_calibration_effects(predictions, bootstrap_replicates=500)
    table, summary = analysis.phi_clip_sensitivity(
        frame,
        predictions,
        matched,
        bootstrap_replicates=500,
    )
    assert tuple(table["phi_absolute_limit"]) == analysis.PHI_CLIP_SENSITIVITY_LIMITS
    assert list(table["n_origins_clipped"]) == [33, 0, 0, 0]
    assert table["non_claim_selecting"].all()
    assert table["post_outcome_nonselecting_addition"].all()
    assert table["predictive_gate_passed"].eq(False).all()
    assert table["dependence_sensitive"].eq(True).all()
    assert summary["primary_n_origins_clipped"] == 0
    assert summary["all_predictive_gate_decisions_match_primary"] is True
    assert summary["all_dependence_sensitivity_categories_match_primary"] is True
    assert summary["maximum_abs_eta_change_vs_primary_in_primary_sd"] < 0.13
    assert summary["scientific_outcomes_are_not_implementation_pass_fail_checks"] is True

    primary = table.set_index("phi_absolute_limit").loc[0.9]
    for inactive_limit in (0.75, 0.99):
        inactive = table.set_index("phi_absolute_limit").loc[inactive_limit]
        for column in (
            "cumulative_delta_elpd_graph_minus_identity",
            "moving_block_90pct_mean_delta_elpd_lower",
            "moving_block_90pct_mean_delta_elpd_upper",
            "crps_identity_minus_graph",
            "cumulative_delta_elpd_graph_minus_naive",
        ):
            assert inactive[column] == primary[column]

    with pytest.raises(ValueError, match="include the primary bound"):
        analysis.phi_clip_sensitivity(
            frame,
            predictions,
            matched,
            bootstrap_replicates=10,
            limits=(0.5, 0.99),
        )


def test_full_claim_statuses_are_terminal_and_bounded(frame: pd.DataFrame) -> None:
    predictions = analysis._model_prediction_rows(frame)
    matched = analysis.matched_calibration_effects(predictions, bootstrap_replicates=5000)
    _, primary_geometry = analysis.primary_model_geometry_audit(frame, full=False)
    claims = analysis.evaluate_claims(
        predictions,
        matched,
        primary_geometry=primary_geometry,
        geometry_checks_passed=True,
        bootstrap_replicates=5000,
        full=True,
        registered_fixture=True,
    )
    indexed = {claim["claim_id"]: claim for claim in claims}
    assert indexed["FU-PRED"]["status"] == "NOT_SUPPORTED"
    assert indexed["FU-DEP"]["status"] == "SUPPORTED"
    assert indexed["FU-DEP"]["conclusion"] == "DEPENDENCE_SENSITIVE"
    assert indexed["FU-GEOM"]["status"] == "SUPPORTED"
    assert indexed["FU-GEOM"]["conclusion"] == "QUANTIFIED"
    assert indexed["FU-GEOM"]["max_abs_eta_shift_in_graph_sd"] > 0.0


def test_study_runner_refuses_completed_output_without_force(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "completed"
    output.mkdir()
    (output / "summary.json").write_text("{}\n", encoding="utf-8")
    sentinel = output / "stale-sentinel.txt"
    sentinel.write_text("must not survive force\n", encoding="utf-8")
    calls: list[bool] = []

    def fake_run(*args, **kwargs):
        calls.append(True)
        return {
            "study": "fulton_market",
            "mode": "quick",
            "analysis_status": "registered_retrospective_reanalysis",
            "implementation_checks": {"all_passed": True},
        }

    monkeypatch.setattr(entrypoint, "run_analysis", fake_run)
    with pytest.raises(SystemExit, match="output directory is not empty"):
        entrypoint.main(["--quick", "--output-root", str(output)])
    assert calls == []
    assert sentinel.is_file()

    with pytest.raises(SystemExit, match="unmanaged directory"):
        entrypoint.main(["--quick", "--force", "--output-root", str(output)])
    assert calls == []
    assert sentinel.is_file()

    managed = tmp_path / "managed"
    entrypoint.main(["--quick", "--output-root", str(managed)])
    assert calls == [True]
    managed_sentinel = managed / "stale-sentinel.txt"
    managed_sentinel.write_text("replace the whole managed run\n", encoding="utf-8")
    entrypoint.main(["--quick", "--force", "--output-root", str(managed)])
    assert calls == [True, True]
    assert managed.is_dir()
    assert not managed_sentinel.exists()


def test_time_preserving_placebo_uses_only_lagged_rows(frame: pd.DataFrame) -> None:
    x = analysis.design_matrix(frame)
    first_test = analysis.FIRST_TEST_POSITION
    start = first_test - analysis.CALIBRATION_WINDOW
    placebo_cal = x[start - analysis.PLACEBO_LAG : first_test - analysis.PLACEBO_LAG]
    placebo_test = x[first_test - analysis.PLACEBO_LAG]
    np.testing.assert_array_equal(placebo_cal[0], x[44])
    np.testing.assert_array_equal(placebo_cal[-1], x[59])
    np.testing.assert_array_equal(placebo_test, x[60])
    assert first_test - analysis.PLACEBO_LAG < first_test


def test_callable_detcorr_matches_analytic_diagonal_jacobian() -> None:
    audit = analysis.detcorr_callable_audit()
    if not audit.get("available"):
        if os.environ.get("DETCORR_CASE_STUDIES_REQUIRE_DETCORR") == "1":
            pytest.fail(f"detcorr unavailable: {audit.get('reason')}")
        pytest.skip(f"detcorr unavailable: {audit.get('reason')}")
    assert audit["target_measure"] == "graph"
    assert audit["version_is_at_least_0_10"] is True
    assert audit["absolute_error"] < 2.0e-6


def test_run_writes_strict_deterministic_output_contract(
    frame: pd.DataFrame, tmp_path: Path
) -> None:
    output = tmp_path / "outputs"
    summary = analysis.run_analysis(output, data_path=STUDY_ROOT / "data" / "fish.csv")
    assert summary["analysis_status"] == "registered_retrospective_reanalysis"
    assert summary["claim_bearing_full_run"] is False
    assert summary["scientific_outcomes_are_not_pass_fail_checks"] is True
    assert summary["registered_protocol"]["calibration_window"] == 16
    assert summary["reproducibility"]["registered_protocol_version"] == 2
    assert summary["primary_predictive_metrics"]["n_forecasts"] == 33
    assert summary["input"]["manifest"]["sha256"] == analysis.DATA_SHA256
    if summary["residual_and_coordinate_audits"]["detcorr_callable"].get("available"):
        assert summary["implementation_checks"]["all_passed"] is True
    claims = {row["claim_id"]: row for row in summary["claim_evaluations"]}
    assert claims["FU-PRED"]["status"] == "NOT_SUPPORTED"
    assert claims["FU-DEP"]["status"] == "MIXED"
    assert claims["FU-DEP"]["conclusion"] == "DEPENDENCE_SENSITIVE"
    assert claims["FU-GEOM"]["status"] == "MIXED"
    assert claims["FU-GEOM"]["conclusion"] == "QUANTIFIED"
    allowed_statuses = {"SUPPORTED", "MIXED", "NOT_SUPPORTED", "BLOCKED", "EXPLORATORY"}
    assert {row["status"] for row in claims.values()} <= allowed_statuses
    assert all(row["status"] != "UNTESTED" for row in claims.values())
    for claim in claims.values():
        assert claim["claim_bearing_full_run"] is False
        assert claim["timing_class"] == "registered_reanalysis"
        assert claim["evidence_level"] == "E0_nonclaim_quick"
        assert len(claim["model_config_sha256"]) == 64
        assert claim["data_sha256"] == analysis.DATA_SHA256
        assert set(claim["source_hashes"]) == {
            "fulton_analysis.py",
            "run_analysis.py",
            "REGISTERED_PROTOCOL.json",
            "common/evidence.py",
        }
        for artifact in claim["evidence_artifacts"]:
            path = output / artifact["path"]
            assert path.is_file()
            assert analysis.sha256_file(path) == artifact["sha256"]
        assert claim["phi_clip_sensitivity"]["outcomes_known_when_added"] is True
        assert claim["phi_clip_sensitivity"]["primary_n_origins_clipped"] == 0
        assert any(
            artifact["path"] == "tables/phi_clip_sensitivity.csv"
            for artifact in claim["evidence_artifacts"]
        )
    assert len(list((output / "tables").glob("*.csv"))) == 13
    assert len(list((output / "figures").glob("*.png"))) == 4
    assert (output / "claims" / "claim_evaluations.json").is_file()
    reproducibility = summary["reproducibility"]
    assert len(reproducibility["registered_protocol_sha256"]) == 64
    assert len(reproducibility["configuration_sha256"]) == 64
    assert len(reproducibility["role_manifest_sha256"]) == 64
    assert reproducibility["random_seed"] == 20260711
    assert reproducibility["environment"]["python"]
    prescore = pd.read_csv(output / "tables" / "pre_score_predictions.csv")
    scored = pd.read_csv(output / "tables" / "rolling_predictions.csv")
    assert sorted(prescore["prediction_sha256"]) == sorted(
        scored["prediction_sha256_before_test_score"]
    )
    diagnostics = pd.read_csv(output / "tables" / "initial_reference_residual_diagnostics.csv")
    names = set(diagnostics["diagnostic"])
    assert "stationary_marginal_log_residual_sd_plugin" in names
    assert "derived_log_innovation_sd_plugin" in names
    sensitivity = pd.read_csv(output / "tables" / "phi_clip_sensitivity.csv")
    assert list(sensitivity["n_origins_clipped"]) == [33, 0, 0, 0]
    assert summary["phi_clip_sensitivity"]["primary_clipping_active"] is False
    assert summary["implementation_checks"]["phi_clip_sensitivity_contract_complete"] is True
    geometry = summary["residual_and_coordinate_audits"]["primary_model_direct_log_and_finite_tube"]
    assert geometry["passed"] is True
    text = (output / "summary.json").read_text(encoding="utf-8")
    parsed = json.loads(
        text,
        parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
    )
    assert parsed["input"]["sha256"] == analysis.DATA_SHA256
    assert parsed["input"]["path"] == "studies/fulton_market/data/fish.csv"
    assert "NaN" not in text and "Infinity" not in text
