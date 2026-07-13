from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from examples import verify_examples as verifier
from examples import worked_examples_registry as registry

from detcorr_demo_common import (
    MINIMUM_DETCORR_VERSION,
    SUPPORTED_DETCORR_SPEC,
    UnsupportedDetcorrVersionError,
    require_supported_detcorr,
    validate_detcorr_version,
    write_json,
)

BUNDLE_ROOT = Path(__file__).resolve().parents[2]


def _summary() -> dict:
    return {
        "demo": "contract_fixture",
        "classification": "illustrative synthetic demonstrator",
        "target_measure": "density with respect to d theta",
        "correction_driver": "declared state Jacobian",
        "interpretation_boundary": "demonstrator, not an empirical estimate",
        "correction_log_range_on_grid": 12.5,
        "outcome": {
            "correction_log_range_on_grid": 12.5,
            "graph_mean": 0.75,
            "graph_sd": 0.05,
            "naive_mean": 0.70,
            "corrected_mean": 0.7500001,
            "naive_mean_shift_in_graph_sd": -1.0,
            "corrected_mean_shift_in_graph_sd": 2.0e-6,
            "naive_minus_corrected_mean_in_graph_sd": -1.000002,
            "naive_density_l1_vs_graph": 0.4,
            "corrected_density_l1_vs_graph": 2.0e-6,
        },
        "independent_validation": {
            "route": "direct full-state quadrature",
            "finite_corrected_vs_graph_centered_max_abs_diff": 1.0e-4,
        },
        "detcorr_api_audit": {"callable_abs_error": 1.0e-12},
        "tables": {
            "determinant_audit": [{"method": "gram_slq", "value": 100.0, "centered_abs_error": 4.3}]
        },
        "provenance": {"python_version": "3.12.13"},
    }


def _write_gate_fixture(root: Path, summary: dict, *, committed: bool) -> None:
    output_root = root / "contract_fixture"
    if committed:
        output_root = output_root / "outputs"
    output_root.mkdir(parents=True)
    (output_root / "summary.json").write_text(json.dumps(summary), encoding="utf-8")


def test_semantic_comparator_ignores_environment_sensitive_diagnostics() -> None:
    committed = _summary()
    regenerated = copy.deepcopy(committed)
    regenerated["independent_validation"]["finite_corrected_vs_graph_centered_max_abs_diff"] = (
        8.0e-3
    )
    regenerated["detcorr_api_audit"]["callable_abs_error"] = 7.0e-9
    regenerated["tables"]["determinant_audit"][0].update(
        value=100.03,
        centered_abs_error=4.1,
    )
    regenerated["provenance"] = {"python_version": "3.13.11"}
    regenerated["outcome"]["naive_mean_shift_in_graph_sd"] += 5.0e-6

    verifier._compare_semantic_summaries(
        committed,
        regenerated,
        location="fixture.summary",
    )


def test_semantic_comparator_rejects_material_headline_change() -> None:
    committed = _summary()
    regenerated = copy.deepcopy(committed)
    regenerated["outcome"]["naive_density_l1_vs_graph"] += 2.0e-3

    with pytest.raises(RuntimeError, match="naive_density_l1_vs_graph"):
        verifier._compare_semantic_summaries(
            committed,
            regenerated,
            location="fixture.summary",
        )


def test_semantic_and_publisher_gates_are_distinct(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    committed_root = tmp_path / "committed" / "examples"
    generated_root = tmp_path / "generated"
    committed = _summary()
    regenerated = copy.deepcopy(committed)
    regenerated["detcorr_api_audit"]["callable_abs_error"] = 9.0e-8
    _write_gate_fixture(committed_root, committed, committed=True)
    _write_gate_fixture(generated_root, regenerated, committed=False)

    monkeypatch.setattr(verifier, "EXAMPLES", committed_root)
    monkeypatch.setattr(
        registry,
        "WORKED_EXAMPLES",
        ({"slug": "contract_fixture"},),
    )

    verifier._require_semantic_regeneration(generated_root)
    with pytest.raises(RuntimeError, match=r"changed=\['summary.json'\]"):
        verifier._require_exact_regeneration(generated_root)


@pytest.mark.parametrize(
    "version",
    ["0.10.0", "0.10.0+publisher", "0.10.0.post1", "0.10.1"],
)
def test_supported_detcorr_versions_are_accepted(version: str) -> None:
    validate_detcorr_version(version)


@pytest.mark.parametrize(
    "version",
    ["0.9.9", "0.10.0rc1", "0.10.1rc1", "0.11.0", "1.0.0", "unknown"],
)
def test_unsupported_detcorr_versions_fail_clearly(version: str) -> None:
    with pytest.raises(
        UnsupportedDetcorrVersionError,
        match=r"requires detcorr>=0\.10\.0,<0\.11",
    ):
        validate_detcorr_version(version, context="test gate")


def test_import_gate_rejects_an_installed_old_detcorr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "detcorr", SimpleNamespace(__version__="0.9.9"))
    with pytest.raises(UnsupportedDetcorrVersionError, match="found detcorr 0.9.9"):
        require_supported_detcorr(context="single-example test")


def test_requirements_and_publisher_constraints_declare_version_floor() -> None:
    assert MINIMUM_DETCORR_VERSION == "0.10.0"
    assert SUPPORTED_DETCORR_SPEC == ">=0.10.0,<0.11"
    requirement_files = [BUNDLE_ROOT / "requirements.txt"]
    requirement_files.extend(BUNDLE_ROOT.glob("examples/*/requirements.txt"))
    assert len(requirement_files) == 11
    for path in requirement_files:
        lines = path.read_text(encoding="utf-8").splitlines()
        assert "detcorr>=0.10.0,<0.11" in lines, path


def test_publisher_constraints_are_an_exact_environment_lock() -> None:
    constraints = (BUNDLE_ROOT / "constraints-publisher.txt").read_text(encoding="utf-8")
    assert all(
        not line or line.startswith("#") or "==" in line for line in constraints.splitlines()
    )
    assert {
        line.lower() for line in constraints.splitlines() if line and not line.startswith("#")
    } == {
        "contourpy==1.3.3",
        "cycler==0.12.1",
        "detcorr==0.10.0",
        "fonttools==4.61.1",
        "iniconfig==2.3.0",
        "kiwisolver==1.4.9",
        "matplotlib==3.10.8",
        "numpy==2.3.5",
        "packaging==26.2",
        "pandas==2.2.3",
        "pillow==12.2.0",
        "pluggy==1.6.0",
        "pygments==2.20.0",
        "pyparsing==3.3.2",
        "pytest==8.4.2",
        "python-dateutil==2.9.0.post0",
        "pytz==2026.2",
        "ruff==0.14.10",
        "scipy==1.17.0",
        "six==1.17.0",
        "tzdata==2026.2",
    }


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_worked_json_writer_rejects_nonfinite_values(tmp_path: Path, value: float) -> None:
    with pytest.raises(ValueError, match=r"non-finite JSON value at root\.nested\[0\]"):
        write_json(tmp_path / "summary.json", {"nested": [value]})
    assert not (tmp_path / "summary.json").exists()


def _write_csv(path: Path, rows: list[list[str]]) -> None:
    path.write_text("\n".join(",".join(row) for row in rows) + "\n", encoding="utf-8")


def test_csv_semantic_comparator_accepts_numeric_formatting_and_small_drift(
    tmp_path: Path,
) -> None:
    committed = tmp_path / "committed.csv"
    regenerated = tmp_path / "regenerated.csv"
    _write_csv(committed, [["id", "value", "note"], ["a", "1", "stable"]])
    _write_csv(regenerated, [["id", "value", "note"], ["a", "1.00000001", "stable"]])

    verifier._compare_semantic_csvs(committed, regenerated, location="fixture/table.csv")


def test_csv_semantic_comparator_accepts_supported_slq_diagnostic_drift(
    tmp_path: Path,
) -> None:
    committed = tmp_path / "committed.csv"
    regenerated = tmp_path / "regenerated.csv"
    _write_csv(
        committed,
        [
            ["route", "value", "centered_value", "centered_abs_error", "error_estimate"],
            ["matrix_free_centered_gram_slq", "6185.664", "285.358", "4.187", "12.971"],
        ],
    )
    _write_csv(
        regenerated,
        [
            ["route", "value", "centered_value", "centered_abs_error", "error_estimate"],
            ["matrix_free_centered_gram_slq", "6185.874", "285.625", "4.454", "13.004"],
        ],
    )

    verifier._compare_semantic_csvs(
        committed,
        regenerated,
        location=("materials_screened_lattice_2d3d_jarvis/tables/exact_vs_matrix_free_audit.csv"),
    )


@pytest.mark.parametrize(
    ("location", "route", "column"),
    [
        (
            "different_example/tables/exact_vs_matrix_free_audit.csv",
            "matrix_free_centered_gram_slq",
            "value",
        ),
        (
            "materials_screened_lattice_2d3d_jarvis/tables/exact_vs_matrix_free_audit.csv",
            "exact_sparse_lu",
            "value",
        ),
        (
            "materials_screened_lattice_2d3d_jarvis/tables/exact_vs_matrix_free_audit.csv",
            "matrix_free_centered_gram_slq",
            "analytic_reference",
        ),
    ],
)
def test_slq_diagnostic_tolerance_is_path_and_route_scoped(
    tmp_path: Path,
    location: str,
    route: str,
    column: str,
) -> None:
    committed = tmp_path / "committed.csv"
    regenerated = tmp_path / "regenerated.csv"
    _write_csv(committed, [["route", column], [route, "6000.0"]])
    _write_csv(regenerated, [["route", column], [route, "6000.2"]])

    with pytest.raises(RuntimeError, match="number differs"):
        verifier._compare_semantic_csvs(committed, regenerated, location=location)


@pytest.mark.parametrize(
    ("column", "committed_value", "regenerated_value"),
    [
        ("value", "6000.0", "6000.301"),
        ("centered_value", "285.0", "285.301"),
        ("centered_abs_error", "4.0", "4.301"),
        ("error_estimate", "11.0", "11.051"),
    ],
)
def test_slq_diagnostic_tolerance_rejects_material_change(
    tmp_path: Path,
    column: str,
    committed_value: str,
    regenerated_value: str,
) -> None:
    committed = tmp_path / "committed.csv"
    regenerated = tmp_path / "regenerated.csv"
    _write_csv(
        committed,
        [["route", column], ["matrix_free_centered_gram_slq", committed_value]],
    )
    _write_csv(
        regenerated,
        [["route", column], ["matrix_free_centered_gram_slq", regenerated_value]],
    )

    with pytest.raises(RuntimeError, match="number differs"):
        verifier._compare_semantic_csvs(
            committed,
            regenerated,
            location=(
                "materials_screened_lattice_2d3d_jarvis/tables/exact_vs_matrix_free_audit.csv"
            ),
        )


def test_slq_diagnostic_tolerance_rejects_nonpositive_error_estimate(tmp_path: Path) -> None:
    committed = tmp_path / "committed.csv"
    regenerated = tmp_path / "regenerated.csv"
    _write_csv(
        committed,
        [["route", "error_estimate"], ["matrix_free_centered_gram_slq", "0.02"]],
    )
    _write_csv(
        regenerated,
        [["route", "error_estimate"], ["matrix_free_centered_gram_slq", "-0.01"]],
    )

    with pytest.raises(RuntimeError, match="non-positive SLQ error estimate"):
        verifier._compare_semantic_csvs(
            committed,
            regenerated,
            location=(
                "materials_screened_lattice_2d3d_jarvis/tables/exact_vs_matrix_free_audit.csv"
            ),
        )


@pytest.mark.parametrize(
    "column",
    [
        "finite_vs_hard_tube_centered_error",
        "finite_corrected_vs_graph_centered_error",
    ],
)
def test_screened_lattice_finite_penalty_tolerance_accepts_supported_drift(
    tmp_path: Path,
    column: str,
) -> None:
    committed = tmp_path / "committed.csv"
    regenerated = tmp_path / "regenerated.csv"
    _write_csv(committed, [[column], ["0.003996644079961698"]])
    _write_csv(regenerated, [[column], ["0.003943238279961698"]])

    verifier._compare_semantic_csvs(
        committed,
        regenerated,
        location=(
            "materials_screened_lattice_2d3d_jarvis/tables/finite_penalty_tube_validation.csv"
        ),
    )


@pytest.mark.parametrize(
    ("location", "regenerated_value"),
    [
        (
            "different_example/tables/finite_penalty_tube_validation.csv",
            "0.003943238279961698",
        ),
        (
            "materials_screened_lattice_2d3d_jarvis/tables/finite_penalty_tube_validation.csv",
            "0.003895644079961698",
        ),
    ],
)
def test_screened_lattice_finite_penalty_tolerance_is_scoped_and_bounded(
    tmp_path: Path,
    location: str,
    regenerated_value: str,
) -> None:
    committed = tmp_path / "committed.csv"
    regenerated = tmp_path / "regenerated.csv"
    column = "finite_vs_hard_tube_centered_error"
    _write_csv(committed, [[column], ["0.003996644079961698"]])
    _write_csv(regenerated, [[column], [regenerated_value]])

    with pytest.raises(RuntimeError, match="number differs"):
        verifier._compare_semantic_csvs(
            committed,
            regenerated,
            location=location,
        )


@pytest.mark.parametrize(
    ("regenerated_rows", "message"),
    [
        ([["different", "value"], ["a", "1"]], "header differs"),
        ([["id", "value"], ["a", "1"], ["b", "2"]], "row count differs"),
        ([["id", "value"], ["a"]], "width differs"),
        ([["id", "value"], ["b", "1"]], "text differs"),
        ([["id", "value"], ["1", "1"]], "value type differs"),
        ([["id", "value"], ["a", "1.01"]], "number differs"),
        ([["id", "value"], ["a", "nan"]], "non-finite numeric CSV value"),
    ],
)
def test_csv_semantic_comparator_rejects_incomplete_or_changed_evidence(
    tmp_path: Path,
    regenerated_rows: list[list[str]],
    message: str,
) -> None:
    committed = tmp_path / "committed.csv"
    regenerated = tmp_path / "regenerated.csv"
    _write_csv(committed, [["id", "value"], ["a", "1"]])
    _write_csv(regenerated, regenerated_rows)

    with pytest.raises(RuntimeError, match=message):
        verifier._compare_semantic_csvs(committed, regenerated, location="fixture/table.csv")


def test_csv_semantic_comparator_rejects_matching_nonfinite_values(tmp_path: Path) -> None:
    committed = tmp_path / "committed.csv"
    regenerated = tmp_path / "regenerated.csv"
    rows = [["id", "value"], ["a", "inf"]]
    _write_csv(committed, rows)
    _write_csv(regenerated, rows)

    with pytest.raises(RuntimeError, match="non-finite numeric CSV value"):
        verifier._compare_semantic_csvs(committed, regenerated, location="fixture/table.csv")
