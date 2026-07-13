from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

from examples.build_release_manifests import (
    build_figure_manifest,
    build_validation_metrics,
    build_worked_example_manifest,
)
from examples.worked_examples_registry import (
    WORKED_EXAMPLE_SLUGS,
    WORKED_EXAMPLE_TIERS,
    WORKED_EXAMPLES,
    tier_counts,
)

BUNDLE_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = BUNDLE_ROOT / "examples"

OPEN_DATA_MANIFEST_REQUIRED_FIELDS = {
    "schema_version",
    "dataset_id",
    "bundled_file",
    "sha256",
    "bytes",
    "rows",
    "row_count",
    "source_name",
    "source_citation",
    "source_url_or_accession",
    "source_row_key",
    "retrieval_method",
    "selection_rule",
    "deduplication_rule",
    "known_source_dependence",
    "license_identifier",
    "license_source_url",
    "license_status",
    "rights_basis",
    "redistribution_decision",
    "rights_checked_date",
    "repository_license_applies",
    "third_party_notice",
    "dependent_observation_columns",
    "reference_input_columns",
    "units",
    "study_role",
}

OPEN_DATA_MANIFEST_REQUIRED_TEXT_FIELDS = {
    "dataset_id",
    "bundled_file",
    "sha256",
    "source_name",
    "source_citation",
    "source_url_or_accession",
    "source_row_key",
    "retrieval_method",
    "selection_rule",
    "deduplication_rule",
    "known_source_dependence",
    "license_identifier",
    "license_source_url",
    "license_status",
    "rights_basis",
    "redistribution_decision",
    "rights_checked_date",
    "third_party_notice",
    "study_role",
}


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value!r}")


def _load_strict_json(path: Path):
    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=_reject_constant,
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _public_markdown_documents() -> list[Path]:
    return sorted(path for path in BUNDLE_ROOT.rglob("*.md") if ".pytest_cache" not in path.parts)


def test_all_public_json_artifacts_are_strict() -> None:
    public_json = sorted(BUNDLE_ROOT.rglob("*.json"))
    assert public_json
    for path in public_json:
        _load_strict_json(path)


def test_registry_declares_exactly_ten_examples_split_five_plus_five() -> None:
    assert (BUNDLE_ROOT / "TECHNICAL_DETAILS.md").is_file()
    assert "(TECHNICAL_DETAILS.md)" in (BUNDLE_ROOT / "README.md").read_text(encoding="utf-8")
    assert len(WORKED_EXAMPLES) == 10
    assert len(WORKED_EXAMPLE_SLUGS) == len(set(WORKED_EXAMPLE_SLUGS))
    assert WORKED_EXAMPLE_TIERS == ("illustrative", "open_data")
    assert tier_counts() == {"illustrative": 5, "open_data": 5}

    for row in WORKED_EXAMPLES:
        assert set(row) == {
            "slug",
            "title",
            "tier",
            "data_basis",
            "main_figure",
            "corrected_l1_role",
            "independent_metric",
        }
        assert row["tier"] in WORKED_EXAMPLE_TIERS
        assert row["title"] and row["data_basis"]
        assert row["main_figure"].endswith(".png")
        assert row["corrected_l1_role"]
        assert row["independent_metric"] is None or isinstance(row["independent_metric"], str)
        if row["tier"] == "illustrative":
            assert "synthetic" in row["data_basis"]
        else:
            assert "external" in row["data_basis"] or "public" in row["data_basis"]


def test_active_directories_equal_the_registry_and_have_current_surface() -> None:
    discovered = {
        path.name
        for path in EXAMPLES.iterdir()
        if path.is_dir() and (path / "outputs" / "summary.json").is_file()
    }
    assert discovered == set(WORKED_EXAMPLE_SLUGS)

    for example in WORKED_EXAMPLES:
        root = EXAMPLES / example["slug"]
        for filename in (
            "README.md",
            "TECHNICAL_DETAILS.md",
            "Makefile",
            "requirements.txt",
            "pyproject.toml",
        ):
            assert (root / filename).is_file(), f"missing {filename} in {root}"
        entry_points = list((root / "src").glob("*_demo.py"))
        assert len(entry_points) == 1
        assert not list(root.glob("*.py")), f"loose active Python file in {root}"
        assert list((root / "tests").glob("test_*.py"))
        assert (root / "outputs" / "summary.json").is_file()
        if example["tier"] == "open_data":
            assert (root / "data" / "FIXTURE_MANIFEST.json").is_file()
        assert not list((root / "outputs").rglob("*.md"))
        assert {path.name for path in root.glob("*.md")} == {
            "README.md",
            "TECHNICAL_DETAILS.md",
        }

        figures = sorted((root / "outputs").glob("*.png"))
        assert len(figures) >= 2
        main_figure = root / "outputs" / example["main_figure"]
        assert main_figure.is_file()

        readme_text = (root / "README.md").read_text(encoding="utf-8")
        assert "(TECHNICAL_DETAILS.md)" in readme_text
        embedded = re.findall(r"!\[[^]]*\]\(([^)]+\.png)\)", readme_text)
        assert len(embedded) >= 2, f"{root / 'README.md'} needs two outcome figures"
        assert f"outputs/{example['main_figure']}" in embedded
        for target in embedded:
            assert (root / target).is_file()


def test_per_example_makefiles_use_safe_configurable_output_roots() -> None:
    for example in WORKED_EXAMPLES:
        path = EXAMPLES / example["slug"] / "Makefile"
        text = path.read_text(encoding="utf-8")
        assert "PYTHON ?= python" in text, path
        assert "OUTPUTS ?= $(abspath ../../regenerated/" in text, path
        assert "MPLCONFIGDIR ?= $(or $(TMPDIR),/tmp)/detcorr-matplotlib-cache" in text, path
        assert '--outputs "$(OUTPUTS)"' in text, path
        assert "$(PYTHON) -m pytest -q tests" in text, path
        assert ".PHONY: run test test-integration" in text, path
        assert "rm -rf" not in text, path
        assert "\nclean:" not in text, path


def test_ci_workflows_are_read_only_pinned_and_verify_the_release_surface() -> None:
    root_workflows = sorted((BUNDLE_ROOT / ".github" / "workflows").glob("*.yml"))
    standalone_workflows = sorted(
        (BUNDLE_ROOT / "case_studies" / ".github" / "workflows").glob("*.yml")
    )
    assert {path.name for path in root_workflows} == {"ci.yml", "full-analysis.yml"}
    assert {path.name for path in standalone_workflows} == {"ci.yml", "full-analysis.yml"}
    workflows = root_workflows + standalone_workflows
    for path in workflows:
        text = path.read_text(encoding="utf-8")
        assert "permissions:\n  contents: read" in text, path
        actions = re.findall(r"uses:\s+([^@\s]+)@([^\s]+)", text)
        assert actions, path
        for action, reference in actions:
            assert action.startswith("actions/"), (path, action)
            assert re.fullmatch(r"[0-9a-f]{40}", reference), (path, action, reference)
        assert "ruff check ." in text, path
        assert "ruff format --check ." in text, path
        assert (
            "Olsson-Materials-Modelling/detcorr.git@a414f063515359f70952e2717a506b75e69e4b0b"
        ) in text, path

    ci = (BUNDLE_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "python examples/verify_examples.py" in ci
    assert "--require-detcorr --run-worked-examples" in ci
    assert "python verify_repository.py --output-root reference_results" in ci


def test_open_data_fixture_manifests_are_complete_and_match_payloads() -> None:
    repository_notice = (BUNDLE_ROOT / "THIRD_PARTY_NOTICES.md").resolve()
    open_data_rows = [row for row in WORKED_EXAMPLES if row["tier"] == "open_data"]
    assert len(open_data_rows) == 5

    for row in open_data_rows:
        manifest_path = EXAMPLES / row["slug"] / "data" / "FIXTURE_MANIFEST.json"
        manifest = _load_strict_json(manifest_path)
        assert manifest.keys() >= OPEN_DATA_MANIFEST_REQUIRED_FIELDS, manifest_path
        assert manifest["schema_version"] == 2
        for field in OPEN_DATA_MANIFEST_REQUIRED_TEXT_FIELDS:
            assert isinstance(manifest[field], str) and manifest[field].strip(), (
                f"{field} must be non-empty in {manifest_path}"
            )

        assert manifest["repository_license_applies"] is False
        fixture = manifest_path.parent / manifest["bundled_file"]
        assert fixture.is_file()
        assert manifest["sha256"] == _sha256(fixture)
        assert manifest["bytes"] == fixture.stat().st_size

        with fixture.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fixture_rows = list(reader)
            fixture_columns = set(reader.fieldnames or ())
        assert manifest["rows"] == manifest["row_count"] == len(fixture_rows)
        assert manifest["rows"] > 0
        assert manifest["source_row_key"] in fixture_columns

        dependent = manifest["dependent_observation_columns"]
        reference = manifest["reference_input_columns"]
        assert isinstance(dependent, list) and dependent
        assert isinstance(reference, list) and reference
        assert all(isinstance(column, str) and column for column in dependent + reference)
        assert set(dependent).isdisjoint(reference)
        assert set(dependent + reference) <= fixture_columns
        assert set(dependent) <= manifest["units"].keys()

        notice = (manifest_path.parent / manifest["third_party_notice"]).resolve()
        assert notice == repository_notice
        assert notice.is_file()
        license_file = manifest.get("license_file")
        if license_file is not None:
            assert isinstance(license_file, str) and license_file.strip()
            assert (manifest_path.parent / license_file).resolve().is_file()


def test_common_summary_contract_for_all_ten_active_examples() -> None:
    for example in WORKED_EXAMPLES:
        summary_path = EXAMPLES / example["slug"] / "outputs" / "summary.json"
        summary = _load_strict_json(summary_path)
        assert summary["demo"] == example["slug"]
        for key in (
            "classification",
            "target_measure",
            "correction_driver",
            "correction_log_range_on_grid",
            "outcome",
            "independent_validation",
            "provenance",
        ):
            assert key in summary, f"{key} missing from {summary_path}"
        assert summary["classification"]
        assert summary["target_measure"]
        assert summary["correction_driver"]
        assert summary["independent_validation"]["route"]

        outcome = summary["outcome"]
        for key in (
            "correction_log_range_on_grid",
            "naive_mean_shift_in_graph_sd",
            "corrected_mean_shift_in_graph_sd",
            "naive_minus_corrected_mean_in_graph_sd",
            "naive_density_l1_vs_graph",
            "corrected_density_l1_vs_graph",
        ):
            assert isinstance(outcome[key], (int, float))
            assert math.isfinite(float(outcome[key]))
        assert math.isclose(
            float(outcome["correction_log_range_on_grid"]),
            float(summary["correction_log_range_on_grid"]),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        assert 0.0 <= float(outcome["corrected_density_l1_vs_graph"]) < 1.0e-3
        assert 0.0 <= float(outcome["naive_density_l1_vs_graph"]) <= 2.0


def test_correction_spectrum_covers_control_moderate_large_and_both_signs() -> None:
    rows: dict[str, tuple[dict, dict]] = {}
    for example in WORKED_EXAMPLES:
        summary = _load_strict_json(EXAMPLES / example["slug"] / "outputs" / "summary.json")
        rows[example["slug"]] = (summary, summary["outcome"])

    control_summary, control = rows["calorimetry_constant_determinant_control"]
    assert control_summary["correction_log_range_on_grid"] == 0.0
    assert control["naive_minus_corrected_mean_in_graph_sd"] == 0.0
    assert control["naive_density_l1_vs_graph"] == control["corrected_density_l1_vs_graph"]

    nonconstant = [
        row for slug, row in rows.items() if slug != "calorimetry_constant_determinant_control"
    ]
    for summary, outcome in nonconstant:
        assert summary["correction_log_range_on_grid"] > 5.0
        assert outcome["naive_density_l1_vs_graph"] > outcome["corrected_density_l1_vs_graph"]

    signed = [
        float(outcome["naive_minus_corrected_mean_in_graph_sd"]) for _, outcome in rows.values()
    ]
    magnitudes = [abs(value) for value in signed]
    assert any(value > 0.1 for value in signed)
    assert any(value < -0.1 for value in signed)
    assert any(0.0 < value < 0.3 for value in magnitudes)
    assert any(0.3 <= value < 1.0 for value in magnitudes)
    assert any(1.0 <= value < 3.0 for value in magnitudes)
    assert any(value > 10.0 for value in magnitudes)

    illustrative = [
        abs(rows[row["slug"]][1]["naive_minus_corrected_mean_in_graph_sd"])
        for row in WORKED_EXAMPLES
        if row["tier"] == "illustrative"
    ]
    open_data = [
        abs(rows[row["slug"]][1]["naive_minus_corrected_mean_in_graph_sd"])
        for row in WORKED_EXAMPLES
        if row["tier"] == "open_data"
    ]
    assert min(illustrative) == 0.0 and max(illustrative) > 0.8
    assert min(open_data) > 0.4 and max(open_data) > 10.0


def test_committed_manifests_match_the_active_registry_payload() -> None:
    figure_manifest = build_figure_manifest()
    example_manifest = build_worked_example_manifest()
    metrics = build_validation_metrics()
    assert _load_strict_json(EXAMPLES / "FIGURE_MANIFEST.json") == figure_manifest
    assert _load_strict_json(EXAMPLES / "WORKED_EXAMPLES_MANIFEST.json") == example_manifest
    assert _load_strict_json(EXAMPLES / "outputs_validation_metrics.json") == metrics

    expected_counts = {"illustrative": 5, "open_data": 5}
    assert figure_manifest["example_count"] == 10
    assert figure_manifest["tier_counts"] == expected_counts
    assert example_manifest["schema_version"] == 3
    assert example_manifest["example_count"] == 10
    assert example_manifest["tier_counts"] == expected_counts
    assert Counter(row["tier"] for row in example_manifest["examples"]) == expected_counts
    assert Counter(row["tier"] for row in metrics) == expected_counts
    assert {row["data_basis"] for row in metrics} == {row["data_basis"] for row in WORKED_EXAMPLES}


def test_jarvis_round_trip_l1_is_separate_from_independent_state_integration() -> None:
    expected_independent_errors = {
        "materials_bulk_modulus_samplers_jarvis": 6.797424248361494e-06,
        "materials_bandgap_samplers_jarvis": 6.275551595535944e-05,
        "materials_screened_lattice_2d3d_jarvis": 0.003996644079961698,
    }
    manifest_rows = {row["slug"]: row for row in build_worked_example_manifest()["examples"]}
    for slug, expected_error in expected_independent_errors.items():
        summary = _load_strict_json(EXAMPLES / slug / "outputs" / "summary.json")
        outcome = summary["outcome"]
        validation = summary["independent_validation"]
        row = manifest_rows[slug]

        assert outcome["corrected_density_l1_role"] == (
            "algebraic hard-limit round-trip consistency; not independent validation"
        )
        assert validation["load_bearing_metric"] == ("finite_vs_hard_tube_centered_max_abs_diff")
        if slug == "materials_bandgap_samplers_jarvis":
            assert (
                "independent corrected finite-tube integration"
                in validation["finite_corrected_vs_graph_role"]
            )
            assert validation["finite_corrected_vs_graph_centered_max_abs_diff"] < 7.0e-5
        else:
            assert "not a separate check" in validation["finite_corrected_vs_graph_role"]
        assert math.isclose(
            validation["finite_vs_hard_tube_centered_max_abs_diff"],
            expected_error,
            rel_tol=5.0e-3,
            abs_tol=1.0e-12,
        )
        assert row["corrected_density_l1_role"] == outcome["corrected_density_l1_role"]
        assert row["independent_validation_metric_name"] == validation["load_bearing_metric"]
        assert (
            row["independent_validation_metric_value"]
            == validation["finite_vs_hard_tube_centered_max_abs_diff"]
        )


def test_all_public_markdown_links_resolve() -> None:
    link_pattern = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
    documents = _public_markdown_documents()
    assert documents
    for document in documents:
        for raw_target in link_pattern.findall(document.read_text(encoding="utf-8")):
            target = raw_target.strip().split(maxsplit=1)[0].strip("<>")
            if not target or target.startswith(("#", "http://", "https://", "mailto:")):
                continue
            local_path = target.split("#", maxsplit=1)[0]
            assert (document.parent / local_path).resolve().exists(), (
                f"missing local target {target!r} referenced by {document}"
            )


def test_human_documentation_surface_is_readme_technical_and_legal_notices_only() -> None:
    expected = {
        BUNDLE_ROOT / "README.md",
        BUNDLE_ROOT / "TECHNICAL_DETAILS.md",
        BUNDLE_ROOT / "THIRD_PARTY_NOTICES.md",
        EXAMPLES / "README.md",
        BUNDLE_ROOT / "case_studies" / "README.md",
        BUNDLE_ROOT / "case_studies" / "TECHNICAL_DETAILS.md",
        BUNDLE_ROOT / "case_studies" / "THIRD_PARTY_NOTICES.md",
        BUNDLE_ROOT / "case_studies" / "third_party_licenses" / "CC-BY-4.0-NOTICE.md",
        BUNDLE_ROOT / "case_studies" / "third_party_licenses" / "NIST-PUBLIC-INFORMATION-NOTICE.md",
    }
    expected.update(
        EXAMPLES / row["slug"] / filename
        for row in WORKED_EXAMPLES
        for filename in ("README.md", "TECHNICAL_DETAILS.md")
    )

    registry = _load_strict_json(BUNDLE_ROOT / "case_studies" / "STUDY_REGISTRY.json")
    expected.update(
        BUNDLE_ROOT / "case_studies" / "studies" / row["slug"] / filename
        for row in registry["registered_reanalyses"]
        for filename in ("README.md", "TECHNICAL_DETAILS.md")
    )
    assert set(_public_markdown_documents()) == expected


def test_every_readme_figure_has_a_visible_caption() -> None:
    image_pattern = re.compile(r"!\[[^]]+\]\([^)]+\.png\)")
    for document in _public_markdown_documents():
        if document.name != "README.md":
            continue
        lines = document.read_text(encoding="utf-8").splitlines()
        for index, line in enumerate(lines):
            if image_pattern.fullmatch(line) is None:
                continue
            cursor = index + 1
            while cursor < len(lines) and not lines[cursor].strip():
                cursor += 1
            assert cursor < len(lines) and lines[cursor].startswith("*"), (
                f"{document}:{index + 1} needs a visible italic figure caption"
            )
            while cursor < len(lines) and lines[cursor].strip():
                if lines[cursor].endswith("*"):
                    break
                cursor += 1
            else:
                raise AssertionError(f"{document}:{index + 1} has an unterminated caption")


def test_tutorial_readmes_are_directly_navigable_and_installable() -> None:
    for row in WORKED_EXAMPLES:
        text = (EXAMPLES / row["slug"] / "README.md").read_text(encoding="utf-8")
        assert "[Examples index](../README.md)" in text
        assert "[Repository overview](../../README.md)" in text
        upstream = "Olsson-Materials-Modelling/detcorr.git@a414f063515359f70952e2717a506b75e69e4b0b"
        assert upstream in text


def test_citation_file_has_minimum_public_metadata() -> None:
    text = (BUNDLE_ROOT / "CITATION.cff").read_text(encoding="utf-8")
    assert "cff-version: 1.2.0" in text
    assert "title: detcorr examples and scientific case studies" in text
    assert "name: Measure-Corrected Inference Project" in text
    assert "license: MIT" in text


def test_public_tree_matches_registered_release_surface() -> None:
    assert not (BUNDLE_ROOT / "archive").exists()
    allowed_support_directories = {"detcorr_demo_common", "tests", "__pycache__"}
    active_directories = {
        path.name
        for path in EXAMPLES.iterdir()
        if path.is_dir() and path.name not in allowed_support_directories
    }
    assert active_directories == set(WORKED_EXAMPLE_SLUGS)
    assert not (BUNDLE_ROOT / "benchmarks").exists()

    allowed_top_level_python = {
        "__init__.py",
        "build_release_manifests.py",
        "conftest.py",
        "run_worked_examples.py",
        "verify_examples.py",
        "worked_examples_registry.py",
    }
    assert {path.name for path in EXAMPLES.glob("*.py")} == allowed_top_level_python
