from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import run_studies
import verify_repository

ROOT = Path(__file__).resolve().parents[1]


def test_public_narrative_docs_are_not_computational_study_inputs() -> None:
    study = ROOT / "studies" / "jarvis_bulk_modulus"
    runner_hashes = run_studies._study_input_hashes(study)
    verifier_hashes = verify_repository._file_hashes(
        study,
        relative_to=ROOT,
        exclude_generated=True,
    )

    assert runner_hashes == verifier_hashes
    assert not any(path.endswith("README.md") for path in runner_hashes)
    assert not any(path.endswith("TECHNICAL_DETAILS.md") for path in runner_hashes)
    assert "studies/jarvis_bulk_modulus/src/bulk_modulus_analysis.py" in runner_hashes
    assert "studies/jarvis_bulk_modulus/data/FIXTURE_MANIFEST.json" in runner_hashes


def test_machine_protocol_lock_covers_every_registered_study_and_geometry_rule() -> None:
    protocol = json.loads((ROOT / "PROTOCOL_LOCK.json").read_text(encoding="utf-8"))
    registry = json.loads((ROOT / "STUDY_REGISTRY.json").read_text(encoding="utf-8"))

    registered = {row["slug"] for row in registry["registered_reanalyses"]}
    assert set(protocol["active_studies"]) == registered
    assert protocol["version"] == "0.2-retrospective"
    assert protocol["geometry_contract"]["correction"] == (
        "J_over_q_for_the_declared_graph_measure"
    )
    assert protocol["geometry_contract"]["predictive_scores_validate_geometry"] is False
    assert set(protocol["computational_hash_scope"]["excluded_documents"]) == {
        "README.md",
        "TECHNICAL_DETAILS.md",
    }


def test_runner_and_verifier_hash_the_same_machine_protocol_surface() -> None:
    assert run_studies._protocol_hashes() == verify_repository._protocol_hashes()
    assert set(run_studies._protocol_hashes()) == {
        "PROTOCOL_LOCK.json",
        "STUDY_REGISTRY.json",
    }


def test_detcorr_source_identity_reads_pep610_git_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = tmp_path / "detcorr"
    package.mkdir()
    module = package / "__init__.py"
    module.write_text('__version__ = "0.10.0"\n', encoding="utf-8")
    (package / "core.py").write_text("VALUE = 1\n", encoding="utf-8")
    monkeypatch.setitem(
        sys.modules,
        "detcorr",
        SimpleNamespace(__file__=str(module), __version__="0.10.0"),
    )

    direct_url = json.dumps(
        {
            "url": run_studies.DETCORR_REPOSITORY_URL,
            "vcs_info": {
                "commit_id": run_studies.DETCORR_COMMIT,
                "requested_revision": run_studies.DETCORR_COMMIT,
                "vcs": "git",
            },
        }
    )
    distribution = SimpleNamespace(
        locate_file=lambda _name: package,
        read_text=lambda name: direct_url if name == "direct_url.json" else None,
    )
    monkeypatch.setattr(run_studies.importlib.metadata, "distribution", lambda _name: distribution)

    identity = run_studies._detcorr_source_identity()
    assert identity["install_origin"] == "git"
    assert identity["repository_url"] == run_studies.DETCORR_REPOSITORY_URL
    assert identity["commit_id"] == run_studies.DETCORR_COMMIT
    assert identity["distribution_package_matches_import"] is True
    assert set(identity["package_file_sha256"]) == {"__init__.py", "core.py"}


def test_full_runner_rejects_a_nonpublic_detcorr_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        run_studies,
        "_detcorr_source_identity",
        lambda: {
            "commit_id": None,
            "repository_url": None,
            "package_file_sha256": {"__init__.py": "0" * 64},
            "version": "0.10.0",
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_studies.py", "--full", "--output-root", str(tmp_path / "full")],
    )
    with pytest.raises(SystemExit, match="claim-bearing full runs require"):
        run_studies.main()


def test_verifier_requires_public_commit_only_for_claim_bearing_runs() -> None:
    source_hashes = {"__init__.py": "0" * 64}
    local_manifest = {
        "dependency_sources": {
            "detcorr": {
                "commit_id": None,
                "repository_url": None,
                "package_file_sha256": source_hashes,
            }
        }
    }
    verify_repository._validate_detcorr_source(local_manifest, require_public_commit=False)
    with pytest.raises(RuntimeError, match="DETCORR_SOURCE_LOCK.json"):
        verify_repository._validate_detcorr_source(local_manifest, require_public_commit=True)

    source_lock = json.loads((ROOT / "DETCORR_SOURCE_LOCK.json").read_text(encoding="utf-8"))
    public_manifest = {
        "dependency_sources": {
            "detcorr": {
                "commit_id": source_lock["commit"],
                "distribution_package_matches_import": True,
                "package_file_sha256": source_lock["package_file_sha256"],
                "repository_url": source_lock["repository_url"],
                "version": source_lock["version"],
            }
        }
    }
    verify_repository._validate_detcorr_source(public_manifest, require_public_commit=True)
    shadowed_manifest = json.loads(json.dumps(public_manifest))
    shadowed_manifest["dependency_sources"]["detcorr"]["distribution_package_matches_import"] = (
        False
    )
    with pytest.raises(RuntimeError, match="DETCORR_SOURCE_LOCK.json"):
        verify_repository._validate_detcorr_source(
            shadowed_manifest,
            require_public_commit=True,
        )
    with pytest.raises(RuntimeError, match="source-file hashes"):
        verify_repository._validate_detcorr_source(
            {"dependency_sources": {"detcorr": {}}},
            require_public_commit=False,
        )


def test_runner_cleans_its_temporary_matplotlib_directory(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output_root = tmp_path / "run"
    captured: dict[str, Path] = {}
    row = {
        "entrypoint": "studies/fulton_market/run_analysis.py",
        "role": "test",
        "slug": "fulton_market",
    }
    monkeypatch.delenv("MPLCONFIGDIR", raising=False)
    monkeypatch.setattr(run_studies, "_registered", lambda: [row])
    monkeypatch.setattr(run_studies, "_study_input_hashes", lambda _root: {})
    monkeypatch.setattr(run_studies, "_shared_input_hashes", lambda: {})
    monkeypatch.setattr(run_studies, "_protocol_hashes", lambda: {})
    monkeypatch.setattr(run_studies, "_distribution_versions", lambda: {})
    monkeypatch.setattr(run_studies, "_detcorr_source_identity", lambda: {"version": "0.10.0"})
    monkeypatch.setattr(run_studies.platform, "platform", lambda: "test-platform")
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_studies.py", "--quick", "--output-root", str(output_root)],
    )

    def fake_run(command, *, cwd, env, check):
        del command, cwd
        assert check is True
        captured["matplotlib_config"] = Path(env["MPLCONFIGDIR"])
        assert captured["matplotlib_config"].is_dir()
        study_output = output_root / "fulton_market"
        study_output.mkdir()
        (study_output / "summary.json").write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(run_studies.subprocess, "run", fake_run)

    run_studies.main()

    manifest = json.loads((output_root / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["dependency_sources"]["detcorr"] == {"version": "0.10.0"}
    assert not captured["matplotlib_config"].exists()


def test_reference_comparison_requires_byte_identical_output_trees(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate"
    reference = tmp_path / "reference"
    candidate.mkdir()
    reference.mkdir()
    (candidate / "run_manifest.json").write_text('{"runtime": "candidate"}\n', encoding="utf-8")
    (reference / "run_manifest.json").write_text('{"runtime": "reference"}\n', encoding="utf-8")
    (candidate / "summary.json").write_text('{"value": 1}\n', encoding="utf-8")
    (reference / "summary.json").write_text('{"value": 1}\n', encoding="utf-8")

    verify_repository._compare_output_trees(candidate, reference)

    (candidate / "summary.json").write_text('{"value": 2}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="changed: summary.json"):
        verify_repository._compare_output_trees(candidate, reference)
