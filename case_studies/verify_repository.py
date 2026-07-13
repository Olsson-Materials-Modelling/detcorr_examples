#!/usr/bin/env python3
"""Verify the scientific subproject's scope and generated outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
EXCLUDED_STUDY_PARTS = {"__pycache__", "outputs", "outputs-full", ".pytest_cache"}
NON_COMPUTATIONAL_STUDY_DOCUMENTS = {"README.md", "TECHNICAL_DETAILS.md"}
DETCORR_SOURCE_LOCK_PATH = ROOT / "DETCORR_SOURCE_LOCK.json"
DETCORR_REPOSITORY_URL = "https://github.com/Olsson-Materials-Modelling/detcorr.git"
DETCORR_COMMIT = "a414f063515359f70952e2717a506b75e69e4b0b"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value!r}")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_constant)


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _finite(value: Any, location: str) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _finite(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _finite(child, f"{location}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise RuntimeError(f"non-finite number at {location}")


def _file_hashes(root: Path, *, relative_to: Path, exclude_generated: bool) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix == ".pyc":
            continue
        relative_study = path.relative_to(root)
        if exclude_generated and any(part in EXCLUDED_STUDY_PARTS for part in relative_study.parts):
            continue
        if exclude_generated and path.name in NON_COMPUTATIONAL_STUDY_DOCUMENTS:
            continue
        result[path.relative_to(relative_to).as_posix()] = _sha256(path)
    return result


def _protocol_hashes() -> dict[str, str]:
    names = (
        "PROTOCOL_LOCK.json",
        "STUDY_REGISTRY.json",
    )
    return {name: _sha256(ROOT / name) for name in names}


def _shared_input_hashes() -> dict[str, str]:
    paths = [
        ROOT / "run_studies.py",
        ROOT / "verify_repository.py",
        ROOT / "pyproject.toml",
        ROOT / "requirements.txt",
        ROOT / "requirements-test.txt",
        ROOT / "constraints-publisher.txt",
        DETCORR_SOURCE_LOCK_PATH,
        *sorted((ROOT / "common").rglob("*.py")),
    ]
    return {path.relative_to(ROOT).as_posix(): _sha256(path) for path in paths}


def _expected_claim_source_hashes(slug: str) -> dict[str, str]:
    """Recompute the exact source map written into each claim evaluation."""

    if slug == "fulton_market":
        study = ROOT / "studies" / slug
        paths = {
            "fulton_analysis.py": study / "fulton_analysis.py",
            "run_analysis.py": study / "run_analysis.py",
            "REGISTERED_PROTOCOL.json": study / "REGISTERED_PROTOCOL.json",
            "common/evidence.py": ROOT / "common" / "evidence.py",
        }
    elif slug in {"jarvis_bulk_modulus", "jarvis_band_gaps"}:
        analysis_name = {
            "jarvis_bulk_modulus": "bulk_modulus_analysis.py",
            "jarvis_band_gaps": "bandgap_analysis.py",
        }[slug]
        paths = {
            "study_analysis": ROOT / "studies" / slug / "src" / analysis_name,
            "common_geometry": ROOT / "common" / "jarvis_geometry.py",
            "common_protocol": ROOT / "common" / "jarvis_protocol.py",
            "common_evidence": ROOT / "common" / "evidence.py",
            "common_predictive_mixture": ROOT / "common" / "predictive_mixture.py",
            "protocol_lock": ROOT / "PROTOCOL_LOCK.json",
        }
    else:
        raise RuntimeError(f"no claim-source contract is declared for {slug!r}")
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise RuntimeError(f"{slug} claim-source files are missing: {', '.join(missing)}")
    return {name: _sha256(path) for name, path in paths.items()}


def _validate_detcorr_source(manifest: dict[str, Any], *, require_public_commit: bool) -> None:
    """Validate the recorded installed source without importing the current environment."""

    detcorr_source = manifest.get("dependency_sources", {}).get("detcorr", {})
    source_hashes = detcorr_source.get("package_file_sha256")
    if not isinstance(source_hashes, dict) or not source_hashes:
        raise RuntimeError("run manifest lacks the installed detcorr source-file hashes")
    if not all(re.fullmatch(r"[0-9a-f]{64}", str(value)) for value in source_hashes.values()):
        raise RuntimeError("run manifest has invalid detcorr source-file hashes")
    if require_public_commit:
        source_lock = _load_json(DETCORR_SOURCE_LOCK_PATH)
        if (
            detcorr_source.get("repository_url") != source_lock.get("repository_url")
            or detcorr_source.get("commit_id") != source_lock.get("commit")
            or detcorr_source.get("version") != source_lock.get("version")
            or source_hashes != source_lock.get("package_file_sha256")
            or detcorr_source.get("distribution_package_matches_import") is not True
        ):
            raise RuntimeError(
                "run manifest does not bind the executed package to DETCORR_SOURCE_LOCK.json"
            )


def _compare_output_trees(candidate: Path, reference: Path) -> None:
    """Require fresh scientific artifacts to match the committed reference exactly.

    Each run manifest is validated independently above. Its runtime inventory is
    deliberately excluded from cross-machine byte comparison because it records
    the executing platform rather than a scientific result.
    """

    if not candidate.is_dir():
        raise RuntimeError(f"candidate result directory is missing: {candidate}")
    if not reference.is_dir():
        raise RuntimeError(f"reference result directory is missing: {reference}")
    candidate_hashes = _file_hashes(
        candidate,
        relative_to=candidate,
        exclude_generated=False,
    )
    reference_hashes = _file_hashes(
        reference,
        relative_to=reference,
        exclude_generated=False,
    )
    if "run_manifest.json" not in candidate_hashes or "run_manifest.json" not in reference_hashes:
        raise RuntimeError("candidate and reference must both contain run_manifest.json")
    candidate_hashes.pop("run_manifest.json")
    reference_hashes.pop("run_manifest.json")
    candidate_paths = set(candidate_hashes)
    reference_paths = set(reference_hashes)
    missing = sorted(reference_paths - candidate_paths)
    extra = sorted(candidate_paths - reference_paths)
    changed = sorted(
        path
        for path in candidate_paths & reference_paths
        if candidate_hashes[path] != reference_hashes[path]
    )
    if missing or extra or changed:
        differences = []
        if missing:
            differences.append("missing: " + ", ".join(missing))
        if extra:
            differences.append("extra: " + ", ".join(extra))
        if changed:
            differences.append("changed: " + ", ".join(changed))
        raise RuntimeError(
            "fresh results differ from the committed reference (" + "; ".join(differences) + ")"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs")
    parser.add_argument(
        "--allow-quick-nonclaim",
        action="store_true",
        help="verify a quick implementation run without treating it as release evidence",
    )
    parser.add_argument(
        "--compare-reference",
        type=Path,
        help="also require the complete full output tree to match this reference directory",
    )
    args = parser.parse_args()

    if args.compare_reference is not None and args.allow_quick_nonclaim:
        raise SystemExit("reference comparison is available only for claim-bearing full runs")

    required_files = (
        "README.md",
        "TECHNICAL_DETAILS.md",
        "PROTOCOL_LOCK.json",
        "DETCORR_SOURCE_LOCK.json",
        "STUDY_REGISTRY.json",
        "LICENSE",
        "THIRD_PARTY_NOTICES.md",
        "third_party_licenses/CC-BY-4.0-NOTICE.md",
        "third_party_licenses/GPL-3.0.txt",
        "third_party_licenses/NIST-PUBLIC-INFORMATION-NOTICE.md",
    )
    for name in required_files:
        if not (ROOT / name).is_file():
            raise SystemExit(f"required repository file is missing: {name}")

    registry = _load_json(ROOT / "STUDY_REGISTRY.json")
    forbidden = {
        "materials_bandgap_matbench",
        "materials_screened_lattice_2d3d_jarvis",
    }
    active = {row["slug"] for row in registry["registered_reanalyses"]}
    if active & forbidden:
        raise SystemExit("a non-claim demonstrator was incorrectly registered as an active study")

    output_root = args.output_root.resolve()
    manifest = _load_json(output_root / "run_manifest.json")
    _finite(manifest, "run_manifest")
    expected_mode = "quick" if args.allow_quick_nonclaim else "full"
    if manifest.get("mode") != expected_mode:
        if args.allow_quick_nonclaim:
            raise SystemExit("quick verification requires a quick run manifest")
        raise SystemExit("release verification requires a full run manifest")
    if manifest.get("protocol_hashes") != _protocol_hashes():
        raise SystemExit("run manifest protocol hashes do not match the current repository")
    if manifest.get("shared_input_hashes") != _shared_input_hashes():
        raise SystemExit("run manifest shared code/environment hashes do not match")
    detcorr_version = manifest.get("dependency_versions", {}).get("detcorr")
    if detcorr_version in {None, "not-installed"}:
        raise SystemExit("run manifest does not identify a detcorr version")
    version_match = re.match(r"^(\d+)\.(\d+)(?:\.|$)", str(detcorr_version))
    if version_match is None or tuple(map(int, version_match.groups())) != (0, 10):
        raise SystemExit(f"run manifest has unsupported detcorr version: {detcorr_version!r}")
    try:
        _validate_detcorr_source(
            manifest,
            require_public_commit=not args.allow_quick_nonclaim,
        )
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    observed_studies = {row["slug"] for row in manifest["studies"]}
    if observed_studies != active:
        raise SystemExit(
            f"run manifest does not contain exactly the active studies: {sorted(observed_studies)}"
        )

    for slug in sorted(active):
        study = ROOT / "studies" / slug
        for relative in ("README.md", "run_analysis.py"):
            if not (study / relative).is_file():
                raise SystemExit(f"{slug} is missing {relative}")
        study_manifest = next(row for row in manifest["studies"] if row["slug"] == slug)
        observed_inputs = _file_hashes(study, relative_to=ROOT, exclude_generated=True)
        if study_manifest.get("input_hashes") != observed_inputs:
            raise SystemExit(f"{slug} code/protocol/input hashes changed after the recorded run")
        observed_outputs = _file_hashes(
            output_root / slug,
            relative_to=output_root / slug,
            exclude_generated=False,
        )
        if study_manifest.get("output_hashes") != observed_outputs:
            raise SystemExit(f"{slug} generated output hashes do not match the run manifest")

        summary = _load_json(output_root / slug / "summary.json")
        _finite(summary, f"{slug}.summary")
        expected_claim_bearing = not args.allow_quick_nonclaim
        if summary.get("claim_bearing_full_run") is not expected_claim_bearing:
            run_kind = "claim-bearing full" if expected_claim_bearing else "non-claim-bearing quick"
            raise SystemExit(f"{slug} is not a {run_kind} run on the pinned fixture")
        checks = summary.get("implementation_checks")
        if not isinstance(checks, dict) or checks.get("all_passed") is not True:
            raise SystemExit(f"{slug} implementation checks did not all pass")
        claims = summary.get("claim_evaluations")
        if not isinstance(claims, list) or not claims:
            raise SystemExit(f"{slug} has no machine-readable claim evaluations")
        expected_claim_ids = {
            "fulton_market": {"FU-PRED", "FU-DEP", "FU-GEOM"},
            "jarvis_bulk_modulus": {"JB-PRED", "JB-GEOM"},
            "jarvis_band_gaps": {"JG-PRED", "JG-GEOM", "JG-THRESHOLD"},
        }[slug]
        observed_claim_ids = [claim.get("claim_id") for claim in claims]
        if (
            len(observed_claim_ids) != len(set(observed_claim_ids))
            or set(observed_claim_ids) != expected_claim_ids
        ):
            raise SystemExit(
                f"{slug} claim ledger differs from the declared set: {sorted(expected_claim_ids)}"
            )
        allowed_statuses = {"SUPPORTED", "MIXED", "NOT_SUPPORTED", "BLOCKED", "EXPLORATORY"}
        allowed_timing = {
            "verification",
            "registered_reanalysis",
            "prospective_confirmation",
            "exploratory",
        }
        is_jarvis = slug in {"jarvis_bulk_modulus", "jarvis_band_gaps"}
        expected_model_config_sha256: str | None = None
        expected_data_sha256: str | None = None
        if is_jarvis:
            protocol = summary.get("protocol", {})
            configuration = protocol.get("configuration")
            if not isinstance(configuration, dict):
                raise SystemExit(f"{slug} summary lacks its model configuration")
            expected_model_config_sha256 = _canonical_hash(configuration)
            if protocol.get("model_config_sha256") != expected_model_config_sha256:
                raise SystemExit(f"{slug} summary model configuration hash is inconsistent")
            data_record = summary.get("data", {})
            data_manifest = data_record.get("manifest", {})
            processed_filename = Path(str(data_manifest.get("processed_filename", "")))
            if (
                not processed_filename.name
                or processed_filename.is_absolute()
                or len(processed_filename.parts) != 1
            ):
                raise SystemExit(f"{slug} summary has an unsafe processed data filename")
            fixture_path = study / "data" / processed_filename
            if not fixture_path.is_file():
                raise SystemExit(f"{slug} summary's processed fixture is missing")
            expected_data_sha256 = _sha256(fixture_path)
            if data_record.get("input_sha256") != expected_data_sha256:
                raise SystemExit(f"{slug} summary data hash does not match its bundled fixture")
            expected_protocol_tuple = (
                ("registered_reanalysis", "E1")
                if expected_claim_bearing
                else ("verification", "E0")
            )
            if (protocol.get("timing_class"), protocol.get("evidence_level")) != (
                expected_protocol_tuple
            ):
                raise SystemExit(f"{slug} summary has a mode-inconsistent protocol tuple")
        expected_claim_source_hashes = _expected_claim_source_hashes(slug)
        for claim in claims:
            if claim.get("status") not in allowed_statuses:
                raise SystemExit(
                    f"{slug} claim {claim.get('claim_id')!r} has invalid/unresolved status"
                )
            if claim.get("claim_bearing_full_run") is not expected_claim_bearing:
                raise SystemExit(
                    f"{slug} claim {claim.get('claim_id')!r} has the wrong run-evidence flag"
                )
            if claim.get("timing_class") not in allowed_timing:
                raise SystemExit(
                    f"{slug} claim {claim.get('claim_id')!r} has an invalid timing class"
                )
            required_claim_fields = (
                "protocol_version",
                "evidence_class",
                "evidence_level",
                "model_config_sha256",
                "data_sha256",
                "source_hashes",
                "evidence_artifacts",
            )
            missing = [name for name in required_claim_fields if not claim.get(name)]
            if missing:
                raise SystemExit(
                    f"{slug} claim {claim.get('claim_id')!r} lacks: {', '.join(missing)}"
                )
            for digest_name in ("model_config_sha256", "data_sha256"):
                if not re.fullmatch(r"[0-9a-f]{64}", str(claim[digest_name])):
                    raise SystemExit(
                        f"{slug} claim {claim.get('claim_id')!r} has invalid {digest_name}"
                    )
            if is_jarvis:
                if claim["protocol_version"] != protocol.get("version"):
                    raise SystemExit(
                        f"{slug} claim {claim.get('claim_id')!r} protocol version "
                        "does not match the summary"
                    )
                if claim["model_config_sha256"] != expected_model_config_sha256:
                    raise SystemExit(
                        f"{slug} claim {claim.get('claim_id')!r} model hash does not "
                        "match the summary configuration"
                    )
                if claim["data_sha256"] != expected_data_sha256:
                    raise SystemExit(
                        f"{slug} claim {claim.get('claim_id')!r} data hash does not "
                        "match the bundled fixture"
                    )
                if expected_claim_bearing:
                    expected_claim_tuple = (
                        "registered_reanalysis",
                        "fixture_conditional_inference",
                        "E1",
                    )
                    if claim["status"] == "EXPLORATORY" or "diagnostic_gate_status" in claim:
                        raise SystemExit(
                            f"{slug} full claim {claim.get('claim_id')!r} is diagnostic"
                        )
                else:
                    expected_claim_tuple = (
                        "verification",
                        "implementation_diagnostic",
                        "E0",
                    )
                    if claim["status"] != "EXPLORATORY" or claim.get(
                        "diagnostic_gate_status"
                    ) not in {"SUPPORTED", "MIXED", "NOT_SUPPORTED", "BLOCKED"}:
                        raise SystemExit(
                            f"{slug} quick claim {claim.get('claim_id')!r} lacks a "
                            "terminal diagnostic outcome"
                        )
                observed_claim_tuple = (
                    claim.get("timing_class"),
                    claim.get("evidence_class"),
                    claim.get("evidence_level"),
                )
                if observed_claim_tuple != expected_claim_tuple:
                    raise SystemExit(
                        f"{slug} claim {claim.get('claim_id')!r} has a "
                        "mode-inconsistent evidence tuple"
                    )
            if not isinstance(claim["source_hashes"], dict) or not all(
                re.fullmatch(r"[0-9a-f]{64}", str(value))
                for value in claim["source_hashes"].values()
            ):
                raise SystemExit(
                    f"{slug} claim {claim.get('claim_id')!r} has invalid source hashes"
                )
            if claim["source_hashes"] != expected_claim_source_hashes:
                raise SystemExit(
                    f"{slug} claim {claim.get('claim_id')!r} source hashes do not "
                    "match the current declared source files"
                )
            if not isinstance(claim["evidence_artifacts"], list):
                raise SystemExit(
                    f"{slug} claim {claim.get('claim_id')!r} has invalid evidence artifacts"
                )
            for artifact in claim["evidence_artifacts"]:
                if not isinstance(artifact, dict):
                    raise SystemExit(f"{slug} claim evidence entry is not an object")
                relative = Path(str(artifact.get("path", "")))
                if relative.is_absolute() or ".." in relative.parts:
                    raise SystemExit(f"{slug} claim evidence has an unsafe path: {relative}")
                evidence_path = output_root / slug / relative
                if not evidence_path.is_file() or _sha256(evidence_path) != artifact.get("sha256"):
                    raise SystemExit(
                        f"{slug} claim evidence hash/path mismatch: {relative.as_posix()}"
                    )
                if is_jarvis:
                    expected_artifact_type = "table"
                    if relative.parts and relative.parts[0] == "figures":
                        expected_artifact_type = (
                            "claim_bearing_figure"
                            if expected_claim_bearing
                            else "diagnostic_figure"
                        )
                    if artifact.get("artifact_type") != expected_artifact_type:
                        raise SystemExit(
                            f"{slug} claim evidence has a mode-inconsistent artifact "
                            f"type: {relative.as_posix()}"
                        )

    if args.compare_reference is not None:
        try:
            _compare_output_trees(output_root, args.compare_reference.resolve())
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from exc

    verification_kind = (
        "quick implementation runs" if args.allow_quick_nonclaim else "full reanalyses"
    )
    print(f"verified {len(active)} registered scientific {verification_kind}")
    if args.compare_reference is not None:
        print("fresh full results match the committed reference exactly")
    print("Matbench and screened-lattice demonstrators remain outside the claim-bearing surface")


if __name__ == "__main__":
    main()
