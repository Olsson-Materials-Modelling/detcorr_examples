#!/usr/bin/env python3
"""Run the registered scientific studies without mutating the worked examples."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import tempfile
from contextlib import ExitStack
from pathlib import Path

from common.evidence import prepare_output_directory, strict_json_dump

ROOT = Path(__file__).resolve().parent
REGISTRY_PATH = ROOT / "STUDY_REGISTRY.json"
DETCORR_SOURCE_LOCK_PATH = ROOT / "DETCORR_SOURCE_LOCK.json"
NON_COMPUTATIONAL_STUDY_DOCUMENTS = {"README.md", "TECHNICAL_DETAILS.md"}
DETCORR_REPOSITORY_URL = "https://github.com/Olsson-Materials-Modelling/detcorr.git"
DETCORR_COMMIT = "a414f063515359f70952e2717a506b75e69e4b0b"
DISTRIBUTION_NAMES = (
    "contourpy",
    "cycler",
    "detcorr",
    "detcorr-scientific-case-studies",
    "fonttools",
    "iniconfig",
    "kiwisolver",
    "matplotlib",
    "numpy",
    "packaging",
    "pandas",
    "pillow",
    "pluggy",
    "pygments",
    "pyparsing",
    "pytest",
    "python-dateutil",
    "pytz",
    "ruff",
    "scipy",
    "six",
    "tzdata",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _registered() -> list[dict[str, str]]:
    payload = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    return list(payload["registered_reanalyses"])


def _output_hashes(root: Path) -> dict[str, str]:
    if not root.exists():
        return {}
    return {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _protocol_hashes() -> dict[str, str]:
    paths = (
        "PROTOCOL_LOCK.json",
        "STUDY_REGISTRY.json",
    )
    missing = [name for name in paths if not (ROOT / name).is_file()]
    if missing:
        raise SystemExit(f"protocol is incomplete; missing: {', '.join(missing)}")
    return {name: _sha256(ROOT / name) for name in paths}


def _study_input_hashes(study_root: Path) -> dict[str, str]:
    """Bind computational inputs, excluding public narrative documentation."""

    excluded_parts = {"__pycache__", "outputs", "outputs-full", ".pytest_cache"}
    return {
        path.relative_to(ROOT).as_posix(): _sha256(path)
        for path in sorted(study_root.rglob("*"))
        if path.is_file()
        and not any(part in excluded_parts for part in path.relative_to(study_root).parts)
        and path.name not in NON_COMPUTATIONAL_STUDY_DOCUMENTS
        and path.suffix != ".pyc"
    }


def _shared_input_hashes() -> dict[str, str]:
    """Bind shared analysis code and environment declarations used by every run."""

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
    missing = [path.relative_to(ROOT).as_posix() for path in paths if not path.is_file()]
    if missing:
        raise SystemExit(f"shared analysis surface is incomplete; missing: {', '.join(missing)}")
    return {path.relative_to(ROOT).as_posix(): _sha256(path) for path in paths}


def _distribution_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in DISTRIBUTION_NAMES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            if name == "detcorr":
                try:
                    import detcorr

                    versions[name] = str(detcorr.__version__)
                except Exception:
                    versions[name] = "not-installed"
            else:
                versions[name] = "not-installed"
    return versions


def _detcorr_source_identity() -> dict[str, object]:
    """Record the exact public source and installed Python files used by a run."""

    try:
        import detcorr
    except Exception as exc:  # pragma: no cover - exercised by the CLI error path
        raise RuntimeError("detcorr must be importable before running the studies") from exc

    package_root = Path(detcorr.__file__).resolve().parent
    source_hashes = {
        path.relative_to(package_root).as_posix(): _sha256(path)
        for path in sorted(package_root.rglob("*.py"))
        if path.is_file()
    }
    if not source_hashes:
        raise RuntimeError("the installed detcorr package contains no Python source files")

    repository_url: str | None = None
    commit_id: str | None = None
    requested_revision: str | None = None
    install_origin = "distribution_without_vcs_metadata"
    distribution_package_matches_import = False
    try:
        distribution = importlib.metadata.distribution("detcorr")
        direct_url_text = distribution.read_text("direct_url.json")
        distribution_package_matches_import = (
            Path(distribution.locate_file("detcorr")).resolve() == package_root
        )
    except importlib.metadata.PackageNotFoundError:
        direct_url_text = None
    if direct_url_text:
        direct_url = json.loads(direct_url_text)
        vcs_info = direct_url.get("vcs_info") or {}
        url = str(direct_url.get("url", ""))
        if vcs_info.get("vcs") == "git":
            install_origin = "git"
            repository_url = url
            commit_id = str(vcs_info.get("commit_id") or "") or None
            requested_revision = str(vcs_info.get("requested_revision") or "") or None
        elif direct_url.get("dir_info") is not None:
            install_origin = "local_directory"

    return {
        "commit_id": commit_id,
        "distribution_package_matches_import": distribution_package_matches_import,
        "install_origin": install_origin,
        "package_file_sha256": source_hashes,
        "repository_url": repository_url,
        "requested_revision": requested_revision,
        "version": str(detcorr.__version__),
    }


def _detcorr_source_lock() -> dict[str, object]:
    payload = json.loads(DETCORR_SOURCE_LOCK_PATH.read_text(encoding="utf-8"))
    if (
        payload.get("repository_url") != DETCORR_REPOSITORY_URL
        or payload.get("commit") != DETCORR_COMMIT
        or payload.get("version") != "0.10.0"
    ):
        raise RuntimeError("DETCORR_SOURCE_LOCK.json conflicts with the supported source")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true", help="CI-sized deterministic analysis")
    mode.add_argument("--full", action="store_true", help="prespecified full analysis")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs")
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace the complete output directory instead of refusing an overwrite",
    )
    parser.add_argument(
        "--study",
        action="append",
        default=[],
        help="registered slug to run; repeat for several (default: all)",
    )
    parser.add_argument("--list", action="store_true", help="list registered slugs and exit")
    args = parser.parse_args()

    registered = _registered()
    if args.list:
        for row in registered:
            print(row["slug"])
        return
    if not args.quick and not args.full:
        parser.error("one of --quick or --full is required unless --list is used")

    detcorr_source = _detcorr_source_identity()
    if args.full:
        source_lock = _detcorr_source_lock()
        if (
            detcorr_source["repository_url"] != DETCORR_REPOSITORY_URL
            or detcorr_source["commit_id"] != DETCORR_COMMIT
            or detcorr_source["version"] != source_lock["version"]
            or detcorr_source["package_file_sha256"] != source_lock["package_file_sha256"]
            or detcorr_source["distribution_package_matches_import"] is not True
        ):
            raise SystemExit(
                "claim-bearing full runs require the unshadowed detcorr source locked in "
                "DETCORR_SOURCE_LOCK.json"
            )

    known = {row["slug"] for row in registered}
    selected = set(args.study) if args.study else known
    unknown = selected - known
    if unknown:
        raise SystemExit(f"unknown study slug(s): {', '.join(sorted(unknown))}")

    output_root = prepare_output_directory(args.output_root, force=args.force)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT), environment.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    mode_flag = "--full" if args.full else "--quick"
    executed: list[dict[str, object]] = []

    with ExitStack() as resources:
        if not environment.get("MPLCONFIGDIR"):
            environment["MPLCONFIGDIR"] = resources.enter_context(
                tempfile.TemporaryDirectory(prefix="detcorr-matplotlib-")
            )

        for row in registered:
            slug = row["slug"]
            if slug not in selected:
                continue
            script = ROOT / row["entrypoint"]
            if not script.is_file():
                raise SystemExit(f"registered entrypoint is missing: {script.relative_to(ROOT)}")
            study_output = output_root / slug
            study_inputs = _study_input_hashes(script.parent)
            command = [
                sys.executable,
                str(script),
                mode_flag,
                "--output-root",
                str(study_output),
            ]
            print("$ " + " ".join(command), flush=True)
            subprocess.run(command, cwd=script.parent, env=environment, check=True)
            executed.append(
                {
                    "entrypoint": row["entrypoint"],
                    "input_hashes": study_inputs,
                    "output_hashes": _output_hashes(study_output),
                    "slug": slug,
                }
            )

        strict_json_dump(
            {
                "mode": "full" if args.full else "quick",
                "dependency_sources": {"detcorr": detcorr_source},
                "dependency_versions": _distribution_versions(),
                "protocol_hashes": _protocol_hashes(),
                "python": ".".join(str(part) for part in sys.version_info[:3]),
                "runtime": {
                    "machine": platform.machine(),
                    "platform": platform.platform(),
                    "python_implementation": platform.python_implementation(),
                    "python_version": platform.python_version(),
                },
                "repository_contract": "registered_reanalysis",
                "shared_input_hashes": _shared_input_hashes(),
                "studies": executed,
            },
            output_root / "run_manifest.json",
        )


if __name__ == "__main__":
    main()
