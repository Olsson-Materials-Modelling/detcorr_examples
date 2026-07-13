#!/usr/bin/env python3
"""Run the CI-sized detcorr examples verification checks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import py_compile
import struct
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

EXAMPLES = Path(__file__).resolve().parent
PACKAGE_ROOT = EXAMPLES.parent
COMMON_SRC = EXAMPLES / "detcorr_demo_common" / "src"


@dataclass(frozen=True)
class SemanticTolerance:
    """Numerical acceptance rule for one authoritative scientific result."""

    relative: float
    absolute: float
    rationale: str


# The semantic gate deliberately compares only the scientific identity and the
# headline posterior results promised by every active example.  Numerical audit
# minutiae (finite-difference errors, finite-penalty convergence residuals, SLQ
# estimates, optimizer traces, and provenance) are still written and tested by
# their owning examples, but are not portable release fingerprints.
SEMANTIC_EXACT_FIELDS = (
    ("demo",),
    ("classification",),
    ("target_measure",),
    ("correction_driver",),
    ("interpretation_boundary",),
    ("independent_validation", "route"),
)

SEMANTIC_NUMERIC_FIELDS = {
    ("correction_log_range_on_grid",): SemanticTolerance(
        relative=0.0,
        absolute=1.0e-5,
        rationale="log-correction range (natural-log units)",
    ),
    ("outcome", "correction_log_range_on_grid"): SemanticTolerance(
        relative=0.0,
        absolute=1.0e-5,
        rationale="headline log-correction range (natural-log units)",
    ),
    ("outcome", "graph_mean"): SemanticTolerance(
        relative=0.0,
        absolute=1.0e-6,
        rationale="reported parameter mean",
    ),
    ("outcome", "graph_sd"): SemanticTolerance(
        relative=0.0,
        absolute=1.0e-6,
        rationale="reported parameter standard deviation",
    ),
    ("outcome", "naive_mean"): SemanticTolerance(
        relative=0.0,
        absolute=1.0e-6,
        rationale="reported naive parameter mean",
    ),
    ("outcome", "corrected_mean"): SemanticTolerance(
        relative=0.0,
        absolute=1.0e-6,
        rationale="reported corrected parameter mean",
    ),
    ("outcome", "naive_mean_shift_in_graph_sd"): SemanticTolerance(
        relative=0.0,
        absolute=1.0e-5,
        rationale="headline standardized posterior displacement (graph SD)",
    ),
    ("outcome", "corrected_mean_shift_in_graph_sd"): SemanticTolerance(
        relative=0.0,
        absolute=1.0e-5,
        rationale="headline corrected displacement (graph SD)",
    ),
    ("outcome", "naive_minus_corrected_mean_in_graph_sd"): SemanticTolerance(
        relative=0.0,
        absolute=1.0e-5,
        rationale="headline correction effect (graph SD)",
    ),
    ("outcome", "naive_density_l1_vs_graph"): SemanticTolerance(
        relative=0.0,
        absolute=1.0e-5,
        rationale="headline posterior-density L1 distance",
    ),
    ("outcome", "corrected_density_l1_vs_graph"): SemanticTolerance(
        relative=0.0,
        absolute=1.0e-5,
        rationale="headline corrected-density L1 distance",
    ),
}

_MISSING = object()
CSV_DEFAULT_TOLERANCE = SemanticTolerance(
    relative=5.0e-7,
    absolute=1.0e-8,
    rationale="portable tabular numeric evidence",
)

# The seeded SLQ audit is reproducible at the probe level, while Lanczos
# eigensolves can vary slightly across supported BLAS/LAPACK implementations.
# These bounds are expressed in natural-log units and apply only to the four
# stochastic diagnostic fields on the explicitly named matrix-free route.  The
# exact route, analytic references, configuration, and all posterior results
# retain the default contract.
_SCREENED_LATTICE_AUDIT = (
    "materials_screened_lattice_2d3d_jarvis/tables/exact_vs_matrix_free_audit.csv"
)
_SCREENED_LATTICE_FINITE_PENALTY = (
    "materials_screened_lattice_2d3d_jarvis/tables/finite_penalty_tube_validation.csv"
)
_SCREENED_LATTICE_SLQ_ROUTE = "matrix_free_centered_gram_slq"
CSV_PORTABILITY_TOLERANCES = {
    (_SCREENED_LATTICE_AUDIT, _SCREENED_LATTICE_SLQ_ROUTE, column): SemanticTolerance(
        relative=0.0,
        absolute=0.3,
        rationale="platform-sensitive SLQ log-determinant diagnostic",
    )
    for column in ("value", "centered_value", "centered_abs_error")
}
CSV_PORTABILITY_TOLERANCES[
    (_SCREENED_LATTICE_AUDIT, _SCREENED_LATTICE_SLQ_ROUTE, "error_estimate")
] = SemanticTolerance(
    relative=0.0,
    absolute=0.05,
    rationale="platform-sensitive SLQ diagnostic standard error",
)
# The finite-penalty check evaluates centered residuals after sparse operations
# on log densities of order 1e4 at rho=5e10.  Its 1e-4 portability bound is 40
# times smaller than the declared 4e-3 validation threshold.
for column in (
    "finite_vs_hard_tube_centered_error",
    "finite_corrected_vs_graph_centered_error",
):
    CSV_PORTABILITY_TOLERANCES[(_SCREENED_LATTICE_FINITE_PENALTY, None, column)] = (
        SemanticTolerance(
            relative=0.0,
            absolute=1.0e-4,
            rationale="large-rho sparse finite-penalty validation residual",
        )
    )


def _run(cmd: list[str], *, env: dict[str, str], capture: bool = False) -> None:
    print("$ " + " ".join(cmd), flush=True)
    if not capture:
        subprocess.run(cmd, cwd=PACKAGE_ROOT, env=env, check=True)
        return
    completed = subprocess.run(
        cmd,
        cwd=PACKAGE_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        print(completed.stdout, end="")
        print(completed.stderr, end="", file=sys.stderr)
        raise subprocess.CalledProcessError(completed.returncode, cmd)
    print(f"ok ({len(completed.stdout)} stdout chars)", flush=True)


def _output_hashes(root: Path) -> dict[str, str]:
    """Hash every committed/generated output file by its relative path."""

    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value!r}")


def _load_summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_constant)


def _require_supported_detcorr() -> None:
    """Fail the release gate early with the shared detcorr version contract."""

    if str(COMMON_SRC) not in sys.path:
        sys.path.insert(0, str(COMMON_SRC))
    from detcorr_demo_common import require_supported_detcorr

    require_supported_detcorr(context="the examples release gate")


def _value_at_path(summary: dict, path: tuple[str, ...]):
    value = summary
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return _MISSING
        value = value[key]
    return value


def _field_name(path: tuple[str, ...]) -> str:
    return ".".join(path)


def _compare_semantic_summaries(expected: dict, observed: dict, *, location: str) -> None:
    """Compare the declared portable scientific result surface.

    Exact publisher artifacts have a separate byte-identity gate.  Keeping the
    portable semantic surface explicit prevents environment-sensitive audit
    diagnostics from becoming accidental public API.
    """

    for path in SEMANTIC_EXACT_FIELDS:
        expected_value = _value_at_path(expected, path)
        observed_value = _value_at_path(observed, path)
        field = f"{location}.{_field_name(path)}"
        if expected_value is _MISSING or observed_value is _MISSING:
            raise RuntimeError(f"authoritative field is missing at {field}")
        if expected_value != observed_value:
            raise RuntimeError(
                f"regenerated authoritative value differs at {field}: "
                f"committed={expected_value!r}, regenerated={observed_value!r}"
            )

    for path, tolerance in SEMANTIC_NUMERIC_FIELDS.items():
        expected_value = _value_at_path(expected, path)
        observed_value = _value_at_path(observed, path)
        # Some examples do not report raw parameter moments.  If the committed
        # summary omits an optional field, that field is outside its contract.
        if expected_value is _MISSING:
            continue
        field = f"{location}.{_field_name(path)}"
        if observed_value is _MISSING:
            raise RuntimeError(f"authoritative numeric field is missing at {field}")
        if (
            isinstance(expected_value, bool)
            or not isinstance(expected_value, (int, float))
            or isinstance(observed_value, bool)
            or not isinstance(observed_value, (int, float))
        ):
            raise RuntimeError(f"authoritative numeric field has a non-numeric value at {field}")
        if not math.isclose(
            float(expected_value),
            float(observed_value),
            rel_tol=tolerance.relative,
            abs_tol=tolerance.absolute,
        ):
            raise RuntimeError(
                f"regenerated authoritative result differs at {field}: "
                f"committed={expected_value!r}, regenerated={observed_value!r}; "
                f"allowed rel_tol={tolerance.relative:g}, "
                f"abs_tol={tolerance.absolute:g} for {tolerance.rationale}"
            )


def _png_dimensions(path: Path) -> tuple[int, int]:
    payload = path.read_bytes()
    if len(payload) < 24 or payload[:8] != b"\x89PNG\r\n\x1a\n" or payload[12:16] != b"IHDR":
        raise RuntimeError(f"invalid PNG output: {path}")
    return struct.unpack(">II", payload[16:24])


def _compare_semantic_csvs(expected_path: Path, observed_path: Path, *, location: str) -> None:
    """Compare the complete tabular evidence surface with portable numeric tolerances."""

    with expected_path.open(newline="", encoding="utf-8") as handle:
        expected = list(csv.reader(handle))
    with observed_path.open(newline="", encoding="utf-8") as handle:
        observed = list(csv.reader(handle))
    if not expected or not observed or expected[0] != observed[0]:
        raise RuntimeError(f"regenerated CSV header differs at {location}")
    if len(expected) != len(observed):
        raise RuntimeError(
            f"regenerated CSV row count differs at {location}: "
            f"committed={len(expected) - 1}, regenerated={len(observed) - 1}"
        )
    route_index = expected[0].index("route") if "route" in expected[0] else None
    for row_index, (expected_row, observed_row) in enumerate(
        zip(expected[1:], observed[1:], strict=True),
        start=2,
    ):
        if len(expected_row) != len(observed_row):
            raise RuntimeError(f"regenerated CSV width differs at {location}:{row_index}")
        route = expected_row[route_index] if route_index is not None else None
        for column_index, (expected_value, observed_value) in enumerate(
            zip(expected_row, observed_row, strict=True)
        ):
            column = expected[0][column_index]
            tolerance = CSV_PORTABILITY_TOLERANCES.get(
                (location, route, column),
                CSV_DEFAULT_TOLERANCE,
            )
            try:
                expected_number = float(expected_value)
            except ValueError:
                expected_number = None
            try:
                observed_number = float(observed_value)
            except ValueError:
                observed_number = None

            if expected_number is None or observed_number is None:
                if expected_number is not None or observed_number is not None:
                    raise RuntimeError(
                        f"regenerated CSV value type differs at "
                        f"{location}:{row_index}:{column}: "
                        f"committed={expected_value!r}, regenerated={observed_value!r}"
                    )
                if expected_value == observed_value:
                    continue
                raise RuntimeError(
                    f"regenerated CSV text differs at {location}:{row_index}:{column}: "
                    f"committed={expected_value!r}, regenerated={observed_value!r}"
                )
            if not math.isfinite(expected_number) or not math.isfinite(observed_number):
                raise RuntimeError(
                    f"non-finite numeric CSV value at {location}:{row_index}:{column}"
                )
            if (location, route, column) == (
                _SCREENED_LATTICE_AUDIT,
                _SCREENED_LATTICE_SLQ_ROUTE,
                "error_estimate",
            ) and (expected_number <= 0.0 or observed_number <= 0.0):
                raise RuntimeError(
                    f"non-positive SLQ error estimate at {location}:{row_index}:{column}"
                )
            if not math.isclose(
                expected_number,
                observed_number,
                rel_tol=tolerance.relative,
                abs_tol=tolerance.absolute,
            ):
                raise RuntimeError(
                    f"regenerated CSV number differs at {location}:{row_index}:{column}: "
                    f"committed={expected_number!r}, regenerated={observed_number!r}; "
                    f"allowed rel_tol={tolerance.relative:g}, "
                    f"abs_tol={tolerance.absolute:g} for {tolerance.rationale}"
                )


def _require_semantic_regeneration(generated_root: Path) -> None:
    """Check output surface, figure geometry, and authoritative scientific results."""

    try:
        from .worked_examples_registry import WORKED_EXAMPLES
    except ImportError:  # pragma: no cover - direct script execution
        from worked_examples_registry import WORKED_EXAMPLES

    for example in WORKED_EXAMPLES:
        slug = example["slug"]
        committed_root = EXAMPLES / slug / "outputs"
        regenerated_root = generated_root / slug
        committed_paths = set(_output_hashes(committed_root))
        regenerated_paths = set(_output_hashes(regenerated_root))
        if committed_paths != regenerated_paths:
            raise RuntimeError(
                f"output file set differs for {slug}: "
                f"missing={sorted(committed_paths - regenerated_paths)}, "
                f"extra={sorted(regenerated_paths - committed_paths)}"
            )
        for relative in sorted(path for path in committed_paths if path.endswith(".png")):
            if _png_dimensions(committed_root / relative) != _png_dimensions(
                regenerated_root / relative
            ):
                raise RuntimeError(f"regenerated PNG dimensions differ for {slug}/{relative}")
        for relative in sorted(path for path in committed_paths if path.endswith(".csv")):
            _compare_semantic_csvs(
                committed_root / relative,
                regenerated_root / relative,
                location=f"{slug}/{relative}",
            )
        committed_summary = _load_summary(committed_root / "summary.json")
        regenerated_summary = _load_summary(regenerated_root / "summary.json")
        _compare_semantic_summaries(
            committed_summary,
            regenerated_summary,
            location=f"{slug}.summary",
        )


def _require_exact_regeneration(generated_root: Path) -> None:
    """Reject stale, missing, or byte-different committed outputs."""

    try:
        from .worked_examples_registry import WORKED_EXAMPLES
    except ImportError:  # pragma: no cover - direct script execution
        from worked_examples_registry import WORKED_EXAMPLES

    for example in WORKED_EXAMPLES:
        slug = example["slug"]
        committed = _output_hashes(EXAMPLES / slug / "outputs")
        generated = _output_hashes(generated_root / slug)
        if committed != generated:
            committed_paths = set(committed)
            generated_paths = set(generated)
            missing = sorted(committed_paths - generated_paths)
            extra = sorted(generated_paths - committed_paths)
            changed = sorted(
                path
                for path in committed_paths & generated_paths
                if committed[path] != generated[path]
            )
            raise RuntimeError(
                f"committed outputs differ from regeneration for {slug}: "
                f"missing={missing}, extra={extra}, changed={changed}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-detcorr",
        action="store_true",
        help="fail if detcorr is not importable and enable release/integration pytest mode",
    )
    parser.add_argument(
        "--run-worked-examples",
        action="store_true",
        help="regenerate every active example in a temporary directory before pytest",
    )
    parser.add_argument(
        "--require-byte-identical-outputs",
        action="store_true",
        help=(
            "release-publisher check: require regenerated files to match committed "
            "bytes exactly (requires --run-worked-examples and a pinned environment)"
        ),
    )
    args = parser.parse_args()
    if args.require_byte_identical_outputs and not args.run_worked_examples:
        parser.error("--require-byte-identical-outputs requires --run-worked-examples")

    env = dict(os.environ)
    env.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    pythonpath_entries = [str(PACKAGE_ROOT)]
    pythonpath_entries.append(env.get("PYTHONPATH", ""))
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries).strip(os.pathsep)
    if args.require_detcorr:
        _require_supported_detcorr()
        env["DETCORR_EXAMPLES_REQUIRE_DETCORR"] = "1"

    # Compile to an isolated directory so verification never leaves __pycache__
    # artifacts in the release tree.
    with tempfile.TemporaryDirectory(prefix="detcorr-example-bytecode-") as tmp:
        bytecode_root = Path(tmp)
        for source in sorted(EXAMPLES.rglob("*.py")):
            relative = source.relative_to(EXAMPLES).with_suffix(".pyc")
            destination = bytecode_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            py_compile.compile(str(source), cfile=str(destination), doraise=True)

    if args.run_worked_examples:
        with tempfile.TemporaryDirectory(prefix="detcorr-worked-examples-") as tmp:
            _run(
                [
                    sys.executable,
                    str(EXAMPLES / "run_worked_examples.py"),
                    "--outputs-root",
                    tmp,
                ],
                env=env,
            )
            _require_semantic_regeneration(Path(tmp))
            if args.require_byte_identical_outputs:
                _require_exact_regeneration(Path(tmp))

    pytest_cmd = [sys.executable, "-m", "pytest", str(EXAMPLES)]
    if args.require_detcorr:
        pytest_cmd.append("--require-detcorr")
    _run(pytest_cmd, env=env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
