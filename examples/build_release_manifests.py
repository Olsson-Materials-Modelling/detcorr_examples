#!/usr/bin/env python3
"""Build strict-JSON manifests from the active worked-example registry."""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path
from typing import Any

try:
    from .worked_examples_registry import WORKED_EXAMPLES, tier_counts
except ImportError:  # pragma: no cover - direct script execution
    from worked_examples_registry import WORKED_EXAMPLES, tier_counts

EXAMPLES = Path(__file__).resolve().parent
REPOSITORY = EXAMPLES.parent


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_constant)


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value!r}")


def _png_size(payload: bytes) -> list[int]:
    if payload[:8] != b"\x89PNG\r\n\x1a\n" or payload[12:16] != b"IHDR":
        raise ValueError("expected a PNG payload with an IHDR chunk")
    return list(struct.unpack(">II", payload[16:24]))


def build_figure_manifest() -> dict[str, Any]:
    figures: list[dict[str, Any]] = []
    for example in WORKED_EXAMPLES:
        output_dir = EXAMPLES / example["slug"] / "outputs"
        for path in sorted(output_dir.glob("*.png")):
            payload = path.read_bytes()
            figures.append(
                {
                    "example": example["slug"],
                    "tier": example["tier"],
                    "data_basis": example["data_basis"],
                    "path": path.relative_to(REPOSITORY).as_posix(),
                    "bytes": len(payload),
                    "size": _png_size(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
    return {
        "schema_version": 2,
        "example_count": len(WORKED_EXAMPLES),
        "tier_counts": tier_counts(),
        "figure_count": len(figures),
        "figures": figures,
    }


def build_worked_example_manifest() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for example in WORKED_EXAMPLES:
        root = EXAMPLES / example["slug"]
        summary = _load_json(root / "outputs" / "summary.json")
        outcome = summary["outcome"]
        correction_effect = float(outcome["naive_minus_corrected_mean_in_graph_sd"])
        independent_metric_name = example["independent_metric"]
        independent_metric_value = (
            None
            if independent_metric_name is None
            else float(summary["independent_validation"][independent_metric_name])
        )
        figures = sorted(
            path.relative_to(REPOSITORY).as_posix() for path in (root / "outputs").glob("*.png")
        )
        main_figure = root / "outputs" / example["main_figure"]
        if not main_figure.is_file():
            raise FileNotFoundError(main_figure)
        rows.append(
            {
                "slug": example["slug"],
                "title": example["title"],
                "tier": example["tier"],
                "classification": summary["classification"],
                "data_basis": example["data_basis"],
                "target_measure": summary["target_measure"],
                "correction_driver": summary["correction_driver"],
                "correction_log_range_on_grid": float(summary["correction_log_range_on_grid"]),
                "correction_effect_graph_sd_signed": correction_effect,
                "correction_effect_graph_sd_absolute": abs(correction_effect),
                "naive_density_l1_vs_graph": float(outcome["naive_density_l1_vs_graph"]),
                "corrected_density_l1_vs_graph": float(outcome["corrected_density_l1_vs_graph"]),
                "corrected_density_l1_role": example["corrected_l1_role"],
                "independent_validation_route": summary["independent_validation"]["route"],
                "independent_validation_metric_name": independent_metric_name,
                "independent_validation_metric_value": independent_metric_value,
                "readme": (root / "README.md").relative_to(REPOSITORY).as_posix(),
                "summary": (root / "outputs" / "summary.json").relative_to(REPOSITORY).as_posix(),
                "main_figure": main_figure.relative_to(REPOSITORY).as_posix(),
                "figures": figures,
            }
        )
    return {
        "schema_version": 3,
        "example_count": len(rows),
        "tier_counts": tier_counts(),
        "correction_effect_metric": (
            "(naive posterior mean - corrected posterior mean) / "
            "reduced-graph posterior standard deviation"
        ),
        "validation_metric_note": (
            "corrected_density_l1_vs_graph is interpreted by corrected_density_l1_role; "
            "the separately named independent_validation_metric_value is the load-bearing "
            "numeric check where the summary exposes one"
        ),
        "examples": rows,
    }


def build_validation_metrics() -> list[dict[str, Any]]:
    manifest = build_worked_example_manifest()
    keys = (
        "slug",
        "tier",
        "data_basis",
        "correction_log_range_on_grid",
        "correction_effect_graph_sd_signed",
        "correction_effect_graph_sd_absolute",
        "naive_density_l1_vs_graph",
        "corrected_density_l1_vs_graph",
        "corrected_density_l1_role",
        "independent_validation_route",
        "independent_validation_metric_name",
        "independent_validation_metric_value",
    )
    return [{key: row[key] for key in keys} for row in manifest["examples"]]


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    _write_json(EXAMPLES / "FIGURE_MANIFEST.json", build_figure_manifest())
    _write_json(
        EXAMPLES / "WORKED_EXAMPLES_MANIFEST.json",
        build_worked_example_manifest(),
    )
    _write_json(
        EXAMPLES / "outputs_validation_metrics.json",
        build_validation_metrics(),
    )
    counts = tier_counts()
    tier_text = ", ".join(f"{tier}={count}" for tier, count in counts.items())
    print(f"Wrote release manifests for {len(WORKED_EXAMPLES)} worked examples ({tier_text}).")


if __name__ == "__main__":
    main()
