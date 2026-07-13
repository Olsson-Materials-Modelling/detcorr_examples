#!/usr/bin/env python3
"""CLI for the JARVIS band-gap registered reanalysis."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE / "src"))
sys.path.insert(0, str(REPO))

from bandgap_analysis import run  # noqa: E402

from common.evidence import prepare_output_directory  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=HERE / "outputs")
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace the complete output directory instead of refusing an overwrite",
    )
    parser.add_argument(
        "--data", type=Path, default=HERE / "data" / "jarvis_bandgap_benchmark_fixture.csv"
    )
    parser.add_argument(
        "--data-manifest",
        type=Path,
        default=None,
        help="required provenance/checksum manifest for custom data",
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--quick", action="store_true", help="smaller deterministic grids and 99 permutations"
    )
    modes.add_argument(
        "--full", action="store_true", help="claim-bearing grids and 2,000 permutations"
    )
    parser.add_argument("--permutations", type=int, default=None, help="override permutation count")
    args = parser.parse_args()
    mode = "full" if args.full else "quick" if args.quick else "standard"
    default_permutations = {"quick": 99, "standard": 499, "full": 2000}[mode]
    permutations = default_permutations if args.permutations is None else args.permutations
    if permutations < 19:
        parser.error("--permutations must be at least 19")
    default_data = (HERE / "data" / "jarvis_bandgap_benchmark_fixture.csv").resolve()
    selected_data = args.data.resolve()
    if args.data_manifest is None:
        if selected_data != default_data:
            parser.error("custom --data requires --data-manifest; see TECHNICAL_DETAILS.md")
        selected_manifest = (HERE / "data" / "FIXTURE_MANIFEST.json").resolve()
    else:
        selected_manifest = args.data_manifest.resolve()
    output_root = prepare_output_directory(args.output_root, force=args.force)
    summary = run(
        selected_data,
        output_root,
        mode,
        permutations,
        require_pinned_checksum=selected_data == default_data,
        data_manifest_path=selected_manifest,
    )
    graph = summary["posterior"]["primary_graph"]
    print(f"band gaps: gamma={graph['gamma_mean']:.6f} +/- {graph['gamma_sd']:.6f}")
    print(f"wrote {output_root}")


if __name__ == "__main__":
    main()
