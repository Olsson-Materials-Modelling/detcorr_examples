#!/usr/bin/env python3
"""Run the registered Fulton temporal scientific reanalysis."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = ROOT.parents[1]
for path in (ROOT, REPOSITORY_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fulton_analysis import DATA_MANIFEST_PATH, DATA_PATH, run_analysis  # noqa: E402

from common.evidence import prepare_output_directory  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true", help="CI-sized deterministic run (default)")
    mode.add_argument("--full", action="store_true", help="full registered bootstrap resolution")
    parser.add_argument(
        "--data",
        type=Path,
        default=DATA_PATH,
        help="registered or contract-compatible CSV",
    )
    parser.add_argument(
        "--data-manifest",
        type=Path,
        default=None,
        help="provenance/checksum manifest; required when --data is not the bundled fixture",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPOSITORY_ROOT / "outputs" / "fulton_market",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace a completed study output at the same path",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    selected_data = args.data.resolve()
    if args.data_manifest is None:
        if selected_data != DATA_PATH.resolve():
            parser.error("custom --data requires --data-manifest; see TECHNICAL_DETAILS.md")
        selected_manifest = DATA_MANIFEST_PATH.resolve()
    else:
        selected_manifest = args.data_manifest.resolve()
    try:
        output_root = prepare_output_directory(args.output_root, force=args.force)
    except (FileExistsError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    summary = run_analysis(
        output_root,
        data_path=selected_data,
        data_manifest_path=selected_manifest,
        full=bool(args.full),
    )
    print(
        json.dumps(
            {
                "study": summary["study"],
                "mode": summary["mode"],
                "analysis_status": summary["analysis_status"],
                "implementation_checks": summary["implementation_checks"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
