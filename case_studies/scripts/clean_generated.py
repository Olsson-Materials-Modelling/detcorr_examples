#!/usr/bin/env python3
"""Remove only the explicitly selected generated output root."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.evidence import remove_managed_output_directory  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_root.resolve()
    if root.parent != ROOT or not (root.name == "outputs" or root.name.startswith("outputs-")):
        raise SystemExit(
            "refusing to remove anything except a managed top-level outputs or outputs-* root"
        )
    try:
        remove_managed_output_directory(root)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
