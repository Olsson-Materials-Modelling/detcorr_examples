#!/usr/bin/env python3
"""Write or print the runtime versions used by a scientific analysis."""

from __future__ import annotations

import argparse
import importlib.metadata
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.evidence import strict_json_dump  # noqa: E402
from run_studies import DISTRIBUTION_NAMES, _detcorr_source_identity  # noqa: E402


def distribution_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def report() -> dict[str, object]:
    return {
        "distributions": {name: distribution_version(name) for name in DISTRIBUTION_NAMES},
        "dependency_sources": {"detcorr": _detcorr_source_identity()},
        "machine": platform.machine(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = report()
    if args.output:
        strict_json_dump(payload, args.output)
    else:
        import json

        print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
