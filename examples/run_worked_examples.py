#!/usr/bin/env python3
"""Regenerate every worked example through its public command-line entry point."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from contextlib import ExitStack
from pathlib import Path

try:
    from .worked_examples_registry import (
        WORKED_EXAMPLE_TIERS,
        WORKED_EXAMPLES,
        tier_counts,
    )
except ImportError:  # pragma: no cover - direct script execution
    from worked_examples_registry import WORKED_EXAMPLE_TIERS, WORKED_EXAMPLES, tier_counts

EXAMPLES = Path(__file__).resolve().parent
REPOSITORY = EXAMPLES.parent
COMMON_SRC = EXAMPLES / "detcorr_demo_common" / "src"


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value!r}")


def _entry_point(example_root: Path) -> Path:
    candidates = sorted((example_root / "src").glob("*_demo.py"))
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected one *_demo.py entry point in {example_root}, got {candidates}"
        )
    return candidates[0]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outputs-root",
        type=Path,
        help=(
            "write each example below this directory; by default overwrite each "
            "example's committed outputs directory"
        ),
    )
    parser.add_argument(
        "--tier",
        choices=("all", *WORKED_EXAMPLE_TIERS),
        default="all",
        help="run all active examples or only one declared release tier",
    )
    args = parser.parse_args(argv)
    outputs_root = (
        args.outputs_root.expanduser().resolve() if args.outputs_root is not None else None
    )

    selected = tuple(
        example for example in WORKED_EXAMPLES if args.tier == "all" or example["tier"] == args.tier
    )
    if not selected:
        raise RuntimeError(f"the registry has no examples in tier {args.tier!r}")

    env = dict(os.environ)
    pythonpath = [str(COMMON_SRC), env.get("PYTHONPATH", "")]
    env["PYTHONPATH"] = os.pathsep.join(pythonpath).strip(os.pathsep)
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")

    counts = tier_counts()
    tier_text = ", ".join(f"{tier}={count}" for tier, count in counts.items())
    print(
        f"Active release: {len(WORKED_EXAMPLES)} examples ({tier_text}); "
        f"running {len(selected)} for tier={args.tier}.",
        flush=True,
    )

    with ExitStack() as resources:
        if not env.get("MPLCONFIGDIR"):
            env["MPLCONFIGDIR"] = resources.enter_context(
                tempfile.TemporaryDirectory(prefix="detcorr-worked-matplotlib-")
            )

        for index, example in enumerate(selected, start=1):
            root = EXAMPLES / example["slug"]
            outputs = (
                outputs_root / example["slug"] if outputs_root is not None else root / "outputs"
            )
            command = [
                sys.executable,
                str(_entry_point(root)),
                "--outputs",
                str(outputs),
            ]
            print(
                f"[{index}/{len(selected)}][{example['tier']}] "
                f"{example['slug']} — {example['data_basis']}",
                flush=True,
            )
            print("$ " + " ".join(command), flush=True)
            subprocess.run(command, cwd=REPOSITORY, env=env, check=True)
            summary = outputs / "summary.json"
            if not summary.is_file():
                raise FileNotFoundError(f"worked example did not write {summary}")
            payload = json.loads(
                summary.read_text(encoding="utf-8"),
                parse_constant=_reject_constant,
            )
            if payload.get("demo") != example["slug"]:
                raise RuntimeError(f"worked example wrote an unexpected demo id: {summary}")
            figures = sorted(outputs.glob("*.png"))
            if len(figures) < 2:
                raise RuntimeError(f"worked example wrote fewer than two figures: {outputs}")
            for figure in figures:
                payload = figure.read_bytes()
                if len(payload) < 24 or payload[:8] != b"\x89PNG\r\n\x1a\n":
                    raise RuntimeError(f"worked example wrote an invalid PNG: {figure}")
            main_figure = outputs / example["main_figure"]
            if not main_figure.is_file():
                raise FileNotFoundError(f"worked example did not write {main_figure}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
