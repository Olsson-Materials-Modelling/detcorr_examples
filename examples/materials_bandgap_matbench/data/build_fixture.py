#!/usr/bin/env python3
"""Build the deterministic 100-row fixture from the licensed Matbench dataset."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import re
from pathlib import Path

SOURCE_SHA256 = "1e6816fb8e7132535b76cdac458c4d53944eca97e1ecb68e6a8fd853c9cedc3a"
SOURCE_COLUMNS = ["composition", "gap expt"]
FIXTURE_ROWS = 100
SIMPLE_FORMULA = re.compile(r"(?:[A-Z][a-z]?(?:[0-9]*\.?[0-9]*))+")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def select_rows(payload: dict) -> list[tuple[int, str, float]]:
    if payload.get("columns") != SOURCE_COLUMNS:
        raise ValueError(f"unexpected source columns: {payload.get('columns')!r}")
    indices = payload.get("index")
    data = payload.get("data")
    if not isinstance(indices, list) or not isinstance(data, list) or len(indices) != len(data):
        raise ValueError("source index and data arrays must have equal length")

    selected: list[tuple[int, str, float]] = []
    seen: set[str] = set()
    for source_row, row in zip(indices, data, strict=True):
        if not isinstance(row, list) or len(row) != 2:
            raise ValueError("every source row must contain formula and gap")
        formula = str(row[0]).strip()
        gap = float(row[1])
        if not SIMPLE_FORMULA.fullmatch(formula) or not math.isfinite(gap) or gap <= 0.0:
            continue
        if formula in seen:
            continue
        selected.append((int(source_row), formula, gap))
        seen.add(formula)
        if len(selected) == FIXTURE_ROWS:
            return selected
    raise ValueError(f"source contains only {len(selected)} eligible rows")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source", type=Path, required=True, help="Matbench expt_gap.json.gz")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("matbench_expt_gap_fixture.csv"),
        help="deterministic fixture CSV to write",
    )
    args = parser.parse_args()

    observed_sha256 = sha256_file(args.source)
    if observed_sha256 != SOURCE_SHA256:
        raise SystemExit(
            f"source SHA256 mismatch: expected {SOURCE_SHA256}, observed {observed_sha256}"
        )
    with gzip.open(args.source, "rt", encoding="utf-8") as handle:
        rows = select_rows(json.load(handle))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("source_row", "formula", "band_gap_ev"))
        for source_row, formula, gap in rows:
            writer.writerow((source_row, formula, format(gap, ".12g")))
    print(f"wrote {len(rows)} rows to {args.output}")
    print(f"fixture SHA256: {sha256_file(args.output)}")


if __name__ == "__main__":
    main()
