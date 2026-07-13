"""Numerical helpers for comparable scientific evidence cards.

The routines deliberately know nothing about any application model.  They
summarize already-declared one-dimensional targets and reject non-finite output
rather than silently serializing implementation artefacts.
"""

from __future__ import annotations

import json
import math
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np

# Do not place ``np.trapz`` in ``getattr``'s default argument: Python evaluates
# that argument eagerly, and NumPy 2.5 removed the legacy alias.  The explicit
# branch retains compatibility with NumPy 1.24 while importing cleanly when
# only ``np.trapezoid`` is available.
trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

OUTPUT_MARKER_NAME = ".detcorr-output-root"
OUTPUT_MARKER_CONTENT = "detcorr-case-studies-output-v1\n"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _validated_output_path(path: Path) -> Path:
    """Return a safe absolute output path without following hidden symlinks."""

    requested = Path(os.path.abspath(path.expanduser()))
    for component in (requested, *requested.parents):
        if component.is_symlink():
            raise ValueError(f"refusing an output path with a symlink component: {requested}")
    output = requested.resolve(strict=False)
    filesystem_root = Path(output.anchor)
    home = Path.home().resolve()
    cwd = Path.cwd().resolve()
    protected_roots = (filesystem_root, home, REPOSITORY_ROOT)
    if output in protected_roots or output == cwd or output in cwd.parents:
        raise ValueError(f"refusing a protected output directory: {output}")
    if output.exists() and not output.is_dir():
        raise ValueError(f"output path is not a directory: {output}")
    return output


def _require_managed_output_marker(output: Path) -> None:
    marker = output / OUTPUT_MARKER_NAME
    if marker.is_symlink() or not marker.is_file():
        raise ValueError(
            f"refusing to replace unmanaged directory: {output}; choose a new output path"
        )
    if marker.read_text(encoding="utf-8") != OUTPUT_MARKER_CONTENT:
        raise ValueError(f"refusing directory with an invalid output marker: {output}")


def remove_managed_output_directory(path: Path) -> None:
    """Remove a runner-owned output directory after all safety checks pass."""

    output = _validated_output_path(path)
    if not output.exists():
        return
    _require_managed_output_marker(output)
    shutil.rmtree(output)


def prepare_output_directory(path: Path, *, force: bool = False) -> Path:
    """Create an empty result directory without silently replacing evidence.

    Completed scientific outputs are immutable by default.  ``force=True`` is
    an explicit request to replace the complete directory, never to mix a new
    run with stale products from an earlier one.
    """

    output = _validated_output_path(path)
    if output.exists() and any(output.iterdir()):
        if not force:
            raise FileExistsError(
                f"output directory is not empty: {output}; pass --force to replace it"
            )
        remove_managed_output_directory(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / OUTPUT_MARKER_NAME).write_text(OUTPUT_MARKER_CONTENT, encoding="utf-8")
    return output


def normalized_weights(log_weight: np.ndarray) -> np.ndarray:
    """Return stable normalized weights from a one-dimensional log array."""

    values = np.asarray(log_weight, dtype=float)
    if values.ndim != 1 or values.size < 2 or not np.all(np.isfinite(values)):
        raise ValueError("log_weight must be a finite one-dimensional array")
    shifted = values - float(np.max(values))
    weights = np.exp(shifted)
    total = float(np.sum(weights))
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("log weights cannot be normalized")
    return weights / total


def weighted_quantile(
    values: np.ndarray, weights: np.ndarray, probabilities: np.ndarray
) -> np.ndarray:
    """Compute deterministic weighted quantiles using midpoint plotting positions."""

    x = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    p = np.asarray(probabilities, dtype=float)
    if x.ndim != 1 or w.shape != x.shape or x.size < 2:
        raise ValueError("values and weights must be matching one-dimensional arrays")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(w)) or np.any(w < 0.0):
        raise ValueError("values and weights must be finite and weights nonnegative")
    if np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("probabilities must lie in [0, 1]")
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    ws = w[order]
    total = float(np.sum(ws))
    if total <= 0.0:
        raise ValueError("weights must have positive total")
    positions = (np.cumsum(ws) - 0.5 * ws) / total
    return np.interp(p, positions, xs, left=xs[0], right=xs[-1])


def density_distances(grid: np.ndarray, p: np.ndarray, q: np.ndarray) -> dict[str, float]:
    """Return L1, total-variation, Hellinger, and Jensen--Shannon distances."""

    x = np.asarray(grid, dtype=float)
    p_values = np.asarray(p, dtype=float)
    q_values = np.asarray(q, dtype=float)
    if x.ndim != 1 or p_values.shape != x.shape or q_values.shape != x.shape:
        raise ValueError("grid and densities must be matching one-dimensional arrays")
    if np.any(p_values < 0.0) or np.any(q_values < 0.0):
        raise ValueError("densities must be nonnegative")
    p_values = p_values / trapezoid(p_values, x)
    q_values = q_values / trapezoid(q_values, x)
    l1 = float(trapezoid(np.abs(p_values - q_values), x))
    hellinger = float(math.sqrt(0.5 * trapezoid((np.sqrt(p_values) - np.sqrt(q_values)) ** 2, x)))
    midpoint = 0.5 * (p_values + q_values)

    def _kl(a: np.ndarray, b: np.ndarray) -> float:
        positive = a > 0.0
        integrand = np.zeros_like(a)
        integrand[positive] = a[positive] * np.log(a[positive] / b[positive])
        return float(trapezoid(integrand, x))

    js = 0.5 * (_kl(p_values, midpoint) + _kl(q_values, midpoint))
    return {
        "hellinger": hellinger,
        "jensen_shannon_nats": float(js),
        "l1": l1,
        "total_variation": 0.5 * l1,
    }


def determinant_tilt_path(
    theta: np.ndarray,
    graph_log_density: np.ndarray,
    log_correction: np.ndarray,
    t_values: np.ndarray | None = None,
) -> list[dict[str, float]]:
    """Evaluate the exact exponential path from graph (`t=0`) to naive (`t=1`).

    For `p_t(theta) proportional to p_graph(theta) exp(-t h(theta))`, the
    derivative of the mean is `-Cov_t(theta, h)`.  Returning both sides makes
    the mechanism auditable without treating a large displacement as evidence
    of application-model adequacy.
    """

    x = np.asarray(theta, dtype=float)
    log_graph = np.asarray(graph_log_density, dtype=float)
    h = np.asarray(log_correction, dtype=float)
    if x.ndim != 1 or log_graph.shape != x.shape or h.shape != x.shape:
        raise ValueError("theta, graph_log_density, and log_correction must match")
    if not np.all(np.diff(x) > 0.0):
        raise ValueError("theta must be strictly increasing")
    ts = np.linspace(0.0, 1.0, 21) if t_values is None else np.asarray(t_values, dtype=float)
    if ts.ndim != 1 or np.any((ts < 0.0) | (ts > 1.0)):
        raise ValueError("t_values must be a one-dimensional subset of [0, 1]")

    rows: list[dict[str, float]] = []
    cell_width = np.gradient(x)
    for t in ts:
        weights = normalized_weights(log_graph - float(t) * h + np.log(cell_width))
        mean = float(np.sum(weights * x))
        mean_h = float(np.sum(weights * h))
        covariance = float(np.sum(weights * (x - mean) * (h - mean_h)))
        variance = float(np.sum(weights * (x - mean) ** 2))
        rows.append(
            {
                "mean": mean,
                "mean_derivative_exact": -covariance,
                "sd": math.sqrt(max(variance, 0.0)),
                "t": float(t),
            }
        )
    return rows


def _require_finite_json(value: Any, path: str = "root") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _require_finite_json(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _require_finite_json(child, f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite JSON value at {path}")


def strict_json_dump(payload: Any, path: Path) -> None:
    """Write deterministic UTF-8 JSON after recursively rejecting NaN/Inf."""

    _require_finite_json(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
