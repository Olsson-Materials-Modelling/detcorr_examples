"""Shared numerical and output utilities for the worked detcorr examples.

The module deliberately contains no application model. It provides only grid,
validation, output, provenance, plotting, and optional detcorr helpers.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import sys
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

MINIMUM_DETCORR_VERSION = "0.10.0"
SUPPORTED_DETCORR_SPEC = ">=0.10.0,<0.11"
_MINIMUM_DETCORR_RELEASE = (0, 10, 0)
_SUPPORTED_DETCORR_SERIES = (0, 10)
_RELEASE_VERSION = re.compile(r"^(\d+)\.(\d+)\.(\d+)(.*)$")


class UnsupportedDetcorrVersionError(RuntimeError):
    """Raised when an installed detcorr cannot satisfy the examples' API floor."""


def validate_detcorr_version(version: str, *, context: str = "detcorr examples") -> None:
    """Require a supported detcorr release version.

    The examples intentionally avoid a dependency on ``packaging`` in their
    portable runtime.  detcorr publishes conventional three-part release
    versions, so a small strict parser is sufficient for the declared API
    compatibility floor.
    """

    match = _RELEASE_VERSION.fullmatch(str(version).strip())
    if match is None:
        raise UnsupportedDetcorrVersionError(
            f"{context} requires detcorr{SUPPORTED_DETCORR_SPEC}, but the installed "
            f"version {version!r} is not a parseable three-part release version"
        )
    release = tuple(int(part) for part in match.groups()[:3])
    suffix = match.group(4)
    unsupported_prerelease = bool(suffix and not suffix.startswith(("+", ".post")))
    if (
        release < _MINIMUM_DETCORR_RELEASE
        or release[:2] != _SUPPORTED_DETCORR_SERIES
        or unsupported_prerelease
    ):
        raise UnsupportedDetcorrVersionError(
            f"{context} requires detcorr{SUPPORTED_DETCORR_SPEC}; found detcorr {version}"
        )


def require_supported_detcorr(
    *,
    allow_missing: bool = False,
    context: str = "detcorr examples",
):
    """Import detcorr and enforce the supported API floor.

    ``allow_missing`` exists only for examples whose analytic route can run
    without the optional public-API audit.  A present but unsupported detcorr
    installation is never silently ignored.
    """

    try:
        import detcorr
    except ModuleNotFoundError as exc:
        if allow_missing and exc.name == "detcorr":
            return None
        raise RuntimeError(
            f"{context} requires detcorr{SUPPORTED_DETCORR_SPEC}, but detcorr is not importable"
        ) from exc
    except Exception as exc:  # pragma: no cover - depends on a broken installation
        raise RuntimeError(f"detcorr failed to import for {context}: {exc!r}") from exc

    version = getattr(detcorr, "__version__", None)
    if not isinstance(version, str):
        raise UnsupportedDetcorrVersionError(
            f"{context} requires detcorr{SUPPORTED_DETCORR_SPEC}, but the installed "
            "detcorr does not expose a string __version__"
        )
    validate_detcorr_version(version, context=context)
    return detcorr


# All active entry points import this module.  Therefore a stale detcorr is
# rejected before an example can fail later on a missing or changed API.  A
# genuinely absent installation remains permissible for analytic-only runs.
require_supported_detcorr(allow_missing=True)


@dataclass(frozen=True)
class PosteriorSummary:
    mean: float
    sd: float
    mode: float
    q05: float
    q50: float
    q95: float
    log_normalizer: float


def relative_display_path(path: Path | str, *, root: Path | str | None = None) -> str:
    """Return a portable slash-separated path for summaries and manifests."""

    base = Path.cwd() if root is None else Path(root)
    resolved = Path(path).resolve()
    try:
        relative = resolved.relative_to(base.resolve())
    except ValueError:
        relative = Path(os.path.relpath(resolved, base.resolve()))
    return relative.as_posix()


def reproducibility_provenance(*, example_root: Path | str | None = None) -> dict[str, Any]:
    """Return lightweight provenance fields for regenerated outputs."""

    provenance: dict[str, Any] = {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
    }
    if example_root is not None:
        provenance["example_root"] = relative_display_path(example_root, root=example_root)
    detcorr = require_supported_detcorr(
        allow_missing=True,
        context="output provenance",
    )
    if detcorr is not None:
        provenance["detcorr_version"] = getattr(detcorr, "__version__", "unknown")
    else:  # pragma: no cover - depends on installation
        provenance["detcorr_version"] = None
        provenance["detcorr_import_error"] = f"detcorr{SUPPORTED_DETCORR_SPEC} is not importable"
    return provenance


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: Any, path: str = "root") -> Any:
    """Convert NumPy scalars while rejecting non-finite scientific output."""

    if isinstance(value, Mapping):
        return {str(key): _json_safe(item, f"{path}.{key}") for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item, f"{path}[{index}]") for index, item in enumerate(value)]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite JSON value at {path}")
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), allow_nan=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def write_table(table: pd.DataFrame, csv_path: Path) -> None:
    """Write the authoritative tabular artifact as CSV."""

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(csv_path, index=False)


def _trapezoid(values: np.ndarray, grid: np.ndarray) -> float:
    """Integrate one-dimensional values without requiring NumPy 2."""

    y = np.asarray(values, dtype=float)
    x = np.asarray(grid, dtype=float)
    if y.ndim != 1 or x.ndim != 1 or y.size != x.size or x.size < 2:
        raise ValueError("values and grid must be one-dimensional arrays of equal size >= 2")
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)))


def centered(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        return values.copy()
    return values - float(np.max(values[finite]))


def centered_max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    """Compare centered log densities while requiring identical finite support."""

    ca = centered(np.asarray(a, dtype=float))
    cb = centered(np.asarray(b, dtype=float))
    if ca.shape != cb.shape:
        raise ValueError("arrays must have identical shapes")
    finite_a = np.isfinite(ca)
    finite_b = np.isfinite(cb)
    if not np.array_equal(finite_a, finite_b):
        return float("inf")
    if not finite_a.any():
        return float("nan")
    return float(np.max(np.abs(ca[finite_a] - cb[finite_a])))


def normalize_on_grid(
    grid: np.ndarray,
    logdensity: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, PosteriorSummary]:
    grid = np.asarray(grid, dtype=float)
    logdensity = np.asarray(logdensity, dtype=float)
    if grid.ndim != 1 or logdensity.ndim != 1 or grid.size != logdensity.size:
        raise ValueError("grid and logdensity must be one-dimensional arrays of equal size")
    if grid.size < 2 or not np.all(np.isfinite(grid)) or np.any(np.diff(grid) <= 0.0):
        raise ValueError("grid must contain at least two finite, strictly increasing values")
    finite = np.isfinite(logdensity)
    if not finite.any():
        raise ValueError("logdensity has no finite entries")
    shifted = np.full_like(logdensity, -np.inf, dtype=float)
    max_log = float(np.max(logdensity[finite]))
    shifted[finite] = logdensity[finite] - max_log
    weights = np.exp(shifted)
    area = _trapezoid(weights, grid)
    if area <= 0.0 or not math.isfinite(area):
        raise ValueError("cannot normalize logdensity on grid")
    density = weights / area
    increments = 0.5 * (density[1:] + density[:-1]) * np.diff(grid)
    cdf = np.concatenate([[0.0], np.cumsum(increments)])
    cdf = np.clip(cdf / cdf[-1], 0.0, 1.0)
    mean = _trapezoid(grid * density, grid)
    variance = _trapezoid((grid - mean) ** 2 * density, grid)
    mode = float(grid[int(np.argmax(density))])
    q05, q50, q95 = np.interp([0.05, 0.5, 0.95], cdf, grid)
    return (
        density,
        cdf,
        PosteriorSummary(
            mean=mean,
            sd=math.sqrt(max(variance, 0.0)),
            mode=mode,
            q05=float(q05),
            q50=float(q50),
            q95=float(q95),
            log_normalizer=float(max_log + math.log(area)),
        ),
    )


def density_l1(grid: np.ndarray, density_a: np.ndarray, density_b: np.ndarray) -> float:
    difference = np.abs(np.asarray(density_a, dtype=float) - np.asarray(density_b, dtype=float))
    return _trapezoid(difference, np.asarray(grid, dtype=float))


def summarize_targets(
    grid: np.ndarray,
    logdensities: Mapping[str, np.ndarray],
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, np.ndarray]]:
    densities: dict[str, np.ndarray] = {}
    cdfs: dict[str, np.ndarray] = {}
    summaries: list[dict[str, Any]] = []
    for name, logdensity in logdensities.items():
        density, cdf, summary = normalize_on_grid(grid, logdensity)
        densities[name] = density
        cdfs[name] = cdf
        summaries.append({"target": name, **asdict(summary)})

    graph_name = "graph_reduced" if "graph_reduced" in densities else "graph"
    graph_density = densities.get(graph_name)
    graph_logdensity = logdensities.get(graph_name)
    rows: list[dict[str, Any]] = []
    for summary in summaries:
        name = str(summary["target"])
        row = dict(summary)
        if graph_density is not None:
            row["density_l1_vs_graph"] = density_l1(grid, densities[name], graph_density)
        if graph_logdensity is not None:
            row["centered_logdensity_max_abs_diff_vs_graph"] = centered_max_abs_diff(
                logdensities[name], graph_logdensity
            )
        rows.append(row)
    return pd.DataFrame(rows), densities, cdfs


def select_grid_points(grid: np.ndarray, n_points: int = 11) -> np.ndarray:
    """Deterministically select representative grid values including endpoints."""

    values = np.asarray(grid, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("grid must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(values)) or np.any(np.diff(values) <= 0.0):
        raise ValueError("grid values must be finite and strictly increasing")
    if int(n_points) < 1:
        raise ValueError("n_points must be positive")
    count = min(int(n_points), int(values.size))
    indices = np.unique(np.linspace(0, values.size - 1, count, dtype=int))
    return values[indices]


def centered_validation_table(
    parameter_name: str,
    parameter_values: np.ndarray,
    finite_tube_logdensity: np.ndarray,
    hard_tube_logdensity: np.ndarray,
    finite_corrected_logdensity: np.ndarray,
    graph_logdensity: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Build an independent finite-penalty validation table."""

    values = np.asarray(parameter_values, dtype=float)
    finite_tube = np.asarray(finite_tube_logdensity, dtype=float)
    hard_tube = np.asarray(hard_tube_logdensity, dtype=float)
    finite_corrected = np.asarray(finite_corrected_logdensity, dtype=float)
    graph = np.asarray(graph_logdensity, dtype=float)
    rows = pd.DataFrame(
        {
            parameter_name: values,
            "finite_penalty_tube_logdensity": finite_tube,
            "coarea_hard_tube_logdensity": hard_tube,
            "finite_penalty_plus_logdet": finite_corrected,
            "graph_logdensity": graph,
            "finite_vs_hard_tube_centered_error": centered(finite_tube) - centered(hard_tube),
            "finite_corrected_vs_graph_centered_error": centered(finite_corrected)
            - centered(graph),
        }
    )
    stats = {
        "n_points": int(values.size),
        "finite_vs_hard_tube_centered_max_abs_diff": centered_max_abs_diff(finite_tube, hard_tube),
        "finite_corrected_vs_graph_centered_max_abs_diff": centered_max_abs_diff(
            finite_corrected, graph
        ),
    }
    return rows, stats


def try_detcorr_logdet_diag(
    diagonal: np.ndarray,
    *,
    structure: str = "diagonal",
) -> dict[str, Any]:
    detcorr = require_supported_detcorr(
        allow_missing=True,
        context="the diagonal detcorr API audit",
    )
    if detcorr is None:  # pragma: no cover - depends on installation
        return {
            "detcorr_available": False,
            "reason": f"detcorr{SUPPORTED_DETCORR_SPEC} is not importable",
        }
    result = detcorr.log_correction(
        jacobian=np.asarray(diagonal, dtype=float),
        structure=structure,
    )
    return {
        "detcorr_available": True,
        "result": result.to_dict(),
        "value": float(result.value),
    }


def plot_density(
    ax: Any,
    grid: np.ndarray,
    densities: Mapping[str, np.ndarray],
    labels: Mapping[str, str] | None = None,
) -> None:
    for name, density in densities.items():
        ax.plot(grid, density, label=(labels or {}).get(name, name))
    ax.set_ylabel("posterior density")
    ax.legend(frameon=False)
