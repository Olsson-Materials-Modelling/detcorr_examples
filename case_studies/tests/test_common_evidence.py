from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest

import common.evidence as evidence
from common.evidence import (
    OUTPUT_MARKER_CONTENT,
    OUTPUT_MARKER_NAME,
    _validated_output_path,
    density_distances,
    determinant_tilt_path,
    normalized_weights,
    prepare_output_directory,
)


def test_import_does_not_eagerly_resolve_removed_numpy_trapz(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the NumPy >=2.5 import surface on older compatible NumPy."""

    if not hasattr(np, "trapezoid"):
        pytest.skip("NumPy predates np.trapezoid")
    with monkeypatch.context() as patch:
        patch.delattr(np, "trapz", raising=False)
        reloaded = importlib.reload(evidence)
        assert reloaded.trapezoid is np.trapezoid
    importlib.reload(evidence)


def test_trapezoid_selector_matches_the_available_numpy_api() -> None:
    expected = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    assert evidence.trapezoid is expected


def test_output_directory_is_immutable_without_explicit_force(tmp_path: Path) -> None:
    output = prepare_output_directory(tmp_path / "result")
    (output / "summary.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="--force"):
        prepare_output_directory(output)

    replaced = prepare_output_directory(output, force=True)
    assert replaced == output.resolve()
    assert {path.name for path in replaced.iterdir()} == {OUTPUT_MARKER_NAME}
    assert (replaced / OUTPUT_MARKER_NAME).read_text(encoding="utf-8") == OUTPUT_MARKER_CONTENT


def test_force_refuses_unmanaged_nonempty_directory(tmp_path: Path) -> None:
    output = tmp_path / "unmanaged"
    output.mkdir()
    (output / "evidence.txt").write_text("preserve\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unmanaged"):
        prepare_output_directory(output, force=True)
    assert (output / "evidence.txt").read_text(encoding="utf-8") == "preserve\n"


def test_output_directory_rejects_symlink_target(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "linked-result"
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink component"):
        prepare_output_directory(link, force=True)


def test_output_directory_rejects_symlinked_parent(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "alias"
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink component"):
        prepare_output_directory(link / "results", force=True)
    assert not (real / "results").exists()


def test_output_path_validator_rejects_critical_roots() -> None:
    for protected in (Path("/"), Path.home(), Path.cwd()):
        with pytest.raises(ValueError, match="protected"):
            _validated_output_path(protected)


def test_normalized_weights_are_stable() -> None:
    weights = normalized_weights(np.array([-10_000.0, -10_001.0, -10_002.0]))
    assert np.isclose(weights.sum(), 1.0)
    assert np.all(weights > 0.0)


def test_density_distance_identity_and_separation() -> None:
    grid = np.linspace(-5.0, 5.0, 2001)
    p = np.exp(-0.5 * grid**2)
    q = np.exp(-0.5 * (grid - 1.0) ** 2)
    identity = density_distances(grid, p, p)
    shifted = density_distances(grid, p, q)
    assert identity["total_variation"] < 1.0e-12
    assert shifted["total_variation"] > 0.2
    assert 0.0 < shifted["jensen_shannon_nats"] < 1.0


def test_tilt_path_derivative_matches_finite_difference() -> None:
    theta = np.linspace(-5.0, 5.0, 4001)
    graph_log = -0.5 * theta**2
    correction = 0.7 * theta
    rows = determinant_tilt_path(
        theta,
        graph_log,
        correction,
        t_values=np.array([0.49, 0.50, 0.51]),
    )
    finite_difference = (rows[2]["mean"] - rows[0]["mean"]) / 0.02
    assert np.isclose(finite_difference, rows[1]["mean_derivative_exact"], atol=2.0e-4)
