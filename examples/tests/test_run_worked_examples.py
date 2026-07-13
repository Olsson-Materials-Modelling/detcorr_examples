from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from examples import run_worked_examples as runner


def _minimal_png() -> bytes:
    return b"\x89PNG\r\n\x1a\n" + b"\x00" * 16


def test_relative_outputs_root_is_resolved_before_subprocess_cwd_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    slug = "contract_fixture"
    example = {
        "slug": slug,
        "tier": "illustrative",
        "data_basis": "test fixture",
        "main_figure": "main.png",
    }
    captured: dict[str, object] = {}
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MPLCONFIGDIR", raising=False)
    monkeypatch.setattr(runner, "WORKED_EXAMPLES", (example,))
    monkeypatch.setattr(runner, "_entry_point", lambda _root: Path("fixture_demo.py"))

    def fake_run(command, *, cwd, env, check):
        assert check is True
        captured["cwd"] = cwd
        captured["matplotlib_config"] = Path(env["MPLCONFIGDIR"])
        assert captured["matplotlib_config"].is_dir()
        outputs = Path(command[command.index("--outputs") + 1])
        captured["outputs"] = outputs
        outputs.mkdir(parents=True)
        (outputs / "summary.json").write_text(json.dumps({"demo": slug}), encoding="utf-8")
        (outputs / "main.png").write_bytes(_minimal_png())
        (outputs / "secondary.png").write_bytes(_minimal_png())

    monkeypatch.setattr(runner.subprocess, "run", fake_run)

    assert runner.main(["--outputs-root", "relative-results"]) == 0
    assert captured["cwd"] == runner.REPOSITORY
    assert captured["outputs"] == (tmp_path / "relative-results" / slug).resolve()
    assert not captured["matplotlib_config"].exists()


def test_existing_matplotlib_config_is_preserved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    slug = "contract_fixture"
    configured = tmp_path / "matplotlib-config"
    configured.mkdir()
    monkeypatch.setenv("MPLCONFIGDIR", str(configured))
    monkeypatch.setattr(
        runner,
        "WORKED_EXAMPLES",
        (
            {
                "slug": slug,
                "tier": "illustrative",
                "data_basis": "test fixture",
                "main_figure": "main.png",
            },
        ),
    )
    monkeypatch.setattr(runner, "_entry_point", lambda _root: Path("fixture_demo.py"))

    def fake_run(command, *, cwd, env, check):
        assert cwd == runner.REPOSITORY
        assert check is True
        assert env["MPLCONFIGDIR"] == str(configured)
        outputs = Path(command[command.index("--outputs") + 1])
        outputs.mkdir(parents=True)
        (outputs / "summary.json").write_text(json.dumps({"demo": slug}), encoding="utf-8")
        (outputs / "main.png").write_bytes(_minimal_png())
        (outputs / "secondary.png").write_bytes(_minimal_png())

    monkeypatch.setattr(runner.subprocess, "run", fake_run)

    assert runner.main(["--outputs-root", str(tmp_path / "outputs")]) == 0
    assert configured.is_dir()


def test_temporary_matplotlib_config_is_removed_after_subprocess_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Path] = {}
    monkeypatch.delenv("MPLCONFIGDIR", raising=False)
    monkeypatch.setattr(
        runner,
        "WORKED_EXAMPLES",
        (
            {
                "slug": "contract_fixture",
                "tier": "illustrative",
                "data_basis": "test fixture",
                "main_figure": "main.png",
            },
        ),
    )
    monkeypatch.setattr(runner, "_entry_point", lambda _root: Path("fixture_demo.py"))

    def fake_run(command, *, cwd, env, check):
        del cwd, check
        captured["matplotlib_config"] = Path(env["MPLCONFIGDIR"])
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(runner.subprocess, "run", fake_run)

    with pytest.raises(subprocess.CalledProcessError):
        runner.main(["--outputs-root", str(tmp_path / "outputs")])
    assert not captured["matplotlib_config"].exists()
