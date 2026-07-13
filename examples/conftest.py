from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

COMMON_SRC = Path(__file__).resolve().parent / "detcorr_demo_common" / "src"
if str(COMMON_SRC) not in sys.path:
    sys.path.insert(0, str(COMMON_SRC))

from detcorr_demo_common import (  # noqa: E402
    MINIMUM_DETCORR_VERSION,
    require_supported_detcorr,
)


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--require-detcorr",
        action="store_true",
        default=False,
        help=(
            f"Fail unless detcorr>={MINIMUM_DETCORR_VERSION} is importable. Use this "
            "for release/integration verification; standalone analytic runs may omit it."
        ),
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "detcorr_integration: tests that exercise the installed detcorr API",
    )
    if config.getoption("--require-detcorr"):
        os.environ["DETCORR_EXAMPLES_REQUIRE_DETCORR"] = "1"
        try:
            require_supported_detcorr(context="pytest --require-detcorr")
        except RuntimeError as exc:  # pragma: no cover - depends on environment
            raise pytest.UsageError(
                f"--require-detcorr was supplied, but its version gate failed: {exc}"
            ) from exc


def detcorr_required() -> bool:
    return os.environ.get("DETCORR_EXAMPLES_REQUIRE_DETCORR") == "1"
