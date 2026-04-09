"""Tests for Step 2: Type Safety (mypy)."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from nexa_backtest.validation.mypy_check import MypyCheck
from tests.test_validation.conftest import fixture_path


def _mypy_available() -> bool:
    try:
        subprocess.run(["mypy", "--version"], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


MYPY_AVAILABLE = _mypy_available()
skip_no_mypy = pytest.mark.skipif(not MYPY_AVAILABLE, reason="mypy not installed")


@skip_no_mypy
def test_mypy_passes_clean_file() -> None:
    result = MypyCheck().run(fixture_path("valid_simple_algo.py"))
    assert result.status in ("pass", "warn")


@skip_no_mypy
def test_mypy_does_not_fail_on_untyped_library(tmp_path: Path) -> None:
    """Missing stubs for third-party libraries must not fail the check."""
    algo = tmp_path / "algo.py"
    algo.write_text(
        "import pandas as pd\n"
        "from nexa_backtest.algo import SimpleAlgo\n\n"
        "class A(SimpleAlgo):\n"
        "    def on_setup(self, ctx) -> None:  # type: ignore[override]\n"
        "        df = pd.DataFrame()\n"
    )
    result = MypyCheck().run(str(algo))
    # Should not fail purely because pandas stubs are missing.
    assert result.status != "fail" or all("pandas" not in m for m in result.messages)


def test_mypy_skips_when_not_installed(tmp_path: Path) -> None:
    dummy = tmp_path / "algo.py"
    dummy.write_text("x = 1\n")
    with patch("subprocess.run", side_effect=FileNotFoundError):
        result = MypyCheck().run(str(dummy))
    assert result.status == "skip"
    assert "mypy not found" in result.messages[0]


def test_mypy_skips_on_called_process_error(tmp_path: Path) -> None:
    dummy = tmp_path / "algo.py"
    dummy.write_text("x = 1\n")
    with patch(
        "subprocess.run",
        side_effect=subprocess.CalledProcessError(1, "mypy"),
    ):
        result = MypyCheck().run(str(dummy))
    assert result.status == "skip"
    assert "mypy not found" in result.messages[0]


def test_mypy_reports_errors_from_output(tmp_path: Path) -> None:
    dummy = tmp_path / "algo.py"
    dummy.write_text("x: int = 'hello'\n")

    version_ok = MagicMock(returncode=0)
    mypy_output = MagicMock(
        stdout=f"{dummy}:1: error: Incompatible types in assignment  [assignment]\n",
        returncode=1,
    )
    with patch("subprocess.run", side_effect=[version_ok, mypy_output]):
        result = MypyCheck().run(str(dummy))

    assert result.status == "fail"
    assert any("incompatible" in m.lower() for m in result.messages)


def test_mypy_reports_warnings_from_output(tmp_path: Path) -> None:
    dummy = tmp_path / "algo.py"
    dummy.write_text("x = 1\n")

    version_ok = MagicMock(returncode=0)
    mypy_output = MagicMock(
        stdout=f"{dummy}:1: warning: Something looks off\n",
        returncode=0,
    )
    with patch("subprocess.run", side_effect=[version_ok, mypy_output]):
        result = MypyCheck().run(str(dummy))

    assert result.status == "warn"
    assert any("something" in m.lower() for m in result.messages)


def test_mypy_ignores_findings_from_other_files(tmp_path: Path) -> None:
    dummy = tmp_path / "algo.py"
    other = tmp_path / "other.py"
    dummy.write_text("x = 1\n")

    version_ok = MagicMock(returncode=0)
    # Finding from other.py — must be filtered out.
    mypy_output = MagicMock(
        stdout=f"{other}:1: error: Some error  [misc]\n",
        returncode=1,
    )
    with patch("subprocess.run", side_effect=[version_ok, mypy_output]):
        result = MypyCheck().run(str(dummy))

    # Findings from other files are suppressed → should pass.
    assert result.status == "pass"


def test_mypy_skips_empty_lines_in_output(tmp_path: Path) -> None:
    # Empty lines in mypy stdout trigger the `continue` at line 77.
    dummy = tmp_path / "algo.py"
    dummy.write_text("x: int = 'hello'\n")

    version_ok = MagicMock(returncode=0)
    mypy_output = MagicMock(
        stdout=f"\n{dummy}:1: error: Incompatible types  [assignment]\n\n",
        returncode=1,
    )
    with patch("subprocess.run", side_effect=[version_ok, mypy_output]):
        result = MypyCheck().run(str(dummy))

    assert result.status == "fail"


def test_mypy_suppresses_non_matching_output_lines(tmp_path: Path) -> None:
    # Lines that don't match the mypy output regex hit the `pass` at line 96.
    dummy = tmp_path / "algo.py"
    dummy.write_text("x = 1\n")

    version_ok = MagicMock(returncode=0)
    mypy_output = MagicMock(
        stdout="Found 1 error in 1 file (checked 1 source file)\n",
        returncode=1,
    )
    with patch("subprocess.run", side_effect=[version_ok, mypy_output]):
        result = MypyCheck().run(str(dummy))

    # Non-matching lines are suppressed → no messages.
    assert result.status == "pass"
    assert not result.messages
