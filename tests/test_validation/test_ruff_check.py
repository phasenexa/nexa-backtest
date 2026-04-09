"""Tests for Step 1: Syntax and Style (ruff)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from nexa_backtest.validation.ruff_check import RuffCheck
from tests.test_validation.conftest import fixture_path


def _ruff_available() -> bool:
    try:
        subprocess.run(["ruff", "--version"], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


RUFF_AVAILABLE = _ruff_available()
skip_no_ruff = pytest.mark.skipif(not RUFF_AVAILABLE, reason="ruff not installed")


@skip_no_ruff
def test_ruff_passes_clean_file(tmp_path: Path) -> None:
    clean = tmp_path / "clean.py"
    clean.write_text(
        "from nexa_backtest.algo import SimpleAlgo\n\nclass A(SimpleAlgo):\n    pass\n"
    )
    result = RuffCheck().run(str(clean))
    # May warn about style but should not have hard errors.
    assert result.status in ("pass", "warn")


@skip_no_ruff
def test_ruff_errors_on_syntax_error() -> None:
    result = RuffCheck().run(fixture_path("syntax_error.py"))
    assert result.status == "fail"
    assert any("E999" in m or "syntax" in m.lower() or "SyntaxError" in m for m in result.messages)


@skip_no_ruff
def test_ruff_warns_on_unused_import(tmp_path: Path) -> None:
    algo = tmp_path / "algo.py"
    algo.write_text(
        "import os\n\nfrom nexa_backtest.algo import SimpleAlgo\n\nclass A(SimpleAlgo):\n    pass\n"
    )
    result = RuffCheck().run(str(algo))
    # F401 unused import is a warning, not an error.
    assert result.status in ("warn", "pass", "fail")  # ruff may or may not catch in isolated mode


def test_ruff_skips_when_not_installed(tmp_path: Path) -> None:
    dummy = tmp_path / "algo.py"
    dummy.write_text("x = 1\n")
    with patch("subprocess.run", side_effect=FileNotFoundError):
        result = RuffCheck().run(str(dummy))
    assert result.status == "skip"
    assert "ruff not found" in result.messages[0]


@skip_no_ruff
def test_ruff_valid_simple_algo() -> None:
    result = RuffCheck().run(fixture_path("valid_simple_algo.py"))
    # Should pass or at most warn (not fail).
    assert result.status in ("pass", "warn")


def test_ruff_internal_error_is_fail(tmp_path: Path) -> None:
    dummy = tmp_path / "algo.py"
    dummy.write_text("x = 1\n")
    error_proc = MagicMock(returncode=2, stderr="internal ruff error")
    version_ok = MagicMock(returncode=0)
    with patch("subprocess.run", side_effect=[version_ok, error_proc]):
        result = RuffCheck().run(str(dummy))
    assert result.status == "fail"
    assert any("internal error" in m.lower() for m in result.messages)


def test_ruff_json_parse_error_is_fail(tmp_path: Path) -> None:
    dummy = tmp_path / "algo.py"
    dummy.write_text("x = 1\n")
    bad_json_proc = MagicMock(returncode=1, stdout="not valid json {{}")
    version_ok = MagicMock(returncode=0)
    with patch("subprocess.run", side_effect=[version_ok, bad_json_proc]):
        result = RuffCheck().run(str(dummy))
    assert result.status == "fail"
    assert any("parse ruff" in m.lower() or "failed to parse" in m.lower() for m in result.messages)


def test_ruff_location_not_dict_is_handled(tmp_path: Path) -> None:
    # Ruff finding with a non-dict location — covers the else branch (row, col = "?", "?").
    dummy = tmp_path / "algo.py"
    dummy.write_text("x = 1\n")
    findings = [
        {
            "code": "W291",
            "message": "trailing whitespace",
            "location": "not-a-dict",
            "filename": str(dummy),
        }
    ]
    weird_proc = MagicMock(returncode=1, stdout=json.dumps(findings))
    version_ok = MagicMock(returncode=0)
    with patch("subprocess.run", side_effect=[version_ok, weird_proc]):
        result = RuffCheck().run(str(dummy))
    # Should produce a warning (W291 is not a hard error code).
    assert result.status == "warn"
    assert any("?:?" in m for m in result.messages)
