"""Tests for ``nexa run --validate`` integration."""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from nexa_backtest.cli.main import cli
from tests.test_validation.conftest import fixture_path


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def test_run_validate_blocks_on_failure(runner: CliRunner, tmp_path) -> None:
    """When validation fails, the backtest should not start."""
    result = runner.invoke(
        cli,
        [
            "run",
            fixture_path("no_hooks.py"),
            "--exchange",
            "nordpool",
            "--start",
            "2026-03-01",
            "--end",
            "2026-03-31",
            "--products",
            "NO1_DA",
            "--data-dir",
            str(tmp_path),
            "--validate",
        ],
    )
    # Should fail because no_hooks.py fails interface compliance.
    assert result.exit_code != 0
    assert "validation" in result.output.lower() or "Validation" in result.output


def test_run_without_validate_does_not_run_pipeline(runner: CliRunner, tmp_path) -> None:
    """Without --validate, no validation output should appear."""
    result = runner.invoke(
        cli,
        [
            "run",
            fixture_path("no_hooks.py"),
            "--exchange",
            "nordpool",
            "--start",
            "2026-03-01",
            "--end",
            "2026-03-31",
            "--products",
            "NO1_DA",
            "--data-dir",
            str(tmp_path),
            # No --validate flag.
        ],
    )
    # May fail for data reasons, but not for validation reasons.
    assert "Interface Compliance" not in result.output
    assert "Look-Ahead Bias" not in result.output
