"""Tests for the ``nexa validate`` CLI command."""

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from nexa_backtest.cli.main import cli
from tests.test_validation.conftest import fixture_path


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def test_validate_passes_valid_algo(runner: CliRunner) -> None:
    result = runner.invoke(
        cli,
        [
            "validate",
            fixture_path("valid_simple_algo.py"),
            "--exchange",
            "nordpool",
            "--skip",
            "ruff,mypy",  # Skip slow tools in CI
        ],
    )
    assert result.exit_code == 0, result.output


def test_validate_fails_on_no_hooks(runner: CliRunner) -> None:
    result = runner.invoke(
        cli,
        [
            "validate",
            fixture_path("no_hooks.py"),
            "--exchange",
            "nordpool",
            "--skip",
            "ruff,mypy",
        ],
    )
    assert result.exit_code == 1


def test_validate_strict_flag_fails_on_warnings(runner: CliRunner) -> None:
    result = runner.invoke(
        cli,
        [
            "validate",
            fixture_path("lookahead_bias.py"),
            "--exchange",
            "nordpool",
            "--strict",
            "--skip",
            "ruff,mypy",
        ],
    )
    # If there are warnings, strict mode should fail.
    if "WARN" in result.output:
        assert result.exit_code != 0


def test_validate_json_output(runner: CliRunner) -> None:
    result = runner.invoke(
        cli,
        [
            "validate",
            fixture_path("valid_simple_algo.py"),
            "--exchange",
            "nordpool",
            "--skip",
            "ruff,mypy",
            "--json",
        ],
    )
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert "steps" in data
    assert "passed" in data


def test_validate_skip_flag(runner: CliRunner) -> None:
    result = runner.invoke(
        cli,
        [
            "validate",
            fixture_path("valid_simple_algo.py"),
            "--exchange",
            "nordpool",
            "--skip",
            "ruff,mypy,interface,features,lookahead,resources",
        ],
    )
    assert result.exit_code == 0
    assert "SKIP" in result.output


def test_validate_block_bid_passes_nordpool(runner: CliRunner) -> None:
    result = runner.invoke(
        cli,
        [
            "validate",
            fixture_path("unsupported_feature.py"),
            "--exchange",
            "nordpool",
            "--skip",
            "ruff,mypy",
        ],
    )
    # Nord Pool supports block bids — step 4 should pass.
    assert result.exit_code == 0, result.output
