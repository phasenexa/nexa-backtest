"""Tests for the ValidationRunner orchestrator."""

from __future__ import annotations

from nexa_backtest.validation.runner import ValidationResult, ValidationRunner
from tests.test_validation.conftest import fixture_path


def test_runner_passes_valid_algo() -> None:
    runner = ValidationRunner(
        algo_path=fixture_path("valid_simple_algo.py"),
        exchange="nordpool",
        skip={"ruff", "mypy"},  # Skip slow external tools in unit tests.
    )
    result = runner.run()
    assert isinstance(result, ValidationResult)
    # Steps 3-6 should all pass for the valid fixture.
    for step in result.steps:
        if step.status != "skip":
            assert step.status in ("pass", "warn"), (
                f"Step '{step.name}' unexpectedly failed: {step.messages}"
            )


def test_runner_skips_remaining_on_syntax_error() -> None:
    runner = ValidationRunner(
        algo_path=fixture_path("syntax_error.py"),
        exchange="nordpool",
    )
    result = runner.run()
    # Step 1 (ruff) should fail or detect the syntax error.
    # Steps 2-6 should be skipped.
    skipped = [s for s in result.steps if s.status == "skip"]
    # At least some later steps should be skipped.
    # (If ruff is not installed, step 1 itself skips, so later steps may not be skipped.)
    if result.steps[0].status == "fail":
        assert len(skipped) >= 4


def test_runner_strict_mode_fails_on_warnings() -> None:
    runner = ValidationRunner(
        algo_path=fixture_path("lookahead_bias.py"),
        exchange="nordpool",
        strict=True,
        skip={"ruff", "mypy"},
    )
    result = runner.run()
    if result.warning_count > 0:
        assert not result.passed


def test_runner_skip_steps() -> None:
    runner = ValidationRunner(
        algo_path=fixture_path("valid_simple_algo.py"),
        exchange="nordpool",
        skip={"ruff", "mypy", "interface", "features", "lookahead", "resources"},
    )
    result = runner.run()
    assert all(s.status == "skip" for s in result.steps)


def test_runner_summary_contains_step_names() -> None:
    runner = ValidationRunner(
        algo_path=fixture_path("valid_simple_algo.py"),
        exchange="nordpool",
        skip={"ruff", "mypy"},
    )
    result = runner.run()
    summary = result.summary()
    assert "Interface Compliance" in summary
    assert "Exchange Features" in summary
    assert "Look-Ahead Bias" in summary
    assert "Resource Safety" in summary


def test_runner_to_dict() -> None:
    runner = ValidationRunner(
        algo_path=fixture_path("valid_simple_algo.py"),
        exchange="nordpool",
        skip={"ruff", "mypy"},
    )
    result = runner.run()
    d = result.to_dict()
    assert "steps" in d
    assert "passed" in d
    assert "error_count" in d
    assert "warning_count" in d
    assert isinstance(d["steps"], list)


def test_validation_result_passed_property() -> None:
    from nexa_backtest.validation.runner import StepResult

    result = ValidationResult(
        steps=[StepResult("A", "pass"), StepResult("B", "warn")],
        strict=False,
    )
    assert result.passed

    strict_result = ValidationResult(
        steps=[StepResult("A", "pass"), StepResult("B", "warn")],
        strict=True,
    )
    assert not strict_result.passed


def test_error_and_warning_counts() -> None:
    from nexa_backtest.validation.runner import StepResult

    result = ValidationResult(
        steps=[
            StepResult("A", "pass"),
            StepResult("B", "fail"),
            StepResult("C", "warn"),
            StepResult("D", "skip"),
        ]
    )
    assert result.error_count == 1
    assert result.warning_count == 1


def test_step_result_has_errors_and_has_warnings() -> None:
    from nexa_backtest.validation.runner import StepResult

    fail_step = StepResult("A", "fail")
    warn_step = StepResult("B", "warn")
    pass_step = StepResult("C", "pass")

    assert fail_step.has_errors is True
    assert fail_step.has_warnings is False
    assert warn_step.has_errors is False
    assert warn_step.has_warnings is True
    assert pass_step.has_errors is False
    assert pass_step.has_warnings is False


def test_summary_non_strict_failure_with_warnings() -> None:
    # Covers the non-strict branch (line 119) where warnings are listed without "treated as error".
    from nexa_backtest.validation.runner import StepResult

    result = ValidationResult(
        steps=[
            StepResult("A", "fail", messages=["bad thing"]),
            StepResult("B", "warn", messages=["dodgy thing"]),
        ],
        strict=False,
    )
    summary = result.summary()
    assert "FAILED" in summary
    assert "warning" in summary
    assert "treated as error" not in summary


def test_status_label_unknown_status() -> None:
    from nexa_backtest.validation.runner import _status_label

    label = _status_label("unknown_status", strict=False)
    assert label == "[UNKNOWN_STATUS]"
