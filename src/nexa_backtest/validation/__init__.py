"""Validation pipeline for nexa-backtest algo files.

Runs six static analysis steps before execution to catch common bugs,
type errors, unsupported exchange features, look-ahead bias, and unsafe
resource usage.

Usage::

    from nexa_backtest.validation.runner import ValidationRunner

    runner = ValidationRunner("my_algo.py", exchange="nordpool")
    result = runner.run()
    print(result.summary())
"""

from nexa_backtest.validation.runner import StepResult, ValidationResult, ValidationRunner

__all__ = ["StepResult", "ValidationResult", "ValidationRunner"]
