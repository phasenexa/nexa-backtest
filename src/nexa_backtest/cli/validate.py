"""CLI command: ``nexa validate``.

Runs the six-step validation pipeline against an algo file and prints a
human-readable report. Exits with:

- **0**: All steps passed (warnings are ok).
- **1**: One or more steps failed.
- **2**: Strict mode: warnings treated as errors.
"""

from __future__ import annotations

import json
import os
import sys

import click

from nexa_backtest.validation.runner import ValidationRunner


@click.command("validate")
@click.argument("algo_file", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--exchange",
    required=True,
    help="Target exchange identifier, e.g. 'nordpool' or 'epex_spot'.",
)
@click.option(
    "--strict",
    is_flag=True,
    default=False,
    help="Treat warnings as errors.",
)
@click.option(
    "--skip",
    default="",
    help=(
        "Comma-separated list of steps to skip. "
        "Valid names: ruff, mypy, interface, features, lookahead, resources."
    ),
)
@click.option(
    "--json",
    "output_json",
    is_flag=True,
    default=False,
    help="Output results as JSON instead of human-readable text.",
)
@click.option(
    "--model",
    "model_specs",
    multiple=True,
    help=(
        "Register an ML model for step 7 validation as 'name:path'. "
        "The loader is inferred from the file extension (.onnx, .pkl, .joblib). "
        "Repeat for multiple models."
    ),
)
@click.option(
    "--quiet",
    "-q",
    is_flag=True,
    default=False,
    help="Suppress non-essential output including the first-run notice.",
)
def validate_command(
    algo_file: str,
    exchange: str,
    strict: bool,
    skip: str,
    output_json: bool,
    model_specs: tuple[str, ...],
    quiet: bool,
) -> None:
    """Validate ALGO_FILE against the target exchange before running.

    Runs six static analysis steps (plus step 7 when --model is used):

    \b
    1. Syntax & Style (ruff)
    2. Type Safety (mypy --strict)
    3. Interface Compliance (AST)
    4. Exchange Feature Compatibility
    5. Look-Ahead Bias Detection
    6. Resource Safety
    7. Model Compatibility (optional, when --model is provided)

    Exit codes: 0 = pass, 1 = fail, 2 = strict fail (warnings as errors).
    """
    from nexa_backtest.cli.main import _build_model_registry

    skip_set: set[str] = {s.strip() for s in skip.split(",") if s.strip()}

    model_registry = _build_model_registry(model_specs) if model_specs else None

    exchange_display = exchange.replace("_", " ").title()
    if not output_json:
        click.echo(f"\nValidating {algo_file} against {exchange_display}...\n")

    runner = ValidationRunner(
        algo_path=algo_file,
        exchange=exchange,
        strict=strict,
        skip=skip_set,
        models=model_registry,
    )
    result = runner.run()

    if output_json:
        click.echo(json.dumps(result.to_dict(), indent=2))
    else:
        click.echo(result.summary())
        if result.passed:
            click.echo(f"\nYour algo is ready to run against {exchange_display}.")
        else:
            click.echo(
                "\nFix the issues above before running the backtest.",
                err=True,
            )

    # Exit codes
    if not quiet and not os.environ.get("NEXA_QUIET"):
        from nexa_backtest.cli.notice import maybe_show_notice

        maybe_show_notice()

    if not result.passed:
        if strict and result.warning_count > 0 and result.error_count == 0:
            sys.exit(2)
        sys.exit(1)
