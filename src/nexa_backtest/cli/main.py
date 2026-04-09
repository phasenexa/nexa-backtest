"""CLI entry point for nexa-backtest.

Provides the ``nexa run`` command for running backtests from the command line
without writing a Python driver script.

Usage::

    nexa run examples/my_algo.py \\
        --exchange nordpool \\
        --start 2026-03-01 \\
        --end 2026-03-31 \\
        --products NO1_DA \\
        --data-dir ./data \\
        --capital 100000
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from types import ModuleType

import click

from nexa_backtest.algo import SimpleAlgo
from nexa_backtest.cli.validate import validate_command
from nexa_backtest.engines.backtest import BacktestEngine
from nexa_backtest.exceptions import NexaBacktestError


@click.group()
def cli() -> None:
    """nexa-backtest: backtesting framework for European power markets."""


cli.add_command(validate_command)


@cli.command("run")
@click.argument("algo_file", type=click.Path(exists=True, dir_okay=False))
@click.option("--exchange", required=True, help="Exchange identifier, e.g. 'nordpool'.")
@click.option(
    "--start",
    required=True,
    type=click.DateTime(formats=["%Y-%m-%d"]),
    help="First delivery date (YYYY-MM-DD).",
)
@click.option(
    "--end",
    required=True,
    type=click.DateTime(formats=["%Y-%m-%d"]),
    help="Last delivery date inclusive (YYYY-MM-DD).",
)
@click.option(
    "--products",
    required=True,
    multiple=True,
    help="Product spec, e.g. 'NO1_DA'. Repeat for multiple products.",
)
@click.option(
    "--data-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    help="Directory containing da_prices.parquet and signals/.",
)
@click.option(
    "--capital",
    default=100_000.0,
    show_default=True,
    help="Starting capital in EUR.",
)
@click.option(
    "--output",
    default=None,
    help=(
        "Write report to this file path. Format is inferred from the extension: "
        ".html (HTML report), .json (JSON export), or no extension / directory "
        "(Parquet export). Summary is always printed to stdout."
    ),
)
@click.option(
    "--validate",
    "run_validation",
    is_flag=True,
    default=False,
    help="Run the validation pipeline before starting the backtest.",
)
@click.option(
    "--strict",
    is_flag=True,
    default=False,
    help="With --validate: treat warnings as errors.",
)
def run_command(
    algo_file: str,
    exchange: str,
    start: datetime,
    end: datetime,
    products: tuple[str, ...],
    data_dir: str,
    capital: float,
    output: str | None,
    run_validation: bool,
    strict: bool,
) -> None:
    """Run a backtest from ALGO_FILE and print the PnL summary.

    ALGO_FILE must contain exactly one subclass of SimpleAlgo. If it contains
    multiple subclasses an error is raised.

    Use --output to additionally write a report file.  The format is inferred
    from the file extension: ``.html`` for an HTML report, ``.json`` for JSON,
    or a path without a recognised extension is treated as a directory for
    Parquet output.

    Use --validate to run the six-step validation pipeline before the backtest
    starts. The backtest will not run if validation fails.
    """
    if run_validation:
        from nexa_backtest.validation.runner import ValidationRunner

        click.echo(f"\nValidating {algo_file} against {exchange}...\n")
        runner = ValidationRunner(algo_path=algo_file, exchange=exchange, strict=strict)
        val_result = runner.run()
        click.echo(val_result.summary())
        if not val_result.passed:
            raise click.ClickException("Validation failed. Fix the issues above before running.")
        click.echo("")

    try:
        algo_class = _load_algo_class(algo_file)
    except NexaBacktestError as exc:
        raise click.ClickException(str(exc)) from exc

    algo = algo_class()

    engine = BacktestEngine(
        algo=algo,
        exchange=exchange,
        start=start.date(),
        end=end.date(),
        products=list(products),
        data_dir=Path(data_dir),
        capital=Decimal(str(capital)),
    )

    try:
        result = engine.run()
    except NexaBacktestError as exc:
        raise click.ClickException(str(exc)) from exc
    except Exception as exc:
        raise click.ClickException(f"Unexpected error during backtest: {exc}") from exc

    click.echo(result.summary())

    if output is not None:
        output_path = Path(output)
        suffix = output_path.suffix.lower()
        try:
            if suffix == ".html":
                result.to_html(output)
                click.echo(f"HTML report written to: {output}")
            elif suffix == ".json":
                result.to_json(output)
                click.echo(f"JSON export written to: {output}")
            else:
                result.to_parquet(output)
                click.echo(f"Parquet export written to: {output}/")
        except Exception as exc:
            raise click.ClickException(f"Failed to write output to '{output}': {exc}") from exc


# ------------------------------------------------------------------
# Algo discovery helpers
# ------------------------------------------------------------------


def _load_module(path: str) -> ModuleType:
    """Import a Python file as a module using importlib."""
    file_path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location("_nexa_user_algo", file_path)
    if spec is None or spec.loader is None:
        raise click.ClickException(f"Cannot load module from '{path}'.")

    # Add the algo's parent directory to sys.path so relative imports work
    parent = str(file_path.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_algo_class(path: str) -> type[SimpleAlgo]:
    """Find the unique :class:`~nexa_backtest.algo.SimpleAlgo` subclass in ``path``.

    Args:
        path: Path to a Python file.

    Returns:
        The single :class:`~nexa_backtest.algo.SimpleAlgo` subclass found.

    Raises:
        :class:`click.ClickException`: If zero or multiple subclasses are found.
    """
    module = _load_module(path)

    subclasses: list[type[SimpleAlgo]] = [
        obj
        for _, obj in inspect.getmembers(module, inspect.isclass)
        if issubclass(obj, SimpleAlgo) and obj is not SimpleAlgo
    ]

    if len(subclasses) == 0:
        raise click.ClickException(
            f"No SimpleAlgo subclass found in '{path}'. Define a class that extends SimpleAlgo."
        )
    if len(subclasses) > 1:
        names = ", ".join(c.__name__ for c in subclasses)
        raise click.ClickException(
            f"Multiple SimpleAlgo subclasses found in '{path}': {names}. "
            "Move the unused classes to a separate file."
        )

    return subclasses[0]


def find_algo_class(path: str) -> type[SimpleAlgo]:
    """Public wrapper around :func:`_load_algo_class` for use in tests.

    Args:
        path: Path to a Python file containing a SimpleAlgo subclass.

    Returns:
        The discovered :class:`~nexa_backtest.algo.SimpleAlgo` subclass.
    """
    return _load_algo_class(path)
