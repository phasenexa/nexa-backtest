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
import os
import sys
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from types import ModuleType
from typing import Any

import click

from nexa_backtest.algo import SimpleAlgo
from nexa_backtest.cli.validate import validate_command
from nexa_backtest.engines.backtest import BacktestEngine
from nexa_backtest.engines.shared import MAX_ALGOS, SharedReplayEngine
from nexa_backtest.exceptions import AlgoError, NexaBacktestError
from nexa_backtest.models.registry import ModelRegistry


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
@click.option(
    "--model",
    "model_specs",
    multiple=True,
    help=(
        "Register an ML model as 'name:path'. The loader is inferred from the extension "
        "(.onnx → ONNX, .pkl/.joblib → scikit-learn). Repeat for multiple models."
    ),
)
@click.option(
    "--quiet",
    "-q",
    is_flag=True,
    default=False,
    help="Suppress non-essential output including the first-run notice.",
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
    model_specs: tuple[str, ...],
    quiet: bool,
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

    Use --model name:path to register ML models. The loader is inferred from
    the file extension (.onnx → ONNX, .pkl/.joblib → scikit-learn).
    """
    model_registry = _build_model_registry(model_specs) if model_specs else None

    if run_validation:
        from nexa_backtest.validation.runner import ValidationRunner

        click.echo(f"\nValidating {algo_file} against {exchange}...\n")
        runner = ValidationRunner(
            algo_path=algo_file, exchange=exchange, strict=strict, models=model_registry
        )
        val_result = runner.run()
        click.echo(val_result.summary())
        if not val_result.passed:
            raise click.ClickException("Validation failed. Fix the issues above before running.")
        click.echo("")

    try:
        algo_or_class = _load_algo(algo_file)
    except (NexaBacktestError, AlgoError) as exc:
        raise click.ClickException(str(exc)) from exc

    algo = algo_or_class() if isinstance(algo_or_class, type) else algo_or_class

    engine = BacktestEngine(
        algo=algo,
        exchange=exchange,
        start=start.date(),
        end=end.date(),
        products=list(products),
        data_dir=Path(data_dir),
        capital=Decimal(str(capital)),
        models=model_registry,
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

    if not quiet and not os.environ.get("NEXA_QUIET"):
        from nexa_backtest.cli.notice import maybe_show_notice

        maybe_show_notice()


@cli.command("compare")
@click.argument("algo_specs", nargs=-1, required=True)
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
    help="Directory containing market data files.",
)
@click.option(
    "--capital",
    default=100_000.0,
    show_default=True,
    help="Starting capital in EUR (applied equally to all algos).",
)
@click.option(
    "--output",
    default=None,
    help=(
        "Write comparison report to this file path. Format is inferred from the "
        "extension: .html (HTML report) or .json (JSON export). "
        "Summary is always printed to stdout."
    ),
)
@click.option(
    "--quiet",
    "-q",
    is_flag=True,
    default=False,
    help="Suppress non-essential output including the first-run notice.",
)
def compare_command(
    algo_specs: tuple[str, ...],
    exchange: str,
    start: datetime,
    end: datetime,
    products: tuple[str, ...],
    data_dir: str,
    capital: float,
    output: str | None,
    quiet: bool,
) -> None:
    """Run multiple algos against the same data and compare results.

    ALGO_SPECS are algo file paths, optionally prefixed with a display name
    and/or a class name:

    \b
        # Separate files, one class each
        nexa compare conservative:algos/conservative.py aggressive:algos/aggressive.py \\
            --exchange nordpool --start 2026-03-01 --end 2026-03-31 \\
            --products NO1_DA --data-dir ./data

    \b
        # Single file containing multiple SimpleAlgo subclasses
        nexa compare \\
            conservative:examples/multi_algo_comparison.py:ConservativeAlgo \\
            moderate:examples/multi_algo_comparison.py:ModerateAlgo \\
            aggressive:examples/multi_algo_comparison.py:AggressiveAlgo \\
            --exchange nordpool --start 2026-03-01 --end 2026-03-31 \\
            --products NO1_DA --data-dir ./data

    If no display name is given (just a path), the filename without extension
    is used as the display name.  Maximum 8 algos.
    """
    if len(algo_specs) > MAX_ALGOS:
        raise click.ClickException(
            f"Too many algos: got {len(algo_specs)}, maximum is {MAX_ALGOS}.  "
            "More than that makes the comparison report unreadable."
        )
    if len(algo_specs) < 2:
        raise click.ClickException("At least 2 algo specs are required for a comparison.")

    algos: dict[str, Any] = {}
    for spec in algo_specs:
        parts = spec.split(":", 2)
        if len(parts) == 3:
            display_name, algo_path, class_name = parts
        elif len(parts) == 2:
            display_name, algo_path = parts
            class_name = None
        else:
            algo_path = spec
            display_name = Path(algo_path).stem
            class_name = None

        if display_name in algos:
            raise click.ClickException(
                f"Duplicate display name '{display_name}'.  "
                "Use 'name:path' or 'name:path:ClassName' format to provide unique names."
            )

        try:
            algo_or_class = _load_algo(algo_path, class_name=class_name)
        except (NexaBacktestError, AlgoError) as exc:
            raise click.ClickException(str(exc)) from exc

        algo = algo_or_class() if isinstance(algo_or_class, type) else algo_or_class
        algos[display_name] = algo

    engine = SharedReplayEngine(
        algos=algos,
        exchange=exchange,
        start=start.date(),
        end=end.date(),
        products=list(products),
        data_dir=Path(data_dir),
        initial_capital=Decimal(str(capital)),
    )

    try:
        comparison = engine.run()
    except NexaBacktestError as exc:
        raise click.ClickException(str(exc)) from exc
    except Exception as exc:
        raise click.ClickException(f"Unexpected error during comparison: {exc}") from exc

    click.echo(comparison.summary())

    if output is not None:
        output_path = Path(output)
        suffix = output_path.suffix.lower()
        try:
            if suffix == ".html":
                comparison.to_html(output)
                click.echo(f"HTML comparison report written to: {output}")
            elif suffix == ".json":
                comparison.to_json(output)
                click.echo(f"JSON comparison export written to: {output}")
            else:
                raise click.ClickException(
                    f"Unsupported output format '{suffix}'.  "
                    "Use .html or .json for comparison reports."
                )
        except click.ClickException:
            raise
        except Exception as exc:
            raise click.ClickException(
                f"Failed to write comparison output to '{output}': {exc}"
            ) from exc

    if not quiet and not os.environ.get("NEXA_QUIET"):
        from nexa_backtest.cli.notice import maybe_show_notice

        maybe_show_notice()


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


def _load_algo(path: str, *, class_name: str | None = None) -> type[SimpleAlgo] | Any:
    """Find a runnable algo in ``path``.

    Accepts either a unique :class:`~nexa_backtest.algo.SimpleAlgo` subclass
    or a unique ``@algo``-decorated async function.  ``@algo`` functions take
    priority when both are present; an error is raised if multiple candidates
    of either kind are found.

    When ``class_name`` is provided, the discovered subclasses are filtered to
    the one with that exact name.  This allows a single file containing
    multiple :class:`~nexa_backtest.algo.SimpleAlgo` subclasses to be used
    with the ``nexa compare`` command via the ``name:path:ClassName`` spec
    format.

    Args:
        path: Path to a Python file.
        class_name: Optional name of the specific ``SimpleAlgo`` subclass to
            load.  When omitted the file must contain exactly one subclass.

    Returns:
        A :class:`~nexa_backtest.algo.SimpleAlgo` subclass (to be instantiated
        by the caller) or an ``@algo``-decorated callable (ready to pass
        directly to :class:`~nexa_backtest.engines.backtest.BacktestEngine`).

    Raises:
        :class:`click.ClickException`: If no valid algo is found, if multiple
            candidates are found without a ``class_name`` selector, or if the
            named class does not exist in the file.
    """
    module = _load_module(path)

    algo_fns = [
        obj
        for _, obj in inspect.getmembers(module, inspect.isfunction)
        if getattr(obj, "_is_algo", False)
    ]
    subclasses: list[type[SimpleAlgo]] = [
        obj
        for _, obj in inspect.getmembers(module, inspect.isclass)
        if issubclass(obj, SimpleAlgo) and obj is not SimpleAlgo
    ]

    if algo_fns:
        if len(algo_fns) > 1:
            names = ", ".join(f.__name__ for f in algo_fns)
            raise click.ClickException(
                f"Multiple @algo functions found in '{path}': {names}. "
                "Move the unused functions to a separate file."
            )
        return algo_fns[0]

    if class_name is not None:
        matched = [c for c in subclasses if c.__name__ == class_name]
        if not matched:
            available = ", ".join(c.__name__ for c in subclasses) if subclasses else "none"
            raise click.ClickException(
                f"Class '{class_name}' not found in '{path}'. "
                f"Available SimpleAlgo subclasses: {available}."
            )
        return matched[0]

    if subclasses:
        if len(subclasses) > 1:
            names = ", ".join(c.__name__ for c in subclasses)
            raise click.ClickException(
                f"Multiple SimpleAlgo subclasses found in '{path}': {names}. "
                "Use 'name:path:ClassName' format to select a specific class, "
                "or move the unused classes to a separate file."
            )
        return subclasses[0]

    raise click.ClickException(
        f"No algo found in '{path}'. "
        "Either decorate an async function with @algo or define a class that extends SimpleAlgo."
    )


def find_algo_class(path: str) -> type[SimpleAlgo]:
    """Public wrapper for use in tests (SimpleAlgo only).

    Args:
        path: Path to a Python file containing a SimpleAlgo subclass.

    Returns:
        The discovered :class:`~nexa_backtest.algo.SimpleAlgo` subclass.
    """
    result = _load_algo(path)
    if not (isinstance(result, type) and issubclass(result, SimpleAlgo)):
        raise click.ClickException(
            f"Expected a SimpleAlgo subclass in '{path}', got an @algo function."
        )
    return result


def _build_model_registry(specs: tuple[str, ...]) -> ModelRegistry:
    """Parse ``name:path`` model specs and build a :class:`ModelRegistry`.

    The loader type is inferred from the file extension:

    - ``.onnx`` → :class:`~nexa_backtest.models.onnx.ONNXModel`
    - ``.pkl``, ``.joblib`` → :class:`~nexa_backtest.models.sklearn.SklearnModel`

    For scikit-learn models, a JSON sidecar file at ``<path>.json`` (or
    ``<stem>.json`` beside the model) is read for ``input_schema``,
    ``output_schema``, and ``feature_order`` if present.

    Args:
        specs: Tuple of ``"name:path"`` strings.

    Returns:
        Populated :class:`~nexa_backtest.models.registry.ModelRegistry`.

    Raises:
        :class:`click.ClickException`: On parse errors or unsupported formats.
    """
    import json as _json

    from nexa_backtest.models.onnx import ONNXModel
    from nexa_backtest.models.registry import ModelRegistry
    from nexa_backtest.models.sklearn import SklearnModel

    registry = ModelRegistry()

    for spec in specs:
        if ":" not in spec:
            raise click.ClickException(
                f"Invalid --model spec '{spec}'. Expected format: 'name:path/to/model.onnx'."
            )
        name, _, raw_path = spec.partition(":")
        name = name.strip()
        model_path = Path(raw_path.strip())
        suffix = model_path.suffix.lower()

        if suffix == ".onnx":
            registry.register(
                ONNXModel(
                    name=name,
                    path=model_path,
                    input_schema={},
                    output_schema={},
                )
            )
        elif suffix in {".pkl", ".joblib"}:
            sidecar = model_path.with_suffix(".json")
            if not sidecar.exists():
                sidecar = model_path.parent / (model_path.stem + ".json")
            input_schema: dict[str, type] = {}
            output_schema: dict[str, type] = {}
            feature_order: list[str] = []
            if sidecar.exists():
                try:
                    data = _json.loads(sidecar.read_text(encoding="utf-8"))
                    _type_map: dict[str, type] = {"float": float, "int": int, "str": str}
                    input_schema = {
                        k: _type_map.get(v, float) for k, v in data.get("input_schema", {}).items()
                    }
                    output_schema = {
                        k: _type_map.get(v, float) for k, v in data.get("output_schema", {}).items()
                    }
                    feature_order = data.get("feature_order", list(input_schema.keys()))
                except Exception as exc:
                    raise click.ClickException(
                        f"Failed to read JSON sidecar for model '{name}': {exc}"
                    ) from exc
            else:
                feature_order = list(input_schema.keys())
            registry.register(
                SklearnModel(
                    name=name,
                    path=model_path,
                    input_schema=input_schema,
                    output_schema=output_schema,
                    feature_order=feature_order,
                )
            )
        else:
            raise click.ClickException(
                f"Unsupported model file extension '{suffix}' for model '{name}'. "
                "Supported: .onnx, .pkl, .joblib"
            )

    return registry
