"""Validation step 7: model compatibility check.

Inspects the algo source for ``ctx.predict()`` calls using AST analysis, then
validates that each referenced model is registered and loads correctly.

This step is skipped entirely when no ``ModelRegistry`` is provided **and**
the algo does not call ``ctx.predict()``.
"""

from __future__ import annotations

import ast
import time
from pathlib import Path

from nexa_backtest.models.registry import ModelRegistry
from nexa_backtest.validation.runner import StepResult


class ModelCheck:
    """Validation step 7: check model registry compatibility.

    Scans the algo file for ``ctx.predict(name, ...)`` calls and verifies:

    1. If the algo calls ``ctx.predict()``, a ``ModelRegistry`` was provided.
    2. Each model name referenced in the source is registered.
    3. Every registered model loads and its schema is self-consistent.

    If no models are registered **and** the algo doesn't call
    ``ctx.predict()``, the step is skipped.
    """

    def run(
        self,
        algo_path: str,
        models: ModelRegistry | None,
    ) -> StepResult:
        """Execute the model compatibility check.

        Args:
            algo_path: Path to the algo Python file.
            models: Optional model registry to validate against.

        Returns:
            A :class:`~nexa_backtest.validation.runner.StepResult` with
            status ``"pass"``, ``"fail"``, or ``"skip"``.
        """
        start = time.monotonic()
        step_name = "Model Compatibility"

        source = Path(algo_path).read_text(encoding="utf-8")
        referenced_names = _find_predict_calls(source)

        has_registry = models is not None and bool(models.list_models())
        algo_calls_predict = bool(referenced_names)

        if not algo_calls_predict and not has_registry:
            return StepResult(name=step_name, status="skip", duration_ms=0)

        messages: list[str] = []
        failed = False

        if algo_calls_predict and models is None:
            elapsed = int((time.monotonic() - start) * 1000)
            messages.append(
                "Algo calls ctx.predict() but no ModelRegistry was provided. "
                "Pass --model name:path to register models."
            )
            return StepResult(name=step_name, status="fail", messages=messages, duration_ms=elapsed)

        if models is not None:
            registered = set(models.list_models())

            for ref_name in sorted(referenced_names):
                if ref_name not in registered:
                    failed = True
                    messages.append(
                        f"ctx.predict('{ref_name}', ...) found in source but '{ref_name}' "
                        f"is not registered. Registered models: {sorted(registered) or '(none)'}."
                    )

            for result in models.validate_all():
                if result.valid:
                    detail = f"  {result.model_name}: loaded in {result.load_time_ms}ms"
                    if result.actual_inputs:
                        detail += f", inputs: {result.actual_inputs}"
                    if result.actual_outputs:
                        detail += f", outputs: {result.actual_outputs}"
                    messages.append(detail)
                else:
                    failed = True
                    messages.append(f"  {result.model_name}: FAILED — {result.error}")

        elapsed = int((time.monotonic() - start) * 1000)
        return StepResult(
            name=step_name,
            status="fail" if failed else "pass",
            messages=messages,
            duration_ms=elapsed,
        )


def _find_predict_calls(source: str) -> list[str]:
    """Extract string literal model names from ``ctx.predict(name, ...)`` calls.

    Only picks up calls where the first argument is a string literal.
    Dynamic names (variables) are not detected.

    Args:
        source: Python source code as a string.

    Returns:
        List of unique model names found in the source.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    names: dict[str, None] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (
            isinstance(func, ast.Attribute)
            and func.attr == "predict"
            and isinstance(func.value, ast.Name)
            and func.value.id == "ctx"
        ):
            continue
        if (
            node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            names[node.args[0].value] = None

    return list(names)
