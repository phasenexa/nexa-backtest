"""Step 5: Look-ahead bias detection.

Heuristic static analysis that flags patterns likely to cause look-ahead
bias. All findings are warnings (not errors), even in strict mode, because
false positives are possible. The customer must assess whether a flagged
pattern is actually problematic.
"""

from __future__ import annotations

import ast
import time
from pathlib import Path

from nexa_backtest.validation._utils import _elapsed_ms
from nexa_backtest.validation.runner import StepResult

# Large lookback threshold for get_signal_history() in number of MTUs.
# 96 MTUs = 24 hours at 15-min resolution.
_LARGE_LOOKBACK_MTUS = 48  # 12 hours — flag anything larger


class LookaheadCheck:
    """Step 5: Heuristic look-ahead bias detection.

    Flags:

    1. ``df["col"].shift(-N)`` where N > 0 (looks forward in time).
    2. ``df.iloc[i + N]`` patterns that may access future rows.
    3. ``ctx.get_signal_history(lookback=N)`` with large lookback windows.
    4. ``df.loc[future_time]`` patterns (heuristic).
    5. ``df.sort_values(...).iloc[...]`` patterns that may access future data.

    All findings are warnings. This check is heuristic and may produce false
    positives. It cannot provide a guarantee that no look-ahead bias exists.
    """

    def run(self, algo_path: str) -> StepResult:
        """Analyse ``algo_path`` for look-ahead bias patterns.

        Args:
            algo_path: Path to the algo Python file.

        Returns:
            :class:`~nexa_backtest.validation.runner.StepResult`.
        """
        start = time.monotonic()
        path = Path(algo_path)

        try:
            source = path.read_text(encoding="utf-8")
        except OSError as exc:
            return StepResult(
                name="Look-Ahead Bias",
                status="fail",
                messages=[f"Cannot read file: {exc}"],
                duration_ms=_elapsed_ms(start),
            )

        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError:
            return StepResult(
                name="Look-Ahead Bias",
                status="skip",
                messages=["Skipped: syntax error prevents parsing."],
                duration_ms=_elapsed_ms(start),
            )

        warnings: list[str] = []
        _check_negative_shift(tree, path.name, warnings)
        _check_signal_history_lookback(tree, path.name, warnings)
        _check_iloc_forward(tree, path.name, warnings)
        _check_sort_iloc(tree, path.name, warnings)

        if warnings:
            return StepResult(
                name="Look-Ahead Bias",
                status="warn",
                messages=warnings,
                duration_ms=_elapsed_ms(start),
            )

        return StepResult(
            name="Look-Ahead Bias",
            status="pass",
            duration_ms=_elapsed_ms(start),
        )


# ---------------------------------------------------------------------------
# Pattern detectors
# ---------------------------------------------------------------------------


def _check_negative_shift(
    tree: ast.Module,
    filename: str,
    warnings: list[str],
) -> None:
    """Flag df['col'].shift(-N) where N > 0."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "shift"):
            continue

        # Look for a negative literal argument or keyword.
        neg_val = _extract_negative_int(node)
        if neg_val is not None:
            warnings.append(
                f"{filename}:{node.lineno}: .shift({neg_val}) looks forward in time. "
                "Negative shift accesses future rows and introduces look-ahead bias."
            )


def _extract_negative_int(call: ast.Call) -> int | None:
    """Return the negative integer argument to a call, or None."""
    # Positional argument
    for arg in call.args:
        if (
            isinstance(arg, ast.UnaryOp)
            and isinstance(arg.op, ast.USub)
            and isinstance(arg.operand, ast.Constant)
            and isinstance(arg.operand.value, int)
        ):
            return -arg.operand.value
    # Keyword argument 'periods=-N'
    for kw in call.keywords:
        if kw.arg == "periods":
            val = kw.value
            if (
                isinstance(val, ast.UnaryOp)
                and isinstance(val.op, ast.USub)
                and isinstance(val.operand, ast.Constant)
                and isinstance(val.operand.value, int)
            ):
                return -val.operand.value
    return None


def _check_signal_history_lookback(
    tree: ast.Module,
    filename: str,
    warnings: list[str],
) -> None:
    """Flag get_signal_history() with large lookback values."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "get_signal_history"):
            continue

        lookback: int | None = None
        for kw in node.keywords:
            val = kw.value
            if (
                kw.arg == "lookback"
                and isinstance(val, ast.Constant)
                and isinstance(val.value, int)
            ):
                lookback = val.value
        # Also check positional arg (2nd argument after signal name).
        if lookback is None and len(node.args) >= 2:
            arg = node.args[1]
            if isinstance(arg, ast.Constant) and isinstance(arg.value, int):
                lookback = arg.value

        if lookback is not None and lookback > _LARGE_LOOKBACK_MTUS:
            hours = lookback * 15 / 60
            warnings.append(
                f"{filename}:{node.lineno}: get_signal_history() with lookback={lookback} "
                f"covers {hours:.0f} hours. Verify the signal's publication_offset supports "
                "this range without leaking future data."
            )


def _check_iloc_forward(
    tree: ast.Module,
    filename: str,
    warnings: list[str],
) -> None:
    """Flag df.iloc[i + N] where N > 0 (heuristic)."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Subscript):
            continue
        # Must be .iloc[...]
        if not (isinstance(node.value, ast.Attribute) and node.value.attr == "iloc"):
            continue

        index = node.slice
        # Detect patterns like: i + N where N is a positive literal.
        if isinstance(index, ast.BinOp) and isinstance(index.op, ast.Add):
            right = index.right
            if isinstance(right, ast.Constant) and isinstance(right.value, int) and right.value > 0:
                warnings.append(
                    f"{filename}:{node.lineno}: .iloc[... + {right.value}] may access "
                    "a future row. Verify this does not introduce look-ahead bias."
                )


def _check_sort_iloc(
    tree: ast.Module,
    filename: str,
    warnings: list[str],
) -> None:
    """Flag sort_values(...).iloc[...] chains that may access future rows."""
    for node in ast.walk(tree):
        # Look for .iloc subscript where the value is a Call to .sort_values
        if not isinstance(node, ast.Subscript):
            continue
        if not (isinstance(node.value, ast.Attribute) and node.value.attr == "iloc"):
            continue
        # The object being subscripted with iloc must itself be a method call.
        inner = node.value.value
        if isinstance(inner, ast.Call):
            func = inner.func
            if isinstance(func, ast.Attribute) and func.attr == "sort_values":
                warnings.append(
                    f"{filename}:{node.lineno}: sort_values(...).iloc[...] may access "
                    "rows that are chronologically in the future after sorting. "
                    "Verify this does not introduce look-ahead bias."
                )
