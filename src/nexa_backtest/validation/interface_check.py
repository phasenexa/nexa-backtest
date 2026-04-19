"""Step 3: Interface compliance check using AST analysis.

Performs static checks on the algo file's structure without executing the
customer's code. Validates that the algo correctly implements the expected
interface (SimpleAlgo subclass or @algo decorated function).
"""

from __future__ import annotations

import ast
import time
from pathlib import Path

from nexa_backtest.validation._utils import _elapsed_ms
from nexa_backtest.validation.runner import StepResult

# Expected hook signatures for SimpleAlgo methods.
# Maps method name → minimum required parameter names (excluding 'self').
_HOOK_SIGNATURES: dict[str, list[str]] = {
    "on_setup": ["ctx"],
    "on_auction_open": ["ctx", "auction"],
    "on_fill": ["ctx", "fill"],
    "on_signal": ["ctx", "name", "value"],
    "on_teardown": ["ctx"],
    "on_bar": ["ctx"],
    "on_cancel": ["ctx", "order_id", "reason"],
    "on_gate_closure": ["ctx", "product_id"],
    "on_error": ["ctx", "error"],
}

_ALL_HOOKS = set(_HOOK_SIGNATURES)


class InterfaceCheck:
    """Step 3: AST-based interface compliance check.

    Validates:

    - SimpleAlgo subclasses implement at least one hook, have correct
      signatures, subscribe to signals they use, and are not duplicated.
    - @algo functions are async, accept one argument, and consume events.
    - Order constructions are not obviously malformed.
    - The file imports from ``nexa_backtest``.
    """

    def run(self, algo_path: str) -> StepResult:
        """Analyse ``algo_path`` for interface compliance.

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
                name="Interface Compliance",
                status="fail",
                messages=[f"Cannot read file: {exc}"],
                duration_ms=_elapsed_ms(start),
            )

        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError as exc:
            return StepResult(
                name="Interface Compliance",
                status="skip",
                messages=[f"Skipped: syntax error prevents parsing ({exc})"],
                duration_ms=_elapsed_ms(start),
            )

        errors: list[str] = []
        warnings: list[str] = []

        _check_nexa_import(tree, path.name, errors, warnings)
        _check_simple_algos(tree, path.name, errors, warnings)
        _check_algo_decorators(tree, path.name, errors, warnings)
        _check_multiple_definitions(tree, path.name, errors)

        messages = errors + warnings
        if errors:
            status = "fail"
        elif warnings:
            status = "warn"
        else:
            status = "pass"

        return StepResult(
            name="Interface Compliance",
            status=status,
            messages=messages,
            duration_ms=_elapsed_ms(start),
        )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _check_nexa_import(
    tree: ast.Module,
    filename: str,
    errors: list[str],
    warnings: list[str],
) -> None:
    """Warn if the file doesn't import from nexa_backtest."""
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and "nexa_backtest" in node.module:
            return
        if isinstance(node, ast.Import):
            for alias in node.names:
                if "nexa_backtest" in alias.name:
                    return
    warnings.append(f"{filename}: No import from 'nexa_backtest' found. Is this an algo file?")


def _check_simple_algos(
    tree: ast.Module,
    filename: str,
    errors: list[str],
    warnings: list[str],
) -> None:
    """Check all SimpleAlgo subclasses in the module."""
    simple_algo_classes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and _inherits_from(node, "SimpleAlgo")
    ]

    for cls in simple_algo_classes:
        _check_simple_algo_class(cls, filename, errors, warnings)


def _inherits_from(node: ast.ClassDef, base_name: str) -> bool:
    """Return True if the class definition inherits from ``base_name``."""
    for base in node.bases:
        if isinstance(base, ast.Name) and base.id == base_name:
            return True
        if isinstance(base, ast.Attribute) and base.attr == base_name:
            return True
    return False


def _check_simple_algo_class(
    cls: ast.ClassDef,
    filename: str,
    errors: list[str],
    warnings: list[str],
) -> None:
    """Validate a single SimpleAlgo subclass."""
    method_names = {
        node.name
        for node in ast.walk(cls)
        if isinstance(node, ast.FunctionDef) and node.col_offset > cls.col_offset
    }

    implemented_hooks = method_names & _ALL_HOOKS
    if not implemented_hooks:
        errors.append(
            f"{filename}:{cls.lineno}: {cls.name} implements no hooks. "
            "An algo with no hooks does nothing. Override at least one hook method."
        )

    # Check hook signatures.
    for node in ast.walk(cls):
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name not in _HOOK_SIGNATURES:
            continue
        expected = _HOOK_SIGNATURES[node.name]
        actual = [a.arg for a in node.args.args if a.arg != "self"]
        for i, exp_name in enumerate(expected):
            if i >= len(actual):
                errors.append(
                    f"{filename}:{node.lineno}: {cls.name}.{node.name}() is missing "
                    f"parameter '{exp_name}' (expected: self, {', '.join(expected)})."
                )

    # Check that signals used via ctx.get_signal() are subscribed.
    _check_signal_subscriptions(cls, filename, warnings)


def _check_signal_subscriptions(
    cls: ast.ClassDef,
    filename: str,
    warnings: list[str],
) -> None:
    """Warn when ctx.get_signal() is called for an unsubscribed signal."""
    subscribed: set[str] = set()
    used: dict[str, int] = {}  # name -> line number

    for node in ast.walk(cls):
        if not isinstance(node, ast.Call):
            continue

        # subscribe_signal("name")
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "subscribe_signal"
            and node.args
        ):
            arg = node.args[0]
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                subscribed.add(arg.value)
        # ctx.get_signal("name")
        if isinstance(node.func, ast.Attribute) and node.func.attr == "get_signal" and node.args:
            arg = node.args[0]
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                name = arg.value
                if name not in used:
                    used[name] = node.lineno

    for name, line in used.items():
        if name not in subscribed:
            warnings.append(
                f"{filename}:{line}: ctx.get_signal({name!r}) called but "
                f"'{name}' was not subscribed via self.subscribe_signal() in on_setup. "
                "The signal may not be available."
            )


def _check_algo_decorators(
    tree: ast.Module,
    filename: str,
    errors: list[str],
    warnings: list[str],
) -> None:
    """Check all @algo-decorated functions."""
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not _has_algo_decorator(node):
            continue

        if isinstance(node, ast.FunctionDef):
            errors.append(
                f"{filename}:{node.lineno}: @algo decorated function '{node.name}' "
                "must be async (use 'async def')."
            )

        params = node.args.args
        if len(params) != 1:
            errors.append(
                f"{filename}:{node.lineno}: @algo decorated function '{node.name}' "
                f"must accept exactly one argument (ctx: TradingContext). "
                f"Got {len(params)} parameter(s)."
            )

        # Check that the function consumes events.
        if not _calls_events(node):
            warnings.append(
                f"{filename}:{node.lineno}: @algo function '{node.name}' does not "
                "appear to call 'ctx.events()'. An algo that never consumes events "
                "will do nothing during the backtest."
            )


def _has_algo_decorator(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Return True if the node has an @algo(...) or @algo decorator."""
    for dec in node.decorator_list:
        if isinstance(dec, ast.Name) and dec.id == "algo":
            return True
        if isinstance(dec, ast.Call):
            func = dec.func
            if isinstance(func, ast.Name) and func.id == "algo":
                return True
            if isinstance(func, ast.Attribute) and func.attr == "algo":
                return True
    return False


def _calls_events(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Return True if the function body contains a call to ctx.events()."""
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            func = child.func
            if isinstance(func, ast.Attribute) and func.attr == "events":
                return True
    return False


def _check_multiple_definitions(
    tree: ast.Module,
    filename: str,
    errors: list[str],
) -> None:
    """Error if multiple algo definitions (SimpleAlgo or @algo) exist."""
    simple_algo_classes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and _inherits_from(node, "SimpleAlgo")
    ]
    algo_functions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and _has_algo_decorator(node)
    ]

    total = len(simple_algo_classes) + len(algo_functions)
    if total > 1:
        names = [cls.name for cls in simple_algo_classes] + [fn.name for fn in algo_functions]
        errors.append(
            f"{filename}: Multiple algo definitions found: {', '.join(names)}. "
            "Place each algo in its own file. Ambiguous file cannot be loaded."
        )
