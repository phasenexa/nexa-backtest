"""Step 6: Resource safety check.

Flags operations that would behave differently in backtest vs live mode,
or that could cause the backtest to hang or produce wrong results.
"""

from __future__ import annotations

import ast
import time
from pathlib import Path

from nexa_backtest.validation._utils import _elapsed_ms
from nexa_backtest.validation.runner import StepResult

# Methods/attributes that indicate time.sleep or asyncio.sleep.
_SLEEP_PATTERNS = {
    ("time", "sleep"),
    ("asyncio", "sleep"),
}

# Network call patterns: module + function/class.
_NETWORK_PATTERNS = {
    ("requests", "get"),
    ("requests", "post"),
    ("requests", "put"),
    ("requests", "delete"),
    ("requests", "head"),
    ("requests", "request"),
    ("urllib", "request"),
    ("urllib.request", "urlopen"),
    ("httpx", "get"),
    ("httpx", "post"),
    ("httpx", "request"),
    ("aiohttp", "ClientSession"),
}

# Module-level imports that suggest network use.
_NETWORK_MODULES = {"requests", "httpx", "aiohttp", "urllib"}

# datetime.now / datetime.utcnow
_WALL_CLOCK_PATTERNS = {
    ("datetime", "now"),
    ("datetime", "utcnow"),
    ("datetime", "today"),
}

# File I/O patterns (inside hook methods — warns when used in hot path).
_FILE_IO_NAMES = {"open", "read_csv", "read_parquet", "read_text", "read_bytes"}

# SimpleAlgo hot-path hooks (everything except on_setup).
_HOT_PATH_HOOKS = {
    "on_auction_open",
    "on_fill",
    "on_signal",
    "on_teardown",
    "on_bar",
    "on_cancel",
    "on_gate_closure",
    "on_error",
}

# Threading patterns.
_THREADING_PATTERNS = {
    ("threading", "Thread"),
    ("concurrent.futures", "ThreadPoolExecutor"),
    ("concurrent.futures", "ProcessPoolExecutor"),
}


class ResourceCheck:
    """Step 6: Resource safety check.

    Flags:

    - ``time.sleep()`` / ``asyncio.sleep()``: errors (use ctx.wait() instead).
    - Network calls (requests, httpx, aiohttp): errors.
    - ``datetime.now()`` / ``datetime.utcnow()``: errors (use ctx.now()).
    - File I/O inside hook methods other than on_setup: warnings.
    - Threading (threading.Thread, concurrent.futures): warnings.
    - Mutable module-level variables: warnings.
    """

    def run(self, algo_path: str) -> StepResult:
        """Analyse ``algo_path`` for resource safety issues.

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
                name="Resource Safety",
                status="fail",
                messages=[f"Cannot read file: {exc}"],
                duration_ms=_elapsed_ms(start),
            )

        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError:
            return StepResult(
                name="Resource Safety",
                status="skip",
                messages=["Skipped: syntax error prevents parsing."],
                duration_ms=_elapsed_ms(start),
            )

        errors: list[str] = []
        warnings: list[str] = []

        _check_sleep_calls(tree, path.name, errors)
        _check_network_calls(tree, path.name, errors)
        _check_wall_clock(tree, path.name, errors)
        _check_file_io_in_hot_path(tree, path.name, warnings)
        _check_threading(tree, path.name, warnings)
        _check_mutable_globals(tree, path.name, warnings)

        messages = errors + warnings
        if errors:
            status = "fail"
        elif warnings:
            status = "warn"
        else:
            status = "pass"

        return StepResult(
            name="Resource Safety",
            status=status,
            messages=messages,
            duration_ms=_elapsed_ms(start),
        )


# ---------------------------------------------------------------------------
# Pattern detectors
# ---------------------------------------------------------------------------


def _check_sleep_calls(
    tree: ast.Module,
    filename: str,
    errors: list[str],
) -> None:
    """Flag time.sleep() and asyncio.sleep()."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            pair = (func.value.id, func.attr)
            if pair in _SLEEP_PATTERNS:
                errors.append(
                    f"{filename}:{node.lineno}: {func.value.id}.{func.attr}() pauses "
                    "the real clock, not the simulated clock. The backtest will "
                    "actually wait. Use ctx.wait() for simulated delays."
                )


def _check_network_calls(
    tree: ast.Module,
    filename: str,
    errors: list[str],
) -> None:
    """Flag network library calls."""
    # Check imports first to identify network library usage.
    imported_modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_modules.add(alias.name.split(".")[0])
        if isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.add(node.module.split(".")[0])

    for mod in _NETWORK_MODULES:
        if mod in imported_modules:
            errors.append(
                f"{filename}: '{mod}' is imported. Network calls do not work in "
                "backtest mode (no real network in simulated time). "
                "Remove network calls or guard them with a live-mode check."
            )
            return  # One error per library is enough.


def _check_wall_clock(
    tree: ast.Module,
    filename: str,
    errors: list[str],
) -> None:
    """Flag datetime.now(), datetime.utcnow(), datetime.today()."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute):
            continue
        attr = func.attr
        if attr not in ("now", "utcnow", "today"):
            continue
        # Check the object is 'datetime' (Name) or 'datetime.datetime' (Attribute).
        obj = func.value
        is_datetime = (isinstance(obj, ast.Name) and obj.id == "datetime") or (
            isinstance(obj, ast.Attribute)
            and obj.attr == "datetime"
            and isinstance(obj.value, ast.Name)
            and obj.value.id == "datetime"
        )
        if is_datetime:
            errors.append(
                f"{filename}:{node.lineno}: datetime.{attr}() returns wall-clock time. "
                "Use ctx.now() to get the simulated clock time. "
                "Wall-clock time in a backtest produces wrong results."
            )


def _check_file_io_in_hot_path(
    tree: ast.Module,
    filename: str,
    warnings: list[str],
) -> None:
    """Warn about file I/O calls inside hot-path hook methods."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue

        for item in ast.walk(node):
            if not isinstance(item, ast.FunctionDef):
                continue
            if item.name not in _HOT_PATH_HOOKS:
                continue
            # Scan this method's body for file I/O.
            for child in ast.walk(item):
                if not isinstance(child, ast.Call):
                    continue
                func = child.func
                # open() builtin
                if isinstance(func, ast.Name) and func.id == "open":
                    warnings.append(
                        f"{filename}:{child.lineno}: open() called inside "
                        f"{item.name}(). File I/O in the hot path runs on every "
                        "tick. Move to on_setup() to load data once."
                    )
                # pd.read_csv(), pd.read_parquet(), Path.read_text(), etc.
                if isinstance(func, ast.Attribute) and func.attr in _FILE_IO_NAMES:
                    warnings.append(
                        f"{filename}:{child.lineno}: .{func.attr}() called inside "
                        f"{item.name}(). File I/O in the hot path runs on every "
                        "tick. Move to on_setup() to load data once."
                    )

    # Also check @algo functions: flag any file I/O inside the event loop.
    for node in ast.walk(tree):
        if not isinstance(node, ast.AsyncFunctionDef):
            continue
        if not _has_algo_decorator(node):
            continue

        # Look for async for loops (the event loop) and find file I/O inside.
        for child in ast.walk(node):
            if not isinstance(child, ast.AsyncFor):
                continue
            for inner in ast.walk(child):
                if not isinstance(inner, ast.Call):
                    continue
                func = inner.func
                if isinstance(func, ast.Name) and func.id == "open":
                    warnings.append(
                        f"{filename}:{inner.lineno}: open() called inside the "
                        "@algo event loop. Move file I/O before the loop."
                    )
                if isinstance(func, ast.Attribute) and func.attr in _FILE_IO_NAMES:
                    warnings.append(
                        f"{filename}:{inner.lineno}: .{func.attr}() called inside the "
                        "@algo event loop. Move file I/O before the loop."
                    )


def _has_algo_decorator(node: ast.AsyncFunctionDef | ast.FunctionDef) -> bool:
    for dec in node.decorator_list:
        if isinstance(dec, ast.Name) and dec.id == "algo":
            return True
        if isinstance(dec, ast.Call):
            func = dec.func
            if isinstance(func, ast.Name) and func.id == "algo":
                return True
    return False


def _check_threading(
    tree: ast.Module,
    filename: str,
    warnings: list[str],
) -> None:
    """Warn about threading usage."""
    imported_thread_modules: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if "threading" in alias.name or "concurrent" in alias.name:
                    imported_thread_modules.add(alias.name)
        if (
            isinstance(node, ast.ImportFrom)
            and node.module
            and ("threading" in node.module or "concurrent" in node.module)
        ):
            imported_thread_modules.add(node.module)

    for mod in imported_thread_modules:
        warnings.append(
            f"{filename}: '{mod}' is imported. Threading is not safe with the "
            "simulated clock and may cause non-deterministic backtest results."
        )


def _check_mutable_globals(
    tree: ast.Module,
    filename: str,
    warnings: list[str],
) -> None:
    """Warn about mutable module-level variables (lists, dicts, sets)."""
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        val = node.value
        if isinstance(val, (ast.List, ast.Dict, ast.Set)):
            # Get the first target name.
            target = node.targets[0]
            name = target.id if isinstance(target, ast.Name) else "?"
            warnings.append(
                f"{filename}:{node.lineno}: Module-level mutable variable '{name}' "
                "({}) may leak state between backtest runs. "
                "Move state into the algo class or function.".format(type(val).__name__.lower())
            )
