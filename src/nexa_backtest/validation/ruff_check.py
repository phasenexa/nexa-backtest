"""Step 1: Syntax and style check using ruff.

Shells out to ``ruff check`` with JSON output and classifies each finding
as an error (syntax errors, undefined names) or warning (style, unused
imports, etc.).
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

from nexa_backtest.validation._utils import _elapsed_ms
from nexa_backtest.validation.runner import StepResult

# Ruff rule codes that indicate hard errors (not merely style issues).
# E999 = syntax error, F821 = undefined name, F401 = unused import (warn only)
_ERROR_CODES = frozenset(
    {
        "E999",  # SyntaxError (parse failure)
        "F821",  # Undefined name
        "F811",  # Redefinition of unused name
        "F401",  # Unused import (treat as warning, but here we classify below)
    }
)

# Codes that should be reported as errors rather than warnings.
# Note: modern ruff uses "invalid-syntax" for syntax errors, older used "E999".
_HARD_ERROR_CODES = frozenset({"E999", "invalid-syntax", "F821"})

# Bundled ruff configuration for validating algo files.
# Kept as a dict so no external toml file is required.
_RUFF_CONFIG: dict[str, object] = {
    "target-version": "py311",
    "line-length": 120,
    "lint": {
        "select": ["E", "F", "W", "I", "N", "UP", "B", "SIM"],
    },
}


class RuffCheck:
    """Step 1: Syntax and style check using ruff.

    Shells out to ``ruff check --output-format json`` and classifies
    findings as errors (syntax, undefined names) or warnings (style).

    If ruff is not installed the step is skipped with an informative message.
    """

    def run(self, algo_path: str) -> StepResult:
        """Run ruff against ``algo_path``.

        Args:
            algo_path: Path to the algo Python file.

        Returns:
            :class:`~nexa_backtest.validation.runner.StepResult`.
        """
        start = time.monotonic()

        # Check ruff is available
        try:
            subprocess.run(
                ["ruff", "--version"],
                capture_output=True,
                check=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            return StepResult(
                name="Syntax & Style (ruff)",
                status="skip",
                messages=["ruff not found. Install with: pip install ruff"],
                duration_ms=_elapsed_ms(start),
            )

        # Write a temporary ruff config so we use the nexa config, not the
        # project's own ruff config (which may not exist or may differ).
        config_args = _build_config_args()

        cmd = ["ruff", "check", "--output-format", "json", *config_args, algo_path]
        proc = subprocess.run(cmd, capture_output=True, text=True)

        duration = _elapsed_ms(start)

        # ruff exits 1 when issues found, 0 when clean, 2 on internal error.
        if proc.returncode == 2:
            return StepResult(
                name="Syntax & Style (ruff)",
                status="fail",
                messages=[f"ruff internal error: {proc.stderr.strip()}"],
                duration_ms=duration,
            )

        findings: list[dict[str, object]] = []
        if proc.stdout.strip():
            try:
                findings = json.loads(proc.stdout)
            except json.JSONDecodeError:
                return StepResult(
                    name="Syntax & Style (ruff)",
                    status="fail",
                    messages=[f"Failed to parse ruff output: {proc.stdout[:200]}"],
                    duration_ms=duration,
                )

        if not findings:
            return StepResult(
                name="Syntax & Style (ruff)",
                status="pass",
                duration_ms=duration,
            )

        errors: list[str] = []
        warnings: list[str] = []

        for finding in findings:
            code = str(finding.get("code", ""))
            message = str(finding.get("message", ""))
            location = finding.get("location", {})
            if isinstance(location, dict):
                row = location.get("row", "?")
                col = location.get("column", "?")
            else:
                row, col = "?", "?"
            filename = Path(str(finding.get("filename", algo_path))).name
            text = f"{filename}:{row}:{col}: {code} {message}"

            if code in _HARD_ERROR_CODES:
                errors.append(text)
            else:
                warnings.append(text)

        messages = errors + warnings
        if errors:
            status = "fail"
        elif warnings:
            status = "warn"
        else:
            status = "pass"

        return StepResult(
            name="Syntax & Style (ruff)",
            status=status,
            messages=messages,
            duration_ms=duration,
        )


def _build_config_args() -> list[str]:
    """Return CLI args that apply the bundled nexa ruff config.

    We pass the select list inline so no config file is required.
    """
    select = "E,F,W,I,N,UP,B,SIM"
    return [
        "--select",
        select,
        "--line-length",
        "120",
        "--target-version",
        "py311",
        # Ignore the project's own config to avoid conflicts.
        "--no-cache",
        "--isolated",
    ]
