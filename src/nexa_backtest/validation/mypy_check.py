"""Step 2: Type safety check using mypy.

Shells out to ``mypy --strict`` and classifies findings as errors or warnings.
Missing stubs for third-party libraries are ignored; only findings in the
algo file itself are reported.
"""

from __future__ import annotations

import re
import subprocess
import time
from pathlib import Path

from nexa_backtest.validation.runner import StepResult

# mypy output line pattern: filename:line: severity: message  [error-code]
_MYPY_LINE = re.compile(
    r"^(?P<file>[^:]+):(?P<line>\d+): (?P<severity>error|warning|note): "
    r"(?P<message>.+?)(?:\s+\[(?P<code>[^\]]+)\])?$"
)


class MypyCheck:
    """Step 2: Type safety check using mypy --strict.

    Shells out to ``mypy --strict --ignore-missing-imports``. Only findings
    that originate from the algo file itself are reported; missing stubs for
    third-party libraries are silently ignored.

    If mypy is not installed the step is skipped with an informative message.
    """

    def run(self, algo_path: str) -> StepResult:
        """Run mypy against ``algo_path``.

        Args:
            algo_path: Path to the algo Python file.

        Returns:
            :class:`~nexa_backtest.validation.runner.StepResult`.
        """
        start = time.monotonic()

        try:
            subprocess.run(
                ["mypy", "--version"],
                capture_output=True,
                check=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            return StepResult(
                name="Type Safety (mypy)",
                status="skip",
                messages=["mypy not found. Install with: pip install mypy"],
                duration_ms=_elapsed_ms(start),
            )

        cmd = [
            "mypy",
            "--strict",
            "--ignore-missing-imports",
            "--no-error-summary",
            "--no-pretty",
            algo_path,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        duration = _elapsed_ms(start)

        algo_file = Path(algo_path).resolve()
        errors: list[str] = []
        warnings: list[str] = []

        for line in proc.stdout.splitlines():
            line = line.strip()
            if not line:
                continue

            # Only report findings from the algo file itself.
            m = _MYPY_LINE.match(line)
            if m:
                finding_file = Path(m.group("file")).resolve()
                if finding_file != algo_file:
                    continue
                severity = m.group("severity")
                text = f"{Path(algo_path).name}:{m.group('line')}: {m.group('message')}"
                if m.group("code"):
                    text += f"  [{m.group('code')}]"
                if severity == "error":
                    errors.append(text)
                else:
                    warnings.append(text)
            else:
                # Lines that don't match the pattern (e.g. "Found N errors in ...")
                # are suppressed. --no-error-summary should prevent most of these.
                pass

        messages = errors + warnings
        if errors:
            status = "fail"
        elif warnings:
            status = "warn"
        else:
            status = "pass"

        return StepResult(
            name="Type Safety (mypy)",
            status=status,
            messages=messages,
            duration_ms=duration,
        )


def _elapsed_ms(start: float) -> int:
    return int((time.monotonic() - start) * 1000)
