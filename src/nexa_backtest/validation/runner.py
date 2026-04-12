"""Validation pipeline orchestrator.

Runs all six validation steps in sequence and aggregates the results.
Steps are independent: a failure in one step does not prevent later steps
from running, *except* a syntax error in step 1 (ruff) which means the
file cannot be parsed and all remaining steps are skipped.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class StepResult:
    """Result from a single validation step.

    Attributes:
        name: Human-readable step name, e.g. ``"Syntax & Style (ruff)"``.
        status: One of ``"pass"``, ``"fail"``, ``"warn"``, or ``"skip"``.
        messages: Individual findings. Each message is one issue.
        duration_ms: How long the step took, in milliseconds.
    """

    name: str
    status: str  # "pass", "fail", "warn", "skip"
    messages: list[str] = field(default_factory=list)
    duration_ms: int = 0

    @property
    def has_errors(self) -> bool:
        """Return True if this step failed."""
        return self.status == "fail"

    @property
    def has_warnings(self) -> bool:
        """Return True if this step produced warnings."""
        return self.status == "warn"


@dataclass
class ValidationResult:
    """Aggregated result from all validation steps.

    Attributes:
        steps: Results from each step in order.
        strict: Whether warnings were treated as errors.
    """

    steps: list[StepResult]
    strict: bool = False

    @property
    def passed(self) -> bool:
        """Return True if no errors exist (warnings are ok unless strict mode).

        Returns:
            ``True`` if the algo is safe to run.
        """
        for step in self.steps:
            if step.status == "fail":
                return False
            if self.strict and step.status == "warn":
                return False
        return True

    @property
    def error_count(self) -> int:
        """Return the number of steps that failed with errors.

        Returns:
            Count of ``"fail"`` steps.
        """
        return sum(1 for s in self.steps if s.status == "fail")

    @property
    def warning_count(self) -> int:
        """Return the number of steps that produced warnings.

        Returns:
            Count of ``"warn"`` steps.
        """
        return sum(1 for s in self.steps if s.status == "warn")

    def summary(self) -> str:
        """Return a human-readable summary of all steps.

        Returns:
            Multi-line string suitable for CLI output.
        """
        lines: list[str] = []
        total = len(self.steps)
        for i, step in enumerate(self.steps, 1):
            label = _status_label(step.status, self.strict)
            lines.append(f"Step {i}/{total}: {step.name:<35} {label}  ({step.duration_ms}ms)")
            for msg in step.messages:
                lines.append(f"  {msg}")

        lines.append("")
        if self.passed:
            parts: list[str] = []
            if self.warning_count:
                n = self.warning_count
                parts.append(f"{n} warning{'s' if n != 1 else ''}")
            summary_parts = "PASSED"
            if parts:
                summary_parts += f" ({', '.join(parts)})"
            lines.append(f"Result: {summary_parts}")
        else:
            errors = self.error_count
            warns = self.warning_count
            parts = []
            if errors:
                parts.append(f"{errors} error{'s' if errors != 1 else ''}")
            if warns:
                if self.strict:
                    parts.append(f"{warns} warning{'s' if warns != 1 else ''} treated as error")
                else:
                    parts.append(f"{warns} warning{'s' if warns != 1 else ''}")
            lines.append(f"Result: FAILED ({', '.join(parts)})")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, object]:
        """Serialise to a plain dict for JSON output.

        Returns:
            Dict with ``steps``, ``passed``, ``error_count``, ``warning_count``.
        """
        return {
            "passed": self.passed,
            "strict": self.strict,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "steps": [
                {
                    "name": s.name,
                    "status": s.status,
                    "messages": s.messages,
                    "duration_ms": s.duration_ms,
                }
                for s in self.steps
            ],
        }


def _status_label(status: str, strict: bool) -> str:
    if status == "pass":
        return "[PASS]"
    if status == "fail":
        return "[FAIL]"
    if status == "warn":
        return "[WARN -> FAIL (strict)]" if strict else "[WARN]"
    if status == "skip":
        return "[SKIP]"
    return f"[{status.upper()}]"


class ValidationRunner:
    """Orchestrates the validation pipeline (six mandatory steps + optional step 7).

    Runs each step in sequence. If step 1 (syntax check) produces a syntax
    error, remaining steps are skipped because the file cannot be parsed by
    the AST-based checks.

    Step 7 (Model Compatibility) is included when a ``ModelRegistry`` is
    provided **or** when the algo source contains ``ctx.predict()`` calls.

    Args:
        algo_path: Path to the algo Python file.
        exchange: Exchange identifier, e.g. ``"nordpool"`` or ``"epex_spot"``.
        strict: Treat warnings as errors.
        skip: Names of steps to skip, e.g. ``{"ruff", "mypy"}``.
        models: Optional model registry. When provided, step 7 validates
            every registered model and cross-references names used in
            ``ctx.predict()`` calls.
    """

    def __init__(
        self,
        algo_path: str,
        exchange: str,
        strict: bool = False,
        skip: set[str] | None = None,
        models: object | None = None,
    ) -> None:
        self._algo_path = algo_path
        self._exchange = exchange
        self._strict = strict
        self._skip = skip or set()
        self._models = models  # ModelRegistry | None

    def run(self) -> ValidationResult:
        """Execute all validation steps and return the aggregated result.

        Returns:
            :class:`ValidationResult` containing per-step results.
        """
        # Import here to avoid circular imports at module load time.
        from nexa_backtest.validation.feature_check import FeatureCheck
        from nexa_backtest.validation.interface_check import InterfaceCheck
        from nexa_backtest.validation.lookahead_check import LookaheadCheck
        from nexa_backtest.validation.mypy_check import MypyCheck
        from nexa_backtest.validation.resource_check import ResourceCheck
        from nexa_backtest.validation.ruff_check import RuffCheck

        steps: list[StepResult] = []
        syntax_error = False

        # Step 1: Syntax & Style (ruff)
        ruff = RuffCheck()
        if "ruff" in self._skip:
            steps.append(StepResult(name="Syntax & Style (ruff)", status="skip"))
        else:
            result = ruff.run(self._algo_path)
            steps.append(result)
            # A syntax error means we can't parse the file — skip remaining AST steps.
            if result.status == "fail" and any(
                "syntax" in m.lower() or "SyntaxError" in m or "E999" in m or "invalid-syntax" in m
                for m in result.messages
            ):
                syntax_error = True

        # Steps 2-6: skip if syntax error.
        remaining: list[tuple[str, str, object]] = [
            ("mypy", "Type Safety (mypy)", MypyCheck()),
            ("interface", "Interface Compliance", InterfaceCheck()),
            ("features", "Exchange Features", FeatureCheck()),
            ("lookahead", "Look-Ahead Bias", LookaheadCheck()),
            ("resources", "Resource Safety", ResourceCheck()),
        ]

        for step_key, step_name, _checker in remaining:
            if syntax_error:
                steps.append(
                    StepResult(
                        name=step_name,
                        status="skip",
                        messages=["Skipped: syntax error in step 1 prevents parsing."],
                    )
                )
                continue

            if step_key in self._skip:
                steps.append(StepResult(name=step_name, status="skip"))
                continue

            if step_key == "mypy":
                from nexa_backtest.validation.mypy_check import MypyCheck as MC

                steps.append(MC().run(self._algo_path))
            elif step_key == "interface":
                from nexa_backtest.validation.interface_check import InterfaceCheck as IC

                steps.append(IC().run(self._algo_path))
            elif step_key == "features":
                from nexa_backtest.validation.feature_check import FeatureCheck as FC

                steps.append(FC().run(self._algo_path, self._exchange))
            elif step_key == "lookahead":
                from nexa_backtest.validation.lookahead_check import LookaheadCheck as LC

                steps.append(LC().run(self._algo_path))
            elif step_key == "resources":
                from nexa_backtest.validation.resource_check import ResourceCheck as RC

                steps.append(RC().run(self._algo_path))

        # Step 7 (optional): Model Compatibility
        if not syntax_error and "models" not in self._skip:
            from nexa_backtest.models.registry import ModelRegistry
            from nexa_backtest.validation.model_check import ModelCheck

            registry: ModelRegistry | None = (
                self._models if isinstance(self._models, ModelRegistry) else None
            )
            result7 = ModelCheck().run(self._algo_path, registry)
            if result7.status != "skip":
                steps.append(result7)

        return ValidationResult(steps=steps, strict=self._strict)
