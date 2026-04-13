"""Base protocol and shared types for the ML model system.

All model loaders (ONNX, scikit-learn, etc.) implement the
:class:`ModelProvider` protocol so they can be registered in
:class:`~nexa_backtest.models.registry.ModelRegistry` and called uniformly
from :meth:`~nexa_backtest.context.TradingContext.predict`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ModelProvider(Protocol):
    """Any ML model that can make predictions.

    Implementations must be lazy: the model file should not be loaded at
    construction time. Loading happens on the first :meth:`predict` or
    :meth:`validate` call.
    """

    @property
    def name(self) -> str:
        """Unique name used to look up this model in the registry."""
        ...

    @property
    def input_schema(self) -> dict[str, type]:
        """Expected input features.

        Keys are feature names; values are Python types (``float``, ``int``).
        """
        ...

    @property
    def output_schema(self) -> dict[str, type]:
        """Expected output fields.

        Keys are output field names; values are Python types.
        """
        ...

    def predict(self, features: dict[str, Any]) -> dict[str, Any]:
        """Run inference with the supplied feature values.

        Args:
            features: Input feature dictionary. Keys must match
                :attr:`input_schema`. Extra keys are ignored with a warning.

        Returns:
            Output dictionary whose keys match :attr:`output_schema`.

        Raises:
            :class:`~nexa_backtest.exceptions.ModelLoadError`: If the model
                file cannot be loaded.
            :class:`~nexa_backtest.exceptions.ModelInputError`: If required
                features are missing or have the wrong type.
            :class:`~nexa_backtest.exceptions.ModelInferenceError`: If the
                underlying runtime raises an error during inference.
        """
        ...

    def validate(self) -> ModelValidationResult:
        """Load the model and verify its inputs/outputs match the declared schema.

        Returns:
            :class:`ModelValidationResult` describing whether the model is
            compatible with its declared schema.
        """
        ...


@dataclass(frozen=True)
class ModelValidationResult:
    """Result from validating a single registered model.

    Attributes:
        model_name: Name of the model that was validated.
        valid: ``True`` if the model loaded and the schema matched.
        error: Human-readable error description, or ``None`` if valid.
        actual_inputs: Input names reported by the model itself, or ``None``
            if the model could not be loaded.
        actual_outputs: Output names reported by the model itself, or ``None``
            if the model could not be loaded.
        load_time_ms: Time taken to load and validate the model in milliseconds.
    """

    model_name: str
    valid: bool
    error: str | None
    actual_inputs: list[str] | None
    actual_outputs: list[str] | None
    load_time_ms: int
