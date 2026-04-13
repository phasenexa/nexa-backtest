"""Central registry for ML models.

Models are registered by name and looked up via
:meth:`~nexa_backtest.context.TradingContext.predict` at inference time.
Registration is cheap; model files are loaded lazily on first use.
"""

from __future__ import annotations

from nexa_backtest.exceptions import ModelNotFoundError
from nexa_backtest.models.base import ModelProvider, ModelValidationResult


class ModelRegistry:
    """Registry for ML models used by trading algorithms.

    Register models before constructing a
    :class:`~nexa_backtest.engines.backtest.BacktestEngine`. Each model must
    have a unique name. The registry does **not** load model files at
    registration time — loading happens on the first
    :meth:`~nexa_backtest.models.base.ModelProvider.predict` or
    :meth:`~nexa_backtest.models.base.ModelProvider.validate` call.

    Example::

        registry = ModelRegistry()
        registry.register(ONNXModel("price_predictor", "models/price.onnx", ...))
        engine = BacktestEngine(..., models=registry)
    """

    def __init__(self) -> None:
        self._models: dict[str, ModelProvider] = {}

    def register(self, model: ModelProvider) -> None:
        """Register a model. The name must be unique within this registry.

        Args:
            model: Any object implementing the
                :class:`~nexa_backtest.models.base.ModelProvider` protocol.

        Raises:
            ValueError: If a model with the same name is already registered.
        """
        if model.name in self._models:
            raise ValueError(
                f"Model '{model.name}' is already registered. "
                "Use a different name or remove the existing model first."
            )
        self._models[model.name] = model

    def get(self, name: str) -> ModelProvider:
        """Return the model registered under ``name``.

        Args:
            name: The model name used when registering.

        Returns:
            The :class:`~nexa_backtest.models.base.ModelProvider` instance.

        Raises:
            :class:`~nexa_backtest.exceptions.ModelNotFoundError`: If no model
                with this name is registered.
        """
        if name not in self._models:
            registered = list(self._models.keys())
            if registered:
                hint = f" Registered models: {registered}."
            else:
                hint = " No models are registered."
            raise ModelNotFoundError(f"Model '{name}' is not registered.{hint}")
        return self._models[name]

    def has(self, name: str) -> bool:
        """Return ``True`` if a model with ``name`` is registered.

        Args:
            name: The model name to check.

        Returns:
            ``True`` if registered, ``False`` otherwise.
        """
        return name in self._models

    def list_models(self) -> list[str]:
        """Return a sorted list of registered model names.

        Returns:
            Sorted list of model names.
        """
        return sorted(self._models.keys())

    def validate_all(self) -> list[ModelValidationResult]:
        """Load and validate every registered model.

        Calls :meth:`~nexa_backtest.models.base.ModelProvider.validate` on
        each model in registration order. All models are checked even if an
        earlier one fails.

        Returns:
            List of :class:`~nexa_backtest.models.base.ModelValidationResult`
            objects, one per registered model.
        """
        return [model.validate() for model in self._models.values()]
