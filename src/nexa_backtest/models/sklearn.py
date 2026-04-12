"""Scikit-learn model loader for nexa-backtest.

Loads pickle or joblib model files and exposes them via the
:class:`~nexa_backtest.models.base.ModelProvider` protocol.

.. warning::

    Pickle files can execute arbitrary code when loaded.  Only use models
    you trust.  For hosted environments, export to ONNX instead.

Requires the ``nexa-backtest[ml]`` extra::

    pip install nexa-backtest[ml]
"""

from __future__ import annotations

import logging
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from nexa_backtest.exceptions import ModelInferenceError, ModelInputError, ModelLoadError
from nexa_backtest.models.base import ModelValidationResult

logger = logging.getLogger(__name__)


class SklearnModel:
    """Load and run a scikit-learn model from a pickle or joblib file.

    .. warning::

        Pickle files can execute arbitrary Python code when loaded.  Only use
        models from sources you trust.  For hosted environments, prefer ONNX.

    The model is loaded lazily on the first :meth:`predict` or
    :meth:`validate` call.  A warning is logged on first load.

    Because scikit-learn's ``predict()`` takes a positional 2-D numpy array,
    the caller must supply ``feature_order`` so that features can be reliably
    placed in the correct column order.

    Args:
        name: Unique name for this model in the registry.
        path: Path to the ``.pkl`` or ``.joblib`` file.
        input_schema: Expected input features. Keys are feature names; values
            are Python types (``float`` or ``int``).
        output_schema: Expected output fields. Keys are output names; values
            are Python types.
        feature_order: Ordered list of feature names matching the column order
            the model was trained on.  Must cover every key in
            ``input_schema``.

    Example::

        model = SklearnModel(
            name="vol_classifier",
            path="models/volatility.pkl",
            input_schema={"spread_1h": float, "volume_1h": float},
            output_schema={"regime": str},
            feature_order=["spread_1h", "volume_1h"],
        )
    """

    def __init__(
        self,
        name: str,
        path: str | Path,
        input_schema: dict[str, type],
        output_schema: dict[str, type],
        feature_order: list[str],
    ) -> None:
        self._name = name
        self._path = Path(path)
        self._input_schema = input_schema
        self._output_schema = output_schema
        self._feature_order = feature_order
        self._model: Any = None

    @property
    def name(self) -> str:
        """Unique name used to look up this model in the registry."""
        return self._name

    @property
    def input_schema(self) -> dict[str, type]:
        """Expected input feature names and their Python types."""
        return self._input_schema

    @property
    def output_schema(self) -> dict[str, type]:
        """Expected output field names and their Python types."""
        return self._output_schema

    def _load(self) -> Any:
        """Load and cache the scikit-learn model.

        Logs a security warning on first load.

        Returns:
            The loaded scikit-learn model object.

        Raises:
            :class:`~nexa_backtest.exceptions.ModelLoadError`: If the file
                does not exist or cannot be unpickled.
        """
        if self._model is not None:
            return self._model

        if not self._path.exists():
            raise ModelLoadError(
                f"Scikit-learn model file not found: '{self._path}'. "
                "Check the path passed to SklearnModel."
            )

        warnings.warn(
            f"Loading pickle model '{self._name}'. Pickle files can execute "
            "arbitrary code. Only use models from trusted sources.",
            stacklevel=3,
        )

        try:
            import joblib  # type: ignore[import-untyped]

            self._model = joblib.load(str(self._path))
        except ImportError:
            import pickle

            with open(self._path, "rb") as fh:
                self._model = pickle.load(fh)
        except Exception as exc:
            raise ModelLoadError(
                f"Failed to load scikit-learn model '{self._name}' from '{self._path}': {exc}"
            ) from exc

        return self._model

    def predict(self, features: dict[str, Any]) -> dict[str, Any]:
        """Run single-sample inference.

        Features are reordered to match ``feature_order`` before being passed
        to the underlying model's ``predict()`` method.

        Args:
            features: Input feature dictionary. Must contain all keys declared
                in :attr:`input_schema`.  Extra keys are ignored with a
                warning.

        Returns:
            Output dictionary whose keys match :attr:`output_schema`.  For
            models that expose ``predict_proba()`` and declare a matching
            output key, the probability is included too.

        Raises:
            :class:`~nexa_backtest.exceptions.ModelLoadError`: If the model
                file cannot be loaded.
            :class:`~nexa_backtest.exceptions.ModelInputError`: If required
                features are missing.
            :class:`~nexa_backtest.exceptions.ModelInferenceError`: If the
                underlying model raises during ``predict()``.
        """
        model = self._load()

        extra_keys = set(features.keys()) - set(self._input_schema.keys())
        if extra_keys:
            warnings.warn(
                f"Model '{self._name}' received unexpected feature keys: {sorted(extra_keys)}. "
                "They will be ignored.",
                stacklevel=2,
            )

        X = self._build_input_array(features)

        try:
            raw = model.predict(X)
        except Exception as exc:
            raise ModelInferenceError(
                f"Model '{self._name}' raised an error during inference: {exc}"
            ) from exc

        output_keys = list(self._output_schema.keys())
        result: dict[str, Any] = {}

        raw_value = raw[0] if hasattr(raw, "__len__") else raw
        if output_keys:
            result[output_keys[0]] = raw_value

        # If the model supports predict_proba and the schema has a matching key,
        # include the class probabilities.
        if hasattr(model, "predict_proba") and len(output_keys) > 1:
            try:
                proba = model.predict_proba(X)
                result[output_keys[1]] = (
                    proba[0].tolist() if hasattr(proba[0], "tolist") else proba[0]
                )
            except Exception:  # broad catch intentional: predict_proba is best-effort
                pass

        return result

    def _build_input_array(self, features: dict[str, Any]) -> np.ndarray:
        """Build the ordered 2-D input array expected by scikit-learn.

        Args:
            features: Raw feature values from the caller.

        Returns:
            Shape ``(1, n_features)`` float64 numpy array.

        Raises:
            :class:`~nexa_backtest.exceptions.ModelInputError`: If a required
                feature is missing.
        """
        row: list[float] = []
        for feat_name in self._feature_order:
            if feat_name not in features:
                raise ModelInputError(
                    f"Model '{self._name}' requires feature '{feat_name}' "
                    f"but it was not provided. Required features: {self._feature_order}"
                )
            row.append(float(features[feat_name]))
        return np.array([row], dtype=np.float64)

    def validate(self) -> ModelValidationResult:
        """Load the model and verify basic compatibility.

        For scikit-learn models the only structural check we can perform is
        confirming the model file loads and exposes a ``predict()`` method.
        There is no introspectable metadata equivalent to ONNX's input/output
        names.

        Returns:
            :class:`~nexa_backtest.models.base.ModelValidationResult` with
            ``actual_inputs`` set to ``feature_order`` and ``actual_outputs``
            set to the declared output schema keys.
        """
        start = time.monotonic()
        try:
            model = self._load()
        except (ModelLoadError, Exception) as exc:
            elapsed = int((time.monotonic() - start) * 1000)
            return ModelValidationResult(
                model_name=self._name,
                valid=False,
                error=str(exc),
                actual_inputs=None,
                actual_outputs=None,
                load_time_ms=elapsed,
            )

        elapsed = int((time.monotonic() - start) * 1000)

        if not hasattr(model, "predict"):
            return ModelValidationResult(
                model_name=self._name,
                valid=False,
                error=f"Loaded object does not have a predict() method: {type(model)}",
                actual_inputs=self._feature_order,
                actual_outputs=list(self._output_schema.keys()),
                load_time_ms=elapsed,
            )

        return ModelValidationResult(
            model_name=self._name,
            valid=True,
            error=None,
            actual_inputs=self._feature_order,
            actual_outputs=list(self._output_schema.keys()),
            load_time_ms=elapsed,
        )
