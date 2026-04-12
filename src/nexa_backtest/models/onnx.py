"""ONNX model loader for nexa-backtest.

Wraps ``onnxruntime.InferenceSession`` to provide a
:class:`~nexa_backtest.models.base.ModelProvider`-compatible interface.
ONNX is the recommended model format: portable, fast, and does not execute
arbitrary Python code on load.

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

# Type alias for onnxruntime to allow lazy import
_OrtSession = Any


class ONNXModel:
    """Load and run an ONNX model via ``onnxruntime``.

    The model file is loaded lazily on the first :meth:`predict` or
    :meth:`validate` call, not at construction time.

    Feature values are converted to numpy arrays automatically:

    - ``float`` → ``np.float32``
    - ``int`` → ``np.int64``
    - ``str`` → raises :class:`~nexa_backtest.exceptions.ModelInputError`
      (ONNX does not natively handle strings; encode before passing).

    Single-sample inference only. Each feature must be a scalar value;
    batch inference is not supported.

    Args:
        name: Unique name for this model in the registry.
        path: Path to the ``.onnx`` file.
        input_schema: Expected input features. Keys are feature names;
            values are Python types (``float`` or ``int``).
        output_schema: Expected output fields. Keys are output names;
            values are Python types.

    Example::

        model = ONNXModel(
            name="price_predictor",
            path="models/price.onnx",
            input_schema={"wind": float, "load": float, "hour": int},
            output_schema={"price_forecast": float},
        )
        result = model.predict({"wind": 3000.0, "load": 5000.0, "hour": 12})
        print(result["price_forecast"])
    """

    def __init__(
        self,
        name: str,
        path: str | Path,
        input_schema: dict[str, type],
        output_schema: dict[str, type],
    ) -> None:
        self._name = name
        self._path = Path(path)
        self._input_schema = input_schema
        self._output_schema = output_schema
        self._session: _OrtSession | None = None

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

    def _load(self) -> _OrtSession:
        """Load and cache the ONNX runtime session.

        Returns:
            An ``onnxruntime.InferenceSession`` instance.

        Raises:
            :class:`~nexa_backtest.exceptions.ModelLoadError`: If the file
                does not exist or cannot be parsed as a valid ONNX model.
        """
        if self._session is not None:
            return self._session

        if not self._path.exists():
            raise ModelLoadError(
                f"ONNX model file not found: '{self._path}'. Check the path passed to ONNXModel."
            )

        try:
            import onnxruntime as ort  # type: ignore[import-untyped]

            self._session = ort.InferenceSession(str(self._path))
            return self._session
        except ImportError as exc:
            raise ModelLoadError(
                "onnxruntime is not installed. Install it with: pip install nexa-backtest[ml]"
            ) from exc
        except Exception as exc:
            raise ModelLoadError(
                f"Failed to load ONNX model '{self._name}' from '{self._path}': {exc}"
            ) from exc

    def predict(self, features: dict[str, Any]) -> dict[str, Any]:
        """Run single-sample inference.

        Args:
            features: Input feature dictionary. Must contain all keys declared
                in :attr:`input_schema`. Extra keys are ignored with a warning.

        Returns:
            Output dictionary whose keys match :attr:`output_schema`.

        Raises:
            :class:`~nexa_backtest.exceptions.ModelLoadError`: If the model
                file cannot be loaded.
            :class:`~nexa_backtest.exceptions.ModelInputError`: If required
                features are missing or have an unsupported type (``str``).
            :class:`~nexa_backtest.exceptions.ModelInferenceError`: If
                ``onnxruntime`` raises during inference.
        """
        session = self._load()

        extra_keys = set(features.keys()) - set(self._input_schema.keys())
        if extra_keys and self._input_schema:
            warnings.warn(
                f"Model '{self._name}' received unexpected feature keys: {sorted(extra_keys)}. "
                "They will be ignored.",
                stacklevel=2,
            )

        inputs = self._build_inputs(features)

        try:
            outputs = session.run(None, inputs)
        except Exception as exc:
            raise ModelInferenceError(
                f"Model '{self._name}' raised an error during inference: {exc}"
            ) from exc

        if self._output_schema:
            output_keys = list(self._output_schema.keys())
        else:
            output_keys = [out.name for out in session.get_outputs()]
        return {key: float(outputs[i].flat[0]) for i, key in enumerate(output_keys)}

    def _build_inputs(self, features: dict[str, Any]) -> dict[str, np.ndarray]:
        """Convert the features dict to the numpy arrays expected by onnxruntime.

        When ``input_schema`` is empty, all supplied feature values are passed
        through to the model, with types inferred from the Python values.

        Args:
            features: Raw feature values from the caller.

        Returns:
            Dict mapping ONNX input names to shaped numpy arrays.

        Raises:
            :class:`~nexa_backtest.exceptions.ModelInputError`: If a required
                feature is missing or its type is unsupported.
        """
        schema = self._input_schema or {k: type(v) for k, v in features.items()}

        inputs: dict[str, np.ndarray] = {}
        for feat_name, feat_type in schema.items():
            if feat_name not in features:
                raise ModelInputError(
                    f"Model '{self._name}' requires feature '{feat_name}' "
                    f"but it was not provided. Required features: {list(schema.keys())}"
                )
            value = features[feat_name]
            if feat_type is str or isinstance(value, str):
                raise ModelInputError(
                    f"Model '{self._name}': feature '{feat_name}' has type str. "
                    "ONNX does not natively handle strings. Encode the value before passing."
                )
            if feat_type is int or isinstance(value, int):
                inputs[feat_name] = np.array([[int(value)]], dtype=np.int64)
            else:
                inputs[feat_name] = np.array([[float(value)]], dtype=np.float32)
        return inputs

    def validate(self) -> ModelValidationResult:
        """Load the ONNX file and check its inputs/outputs match the declared schema.

        Returns:
            :class:`~nexa_backtest.models.base.ModelValidationResult` with
            ``valid=True`` if the model loaded and the schema matched, or
            ``valid=False`` with an error message otherwise.
        """
        start = time.monotonic()
        try:
            session = self._load()
        except ModelLoadError as exc:
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
        actual_inputs = [inp.name for inp in session.get_inputs()]
        actual_outputs = [out.name for out in session.get_outputs()]

        declared_inputs = list(self._input_schema.keys())
        declared_outputs = list(self._output_schema.keys())

        errors: list[str] = []
        missing_inputs = set(declared_inputs) - set(actual_inputs)
        if missing_inputs:
            errors.append(
                f"Declared input(s) not found in model: {sorted(missing_inputs)}. "
                f"Model has: {actual_inputs}"
            )
        missing_outputs = set(declared_outputs) - set(actual_outputs)
        if missing_outputs:
            errors.append(
                f"Declared output(s) not found in model: {sorted(missing_outputs)}. "
                f"Model has: {actual_outputs}"
            )

        if errors:
            return ModelValidationResult(
                model_name=self._name,
                valid=False,
                error="; ".join(errors),
                actual_inputs=actual_inputs,
                actual_outputs=actual_outputs,
                load_time_ms=elapsed,
            )

        return ModelValidationResult(
            model_name=self._name,
            valid=True,
            error=None,
            actual_inputs=actual_inputs,
            actual_outputs=actual_outputs,
            load_time_ms=elapsed,
        )
