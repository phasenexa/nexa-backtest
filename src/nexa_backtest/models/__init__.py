"""ML model registry for nexa-backtest.

Provides a framework-agnostic interface for registering and calling ML models
from within trading algorithms. ONNX is the recommended format; scikit-learn
pickle/joblib is supported with a security warning.
"""

from nexa_backtest.models.base import ModelProvider, ModelValidationResult
from nexa_backtest.models.registry import ModelRegistry

__all__ = [
    "ModelProvider",
    "ModelRegistry",
    "ModelValidationResult",
]
