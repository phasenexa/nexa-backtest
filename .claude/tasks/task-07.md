# Task 07: ML Model Registry

## Goal

Let customers use trained ML models inside their algos. Register a
model by name, call `ctx.predict()` with a features dict, get a
result. The algo doesn't know or care what framework the model was
trained in.

After this task, a quant can train an XGBoost price forecaster in
a Jupyter notebook, export it to ONNX, register it with the
backtester, and use its predictions to make trading decisions. Same
model works in backtest, paper, and live modes.

---

## What to build

### 1. `models/registry.py` - Model Registry

Central registry that holds named models and provides lookup.

```python
class ModelRegistry:
    """Registry for ML models used by trading algos."""

    def register(self, model: ModelProvider) -> None:
        """Register a model. Name must be unique."""

    def get(self, name: str) -> ModelProvider:
        """Get a model by name. Raises ModelNotFoundError if missing."""

    def has(self, name: str) -> bool: ...

    def list_models(self) -> list[str]: ...

    def validate_all(self) -> list[ModelValidationResult]:
        """Load and validate every registered model. Returns a list
        of results, one per model, each indicating whether the model
        loaded correctly and whether its actual inputs/outputs match
        the declared schema."""
```

### 2. `models/base.py` - Model Protocol and Types

```python
class ModelProvider(Protocol):
    """Any ML model that can make predictions."""

    @property
    def name(self) -> str: ...

    @property
    def input_schema(self) -> dict[str, type]:
        """Expected input features. Keys are feature names, values
        are Python types (float, int, str)."""
        ...

    @property
    def output_schema(self) -> dict[str, type]:
        """Expected output fields."""
        ...

    def predict(self, features: dict[str, Any]) -> dict[str, Any]:
        """Run inference. Input must match input_schema. Output
        will match output_schema."""
        ...

    def validate(self) -> ModelValidationResult:
        """Load the model and verify it works. Called during
        registry.validate_all() and optionally at startup."""
        ...

@dataclass(frozen=True)
class ModelValidationResult:
    model_name: str
    valid: bool
    error: str | None              # None if valid
    actual_inputs: list[str] | None   # What the model actually expects
    actual_outputs: list[str] | None  # What the model actually produces
    load_time_ms: int
```

### 3. `models/onnx.py` - ONNX Model Loader

The recommended model format. Wraps `onnxruntime.InferenceSession`.

```python
class ONNXModel:
    """Load and run an ONNX model.

    Args:
        name: Unique name for this model in the registry.
        path: Path to the .onnx file.
        input_schema: Expected input features and their types.
        output_schema: Expected output fields and their types.
    """

    def __init__(
        self,
        name: str,
        path: str | Path,
        input_schema: dict[str, type],
        output_schema: dict[str, type],
    ) -> None: ...

    def predict(self, features: dict[str, Any]) -> dict[str, Any]:
        """Run inference.

        1. Validate features against input_schema
        2. Convert features dict to numpy arrays (ONNX runtime
           expects numpy input)
        3. Run onnxruntime.InferenceSession.run()
        4. Convert output to dict matching output_schema
        """

    def validate(self) -> ModelValidationResult:
        """Load the ONNX file, inspect its input/output names,
        and check they match the declared schema."""
```

Implementation details:

- Use `onnxruntime.InferenceSession` for inference. This is a
  required dependency when using `nexa-backtest[ml]`.
- The model file is loaded lazily on first `predict()` call or
  explicitly on `validate()`. Don't load at registration time
  (it might be expensive and the user might not call predict).
- Feature values are converted to numpy arrays with the correct
  dtype: `float` -> `np.float32`, `int` -> `np.int64`,
  `str` -> raises error (ONNX doesn't natively handle strings,
  encode before passing).
- ONNX models expect input as a dict of `{input_name: np.ndarray}`.
  Each input is a 1D array of length 1 (single sample inference).
  Batch inference is out of scope for now.
- Output is a list of numpy arrays. Map them to the output_schema
  keys by position (ONNX outputs are ordered).

Error handling:

- Model file doesn't exist: `ModelLoadError`
- Features dict missing a key: `ModelInputError`
- Features dict has extra keys: warning (ignore extras)
- Feature type mismatch: `ModelInputError`
- ONNX runtime error: wrap in `ModelInferenceError`

### 4. `models/sklearn.py` - Scikit-learn Model Loader

Loads pickle or joblib files. Works but carries a security warning.

```python
class SklearnModel:
    """Load and run a scikit-learn model from pickle/joblib.

    WARNING: Pickle files can execute arbitrary code when loaded.
    Only use with models you trust. For hosted environments, prefer
    ONNX format.

    Args:
        name: Unique name for this model in the registry.
        path: Path to .pkl or .joblib file.
        input_schema: Expected input features and their types.
        output_schema: Expected output fields and their types.
        feature_order: Ordered list of feature names matching the
            order the model was trained on. Required because sklearn
            models expect a positional array, not a dict.
    """

    def __init__(
        self,
        name: str,
        path: str | Path,
        input_schema: dict[str, type],
        output_schema: dict[str, type],
        feature_order: list[str],
    ) -> None: ...
```

Implementation details:

- Load with `joblib.load()` (handles both pickle and joblib format).
  Fall back to `pickle.load()` if joblib is not installed.
- `feature_order` is required because sklearn's `predict()` takes a
  2D numpy array where column order must match training data. The
  features dict is reordered to match `feature_order` before
  conversion to array.
- Log a warning on first load: "Loading pickle model '{name}'. Pickle
  files can execute arbitrary code. Only use models from trusted
  sources."
- The model must have a `predict()` method (standard sklearn interface).
  If it has `predict_proba()`, expose it via a separate output field
  if declared in output_schema.

### 5. Wire `ctx.predict()` into the Backtest Engine

Update the backtest context implementation to support model inference:

```python
# In the backtest context:
def predict(self, model_name: str, features: dict[str, Any]) -> dict[str, Any]:
    """Run inference on a registered model.

    Args:
        model_name: Name of the model in the registry.
        features: Input features matching the model's input_schema.

    Returns:
        Dict matching the model's output_schema.

    Raises:
        ModelNotFoundError: Model not registered.
        ModelInputError: Features don't match schema.
        ModelInferenceError: Model failed during prediction.
    """
    model = self._model_registry.get(model_name)
    return model.predict(features)
```

Update `BacktestEngine` to accept a `ModelRegistry`:

```python
result = BacktestEngine(
    algo=algo,
    exchange="nordpool",
    start=date(2026, 3, 1),
    end=date(2026, 3, 31),
    products=["NO1_DA"],
    signals=[...],
    models=models,        # ModelRegistry instance
    initial_capital=100_000,
).run()
```

If `models` is not provided, `ctx.predict()` raises
`ModelNotFoundError` with a message explaining that no models
were registered.

### 6. Model Validation Step in the Pipeline

Add an optional step 7 to the validation pipeline from task 06.
This checks model compatibility before the backtest runs.

Update `validation/runner.py` to include model validation when
a model registry is provided:

```bash
$ nexa validate my_algo.py --exchange nordpool \
    --model price_predictor:models/xgboost.onnx

Step 1/6: Syntax & Style (ruff)        [PASS]
Step 2/6: Type Safety (mypy)           [PASS]
Step 3/6: Interface Compliance         [PASS]
Step 4/6: Exchange Features            [PASS]
Step 5/6: Look-Ahead Bias             [PASS]
Step 6/6: Resource Safety              [PASS]
Step 7/7: Model Compatibility          [PASS]
  price_predictor: ONNX model loaded (42ms)
  Inputs: wind (float), load (float), hour (int) [OK]
  Outputs: price_forecast (float) [OK]

Result: PASSED
```

Create `validation/model_check.py`:

```python
class ModelCheck:
    def run(
        self,
        algo_path: str,
        models: ModelRegistry | None,
    ) -> StepResult: ...
```

Checks:

- Does the algo call `ctx.predict()`? If yes, is a model registry
  provided?
- For each `ctx.predict("name", ...)` call found in the AST, is
  "name" registered in the model registry?
- Does each registered model load and validate successfully?
- Do the model's actual inputs/outputs match its declared schema?

If no models are registered and the algo doesn't call `ctx.predict()`,
skip this step entirely.

### 7. Update CLI

Add `--model` flag to both `nexa run` and `nexa validate`:

```bash
# Register models via CLI
nexa run my_algo.py --exchange nordpool \
    --model price_predictor:models/xgboost.onnx \
    --model vol_classifier:models/volatility.pkl \
    --start 2026-03-01 --end 2026-03-31

# Validate with models
nexa validate my_algo.py --exchange nordpool \
    --model price_predictor:models/xgboost.onnx
```

The `--model` flag format is `name:path`. The loader type is
inferred from the file extension:
- `.onnx` -> ONNXModel
- `.pkl`, `.joblib` -> SklearnModel

For sklearn models registered via CLI, `feature_order` can't be
specified on the command line. In this case, infer it from the
model's input_schema key order (dict ordering in Python 3.7+).
Alternatively, accept a JSON sidecar file: if
`models/volatility.pkl` exists and `models/volatility.json` also
exists, read the JSON for `input_schema`, `output_schema`, and
`feature_order`.

```json
{
    "input_schema": {"spread_1h": "float", "volume_1h": "float"},
    "output_schema": {"regime": "str"},
    "feature_order": ["spread_1h", "volume_1h"]
}
```

### 8. Example Algo with Model

Create `examples/ml_da_algo.py`:

```python
"""
Example: use an ONNX price prediction model to inform DA bidding.

Trains a simple model, exports to ONNX, then backtests an algo
that uses it.
"""
```

The example should:

1. Generate synthetic training data from the test fixture (use
   yesterday's price + hour of day as features, today's price as
   target)
2. Train a simple model (linear regression or decision tree from
   sklearn)
3. Export to ONNX using `skl2onnx`
4. Register the model and run a backtest
5. Compare PnL with and without the model

This doubles as a tutorial for the ML workflow. Include comments
explaining each step.

Also create `tests/generate_model_fixtures.py` that produces a
minimal ONNX model fixture for use across task 07 tests:

```python
"""Generate a minimal ONNX model fixture for testing.

Creates a simple linear regression model:
  Input: wind (float), load (float), hour (int)
  Output: price_forecast (float)

The model is intentionally trivial. It exists to test the ML
pipeline plumbing, not to make good predictions.

Run: python tests/generate_model_fixtures.py
Output: tests/fixtures/models/simple_predictor.onnx
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType, Int64TensorType

def generate() -> None:
    # Tiny training set (10 samples, deterministic)
    rng = np.random.default_rng(seed=42)
    wind = rng.uniform(1000, 5000, size=10).astype(np.float32)
    load = rng.uniform(3000, 8000, size=10).astype(np.float32)
    hour = rng.integers(0, 24, size=10).astype(np.int64)

    # Target: a simple linear combination (not realistic, doesn't matter)
    price = 30.0 + 0.005 * load - 0.003 * wind + 0.5 * hour
    price = price.astype(np.float32)

    # Train
    X = np.column_stack([wind, load, hour.astype(np.float32)])
    model = LinearRegression()
    model.fit(X, price)

    # Export to ONNX
    initial_types = [
        ("wind", FloatTensorType([None, 1])),
        ("load", FloatTensorType([None, 1])),
        ("hour", Int64TensorType([None, 1])),
    ]
    onnx_model = convert_sklearn(
        model,
        initial_types=initial_types,
        target_opset=15,
    )

    # Save
    output_path = Path(__file__).parent / "fixtures" / "models" / "simple_predictor.onnx"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    print(f"Model saved to {output_path}")
    print(f"Size: {output_path.stat().st_size} bytes")

    # Verify it loads
    import onnxruntime as ort
    session = ort.InferenceSession(str(output_path))
    inputs = {
        "wind": np.array([[3000.0]], dtype=np.float32),
        "load": np.array([[5000.0]], dtype=np.float32),
        "hour": np.array([[12]], dtype=np.int64),
    }
    result = session.run(None, inputs)
    print(f"Test prediction: {result[0][0]:.2f} EUR/MWh")


if __name__ == "__main__":
    generate()
```

Run the script once and commit the `.onnx` file to the repo. Add
`skl2onnx` and `scikit-learn` to test dependencies only (not
runtime deps).

---

## Tests

1. **ModelRegistry**: register, get, has, list. Verify duplicate
   name raises error. Verify get on missing name raises
   ModelNotFoundError.

2. **ONNXModel - happy path**: load the test fixture model, call
   predict with valid features, verify output matches expected
   schema.

3. **ONNXModel - validation**: call validate(), verify it reports
   correct actual_inputs/outputs and matches declared schema.

4. **ONNXModel - schema mismatch**: declare input_schema with a
   feature name that doesn't match the model. Call validate(),
   verify it reports the mismatch.

5. **ONNXModel - missing features**: call predict() with a features
   dict missing a required key. Verify ModelInputError.

6. **ONNXModel - missing file**: create ONNXModel with a
   non-existent path. Call predict() or validate(). Verify
   ModelLoadError.

7. **SklearnModel - happy path**: create a tiny sklearn model
   (LinearRegression), save as pickle, load via SklearnModel,
   predict, verify output.

8. **SklearnModel - security warning**: load a pickle model, verify
   a warning is logged about arbitrary code execution.

9. **SklearnModel - feature_order**: verify features are reordered
   to match feature_order before prediction. Pass features in
   wrong order, verify the model still gets them correctly.

10. **ctx.predict() integration**: write a SimpleAlgo that calls
    ctx.predict() in on_auction_open. Register a model, run a
    backtest, verify the model was called and influenced trading
    decisions.

11. **No models registered**: write an algo that calls ctx.predict().
    Run without registering models. Verify ModelNotFoundError with
    a clear message.

12. **Model validation step**: run `nexa validate` with --model
    flag. Verify step 7 appears and checks the model. Test with a
    valid model (pass) and a model with schema mismatch (fail).

13. **CLI --model flag**: test with CliRunner that models are
    registered from the --model flag and the backtest runs.

14. **JSON sidecar**: create a .pkl model with a .json sidecar.
    Register via CLI. Verify feature_order and schemas are read
    from the JSON.

---

## Dependencies

Add to optional extras in `pyproject.toml`:

```toml
[project.optional-dependencies]
ml = [
    "onnxruntime>=1.16",
    "joblib>=1.3",
]
```

`skl2onnx` is only needed for the example script (exporting sklearn
to ONNX), not for the library itself. Mention it in the example's
docstring but don't add it as a dependency.

`scikit-learn` is a test dependency (for creating fixture models)
but not a runtime dependency of the library.

---

## What NOT to build

- Batch inference (multiple samples at once). Single-sample only.
- Model training or retraining pipelines
- PyTorch loader (export to ONNX instead)
- TensorFlow loader (export to ONNX instead)
- Model versioning or A/B testing between model versions
- Feature engineering helpers
- Model performance metrics (accuracy, RMSE, etc.)
- Hosted model serving
- Sandboxed pickle execution for hosted environments (stage 4)

---

## Acceptance criteria

1. `make ci` passes
2. A customer can register an ONNX model, call `ctx.predict()`
   from their algo, and use the prediction to make trading decisions
3. A customer can register a sklearn pickle model with the same
   interface (with security warning logged)
4. `ModelRegistry.validate_all()` loads each model and verifies
   schema compatibility
5. `nexa validate --model` checks model compatibility as step 7
6. `nexa run --model` registers models from the CLI
7. The example algo demonstrates the full ML workflow (train,
   export, register, backtest)
8. Missing features, wrong types, and missing models all produce
   clear error messages
9. The ONNX model fixture generator (`tests/generate_model_fixtures.py`)
   produces a valid model that loads and runs correctly
10. All new types have type hints and frozen Pydantic models where
    appropriate
11. All new public API has Google-style docstrings
