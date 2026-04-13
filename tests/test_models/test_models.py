"""Tests for the ML model registry (task 07).

Covers ModelRegistry, ONNXModel, SklearnModel, ctx.predict() integration,
the model validation step, and the --model CLI flag.
"""

from __future__ import annotations

import dataclasses
import pickle
import warnings
from datetime import UTC
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from nexa_backtest.exceptions import (
    ModelInputError,
    ModelLoadError,
    ModelNotFoundError,
)
from nexa_backtest.models.base import ModelValidationResult
from nexa_backtest.models.onnx import ONNXModel
from nexa_backtest.models.registry import ModelRegistry
from nexa_backtest.models.sklearn import SklearnModel

# Path to the pre-built ONNX fixture
FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "models"
ONNX_FIXTURE = FIXTURE_DIR / "simple_predictor.onnx"

# Shared input schema for the fixture model
FIXTURE_INPUT_SCHEMA: dict[str, type] = {"wind": float, "load": float, "hour": float}
FIXTURE_OUTPUT_SCHEMA: dict[str, type] = {"variable": float}
FIXTURE_FEATURES: dict[str, float] = {"wind": 3000.0, "load": 5000.0, "hour": 12.0}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sklearn_model(tmp_path: Path) -> tuple[Path, object]:
    """Train a tiny sklearn model and pickle it. Returns (path, model)."""
    from sklearn.linear_model import LinearRegression  # type: ignore[import-untyped]

    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)
    y = np.array([1.0, 2.0, 3.0])
    model = LinearRegression()
    model.fit(X, y)
    pkl_path = tmp_path / "test_model.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(model, f)
    return pkl_path, model


# ===========================================================================
# 1. ModelRegistry
# ===========================================================================


class TestModelRegistry:
    def test_register_and_get(self) -> None:
        registry = ModelRegistry()
        model = ONNXModel("m1", ONNX_FIXTURE, FIXTURE_INPUT_SCHEMA, FIXTURE_OUTPUT_SCHEMA)
        registry.register(model)
        assert registry.get("m1") is model

    def test_has(self) -> None:
        registry = ModelRegistry()
        model = ONNXModel("m1", ONNX_FIXTURE, FIXTURE_INPUT_SCHEMA, FIXTURE_OUTPUT_SCHEMA)
        assert not registry.has("m1")
        registry.register(model)
        assert registry.has("m1")

    def test_list_models(self) -> None:
        registry = ModelRegistry()
        registry.register(ONNXModel("b", ONNX_FIXTURE, {}, {}))
        registry.register(ONNXModel("a", ONNX_FIXTURE, {}, {}))
        assert registry.list_models() == ["a", "b"]  # sorted

    def test_duplicate_name_raises(self) -> None:
        registry = ModelRegistry()
        registry.register(ONNXModel("m1", ONNX_FIXTURE, {}, {}))
        with pytest.raises(ValueError, match="already registered"):
            registry.register(ONNXModel("m1", ONNX_FIXTURE, {}, {}))

    def test_get_missing_raises(self) -> None:
        registry = ModelRegistry()
        with pytest.raises(ModelNotFoundError, match="not registered"):
            registry.get("nonexistent")

    def test_get_missing_shows_registered(self) -> None:
        registry = ModelRegistry()
        registry.register(ONNXModel("existing", ONNX_FIXTURE, {}, {}))
        with pytest.raises(ModelNotFoundError, match="existing"):
            registry.get("missing")

    def test_get_empty_registry_hint(self) -> None:
        registry = ModelRegistry()
        with pytest.raises(ModelNotFoundError, match="No models are registered"):
            registry.get("anything")

    def test_validate_all_empty(self) -> None:
        registry = ModelRegistry()
        assert registry.validate_all() == []

    def test_validate_all_returns_all_results(self) -> None:
        registry = ModelRegistry()
        registry.register(
            ONNXModel("m1", ONNX_FIXTURE, FIXTURE_INPUT_SCHEMA, FIXTURE_OUTPUT_SCHEMA)
        )
        registry.register(
            ONNXModel("m2", ONNX_FIXTURE, FIXTURE_INPUT_SCHEMA, FIXTURE_OUTPUT_SCHEMA)
        )
        results = registry.validate_all()
        assert len(results) == 2
        assert all(isinstance(r, ModelValidationResult) for r in results)


# ===========================================================================
# 2. ONNXModel — happy path
# ===========================================================================


class TestONNXModelHappyPath:
    def test_predict_returns_output(self) -> None:
        model = ONNXModel("pred", ONNX_FIXTURE, FIXTURE_INPUT_SCHEMA, FIXTURE_OUTPUT_SCHEMA)
        result = model.predict(FIXTURE_FEATURES)
        assert "variable" in result
        assert isinstance(result["variable"], float)

    def test_predict_value_reasonable(self) -> None:
        model = ONNXModel("pred", ONNX_FIXTURE, FIXTURE_INPUT_SCHEMA, FIXTURE_OUTPUT_SCHEMA)
        result = model.predict(FIXTURE_FEATURES)
        # 45.2 - 0.003*3000 + 0.005*5000 + 0.4*12 = 45.2 - 9 + 25 + 4.8 = 66.0
        assert abs(result["variable"] - 66.0) < 0.1

    def test_predict_extra_keys_ignored_with_warning(self) -> None:
        model = ONNXModel("pred", ONNX_FIXTURE, FIXTURE_INPUT_SCHEMA, FIXTURE_OUTPUT_SCHEMA)
        features = {**FIXTURE_FEATURES, "extra_key": 99.0}
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = model.predict(features)
        assert any("extra_key" in str(warning.message) for warning in w)
        assert "variable" in result

    def test_name_property(self) -> None:
        model = ONNXModel("mymodel", ONNX_FIXTURE, FIXTURE_INPUT_SCHEMA, FIXTURE_OUTPUT_SCHEMA)
        assert model.name == "mymodel"

    def test_schema_properties(self) -> None:
        model = ONNXModel("m", ONNX_FIXTURE, FIXTURE_INPUT_SCHEMA, FIXTURE_OUTPUT_SCHEMA)
        assert model.input_schema == FIXTURE_INPUT_SCHEMA
        assert model.output_schema == FIXTURE_OUTPUT_SCHEMA

    def test_session_cached_on_second_call(self) -> None:
        model = ONNXModel("m", ONNX_FIXTURE, FIXTURE_INPUT_SCHEMA, FIXTURE_OUTPUT_SCHEMA)
        model.predict(FIXTURE_FEATURES)
        session_ref = model._session
        model.predict(FIXTURE_FEATURES)
        assert model._session is session_ref  # same object, not reloaded


# ===========================================================================
# 3. ONNXModel — validation
# ===========================================================================


class TestONNXModelValidation:
    def test_validate_pass(self) -> None:
        model = ONNXModel("pred", ONNX_FIXTURE, FIXTURE_INPUT_SCHEMA, FIXTURE_OUTPUT_SCHEMA)
        result = model.validate()
        assert result.valid is True
        assert result.error is None
        assert result.model_name == "pred"
        assert result.actual_inputs == ["wind", "load", "hour"]
        assert result.actual_outputs == ["variable"]
        assert result.load_time_ms >= 0

    def test_validate_reports_load_time(self) -> None:
        model = ONNXModel("pred", ONNX_FIXTURE, FIXTURE_INPUT_SCHEMA, FIXTURE_OUTPUT_SCHEMA)
        result = model.validate()
        assert isinstance(result.load_time_ms, int)

    def test_validate_frozen_dataclass(self) -> None:
        model = ONNXModel("pred", ONNX_FIXTURE, FIXTURE_INPUT_SCHEMA, FIXTURE_OUTPUT_SCHEMA)
        result = model.validate()
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.valid = False  # type: ignore[misc]


# ===========================================================================
# 4. ONNXModel — schema mismatch
# ===========================================================================


class TestONNXModelSchemaMismatch:
    def test_validate_input_mismatch(self) -> None:
        wrong_schema = {"wind": float, "nonexistent_feature": float}
        model = ONNXModel("pred", ONNX_FIXTURE, wrong_schema, FIXTURE_OUTPUT_SCHEMA)
        result = model.validate()
        assert result.valid is False
        assert "nonexistent_feature" in (result.error or "")
        assert result.actual_inputs == ["wind", "load", "hour"]

    def test_validate_output_mismatch(self) -> None:
        wrong_output = {"bad_output": float}
        model = ONNXModel("pred", ONNX_FIXTURE, FIXTURE_INPUT_SCHEMA, wrong_output)
        result = model.validate()
        assert result.valid is False
        assert "bad_output" in (result.error or "")


# ===========================================================================
# 5. ONNXModel — missing features
# ===========================================================================


class TestONNXModelMissingFeatures:
    def test_predict_missing_key_raises(self) -> None:
        model = ONNXModel("pred", ONNX_FIXTURE, FIXTURE_INPUT_SCHEMA, FIXTURE_OUTPUT_SCHEMA)
        with pytest.raises(ModelInputError, match="wind"):
            model.predict({"load": 5000.0, "hour": 12.0})  # missing wind

    def test_predict_str_value_raises(self) -> None:
        model = ONNXModel("pred", ONNX_FIXTURE, {"wind": str}, FIXTURE_OUTPUT_SCHEMA)
        with pytest.raises(ModelInputError, match="str"):
            model.predict({"wind": "a_string"})


# ===========================================================================
# 6. ONNXModel — missing file
# ===========================================================================


class TestONNXModelMissingFile:
    def test_predict_missing_file_raises(self) -> None:
        model = ONNXModel(
            "pred", "/does/not/exist.onnx", FIXTURE_INPUT_SCHEMA, FIXTURE_OUTPUT_SCHEMA
        )
        with pytest.raises(ModelLoadError, match="not found"):
            model.predict(FIXTURE_FEATURES)

    def test_validate_missing_file_raises(self) -> None:
        model = ONNXModel(
            "pred", "/does/not/exist.onnx", FIXTURE_INPUT_SCHEMA, FIXTURE_OUTPUT_SCHEMA
        )
        result = model.validate()
        assert result.valid is False
        assert "not found" in (result.error or "")
        assert result.actual_inputs is None


# ===========================================================================
# 7. SklearnModel — happy path
# ===========================================================================


class TestSklearnModelHappyPath:
    def test_predict_returns_output(self, tmp_path: Path) -> None:
        pkl_path, _ = _make_sklearn_model(tmp_path)
        model = SklearnModel(
            name="sk",
            path=pkl_path,
            input_schema={"x": float, "y": float},
            output_schema={"prediction": float},
            feature_order=["x", "y"],
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = model.predict({"x": 1.0, "y": 2.0})
        assert "prediction" in result
        assert isinstance(result["prediction"], float)

    def test_validate_valid(self, tmp_path: Path) -> None:
        pkl_path, _ = _make_sklearn_model(tmp_path)
        model = SklearnModel(
            name="sk",
            path=pkl_path,
            input_schema={"x": float, "y": float},
            output_schema={"prediction": float},
            feature_order=["x", "y"],
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = model.validate()
        assert result.valid is True
        assert result.actual_inputs == ["x", "y"]
        assert result.actual_outputs == ["prediction"]


# ===========================================================================
# 8. SklearnModel — security warning
# ===========================================================================


class TestSklearnModelSecurityWarning:
    def test_security_warning_emitted(self, tmp_path: Path) -> None:
        pkl_path, _ = _make_sklearn_model(tmp_path)
        model = SklearnModel(
            name="sk",
            path=pkl_path,
            input_schema={"x": float, "y": float},
            output_schema={"prediction": float},
            feature_order=["x", "y"],
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model.predict({"x": 1.0, "y": 2.0})
        assert any("arbitrary code" in str(warning.message).lower() for warning in w)

    def test_warning_logged_only_on_first_load(self, tmp_path: Path) -> None:
        pkl_path, _ = _make_sklearn_model(tmp_path)
        model = SklearnModel(
            name="sk",
            path=pkl_path,
            input_schema={"x": float, "y": float},
            output_schema={"prediction": float},
            feature_order=["x", "y"],
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model.predict({"x": 1.0, "y": 2.0})
            count_first = len(w)
            model.predict({"x": 2.0, "y": 3.0})
        # second call should not emit another warning (model already loaded)
        assert len(w) == count_first


# ===========================================================================
# 9. SklearnModel — feature_order
# ===========================================================================


class TestSklearnModelFeatureOrder:
    def test_feature_order_enforced(self, tmp_path: Path) -> None:
        """Features passed in wrong order should still produce the correct result."""
        from sklearn.linear_model import LinearRegression  # type: ignore[import-untyped]

        # Train: y = 1*x1 + 2*x2 (x1 is first column)
        X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float64)
        y = np.array([1.0, 2.0, 3.0])
        lr = LinearRegression(fit_intercept=False)
        lr.fit(X, y)

        pkl_path = tmp_path / "ordered.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(lr, f)

        model = SklearnModel(
            name="sk",
            path=pkl_path,
            input_schema={"x1": float, "x2": float},
            output_schema={"pred": float},
            feature_order=["x1", "x2"],  # x1 first
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Pass in reverse order — feature_order should reorder correctly
            result = model.predict({"x2": 1.0, "x1": 2.0})
        # Expected: 1*2 + 2*1 = 4
        assert abs(result["pred"] - 4.0) < 0.01

    def test_missing_feature_raises(self, tmp_path: Path) -> None:
        pkl_path, _ = _make_sklearn_model(tmp_path)
        model = SklearnModel(
            name="sk",
            path=pkl_path,
            input_schema={"x": float, "y": float},
            output_schema={"prediction": float},
            feature_order=["x", "y"],
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(ModelInputError, match="y"):
                model.predict({"x": 1.0})  # missing y

    def test_extra_keys_ignored_with_warning(self, tmp_path: Path) -> None:
        pkl_path, _ = _make_sklearn_model(tmp_path)
        model = SklearnModel(
            name="sk",
            path=pkl_path,
            input_schema={"x": float, "y": float},
            output_schema={"prediction": float},
            feature_order=["x", "y"],
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model.predict({"x": 1.0, "y": 2.0, "z": 9.9})
        assert any("z" in str(warning.message) for warning in w)

    def test_sklearn_missing_file(self, tmp_path: Path) -> None:
        model = SklearnModel(
            name="sk",
            path=tmp_path / "nonexistent.pkl",
            input_schema={},
            output_schema={},
            feature_order=[],
        )
        result = model.validate()
        assert result.valid is False
        assert "not found" in (result.error or "")


# ===========================================================================
# 10. ctx.predict() integration
# ===========================================================================


class TestCtxPredictIntegration:
    def test_predict_called_in_algo(self, tmp_path: Path) -> None:
        """SimpleAlgo calls ctx.predict() in on_auction_open; model influences order price."""
        from decimal import Decimal

        import pandas as pd
        import pyarrow as pa
        import pyarrow.parquet as pq

        from nexa_backtest.algo import SimpleAlgo
        from nexa_backtest.context import TradingContext
        from nexa_backtest.engines.backtest import BacktestEngine
        from nexa_backtest.types import AuctionInfo, Order

        # Build minimal DA fixture
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        from datetime import datetime

        ts = [
            datetime(2026, 3, 1, 12, 0, tzinfo=UTC),
            datetime(2026, 3, 2, 12, 0, tzinfo=UTC),
        ]
        df = pd.DataFrame(
            {
                "timestamp": ts,
                "zone": ["NO1", "NO1"],
                "price_eur_mwh": [50.0, 55.0],
                "volume_mwh": [1000.0, 1000.0],
            }
        )
        table = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_table(table, data_dir / "da_prices.parquet")

        prices_seen: list[float] = []

        class PredictAlgo(SimpleAlgo):
            def on_auction_open(self, ctx: TradingContext, auction: AuctionInfo) -> None:
                result = ctx.predict("predictor", {"wind": 3000.0, "load": 5000.0, "hour": 12.0})
                predicted_price = result["variable"]
                prices_seen.append(predicted_price)
                ctx.place_order(
                    Order.buy(
                        product_id=auction.product_id,
                        price_eur_mwh=Decimal(str(predicted_price)),
                        volume_mw=Decimal("1.0"),
                    )
                )

        registry = ModelRegistry()
        registry.register(
            ONNXModel("predictor", ONNX_FIXTURE, FIXTURE_INPUT_SCHEMA, FIXTURE_OUTPUT_SCHEMA)
        )

        engine = BacktestEngine(
            algo=PredictAlgo(),
            exchange="nordpool",
            start=__import__("datetime").date(2026, 3, 1),
            end=__import__("datetime").date(2026, 3, 2),
            products=["NO1_DA"],
            data_dir=data_dir,
            capital=Decimal("100000"),
            models=registry,
        )
        result = engine.run()
        assert len(prices_seen) == 2
        assert all(abs(p - 66.0) < 0.1 for p in prices_seen)
        assert len(result.fills) >= 0  # any fills are fine; key is predict() was called


# ===========================================================================
# 11. No models registered
# ===========================================================================


class TestNoModelsRegistered:
    def test_predict_without_registry_raises(self, tmp_path: Path) -> None:
        from decimal import Decimal

        import pandas as pd
        import pyarrow as pa
        import pyarrow.parquet as pq

        from nexa_backtest.algo import SimpleAlgo
        from nexa_backtest.context import TradingContext
        from nexa_backtest.engines.backtest import BacktestEngine
        from nexa_backtest.types import AuctionInfo

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        from datetime import datetime

        df = pd.DataFrame(
            {
                "timestamp": [datetime(2026, 3, 1, 12, 0, tzinfo=UTC)],
                "zone": ["NO1"],
                "price_eur_mwh": [50.0],
                "volume_mwh": [1000.0],
            }
        )
        table = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_table(table, data_dir / "da_prices.parquet")

        error_raised: list[Exception] = []

        class NeedsModelAlgo(SimpleAlgo):
            def on_auction_open(self, ctx: TradingContext, auction: AuctionInfo) -> None:
                try:
                    ctx.predict("mymodel", {})
                except ModelNotFoundError as exc:
                    error_raised.append(exc)

        engine = BacktestEngine(
            algo=NeedsModelAlgo(),
            exchange="nordpool",
            start=__import__("datetime").date(2026, 3, 1),
            end=__import__("datetime").date(2026, 3, 1),
            products=["NO1_DA"],
            data_dir=data_dir,
            capital=Decimal("100000"),
            models=None,  # no registry
        )
        engine.run()
        assert len(error_raised) == 1
        assert "ModelRegistry" in str(error_raised[0]) or "no ModelRegistry" in str(error_raised[0])


# ===========================================================================
# 12. Model validation step (validation pipeline)
# ===========================================================================


class TestModelValidationStep:
    def test_step_skipped_when_no_models_and_no_predict_calls(self, tmp_path: Path) -> None:
        """Step 7 must not appear when algo doesn't call ctx.predict() and no registry given."""
        algo_src = "from nexa_backtest.algo import SimpleAlgo\n\nclass A(SimpleAlgo): pass\n"
        algo_file = tmp_path / "plain_algo.py"
        algo_file.write_text(algo_src)

        from nexa_backtest.validation.runner import ValidationRunner

        result = ValidationRunner(algo_path=str(algo_file), exchange="nordpool").run()
        step_names = [s.name for s in result.steps]
        assert "Model Compatibility" not in step_names

    def test_step_fails_when_predict_called_but_no_registry(self, tmp_path: Path) -> None:
        algo_src = (
            "from nexa_backtest.algo import SimpleAlgo\n"
            "from nexa_backtest.context import TradingContext\n"
            "from nexa_backtest.types import AuctionInfo\n\n"
            "class A(SimpleAlgo):\n"
            "    def on_auction_open(self, ctx: TradingContext, auction: AuctionInfo) -> None:\n"
            "        ctx.predict('mymodel', {})\n"
        )
        algo_file = tmp_path / "predict_algo.py"
        algo_file.write_text(algo_src)

        from nexa_backtest.validation.runner import ValidationRunner

        result = ValidationRunner(algo_path=str(algo_file), exchange="nordpool").run()
        model_step = next(s for s in result.steps if s.name == "Model Compatibility")
        assert model_step.status == "fail"
        assert any(
            "ModelRegistry" in m or "no ModelRegistry" in m or "--model" in m
            for m in model_step.messages
        )

    def test_step_passes_with_valid_model(self, tmp_path: Path) -> None:
        algo_src = (
            "from nexa_backtest.algo import SimpleAlgo\n"
            "from nexa_backtest.context import TradingContext\n"
            "from nexa_backtest.types import AuctionInfo\n\n"
            "class A(SimpleAlgo):\n"
            "    def on_auction_open(self, ctx: TradingContext, auction: AuctionInfo) -> None:\n"
            "        ctx.predict('predictor', {})\n"
        )
        algo_file = tmp_path / "predict_algo.py"
        algo_file.write_text(algo_src)

        registry = ModelRegistry()
        registry.register(
            ONNXModel("predictor", ONNX_FIXTURE, FIXTURE_INPUT_SCHEMA, FIXTURE_OUTPUT_SCHEMA)
        )

        from nexa_backtest.validation.runner import ValidationRunner

        result = ValidationRunner(
            algo_path=str(algo_file), exchange="nordpool", models=registry
        ).run()
        model_step = next(s for s in result.steps if s.name == "Model Compatibility")
        assert model_step.status == "pass"

    def test_step_fails_with_unregistered_model_name(self, tmp_path: Path) -> None:
        algo_src = (
            "from nexa_backtest.algo import SimpleAlgo\n"
            "from nexa_backtest.context import TradingContext\n"
            "from nexa_backtest.types import AuctionInfo\n\n"
            "class A(SimpleAlgo):\n"
            "    def on_auction_open(self, ctx: TradingContext, auction: AuctionInfo) -> None:\n"
            "        ctx.predict('wrong_name', {})\n"
        )
        algo_file = tmp_path / "predict_algo.py"
        algo_file.write_text(algo_src)

        registry = ModelRegistry()
        registry.register(
            ONNXModel("predictor", ONNX_FIXTURE, FIXTURE_INPUT_SCHEMA, FIXTURE_OUTPUT_SCHEMA)
        )

        from nexa_backtest.validation.runner import ValidationRunner

        result = ValidationRunner(
            algo_path=str(algo_file), exchange="nordpool", models=registry
        ).run()
        model_step = next(s for s in result.steps if s.name == "Model Compatibility")
        assert model_step.status == "fail"
        assert any("wrong_name" in m for m in model_step.messages)

    def test_step_fails_with_invalid_model(self, tmp_path: Path) -> None:
        algo_src = "from nexa_backtest.algo import SimpleAlgo\n\nclass A(SimpleAlgo): pass\n"
        algo_file = tmp_path / "plain_algo.py"
        algo_file.write_text(algo_src)

        registry = ModelRegistry()
        registry.register(ONNXModel("bad", tmp_path / "missing.onnx", {}, {}))

        from nexa_backtest.validation.runner import ValidationRunner

        result = ValidationRunner(
            algo_path=str(algo_file), exchange="nordpool", models=registry
        ).run()
        model_step = next(s for s in result.steps if s.name == "Model Compatibility")
        assert model_step.status == "fail"

    def test_step_skipped_via_skip_flag(self, tmp_path: Path) -> None:
        algo_src = "from nexa_backtest.algo import SimpleAlgo\n\nclass A(SimpleAlgo): pass\n"
        algo_file = tmp_path / "plain_algo.py"
        algo_file.write_text(algo_src)

        registry = ModelRegistry()
        registry.register(ONNXModel("m", ONNX_FIXTURE, FIXTURE_INPUT_SCHEMA, FIXTURE_OUTPUT_SCHEMA))

        from nexa_backtest.validation.runner import ValidationRunner

        result = ValidationRunner(
            algo_path=str(algo_file), exchange="nordpool", models=registry, skip={"models"}
        ).run()
        step_names = [s.name for s in result.steps]
        assert "Model Compatibility" not in step_names


# ===========================================================================
# 13. CLI --model flag (nexa run)
# ===========================================================================


class TestCliModelFlag:
    def test_build_model_registry_onnx(self, tmp_path: Path) -> None:
        from nexa_backtest.cli.main import _build_model_registry

        registry = _build_model_registry((f"predictor:{ONNX_FIXTURE}",))
        assert registry.has("predictor")

    def test_build_model_registry_bad_spec(self) -> None:
        import click

        from nexa_backtest.cli.main import _build_model_registry

        with pytest.raises(click.ClickException, match="Invalid"):
            _build_model_registry(("no_colon_here",))

    def test_build_model_registry_unsupported_extension(self, tmp_path: Path) -> None:
        import click

        from nexa_backtest.cli.main import _build_model_registry

        fake = tmp_path / "model.h5"
        fake.touch()
        with pytest.raises(click.ClickException, match="Unsupported"):
            _build_model_registry((f"m:{fake}",))

    def test_cli_run_model_flag(self, tmp_path: Path) -> None:
        """nexa run --model should register models and allow predict() in the algo."""
        from datetime import datetime

        import pandas as pd
        import pyarrow as pa
        import pyarrow.parquet as pq

        # Write fixture data
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        df = pd.DataFrame(
            {
                "timestamp": [datetime(2026, 3, 1, 12, 0, tzinfo=UTC)],
                "zone": ["NO1"],
                "price_eur_mwh": [50.0],
                "volume_mwh": [1000.0],
            }
        )
        pq.write_table(
            pa.Table.from_pandas(df, preserve_index=False), data_dir / "da_prices.parquet"
        )

        algo_src = (
            "from nexa_backtest.algo import SimpleAlgo\n"
            "from nexa_backtest.context import TradingContext\n"
            "from nexa_backtest.types import AuctionInfo\n\n"
            "class ModelAlgo(SimpleAlgo):\n"
            "    def on_auction_open(self, ctx: TradingContext, auction: AuctionInfo) -> None:\n"
            "        result = ctx.predict('predictor', {'wind': 3000.0, 'load': 5000.0, 'hour': 12.0})\n"  # noqa: E501
            "        assert isinstance(result['variable'], float)\n"
        )
        algo_file = tmp_path / "model_algo.py"
        algo_file.write_text(algo_src)

        from nexa_backtest.cli.main import cli

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "run",
                str(algo_file),
                "--exchange",
                "nordpool",
                "--start",
                "2026-03-01",
                "--end",
                "2026-03-01",
                "--products",
                "NO1_DA",
                "--data-dir",
                str(data_dir),
                "--model",
                f"predictor:{ONNX_FIXTURE}",
            ],
        )
        assert result.exit_code == 0, result.output


# ===========================================================================
# 14. JSON sidecar for sklearn via CLI
# ===========================================================================


class TestJsonSidecar:
    def test_sklearn_json_sidecar(self, tmp_path: Path) -> None:
        import json

        pkl_path, _ = _make_sklearn_model(tmp_path)
        sidecar = pkl_path.with_suffix(".json")
        sidecar.write_text(
            json.dumps(
                {
                    "input_schema": {"x": "float", "y": "float"},
                    "output_schema": {"prediction": "float"},
                    "feature_order": ["x", "y"],
                }
            )
        )

        from nexa_backtest.cli.main import _build_model_registry

        registry = _build_model_registry((f"sk:{pkl_path}",))
        assert registry.has("sk")
        model = registry.get("sk")
        # feature_order should be read from the sidecar
        assert isinstance(model, SklearnModel)
        assert model._feature_order == ["x", "y"]
        assert model.input_schema == {"x": float, "y": float}


# ===========================================================================
# 15. ModelValidationResult — validate_all with mixed results
# ===========================================================================


class TestValidateAll:
    def test_validate_all_mixed(self, tmp_path: Path) -> None:
        registry = ModelRegistry()
        registry.register(
            ONNXModel("good", ONNX_FIXTURE, FIXTURE_INPUT_SCHEMA, FIXTURE_OUTPUT_SCHEMA)
        )
        registry.register(ONNXModel("bad", tmp_path / "missing.onnx", {}, {}))

        results = registry.validate_all()
        assert len(results) == 2
        good = next(r for r in results if r.model_name == "good")
        bad = next(r for r in results if r.model_name == "bad")
        assert good.valid is True
        assert bad.valid is False

    def test_validate_all_collects_all_even_on_failure(self, tmp_path: Path) -> None:
        """All models are validated even if earlier ones fail."""
        registry = ModelRegistry()
        registry.register(ONNXModel("bad1", tmp_path / "m1.onnx", {}, {}))
        registry.register(ONNXModel("bad2", tmp_path / "m2.onnx", {}, {}))
        registry.register(
            ONNXModel("good", ONNX_FIXTURE, FIXTURE_INPUT_SCHEMA, FIXTURE_OUTPUT_SCHEMA)
        )

        results = registry.validate_all()
        assert len(results) == 3


# ===========================================================================
# 16. CLI validate --model flag
# ===========================================================================


class TestCliValidateModelFlag:
    def test_validate_with_model_flag(self, tmp_path: Path) -> None:
        algo_src = (
            "from nexa_backtest.algo import SimpleAlgo\n"
            "from nexa_backtest.context import TradingContext\n"
            "from nexa_backtest.types import AuctionInfo\n\n"
            "class A(SimpleAlgo):\n"
            "    def on_auction_open(self, ctx: TradingContext, auction: AuctionInfo) -> None:\n"
            "        ctx.predict('predictor', {'wind': 1.0, 'load': 2.0, 'hour': 3.0})\n"
        )
        algo_file = tmp_path / "algo.py"
        algo_file.write_text(algo_src)

        from nexa_backtest.cli.main import cli

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "validate",
                str(algo_file),
                "--exchange",
                "nordpool",
                "--skip",
                "mypy",  # skip slow mypy in tests
                "--model",
                f"predictor:{ONNX_FIXTURE}",
            ],
        )
        assert "Model Compatibility" in result.output
        assert "PASS" in result.output or "FAIL" not in result.output


# ===========================================================================
# Fixture generator sanity
# ===========================================================================


class TestFixtureGenerator:
    def test_fixture_file_exists(self) -> None:
        assert ONNX_FIXTURE.exists(), f"ONNX fixture not found at {ONNX_FIXTURE}"

    def test_fixture_loads_and_runs(self) -> None:
        import onnxruntime as ort  # type: ignore[import-untyped]

        session = ort.InferenceSession(str(ONNX_FIXTURE))
        inputs = {
            "wind": np.array([[3000.0]], dtype=np.float32),
            "load": np.array([[5000.0]], dtype=np.float32),
            "hour": np.array([[12.0]], dtype=np.float32),
        }
        result = session.run(None, inputs)
        assert abs(result[0][0][0] - 66.0) < 0.1


# ===========================================================================
# Coverage gap: SklearnModel uncovered branches
# ===========================================================================


class TestSklearnModelCoverageGaps:
    def test_output_schema_property(self, tmp_path: Path) -> None:
        pkl_path, _ = _make_sklearn_model(tmp_path)
        model = SklearnModel(
            name="sk",
            path=pkl_path,
            input_schema={"x": float, "y": float},
            output_schema={"prediction": float},
            feature_order=["x", "y"],
        )
        assert model.output_schema == {"prediction": float}

    def test_pickle_fallback_when_joblib_unavailable(self, tmp_path: Path) -> None:
        """Uses pickle directly when joblib is not installed."""
        import sys
        import unittest.mock

        pkl_path, _ = _make_sklearn_model(tmp_path)
        model = SklearnModel(
            name="sk",
            path=pkl_path,
            input_schema={"x": float, "y": float},
            output_schema={"prediction": float},
            feature_order=["x", "y"],
        )
        with unittest.mock.patch.dict(sys.modules, {"joblib": None}):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = model.predict({"x": 1.0, "y": 2.0})
        assert "prediction" in result

    def test_load_raises_model_load_error_on_corrupt_file(self, tmp_path: Path) -> None:
        """ModelLoadError is raised when the file exists but cannot be unpickled."""
        corrupt = tmp_path / "corrupt.pkl"
        corrupt.write_bytes(b"this is not a pickle file")

        model = SklearnModel(
            name="sk",
            path=corrupt,
            input_schema={},
            output_schema={},
            feature_order=[],
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(ModelLoadError, match="Failed to load"):
                model.predict({})

    def test_predict_raises_inference_error_on_model_failure(self, tmp_path: Path) -> None:
        """ModelInferenceError is raised when the underlying model.predict() raises."""
        import unittest.mock

        from nexa_backtest.exceptions import ModelInferenceError

        pkl_path, _ = _make_sklearn_model(tmp_path)
        model = SklearnModel(
            name="sk",
            path=pkl_path,
            input_schema={"x": float, "y": float},
            output_schema={"pred": float},
            feature_order=["x", "y"],
        )
        mock_underlying = unittest.mock.MagicMock()
        mock_underlying.predict.side_effect = RuntimeError("exploded")
        with unittest.mock.patch.object(model, "_load", return_value=mock_underlying):
            with pytest.raises(ModelInferenceError, match="exploded"):
                model.predict({"x": 1.0, "y": 2.0})

    def test_predict_proba_included_when_available(self, tmp_path: Path) -> None:
        """predict_proba result is included in output when model supports it and schema has 2+ keys."""
        import unittest.mock

        proba_array = np.array([[0.3, 0.7]])
        mock_underlying = unittest.mock.MagicMock()
        mock_underlying.predict.return_value = np.array([1])
        mock_underlying.predict_proba.return_value = proba_array

        pkl_path, _ = _make_sklearn_model(tmp_path)
        model = SklearnModel(
            name="sk",
            path=pkl_path,
            input_schema={"x": float, "y": float},
            output_schema={"label": float, "proba": float},
            feature_order=["x", "y"],
        )
        with unittest.mock.patch.object(model, "_load", return_value=mock_underlying):
            result = model.predict({"x": 1.0, "y": 2.0})

        assert "label" in result
        assert "proba" in result
        assert result["proba"] == [0.3, 0.7]

    def test_predict_proba_exception_swallowed(self, tmp_path: Path) -> None:
        """predict_proba errors are silently ignored (best-effort)."""
        import unittest.mock

        mock_underlying = unittest.mock.MagicMock()
        mock_underlying.predict.return_value = np.array([1])
        mock_underlying.predict_proba.side_effect = RuntimeError("proba broken")

        pkl_path, _ = _make_sklearn_model(tmp_path)
        model = SklearnModel(
            name="sk",
            path=pkl_path,
            input_schema={"x": float, "y": float},
            output_schema={"label": float, "proba": float},
            feature_order=["x", "y"],
        )
        with unittest.mock.patch.object(model, "_load", return_value=mock_underlying):
            result = model.predict({"x": 1.0, "y": 2.0})

        assert "label" in result
        assert "proba" not in result

    def test_validate_fails_when_loaded_object_has_no_predict(self, tmp_path: Path) -> None:
        """validate() returns invalid when the pickled object has no predict() method."""
        import pickle

        pkl_path = tmp_path / "no_predict.pkl"
        with open(pkl_path, "wb") as fh:
            pickle.dump({"not": "a model"}, fh)

        model = SklearnModel(
            name="sk",
            path=pkl_path,
            input_schema={"x": float},
            output_schema={"pred": float},
            feature_order=["x"],
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = model.validate()

        assert result.valid is False
        assert "predict" in (result.error or "").lower()


# ===========================================================================
# Coverage gap: ONNXModel uncovered branches
# ===========================================================================


class TestONNXModelCoverageGaps:
    def test_load_raises_when_onnxruntime_not_installed(self, tmp_path: Path) -> None:
        """ModelLoadError raised with install hint when onnxruntime is missing."""
        import sys
        import unittest.mock

        dummy_onnx = tmp_path / "dummy.onnx"
        dummy_onnx.write_bytes(b"not real onnx")

        model = ONNXModel("m", dummy_onnx, {}, {})
        with unittest.mock.patch.dict(sys.modules, {"onnxruntime": None}):
            model._session = None
            with pytest.raises(ModelLoadError, match="onnxruntime"):
                model._load()

    def test_load_raises_on_corrupt_onnx_file(self, tmp_path: Path) -> None:
        """ModelLoadError raised when the ONNX file exists but is not a valid model."""
        corrupt_onnx = tmp_path / "corrupt.onnx"
        corrupt_onnx.write_bytes(b"this is not a valid onnx model")

        model = ONNXModel("m", corrupt_onnx, {}, {})
        with pytest.raises(ModelLoadError, match="Failed to load"):
            model._load()

    def test_predict_raises_inference_error_on_session_failure(self) -> None:
        """ModelInferenceError is raised when session.run() raises."""
        import unittest.mock

        from nexa_backtest.exceptions import ModelInferenceError

        model = ONNXModel("m", ONNX_FIXTURE, FIXTURE_INPUT_SCHEMA, FIXTURE_OUTPUT_SCHEMA)
        mock_session = unittest.mock.MagicMock()
        mock_session.run.side_effect = RuntimeError("inference exploded")
        model._session = mock_session

        with pytest.raises(ModelInferenceError, match="inference exploded"):
            model.predict(FIXTURE_FEATURES)

    def test_build_inputs_int_type(self) -> None:
        """Integer features are converted to np.int64 arrays."""
        model = ONNXModel(
            "m",
            ONNX_FIXTURE,
            {"hour": int, "wind": float},
            FIXTURE_OUTPUT_SCHEMA,
        )
        inputs = model._build_inputs({"hour": 12, "wind": 3000.0})
        assert inputs["hour"].dtype == np.int64
        assert inputs["wind"].dtype == np.float32


# ===========================================================================
# Coverage gap: _find_predict_calls uncovered branches
# ===========================================================================


class TestFindPredictCalls:
    def test_syntax_error_returns_empty_list(self) -> None:
        """SyntaxError in source returns an empty list rather than raising."""
        from nexa_backtest.validation.model_check import _find_predict_calls

        result = _find_predict_calls("def broken(: pass")
        assert result == []

    def test_non_ctx_predict_calls_ignored(self) -> None:
        """Call nodes that don't match ctx.predict() pattern are skipped."""
        from nexa_backtest.validation.model_check import _find_predict_calls

        source = (
            "def setup():\n"
            "    other.predict('model_name')\n"  # wrong receiver
            "    ctx.fit('model_name')\n"  # wrong method
            "    predict('model_name')\n"  # not a method call
        )
        result = _find_predict_calls(source)
        assert result == []

    def test_ctx_predict_no_args_not_collected(self) -> None:
        """ctx.predict() with no arguments does not add anything to the result."""
        from nexa_backtest.validation.model_check import _find_predict_calls

        source = "ctx.predict()\n"
        result = _find_predict_calls(source)
        assert result == []

    def test_ctx_predict_non_string_first_arg_not_collected(self) -> None:
        """ctx.predict() with a non-string first arg (e.g. a variable) is not collected."""
        from nexa_backtest.validation.model_check import _find_predict_calls

        source = "ctx.predict(model_name, {})\n"
        result = _find_predict_calls(source)
        assert result == []
