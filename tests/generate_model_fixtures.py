"""Generate a minimal ONNX model fixture for testing.

Creates a simple linear regression model with three named inputs:
  Input:  wind (float32), load (float32), hour (float32)
  Output: variable (float32) — price forecast

The model is intentionally trivial. It exists to test the ML pipeline
plumbing, not to make good predictions.  The ONNX graph is built directly
with the ``onnx`` library so we can control the exact input and output names,
which skl2onnx does not support for multi-input LinearRegression conversion.

Run::

    python tests/generate_model_fixtures.py

Output: tests/fixtures/models/simple_predictor.onnx

Requires ``onnx``, ``numpy``, and ``onnxruntime`` (dev dependencies only).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def generate() -> None:
    """Build and save a hand-crafted linear regression ONNX model.

    The model computes::

        price = 45.2 + 0.005 * load - 0.003 * wind + 0.4 * hour

    This is a fixed linear combination; coefficients are pre-baked into the
    graph as initializers so the model is self-contained with no training step.
    """
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    # Pre-baked coefficients (intercept + 3 weights)
    intercept = np.float32(45.2)
    w_wind = np.float32(-0.003)
    w_load = np.float32(0.005)
    w_hour = np.float32(0.4)

    # Nodes: multiply each input by its weight, sum, add intercept
    mul_wind = helper.make_node("Mul", inputs=["wind", "w_wind"], outputs=["t_wind"])
    mul_load = helper.make_node("Mul", inputs=["load", "w_load"], outputs=["t_load"])
    mul_hour = helper.make_node("Mul", inputs=["hour", "w_hour"], outputs=["t_hour"])

    add1 = helper.make_node("Add", inputs=["t_wind", "t_load"], outputs=["t_wl"])
    add2 = helper.make_node("Add", inputs=["t_wl", "t_hour"], outputs=["t_wlh"])
    add3 = helper.make_node("Add", inputs=["t_wlh", "intercept"], outputs=["variable"])

    graph = helper.make_graph(
        nodes=[mul_wind, mul_load, mul_hour, add1, add2, add3],
        name="simple_price_predictor",
        inputs=[
            helper.make_tensor_value_info("wind", TensorProto.FLOAT, [None, 1]),
            helper.make_tensor_value_info("load", TensorProto.FLOAT, [None, 1]),
            helper.make_tensor_value_info("hour", TensorProto.FLOAT, [None, 1]),
        ],
        outputs=[
            helper.make_tensor_value_info("variable", TensorProto.FLOAT, [None, 1]),
        ],
        initializer=[
            numpy_helper.from_array(np.array([w_wind], dtype=np.float32), name="w_wind"),
            numpy_helper.from_array(np.array([w_load], dtype=np.float32), name="w_load"),
            numpy_helper.from_array(np.array([w_hour], dtype=np.float32), name="w_hour"),
            numpy_helper.from_array(np.array([intercept], dtype=np.float32), name="intercept"),
        ],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 15)])
    model.doc_string = "Minimal test fixture for nexa-backtest ML pipeline tests."
    onnx.checker.check_model(model)

    output_path = Path(__file__).parent / "fixtures" / "models" / "simple_predictor.onnx"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(model.SerializeToString())

    print(f"Model saved to {output_path}")
    print(f"Size: {output_path.stat().st_size} bytes")

    import onnxruntime as ort  # type: ignore[import-untyped]

    session = ort.InferenceSession(str(output_path))
    inputs = {
        "wind": np.array([[3000.0]], dtype=np.float32),
        "load": np.array([[5000.0]], dtype=np.float32),
        "hour": np.array([[12.0]], dtype=np.float32),
    }
    result = session.run(None, inputs)
    print(f"Test prediction: {result[0][0][0]:.2f} EUR/MWh")
    print(f"Input names:  {[i.name for i in session.get_inputs()]}")
    print(f"Output names: {[o.name for o in session.get_outputs()]}")


if __name__ == "__main__":
    generate()
