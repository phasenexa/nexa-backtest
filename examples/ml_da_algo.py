"""Example: use an ONNX price prediction model to inform DA bidding.

This script demonstrates the full ML workflow for nexa-backtest:

1. Build a tiny synthetic dataset from historical DA prices.
2. Train a simple scikit-learn model (linear regression).
3. Export the model to ONNX format.
4. Register the model with the backtester.
5. Run two backtests — one naive (bid at clearing price) and one
   model-driven (bid at the predicted price) — and compare PnL.

Prerequisites (install the dev extras)::

    pip install nexa-backtest[ml] scikit-learn skl2onnx

Run::

    python examples/ml_da_algo.py --data-dir ./data/nordpool \\
        --start 2026-03-01 --end 2026-03-31 --zone NO1

The script will train a model from the DA price history in the Parquet
file, then backtest against the same period.  Use a separate
train/test split in production to avoid overfitting.
"""

from __future__ import annotations

import argparse
import tempfile
from datetime import date
from decimal import Decimal
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# 1. Load historical DA prices from the Parquet fixture
# ---------------------------------------------------------------------------


def load_prices(data_dir: Path, zone: str) -> tuple[np.ndarray, np.ndarray]:
    """Load DA clearing prices and return (timestamps_ordinal, prices).

    Args:
        data_dir: Directory containing ``da_prices.parquet``.
        zone: Bidding zone filter, e.g. ``"NO1"``.

    Returns:
        Tuple of (X, y) where X is hour-of-day (float) and y is price (float).
    """
    table = pq.read_table(data_dir / "da_prices.parquet")
    df = table.to_pandas()
    df = df[df["zone"] == zone].copy()
    df = df.sort_values("timestamp")
    df["hour"] = df["timestamp"].dt.hour.astype(float)
    X = df[["hour"]].values.astype(np.float32)
    y = df["price_eur_mwh"].values.astype(np.float32)
    return X, y


# ---------------------------------------------------------------------------
# 2. Train a sklearn model and export to ONNX
# ---------------------------------------------------------------------------


def train_and_export(X: np.ndarray, y: np.ndarray, output_path: Path) -> None:
    """Train a linear regression model and export it to ONNX.

    Args:
        X: Feature array, shape (n, 1). Column: hour of day.
        y: Target array, shape (n,). DA clearing price in EUR/MWh.
        output_path: Where to write the ``.onnx`` file.

    The exported model has a single input tensor named ``"hour"`` of shape
    ``[None, 1]`` and a single output named ``"variable"``.
    """
    from sklearn.linear_model import LinearRegression  # type: ignore[import-untyped]
    from skl2onnx import convert_sklearn  # type: ignore[import-untyped]
    from skl2onnx.common.data_types import FloatTensorType  # type: ignore[import-untyped]

    model = LinearRegression()
    model.fit(X, y)
    print(f"  Trained model: intercept={model.intercept_:.2f}, coef={model.coef_[0]:.4f}")

    # skl2onnx expects a single input for LinearRegression
    onnx_model = convert_sklearn(
        model,
        initial_types=[("hour", FloatTensorType([None, 1]))],
        target_opset=15,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"  Model exported to {output_path} ({output_path.stat().st_size} bytes)")


# ---------------------------------------------------------------------------
# 3. Define trading algos
# ---------------------------------------------------------------------------


def run_backtests(
    data_dir: Path,
    start: date,
    end: date,
    zone: str,
    model_path: Path,
) -> None:
    """Run naive and model-driven backtests and print a comparison.

    Args:
        data_dir: Historical data directory.
        start: Backtest start date.
        end: Backtest end date.
        zone: Bidding zone, e.g. ``"NO1"``.
        model_path: Path to the exported ONNX model.
    """
    from nexa_backtest.algo import SimpleAlgo
    from nexa_backtest.context import TradingContext
    from nexa_backtest.engines.backtest import BacktestEngine
    from nexa_backtest.models.onnx import ONNXModel
    from nexa_backtest.models.registry import ModelRegistry
    from nexa_backtest.types import AuctionInfo, Order

    product = f"{zone}_DA"
    capital = Decimal("100000")

    # -----------------------------------------------------------------------
    # Algo A: naive — always bid 1 MW at the predicted price
    # -----------------------------------------------------------------------
    class NaiveAlgo(SimpleAlgo):
        """Bid 1 MW at a fixed price regardless of market conditions."""

        def on_auction_open(self, ctx: TradingContext, auction: AuctionInfo) -> None:
            ctx.place_order(
                Order.buy(
                    product_id=auction.product_id,
                    price_eur_mwh=Decimal("50"),  # flat bid
                    volume_mw=Decimal("1"),
                )
            )

    # -----------------------------------------------------------------------
    # Algo B: model-driven — bid at the model's predicted price
    # -----------------------------------------------------------------------
    class ModelAlgo(SimpleAlgo):
        """Bid 1 MW at the ONNX model's predicted clearing price.

        The model takes the auction hour of day as input and outputs a
        price forecast.  We bid slightly above the forecast to improve
        fill probability.
        """

        def on_auction_open(self, ctx: TradingContext, auction: AuctionInfo) -> None:
            hour = float(ctx.now().hour)
            # ctx.predict() calls the registered ONNX model
            prediction = ctx.predict("price_model", {"hour": hour})
            predicted_price = Decimal(str(round(prediction["variable"], 2)))
            # Bid 0.50 EUR/MWh above the forecast to improve fill probability
            bid_price = predicted_price + Decimal("0.50")
            ctx.place_order(
                Order.buy(
                    product_id=auction.product_id,
                    price_eur_mwh=bid_price,
                    volume_mw=Decimal("1"),
                )
            )

    # Register the ONNX model (single float input: hour)
    registry = ModelRegistry()
    registry.register(
        ONNXModel(
            name="price_model",
            path=model_path,
            input_schema={"hour": float},
            output_schema={"variable": float},
        )
    )

    print(f"\nRunning naive backtest ({start} to {end})...")
    naive_result = BacktestEngine(
        algo=NaiveAlgo(),
        exchange="nordpool",
        start=start,
        end=end,
        products=[product],
        data_dir=data_dir,
        capital=capital,
    ).run()

    print(f"\nRunning model-driven backtest ({start} to {end})...")
    model_result = BacktestEngine(
        algo=ModelAlgo(),
        exchange="nordpool",
        start=start,
        end=end,
        products=[product],
        data_dir=data_dir,
        capital=capital,
        models=registry,
    ).run()

    # -----------------------------------------------------------------------
    # 5. Compare results
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("COMPARISON: Naive vs Model-Driven")
    print("=" * 60)
    print(f"{'Metric':<30} {'Naive':>14} {'Model-Driven':>14}")
    print("-" * 60)

    def _fmt(val: object) -> str:
        if val is None:
            return "N/A"
        return f"{float(val):>14.2f}"  # type: ignore[arg-type]

    print(f"{'Realised PnL (EUR)':<30} {_fmt(naive_result.pnl.realised_pnl)} {_fmt(model_result.pnl.realised_pnl)}")
    print(f"{'vs VWAP (EUR/MWh)':<30} {_fmt(naive_result.pnl.vs_vwap_eur_mwh)} {_fmt(model_result.pnl.vs_vwap_eur_mwh)}")
    print(f"{'Fills':<30} {len(naive_result.fills):>14} {len(model_result.fills):>14}")
    print(f"{'Sharpe Ratio':<30} {_fmt(naive_result.sharpe_ratio)} {_fmt(model_result.sharpe_ratio)}")
    print(f"{'Max Drawdown (EUR)':<30} {_fmt(naive_result.max_drawdown)} {_fmt(model_result.max_drawdown)}")
    print("=" * 60)

    naive_pnl = float(naive_result.pnl.realised_pnl or 0)
    model_pnl = float(model_result.pnl.realised_pnl or 0)
    if model_pnl > naive_pnl:
        print(f"\n✓ Model-driven algo outperformed naive by {model_pnl - naive_pnl:.2f} EUR")
    else:
        print(f"\n✗ Naive algo outperformed model-driven by {naive_pnl - model_pnl:.2f} EUR")
        print("  (This is expected with a trivial 1-feature linear model.)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse CLI args, train model, run backtests."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", required=True, help="Directory containing da_prices.parquet")
    parser.add_argument("--start", required=True, help="Backtest start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="Backtest end date (YYYY-MM-DD)")
    parser.add_argument("--zone", default="NO1", help="Bidding zone (default: NO1)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)

    # Step 1: Load prices
    print("Step 1: Loading DA prices...")
    X, y = load_prices(data_dir, args.zone)
    print(f"  Loaded {len(y)} price points from zone {args.zone}")

    # Step 2: Train and export
    print("\nStep 2: Training model and exporting to ONNX...")
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "price_model.onnx"
        train_and_export(X, y, model_path)

        # Steps 3-5: Backtest and compare
        print("\nStep 3: Running backtests...")
        run_backtests(data_dir, start, end, args.zone, model_path)


if __name__ == "__main__":
    main()
