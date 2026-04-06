# nexa-backtest

[![CI](https://github.com/phasenexa/nexa-backtest/actions/workflows/ci.yml/badge.svg)](https://github.com/phasenexa/nexa-backtest/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/nexa-backtest)](https://pypi.org/project/nexa-backtest/)
[![Python](https://img.shields.io/pypi/pyversions/nexa-backtest)](https://pypi.org/project/nexa-backtest/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A backtesting framework built for European power markets. Not another equities backtester
with energy bolted on.

Handles day-ahead auctions, intraday auctions, intraday continuous trading, 15-minute MTUs,
block bids, gate closures, and exchange-specific matching rules. Runs your algo against
historical data and tells you: **did it make money? Did it beat VWAP?**

## Why this exists

Every backtesting framework out there assumes continuous order books, tick-by-tick data,
and price-time priority. Energy markets work differently. You have auctions with gate
closures, 96 quarter-hour products per day, block bids that span multiple hours, and
matching algorithms that clear everything at once.

If you have tried backtesting an energy trading strategy with Zipline, Backtrader, or
VectorBT, you know the pain. `nexa-backtest` is the tool those frameworks should have been.

## Features

- **Purpose-built for energy**: DA auctions, ID auctions, IDC continuous, 15-min MTUs
- **One interface, three engines**: same algo code for backtesting, paper trading, and live trading
- **Two API levels**: `SimpleAlgo` for quick experiments, `@algo` decorator for full control
- **Exchange adapters**: Nord Pool, EPEX SPOT, EEX with feature detection and validation
- **Signal system**: weather, price forecasts, load data, carbon prices, or anything custom
- **ML model support**: ONNX, scikit-learn, PyTorch models via `ctx.predict()`
- **Smart data loading**: DA data loaded entirely (tiny), IDC data windowed from Parquet (scalable)
- **Validation pipeline**: ruff + mypy + exchange feature checks + look-ahead bias detection
- **Code protection**: Cython/Nuitka compilation for IP-sensitive hosted environments

## Installation

```bash
pip install nexa-backtest
```

With optional extras:

```bash
pip install nexa-backtest[pandas]     # DataFrame output
pip install nexa-backtest[plot]       # matplotlib/plotly charts
pip install nexa-backtest[ml]         # ONNX + scikit-learn model support
pip install nexa-backtest[data]       # nexa-marketdata integration
pip install nexa-backtest[live]       # nexa-connect for live trading
pip install nexa-backtest[all]        # everything
```

## Quick start

### Your first backtest (20 lines)

```python
from datetime import date
from nexa_backtest import SimpleAlgo, TradingContext, Order, BacktestEngine

class BuyBelowForecast(SimpleAlgo):
    """Buy when DA clearing price is below the wind forecast signal."""

    def on_setup(self, ctx: TradingContext) -> None:
        self.subscribe_signal("da_price_forecast")
        self.subscribe_signal("wind_generation_forecast")

    def on_auction_open(self, ctx: TradingContext, auction) -> None:
        forecast = ctx.get_signal("da_price_forecast").value
        wind = ctx.get_signal("wind_generation_forecast").value

        if wind > 15_000:  # High wind expected, prices likely low
            ctx.place_order(Order.buy(
                product=auction.product_id,
                volume_mw=10,
                price_eur=forecast - 5.0,
            ))

    def on_fill(self, ctx: TradingContext, fill) -> None:
        ctx.log(f"Filled {fill.volume_mw} MW @ {fill.price_eur}")

# Run it
result = BacktestEngine(
    algo=BuyBelowForecast(),
    exchange="nordpool",
    start=date(2026, 3, 1),
    end=date(2026, 3, 31),
    products=["NO1_DA"],
    initial_capital=100_000,
).run()

print(result.summary())
# Total PnL: +12,340.50 EUR
# vs VWAP:   +3.2%
# Sharpe:    1.4
# Win rate:  62%
# Max DD:    -4,200.00 EUR
```

### Full control with @algo

For quants who want to manage their own event loop:

```python
from nexa_backtest import TradingContext, Order, algo

@algo(name="spread_scalper", version="1.0.0")
async def run(ctx: TradingContext) -> None:
    async for event in ctx.events():
        match event:
            case MarketDataUpdate(product_id=pid):
                book = ctx.get_orderbook(pid)
                spread = book.best_ask.price - book.best_bid.price
                if spread > 2.0:
                    ctx.place_order(Order.buy(
                        product=pid,
                        volume_mw=5,
                        price_eur=book.best_bid.price + 0.5,
                    ))

            case GateClosureWarning(product_id=pid, remaining=remaining):
                if remaining.total_seconds() < 300:
                    pos = ctx.get_position(pid)
                    if pos.net_mw != 0:
                        ctx.place_order(Order.market(
                            product=pid,
                            volume_mw=-pos.net_mw,
                        ))
```

### Same algo, three modes

```python
from nexa_backtest import BacktestEngine, PaperEngine, LiveEngine

algo = BuyBelowForecast()

# Backtest: historical replay, simulated matching
result = BacktestEngine(algo=algo, exchange="nordpool", ...).run()

# Paper: live data, simulated matching, no real money
paper = PaperEngine(algo=algo, exchange="nordpool", ...).start()

# Live: real data, real exchange, real money
live = LiveEngine(algo=algo, exchange="nordpool", credentials=...).start()
```

The algo code is identical in all three cases. The only thing that changes is
which engine you pass it to.

### Using signals

```python
from nexa_backtest.signals import (
    DayAheadPriceSignal,
    WindForecastSignal,
    LoadForecastSignal,
)

result = BacktestEngine(
    algo=algo,
    exchange="nordpool",
    start=date(2026, 3, 1),
    end=date(2026, 3, 31),
    signals=[
        DayAheadPriceSignal(zone="NO1"),
        WindForecastSignal(zone="NO1", provider="meteomatics"),
        LoadForecastSignal(zone="NO1"),
    ],
).run()
```

Custom signals implement `SignalProvider`:

```python
from nexa_backtest.signals import SignalProvider, SignalSchema, SignalValue

class MyForecast(SignalProvider):
    name = "my_forecast"
    schema = SignalSchema(
        name="my_forecast",
        dtype=float,
        frequency=timedelta(minutes=15),
        description="Internal price forecast",
        unit="EUR/MWh",
    )

    def __init__(self, data_path: str):
        self._data = pd.read_parquet(data_path)

    def get_value(self, timestamp: datetime) -> SignalValue:
        return SignalValue(
            timestamp=timestamp,
            value=self._data.loc[timestamp, "forecast"],
        )
```

### Using ML models

```python
from nexa_backtest.models import ModelRegistry, ONNXModel

models = ModelRegistry()
models.register(ONNXModel(
    name="price_predictor",
    path="models/price_xgboost.onnx",
    input_schema={"wind": float, "load": float, "hour": int},
    output_schema={"price_forecast": float},
))

result = BacktestEngine(
    algo=algo,
    exchange="nordpool",
    models=models,
    ...
).run()

# In your algo:
prediction = ctx.predict("price_predictor", {
    "wind": ctx.get_signal("wind_forecast").value,
    "load": ctx.get_signal("load_forecast").value,
    "hour": ctx.now().hour,
})
```

### Validation

Catch bugs before they cost you a 10-minute backtest run:

```bash
$ nexa validate my_algo.py --exchange nordpool

Step 1/6: Syntax Check (ruff)
  [PASS] No syntax errors

Step 2/6: Type Check (mypy --strict)
  [PASS] TradingContext protocol satisfied

Step 3/6: Interface Compliance
  [PASS] Required hooks implemented

Step 4/6: Exchange Feature Compatibility
  [PASS] All order types supported by Nord Pool

Step 5/6: Look-ahead Bias Detection
  [PASS] No future data access detected

Step 6/6: Resource Safety
  [WARN] Line 78: time.sleep() detected. Use ctx.wait() instead.

Result: PASSED (1 warning)
```

### PnL analysis

```python
result = engine.run()

# Summary
print(result.summary())

# VWAP comparison
print(result.vwap_analysis())
# Period      | Your Avg | VWAP   | Edge    | Volume
# 2026-03-01  | 42.30    | 43.15  | +0.85   | 120 MW
# 2026-03-02  | 38.90    | 39.20  | +0.30   | 95 MW
# TOTAL       | 44.20    | 44.85  | +0.65   | 3,240 MW

# Export
result.to_parquet("results/march.parquet")
result.to_html("results/march.html")  # full report with charts
trades_df = result.trades.to_dataframe()
```

## Historical data format

Data is stored as Parquet files. DA data is tiny (load entirely). IDC data is
large (windowed replay).

```
data/
  nordpool/
    da_clearing_prices/
      NO1_2025.parquet              # ~1.7 MB, loaded entirely
      NO1_2026.parquet
    idc_orderbook_snapshots/
      NO1_2026_01.parquet           # ~800 MB, windowed replay
      NO1_2026_02.parquet
    idc_events/
      NO1_2026_01.parquet           # ~125 MB, windowed replay
    idc_trades/
      NO1_2026_01.parquet           # ~17 MB, windowed replay
  signals/
    wind_forecast/
      NO1_2026.parquet              # ~50 MB, loaded entirely
```

Use `nexa-marketdata` to fetch and cache historical data, then `nexa-backtest`
replays it:

```python
from nexa_backtest.data import NexaMarketdataLoader

loader = NexaMarketdataLoader(
    source="nordpool",
    zones=["NO1", "NO2"],
    start=date(2025, 10, 1),
    end=date(2026, 3, 31),
)
# Downloads and caches locally as Parquet
```

## Exchange support

| Exchange | DA Auction | ID Auction | IDC Continuous | Status |
|----------|:----------:|:----------:|:--------------:|--------|
| Nord Pool | Yes | Yes | Yes | Planned |
| EPEX SPOT | Yes | Yes | Yes | Planned |
| EEX | Yes | - | - | Planned |

Each exchange adapter declares its capabilities. The validation pipeline checks
your algo uses only supported features before runtime:

```
$ nexa validate my_algo.py --exchange epex_spot

  [FAIL] Feature compatibility:
    - Line 42: Order.block_bid() used, but EPEX SPOT continuous
      does not support block bids.

  1 error. Fix before running.
```

## Code protection

For hosted environments where you do not want to share source code:

```bash
# Compile to native binary (Cython)
$ nexa compile my_algo.py --output my_algo.so

# Upload compiled binary, not source
$ nexa upload my_algo.so --config backtest.yaml
```

| Approach | IP Protection | Performance | Best for |
|----------|:------------:|:-----------:|----------|
| Self-hosted | N/A (local) | Fastest | Most users |
| Cython | Good | Fast | Hosted backtesting |
| Nuitka | Very good | Fast | Maximum protection |
| Container | Excellent | Slower | Enterprise |

## Phase Nexa ecosystem

`nexa-backtest` integrates with the rest of the Phase Nexa toolkit:

```
nexa-marketdata ---- fetches data ------> nexa-backtest (replay)
nexa-bidkit -------- bid construction --> nexa-backtest (order types)
nexa-connect ------- exchange comms ----> nexa-backtest (live engine)
nexa-forecast ------ ML models ---------> nexa-backtest (signals/models)
nexa-mcp ----------- LLM interface -----> nexa-backtest (run from chat)
```

Each piece is independently useful. Together they form a complete trading
development environment.

## Performance

| Scenario | Time | Peak memory |
|----------|------|-------------|
| 1 year, DA, 1 zone | < 1 second | ~50 MB |
| 1 year, DA, 10 zones | < 5 seconds | ~200 MB |
| 1 year, IDC, 1 zone | 1-3 minutes | ~300 MB |
| 1 year, IDC, 5 zones | 3-8 minutes | ~500 MB |
| 4 algos shared, IDC, 5 zones | 5-10 minutes | ~700 MB |

Times assume Parquet on local SSD/NVMe.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, coding standards,
and the PR process.

## License

MIT
