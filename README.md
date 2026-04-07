# nexa-backtest

**Energy market backtesting framework for European power trading.**

[![CI](https://github.com/phasenexa/nexa-backtest/actions/workflows/ci.yml/badge.svg)](https://github.com/phasenexa/nexa-backtest/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/nexa-backtest)](https://pypi.org/project/nexa-backtest/)
[![Python](https://img.shields.io/pypi/pyversions/nexa-backtest)](https://pypi.org/project/nexa-backtest/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

Most backtesting frameworks are built for equities. They assume continuous order books, tick-by-tick data, and price-time priority. European power markets work differently: day-ahead auctions with gate closures, intraday continuous trading, block bids, linked orders, and 15-minute MTUs.

nexa-backtest is purpose-built for this. It replays historical market conditions, runs your trading algorithm against them, and answers two questions: **did it make money?** and **did it beat VWAP?**

## Key Features

- **One interface, three modes.** Your algo runs identically in backtest, paper trading, and live trading. Same code, different engine underneath. Zero changes to go from replay to production.
- **Two API levels.** `SimpleAlgo` with hooks for quick experiments. `@algo` with async event streams for full control.
- **DA + IDC support.** Day-ahead auction matching (price-taker against historical clearing prices) and intraday continuous matching (price-time priority against historical order book).
- **Signal system.** Plug in weather forecasts, DA prices, load forecasts, gas prices, or any time-series data. Built-in look-ahead bias prevention via `publication_offset`.
- **ML model integration.** Register ONNX or scikit-learn models and call `ctx.predict()` from your algo.
- **Exchange adapters.** Nord Pool, EPEX SPOT, EEX. Each adapter declares its capabilities. Use block bids on an exchange that doesn't support them? The validator catches it before you run.
- **Efficient replay.** DA data loads entirely (it's tiny). IDC data uses windowed replay via PyArrow row groups, keeping peak memory at 200-500 MB regardless of replay period.

## Installation

```bash
pip install nexa-backtest
```

With optional extras:

```bash
pip install nexa-backtest[pandas]       # DataFrame output
pip install nexa-backtest[ml]           # ONNX model inference
pip install nexa-backtest[charts]       # Report charts (matplotlib/plotly)
pip install nexa-backtest[marketdata]   # Data fetching via nexa-marketdata
pip install nexa-backtest[live]         # Live trading via nexa-connect
pip install nexa-backtest[all]          # Everything
```

## Walkthrough Notebook

New to backtesting or power markets? Start here:

**[notebooks/backtester_walkthrough.ipynb](notebooks/backtester_walkthrough.ipynb)**

A guided notebook covering: what a backtester is, MTUs, DA auctions, gate closures, the price-taker assumption, writing your first algo, running a backtest, interpreting every result metric, the signal system, and look-ahead bias prevention. Runs end-to-end against the included fixture data with no extra setup.

```bash
jupyter notebook notebooks/backtester_walkthrough.ipynb
```

## Quick Start

### Write an algo

```python
from nexa_backtest import SimpleAlgo, TradingContext, Order

class BuyBelowForecast(SimpleAlgo):
    """Buy when DA clearing price is below our forecast."""

    def on_setup(self, ctx: TradingContext) -> None:
        self.subscribe_signal("price_forecast")
        self.threshold = 5.0  # EUR/MWh

    def on_auction_open(self, ctx: TradingContext, auction: AuctionInfo) -> None:
        forecast = ctx.get_signal("price_forecast").value
        ctx.place_order(Order.buy(
            product=auction.product_id,
            volume_mw=10,
            price_eur=forecast - self.threshold,
        ))

    def on_fill(self, ctx: TradingContext, fill: Fill) -> None:
        ctx.log(f"Filled {fill.volume_mw} MW @ {fill.price_eur}")
```

### Backtest it

```python
from datetime import date
from nexa_backtest import BacktestEngine
from nexa_backtest.signals import CsvSignalProvider

result = BacktestEngine(
    algo=BuyBelowForecast(),
    exchange="nordpool",
    start=date(2026, 3, 1),
    end=date(2026, 3, 31),
    products=["NO1_DA"],
    signals=[
        CsvSignalProvider(
            name="price_forecast",
            path="data/signals/price_forecast.csv",
            unit="EUR/MWh",
            publication_offset=timedelta(hours=12),
        ),
    ],
    initial_capital=100_000,
).run()

print(result.summary())
# Total PnL: +12,340.50 EUR
# vs VWAP: +3.2%
# Win rate: 62%
# Trades: 186
```

Or run from the CLI:

```bash
# Signal CSVs are discovered by convention in {data_dir}/signals/
nexa run my_algo.py \
    --exchange nordpool \
    --start 2026-03-01 \
    --end 2026-03-31 \
    --products NO1_DA \
    --data-dir ./data \
    --capital 100000
```

### Paper trade it (same algo, zero changes)

```python
from nexa_backtest import PaperEngine

paper = PaperEngine(
    algo=BuyBelowForecast(),
    exchange="nordpool",
    products=["NO1_DA"],
    signals=[...],
).start()
```

### Go live (same algo, zero changes)

```python
from nexa_backtest import LiveEngine

live = LiveEngine(
    algo=BuyBelowForecast(),
    exchange="nordpool",
    credentials=NordPoolCredentials.from_env(),
    products=["NO1_DA"],
    signals=[...],
).start()
```

## The Low-Level API

For quants who want full control over the event loop:

```python
from nexa_backtest import TradingContext, algo

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
                if remaining < timedelta(minutes=5):
                    pos = ctx.get_position(pid)
                    if pos.net_mw != 0:
                        ctx.place_order(Order.market(
                            product=pid,
                            volume_mw=-pos.net_mw,
                        ))
```

## Signals

Any time-series data your algo needs. Load a CSV or implement the `SignalProvider` protocol:

```python
from nexa_backtest.signals import CsvSignalProvider

# Load your own forecast data from CSV
forecast = CsvSignalProvider(
    name="my_forecast",
    path="data/signals/my_forecast.csv",
    unit="EUR/MWh",
    description="Our internal price forecast",
    publication_offset=timedelta(hours=12),  # Published 12h before delivery
)

engine = BacktestEngine(
    algo=algo,
    signals=[forecast],
    # ...
)
```

CSV format (simple two-column minimum):

```csv
timestamp,value
2026-03-15T00:00:00+01:00,42.31
2026-03-15T00:15:00+01:00,41.87
2026-03-15T00:30:00+01:00,43.05
```

For full control, implement `SignalProvider` directly:

```python
from nexa_backtest.signals import SignalProvider, SignalSchema, SignalValue

class MyModelForecast(SignalProvider):
    name = "model_forecast"
    schema = SignalSchema(
        name="model_forecast",
        dtype=float,
        frequency=timedelta(minutes=15),
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

Look-ahead bias is prevented automatically. The `publication_offset` controls when forecast values become visible to your algo. A value for delivery period T with `publication_offset=timedelta(hours=6)` was published at T - 6 hours, so it only becomes visible when the simulated clock reaches that publication time.

## ML Models

Register models and call `ctx.predict()` from your algo:

```python
from nexa_backtest.models import ModelRegistry, ONNXModel

models = ModelRegistry()
models.register(ONNXModel(
    name="price_predictor",
    path="models/xgboost_prices.onnx",
    input_schema={"wind": float, "load": float, "hour": int},
    output_schema={"price_forecast": float},
))

engine = BacktestEngine(algo=algo, models=models, ...)

# In your algo:
prediction = ctx.predict("price_predictor", {
    "wind": ctx.get_signal("wind_forecast").value,
    "load": ctx.get_signal("load_forecast").value,
    "hour": ctx.now().hour,
})
```

ONNX is recommended (portable, fast, no arbitrary code execution). Scikit-learn pickle is supported but flagged as a security risk in hosted environments.

## Validation

Catch bugs before they cost you a 10-minute backtest run:

```bash
$ nexa validate my_algo.py --exchange nordpool

Step 1/6: Syntax Check (ruff)          [PASS]
Step 2/6: Type Check (mypy --strict)   [PASS]
Step 3/6: Interface Compliance         [PASS]
Step 4/6: Exchange Feature Compat      [FAIL]
  - Line 42: Order.block_bid() used, but Nord Pool IDC
    does not support block bids.
Step 5/6: Look-ahead Bias Detection    [PASS]
Step 6/6: Resource Safety              [PASS]

1 error. Fix before running.
```

## Historical Data

Backtest data is stored as Parquet files. Use `nexa-marketdata` to fetch and cache data, or bring your own:

```python
from nexa_backtest.data import ParquetLoader, NexaMarketdataLoader

# From local files
loader = ParquetLoader(data_dir="./data/nordpool")

# Or fetch via nexa-marketdata
loader = NexaMarketdataLoader(
    source="nordpool",
    zones=["NO1", "NO2"],
    start=date(2025, 10, 1),
    end=date(2026, 3, 31),
)
```

**Data format examples (shown as CSV for clarity, stored as Parquet):**

DA clearing prices:

```csv
timestamp,zone,price_eur_mwh,volume_mwh
2026-03-15T00:00:00+01:00,NO1,42.31,1250.5
2026-03-15T00:15:00+01:00,NO1,41.87,1180.2
2026-03-15T00:30:00+01:00,NO1,43.05,1310.8
```

IDC events:

```csv
timestamp,event_type,order_id,zone,product_id,side,price_eur_mwh,volume_mw,remaining_mw
2026-03-15T08:12:03.412+01:00,new,ord-88291,NO1,NO1-QH-0900,buy,52.40,5.0,5.0
2026-03-15T08:12:03.987+01:00,new,ord-88292,NO1,NO1-QH-0900,sell,53.10,3.0,3.0
2026-03-15T08:12:04.201+01:00,trade,ord-88293,NO1,NO1-QH-0900,buy,53.10,2.0,0.0
```

## Code Protection

If you don't want to share your algo's source code with a hosted platform:

```bash
# Compile to native shared library (Cython)
$ nexa compile my_algo.py --output my_algo.so

# Or maximum protection (Nuitka)
$ nexa compile my_algo.py --compiler nuitka --output my_algo_binary
```

Upload the compiled binary. We run it but can't read it.

## CLI Reference

```bash
nexa run my_algo.py --exchange nordpool --start 2026-03-01 --end 2026-03-31
nexa validate my_algo.py --exchange nordpool
nexa compile my_algo.py --output my_algo.so
nexa report results.parquet --format html --output report.html
```

## Phase Nexa Ecosystem

nexa-backtest integrates with the wider Phase Nexa toolkit:

| Package | Role | Integration |
|---------|------|-------------|
| [nexa-marketdata](https://github.com/phasenexa/nexa-marketdata) | Market data fetching | Historical data source for backtests |
| [nexa-bidkit](https://github.com/phasenexa/nexa-bidkit) | Bid generation | Order type definitions and validation |
| [nexa-connect](https://github.com/phasenexa/nexa-connect) | Exchange connectivity | Powers the live trading engine |
| [nexa-forecast](https://github.com/phasenexa/nexa-forecast) | Price forecasting | ML models and signal providers |
| [nexa-mcp](https://github.com/phasenexa/nexa-mcp) | LLM interface | Run backtests from chat |

## Implementation Status

| Feature | Status |
|---------|--------|
| Core types and protocols | Task 01 |
| SimpleAlgo with DA hooks | Task 01 |
| BacktestEngine (DA only) | Task 01 |
| Nord Pool DA adapter | Task 01 |
| Parquet data loader (DA) | Task 01 |
| PnL + VWAP analysis | Task 01 |
| Signal system + CSV loader | Task 02 |
| CLI (`nexa run`) | Task 02 |
| IDC continuous matching | Planned (Stage 2) |
| Windowed replay | Planned (Stage 2) |
| `@algo` low-level API | Planned (Stage 2) |
| Built-in signal providers | Planned (Stage 2) |
| EPEX SPOT / EEX adapters | Planned (Stage 2) |
| HTML reports | Planned (Stage 2) |
| Sharpe / drawdown / equity curve | Planned (Stage 2) |
| Validation pipeline | Planned (Stage 3) |
| ML model registry | Planned (Stage 3) |
| Multi-algo replay | Planned (Stage 3) |
| Paper trading engine | Planned (Stage 4) |
| Live trading engine | Planned (Stage 4) |
| Code compilation | Planned (Stage 4) |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, coding standards, and PR workflow.

## License

[MIT](./LICENSE)
