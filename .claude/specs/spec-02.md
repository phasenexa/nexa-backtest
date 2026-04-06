# Task 02: Signals, CSV Signal Loader, and CLI

## Goal

Add the signal system and a basic CLI so a customer can bring their own
forecast data (as CSV), reference it from their algo, and run a backtest
from the command line rather than writing a Python script.

This builds on task-01. All existing types, the DA matching engine, and
the PnL/VWAP analysis remain unchanged.

---

## What to build

### 1. `signals/base.py`

The `SignalProvider` protocol and supporting types:

```python
class SignalSchema:
    """Describes what a signal provides."""
    name: str
    dtype: type          # float, int, str
    frequency: timedelta # How often it updates
    description: str
    unit: str            # "EUR/MWh", "MW", "m/s", "celsius", etc.

class SignalValue:
    """A single signal observation."""
    timestamp: datetime  # When this value is valid for
    value: Any           # The actual value

class SignalProvider(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def schema(self) -> SignalSchema: ...

    def get_value(self, timestamp: datetime) -> SignalValue: ...
    def get_range(self, start: datetime, end: datetime) -> pd.Series: ...
```

Use Pydantic v2 for `SignalSchema` and `SignalValue`.

### 2. `signals/registry.py`

A `SignalRegistry` that holds registered signal providers and provides
lookup by name. Nothing fancy - a dict with validation that the signal
exists when the algo asks for it.

```python
class SignalRegistry:
    def register(self, provider: SignalProvider) -> None: ...
    def get(self, name: str) -> SignalProvider: ...
    def has(self, name: str) -> bool: ...
    def list_signals(self) -> list[str]: ...
```

### 3. `signals/csv_loader.py`

A `CsvSignalProvider` that loads a CSV file and serves it as a signal.
This is the simplest way for a customer to bring their own data without
implementing `SignalProvider` from scratch.

Expected CSV format:
```csv
timestamp,value
2026-03-15T00:00:00+01:00,42.31
2026-03-15T00:15:00+01:00,41.87
2026-03-15T00:30:00+01:00,43.05
```

Columns: `timestamp` (required, parsed as timezone-aware datetime) and
`value` (required, parsed as float). Additional columns are ignored.

The provider must support an optional `publication_offset` parameter to
prevent look-ahead bias. The offset represents how far ahead of the
delivery period the forecast is published. A value with timestamp T
(the period it describes) is visible to the algo at time
T - publication_offset (when it was published).

For example, a wind forecast published 6 hours ahead would use
`publication_offset=timedelta(hours=6)`. The forecast for the 06:00
delivery period was published at 00:00, so it becomes visible when the
simulated clock reaches 00:00. In code: `get_value(current_time)` returns
the latest value where `timestamp <= current_time + publication_offset`.

If `publication_offset` is not set, values are available at their timestamp
(i.e., the value for 06:00 is available at 06:00). This is correct for
actuals/historical data but would be look-ahead bias for forecasts. Log
a warning when no offset is set, suggesting the user consider whether
their data represents forecasts or actuals.

```python
signal = CsvSignalProvider(
    name="my_wind_forecast",
    path="data/wind_forecast_NO1.csv",
    unit="MW",
    description="Wind generation forecast for NO1",
    publication_offset=timedelta(hours=6),
)
```

### 4. Update `context.py`

Add the signal methods to `TradingContext` (they exist as stubs from
task-01, now they need real signatures):

```python
def get_signal(self, name: str) -> SignalValue: ...
def get_signal_history(self, name: str, lookback: int) -> list[SignalValue]: ...
```

### 5. Update `engines/backtest.py`

Wire signals into the `BacktestEngine`:

- Accept a `signals` parameter (list of `SignalProvider` instances)
- Register them in a `SignalRegistry`
- When the algo calls `ctx.get_signal(name)`, look up the provider and
  return the value for the current simulated time, respecting
  `publication_offset`
- When the algo calls `ctx.get_signal_history(name, lookback=N)`, return
  the last N values up to and including the current time

The backtest context implementation needs to enforce the look-ahead bias
rule: at simulated time T, `get_signal` must not return any value whose
publication time is after T.

### 6. Update `algo.py`

Add `subscribe_signal(name)` to `SimpleAlgo` and the `on_signal` hook:

```python
class SimpleAlgo:
    def subscribe_signal(self, name: str) -> None:
        """Register interest in a signal. Called during on_setup."""

    def on_signal(self, ctx: TradingContext, name: str, value: SignalValue) -> None:
        """Called when a subscribed signal updates. Override to react."""
```

For DA backtesting, `on_signal` is called at the start of each auction
period with the latest signal values. The algo can also pull signals
directly via `ctx.get_signal()` from any hook.

### 7. `cli/main.py`

A minimal CLI using `click`:

```bash
# Run a backtest
nexa run examples/simple_da_algo.py \
    --exchange nordpool \
    --start 2026-03-01 \
    --end 2026-03-31 \
    --products NO1_DA \
    --data-dir ./data \
    --capital 100000

# Output: the BacktestResult.summary() text
```

The CLI needs to:
- Accept a path to a Python file containing an algo
- Import the file and find the algo (look for a subclass of `SimpleAlgo`
  or a function decorated with `@algo`)
- Instantiate the `BacktestEngine` with the provided arguments
- Call `.run()` and print `result.summary()`

Use `importlib` to dynamically load the algo module. If the module contains
exactly one `SimpleAlgo` subclass, use it. If it contains multiple, raise
an error asking the user to specify which one.

Register the CLI entry point in `pyproject.toml`:

```toml
[project.scripts]
nexa = "nexa_backtest.cli.main:cli"
```

### 8. Update the example algo

Update `examples/simple_da_algo.py` to use a signal:

```python
class ForecastAlgo(SimpleAlgo):
    """Buy when DA price is below a provided price forecast."""

    def on_setup(self, ctx: TradingContext) -> None:
        self.subscribe_signal("price_forecast")
        self.threshold = 5.0

    def on_auction_open(self, ctx: TradingContext, auction: AuctionInfo) -> None:
        forecast = ctx.get_signal("price_forecast").value
        if forecast is not None:
            ctx.place_order(Order.buy(
                product=auction.product_id,
                volume_mw=10,
                price_eur=Decimal(str(forecast)) - Decimal(str(self.threshold)),
            ))
```

Create a corresponding `examples/data/price_forecast_NO1.csv` with
synthetic forecast values (slightly noisy version of the actual clearing
prices from the test fixture, offset by the publication delay).

The example should be runnable via:
```bash
nexa run examples/simple_da_algo.py \
    --exchange nordpool \
    --start 2026-03-01 \
    --end 2026-03-31 \
    --products NO1_DA \
    --data-dir tests/fixtures \
    --capital 100000
```

---

## How signals are passed to the CLI

For this task, signal CSV files are discovered by convention. The engine
looks in `{data_dir}/signals/` for CSV files matching the signal name:

```
data_dir/
  signals/
    price_forecast.csv
    wind_forecast.csv
```

If the algo subscribes to a signal called "price_forecast", the engine
looks for `{data_dir}/signals/price_forecast.csv`. If the file doesn't
exist, raise a `DataError` with a clear message.

This is deliberately simple. A more flexible signal configuration (YAML
config file, CLI flags per signal, explicit paths) is a later concern.

---

## Tests

1. **signals/base.py**: SignalSchema and SignalValue construction
2. **signals/csv_loader.py**:
   - Load a valid CSV, retrieve values at known timestamps
   - publication_offset prevents future values from being visible
   - Missing file raises DataError
   - Malformed CSV (missing columns, bad timestamps) raises DataError
3. **signals/registry.py**: register, get, has, get missing raises error
4. **backtest.py integration**: algo that uses a signal to make trading
   decisions. Verify that the signal value influences the fills (e.g.,
   algo only buys when forecast is below threshold, verify it doesn't
   buy when forecast is above threshold)
5. **cli/main.py**: test that the CLI loads an algo module, finds the
   SimpleAlgo subclass, and runs without error (use click's CliRunner)
6. **look-ahead bias**: test that a signal with publication_offset does
   NOT return a value before its publication time

---

## What NOT to build

- Built-in signal providers (DayAheadPriceSignal, WindForecastSignal, etc.)
- Signal providers that fetch from APIs
- YAML/JSON signal configuration
- `nexa validate` CLI command
- `nexa compile` CLI command
- `nexa report` CLI command
- Any IDC or windowed replay changes
- HTML report generation

---

## Acceptance criteria

1. `make ci` passes
2. A customer can load a CSV file as a signal and use it in their algo
3. publication_offset correctly prevents look-ahead bias
4. `nexa run` CLI works end-to-end with the example algo
5. The example algo uses a signal to make trading decisions and produces
   a PnL summary
6. All new types have type hints, frozen Pydantic models where appropriate
7. All new public API has Google-style docstrings
