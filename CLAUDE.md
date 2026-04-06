# nexa-backtest

## What this is

A backtesting framework purpose-built for European power markets. It replays
historical market conditions against a customer's trading algorithm and
produces PnL analysis with VWAP benchmarking. Not adapted from equities.
Handles DA auctions, intraday auctions, and intraday continuous markets
with 15-minute MTU resolution natively.

Part of the Phase Nexa ecosystem.

## Audience

Quants, data scientists, and developers at energy trading companies who build
their own trading systems. Two user types:

- Beginners and trading desks experimenting: use `SimpleAlgo` with hooks
- Quants with complex strategies: use the `@algo` decorator with async event streams

Both types use the same `TradingContext` protocol underneath.

## The critical design rule

**The algo must never know whether it is backtesting, paper trading, or live
trading.** Same code, same interface, different engine underneath. The
`TradingContext` protocol is the boundary. Algo code imports nothing
mode-specific. If any change requires the algo to know its execution mode,
the design is broken.

## Code style

- Python 3.11+
- Type hints everywhere, strict mypy compliance
- Pydantic v2 for data models
- pytest for testing
- Ruff for linting and formatting
- No classes where a function will do
- Docstrings on all public API (Google style)
- Tabular numerical data uses pandas DataFrames
- Timezone-aware datetimes only (never naive)
- `Decimal` for all prices and monetary values
- Utilise the Makefile for common developer actions
- Always prefer UK English unless using existing nomenclature popular in energy trading

## Domain context

- MTU = Market Time Unit. EU power markets transitioned to 15-minute MTUs on
  30 Sept 2025. The backtester handles both 15-min and hourly resolution.
- DA = Day-Ahead auction. Bids submitted before gate closure, matched against
  a single clearing price. Use the price-taker assumption: the algo's volume
  did not move the market. Filled if bid >= clearing price (buy) or
  bid <= clearing price (sell).
- IDC = Intraday Continuous. Orders matched against a historical order book
  using price-time priority. More complex than DA, much larger data volumes.
- VWAP = Volume-Weighted Average Price. The primary benchmark. If the algo
  can't beat VWAP, a simple time-weighted execution would do better.
- Gate closure = deadline for submitting/modifying orders for a delivery period.
  Different per exchange and product type. The algo receives a
  `GateClosureWarning` event before this happens.

## Data loading strategy

DA and IDC data have wildly different volumes. Handle them differently:

- **DA data** (~1.7 MB/zone/year): load entirely into memory. No windowing.
- **IDC data** (~10 GB/zone/year): windowed replay via PyArrow row groups.
  Active window covers current hour + configurable lookback (default 4 hours).
  Old chunks evicted as clock advances. Peak memory stays at 200-500 MB
  regardless of replay period length.
- **Signal data** (<1 GB): load entirely. Signal data >1 GB: windowed.

Window transitions happen between MTU boundaries, never mid-event-processing.

When running multiple algos against the same data, use the shared data replay
mode (single data window, multiple algo contexts). Algos advance in lockstep.

## Exchange adapters

Each exchange adapter implements the `ExchangeAdapter` protocol and declares
its `ExchangeCapabilities`. The matching engine behaviour differs per adapter:

- Nord Pool DA: price-taker against historical clearing price
- Nord Pool IDC: price-time priority against historical order book
- EPEX SPOT: same principles, different gate closure rules and product naming
- EEX: same principles, different product structure

When an algo uses a feature not supported by the target exchange (e.g., block
bids on an exchange that doesn't support them), this must fail at validation
time, not at runtime. The `nexa validate` CLI catches these before execution.

## Signal system

Signals are any time-series data the algo consumes: weather forecasts, DA
prices, load forecasts, gas prices, etc. Each signal has a `publication_offset`
that prevents look-ahead bias. At simulated time T, the signal returns the
value that was known at T, not the future value.

Custom signals implement the `SignalProvider` protocol.

## ML model support

Models are registered via `ModelRegistry` and accessed via `ctx.predict()`.
ONNX is the recommended format (portable, fast, no arbitrary code execution).
Scikit-learn pickle/joblib is supported but flagged as a security risk in
hosted environments.

## Validation pipeline

The `nexa validate` CLI runs before execution:

1. Ruff: syntax and style
2. Mypy --strict: type safety and protocol compliance
3. Interface compliance: AST check for required hooks/decorator
4. Exchange feature compatibility: cross-reference against ExchangeCapabilities
5. Look-ahead bias detection: heuristic check for future data access patterns
6. Resource safety: flags time.sleep(), file I/O, network calls outside ctx

All six steps must pass before the algo can run. The CLI returns proper exit
codes for CI integration.

## Code layout

```
src/nexa_backtest/
    __init__.py
    _version.py
    algo.py              # SimpleAlgo base class, @algo decorator
    context.py           # TradingContext protocol definition
    types.py             # Order, Fill, Position, MTU, PriceLevel, etc.
    exceptions.py        # NexaBacktestError hierarchy

    engines/
        backtest.py      # BacktestEngine (simulated clock, historical replay)
        paper.py         # PaperEngine (real clock, live data, simulated fills)
        live.py          # LiveEngine (wraps nexa-connect for real execution)
        clock.py         # SimulatedClock, RealtimeClock
        matching.py      # Simulated matching engines (DA auction, IDC continuous)
        shared.py        # SharedDataReplay for multi-algo runs

    exchanges/
        base.py          # ExchangeAdapter protocol
        capabilities.py  # ExchangeCapabilities dataclass
        nordpool.py      # Nord Pool adapter (DA + IDC)
        epex_spot.py     # EPEX SPOT adapter
        eex.py           # EEX adapter

    signals/
        base.py          # SignalProvider protocol, SignalSchema
        registry.py      # Signal registration and lookup
        builtins.py      # DA price, wind, solar, load, imbalance, gas, carbon

    models/
        registry.py      # ModelRegistry
        onnx.py          # ONNX model loader (onnxruntime)
        sklearn.py       # Scikit-learn model loader (pickle/joblib)

    analysis/
        pnl.py           # PnL calculation engine
        vwap.py          # VWAP benchmark comparison
        metrics.py       # Sharpe, drawdown, win rate, profit factor
        report.py        # HTML/JSON/Parquet report generation

    data/
        loader.py        # ParquetLoader, NexaMarketdataLoader
        schema.py        # Standard Parquet schemas for DA, IDC, signals
        window.py        # SlidingWindow, WindowManager for IDC replay
        cache.py         # LRU NVMe/SSD cache for hosted environments
        manifest.py      # Data file index (timestamp ranges per row group)

    validation/
        runner.py        # Orchestrates all validation steps
        ruff_check.py    # Ruff integration
        mypy_check.py    # Mypy integration
        interface_check.py   # AST-based interface compliance
        feature_check.py     # Exchange feature compatibility
        lookahead_check.py   # Look-ahead bias detection
        resource_check.py    # Resource safety checks

    compile/
        cython_compiler.py   # Cython compilation for IP protection
        nuitka_compiler.py   # Nuitka compilation for IP protection

    cli/
        main.py          # CLI entry point (click or typer)
        validate.py      # nexa validate
        run.py           # nexa run
        compile.py       # nexa compile
        report.py        # nexa report
```

## Implementation sequence

### Stage 1: DA Auction Backtester (minimum viable)

The smallest thing that produces useful results. Scope:

- `types.py` with core domain types (Order, Fill, Position, MTU, PriceLevel)
- `exceptions.py` with error hierarchy
- `context.py` with TradingContext protocol
- `algo.py` with SimpleAlgo base class (on_setup, on_auction_open,
  on_fill, on_gate_closure hooks only)
- `engines/clock.py` with SimulatedClock
- `engines/backtest.py` with BacktestEngine (DA only, full data load)
- `engines/matching.py` with DA auction matching (price-taker)
- `exchanges/base.py` with ExchangeAdapter protocol
- `exchanges/capabilities.py` with ExchangeCapabilities
- `exchanges/nordpool.py` with Nord Pool DA adapter
- `data/loader.py` with ParquetLoader (DA clearing prices only)
- `data/schema.py` with DA clearing price schema
- `analysis/pnl.py` with basic PnL calculation
- `analysis/vwap.py` with VWAP benchmark comparison
- `analysis/metrics.py` with Sharpe, drawdown, win rate
- `cli/main.py` with `nexa run` command (text output)
- One signal: CSV-based custom signal loader (so users can bring
  their own forecast data without implementing SignalProvider)

Does NOT include: IDC, windowed replay, validation pipeline, ML models,
code compilation, HTML reports, paper/live engines, multi-algo replay.

### Stage 2: IDC Continuous + Windowed Replay

- IDC continuous matching engine (price-time priority)
- Windowed data loading (PyArrow row groups, SlidingWindow, manifest)
- `@algo` decorator with async event stream
- Full signal system with publication_offset
- EPEX SPOT and EEX exchange adapters
- HTML report generation

### Stage 3: Intelligence + Quality

- ML model registry (ONNX + sklearn)
- Full validation pipeline (all 6 steps)
- Multi-algo shared data replay
- ExchangeCapabilities with pre-run feature checking

### Stage 4: Production + Hosted

- Paper and live trading engines
- Code protection (Cython, Nuitka)
- Hosted service infrastructure

## Historical data formats

All backtest data is stored as Parquet. Standard schemas:

**DA clearing prices** (loaded entirely, ~1.7 MB/zone/year):
Columns: timestamp (datetime64[ns, UTC]), zone (str), price_eur_mwh (float64),
volume_mwh (float64)

**IDC events** (windowed, ~1.5 GB/zone/year):
Columns: timestamp (datetime64[ns, UTC]), event_type (str: new/modify/cancel/trade),
order_id (str), zone (str), product_id (str), side (str: buy/sell),
price_eur_mwh (float64), volume_mw (float64), remaining_mw (float64)

**IDC trades** (windowed, ~200 MB/zone/year):
Columns: timestamp (datetime64[ns, UTC]), trade_id (str), zone (str),
product_id (str), price_eur_mwh (float64), volume_mw (float64),
aggressor_side (str: buy/sell, optional - may not be available from all
exchange data exports, degrade gracefully when missing)

**Signals** (loaded entirely unless >1 GB):
Columns: published_at (datetime64[ns, UTC]), valid_from (datetime64[ns, UTC]),
valid_to (datetime64[ns, UTC]), zone (str), value (float64), provider (str)

Parquet files are partitioned by zone and month for IDC data, zone and year
for DA data. Row groups sized at ~64 MB after compression.

## Dependencies

Core (required):
- pydantic >= 2.0
- pyarrow (Parquet I/O and windowed replay)
- numpy
- click or typer (CLI)

Optional extras:
- pandas (DataFrame output, installed via `pip install nexa-backtest[pandas]`)
- onnxruntime (ML model inference, `pip install nexa-backtest[ml]`)
- matplotlib or plotly (report charts, `pip install nexa-backtest[charts]`)
- nexa-marketdata (data fetching, `pip install nexa-backtest[marketdata]`)
- nexa-connect (live trading engine, `pip install nexa-backtest[live]`)

## Common pitfalls

- **Look-ahead bias is the #1 backtesting mistake.** Signals must respect
  publication_offset. At time T, the algo can only see data published before T.
- **DA matching is price-taker only.** The algo's bid does not affect the
  clearing price. This is realistic for most participants but not for very
  large portfolios (market impact modelling is a v2 concern).
- **IDC order book data is huge.** Never load a full year into memory. Always
  use windowed replay via the SlidingWindow/WindowManager.
- **Window transitions happen between MTU boundaries.** Never evict or load
  data mid-event-processing.
- **aggressor_side may not be available.** Some exchange data exports (e.g.,
  EPEX) don't include it explicitly. The matching engine must degrade
  gracefully when this field is missing.
- **product_id identifies the delivery period, not the trading session.**
  NO1-QH-0900 is the 09:00-09:15 delivery product. Orders for this product
  might be placed hours before delivery.
- **Gate closure times differ per exchange and product type.** Always use
  ctx.time_to_gate_closure() rather than hardcoding offsets.
- **time.sleep() pauses the real clock, not the simulated one.** The
  validation pipeline catches this. Use ctx.wait() for simulated delays.

## Makefile targets

```
make install          # Install dev dependencies
make test             # Run pytest
make lint             # Run ruff check + ruff format --check
make typecheck        # Run mypy --strict
make ci               # lint + typecheck + test (used by CI workflow)
make test-notebooks   # Validate notebook syntax
make execute-notebooks # Run notebooks end-to-end
```

## Definition of done

A feature is complete when:

1. All new code has type hints and passes mypy --strict
2. All public API has Google-style docstrings
3. Tests cover the happy path and at least one error case
4. `make ci` passes
5. No regressions in existing tests
6. Test coverage should be >80%
7. If the feature adds a new algo hook or TradingContext method, the
   protocol in context.py is updated and all three engine implementations
   (backtest, paper, live) are updated or stubbed
8. Review the README file, check nothing major is missing, add additions if something is identified
