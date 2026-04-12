# Task 08: Multi-Algo Shared Replay and Comparison Reports

## Goal

Run multiple algos against the same historical data in a single pass.
One copy of market data in memory, multiple trading contexts, side-by-side
results. This is the A/B testing feature that lets a trading desk answer
"which strategy is actually better?" without running separate backtests
and manually comparing spreadsheets.

After this task, a customer can run 2-4 algos (or the same algo with
different configurations) against the same data, get a comparison
report with overlaid equity curves and a metrics table, and make an
informed decision about which to put into production.

---

## What to build

### 1. `engines/shared.py` - Shared Data Replay Engine

Orchestrates multiple algos against the same data stream.

```python
class SharedReplayEngine:
    """Run multiple algos against the same historical data.

    Market data is loaded once. Each algo gets its own TradingContext
    with independent positions, orders, and equity tracking. Algos
    advance in lockstep: all process timestamp T before moving to T+1.

    Args:
        algos: Named algos to run. Keys are display names for the
            comparison report, values are algo instances.
        exchange: Target exchange.
        start: Backtest start date.
        end: Backtest end date.
        products: Products to trade.
        signals: Signal providers (shared across all algos).
        models: Model registry (shared across all algos).
        data_dir: Path to historical data.
        initial_capital: Starting capital (same for all algos).
    """

    def __init__(
        self,
        algos: dict[str, SimpleAlgo | AsyncAlgoFunction],
        exchange: str,
        start: date,
        end: date,
        products: list[str],
        signals: list[SignalProvider] | None = None,
        models: ModelRegistry | None = None,
        data_dir: str = "./data",
        initial_capital: Decimal = Decimal("100000"),
    ) -> None: ...

    def run(self) -> ComparisonResult: ...
```

**How the lockstep loop works:**

```text
# Simplified shared replay loop:
#
# 1. Load data (SlidingWindow for IDC, full load for DA)
#    ONE copy, shared across all algo contexts
#
# 2. For each time step (auction period or MTU boundary):
#    a. Advance the shared data window
#    b. For each algo (in registration order):
#       i.   Set the algo's context as active
#       ii.  Dispatch events to this algo
#       iii. Collect orders from this algo
#       iv.  Run matching for this algo's orders
#       v.   Record fills, update positions, snapshot equity
#    c. Move to next time step
#
# 3. Build ComparisonResult from all per-algo results
```

Each algo has its own:

- `TradingContext` (with its own position tracker, order book)
- `AlgoDispatcher` (SimpleAlgo hooks or async event queue)
- Matching engine state (its own resting orders)
- Equity curve

All algos share:

- Market data (the SlidingWindow or loaded DA data)
- Signal providers
- Model registry
- Clock

**Important: algos do not affect each other.** Algo A's orders don't
appear in Algo B's view of the market. Each algo sees the same
historical order book. This is the price-taker assumption applied
per algo.

### 2. `ComparisonResult`

```python
@dataclass
class ComparisonResult:
    """Results from a multi-algo backtest."""
    results: dict[str, BacktestResult]  # Keyed by algo display name
    start_date: date
    end_date: date
    exchange: str
    products: list[str]

    def summary(self) -> str:
        """Side-by-side text summary of all algos."""

    def to_html(self, path: str) -> None:
        """Comparison HTML report."""

    def to_json(self, path: str) -> None:
        """Comparison JSON export."""

    def ranking(self, metric: str = "total_pnl") -> list[str]:
        """Rank algos by a metric. Returns algo names best to worst."""

    @property
    def best(self) -> tuple[str, BacktestResult]:
        """The algo with the highest total PnL."""

    @property
    def worst(self) -> tuple[str, BacktestResult]:
        """The algo with the lowest total PnL."""
```

### 3. Comparison Text Summary

```text
Comparison Results: 2026-03-01 to 2026-03-31 (31 days)
Exchange: Nord Pool | Products: NO1_DA

                    conservative    aggressive      mean_revert
  Total PnL         +8,240.50 EUR   +14,890.20 EUR  +11,340.00 EUR
  vs VWAP           +0.42 EUR/MWh   +0.78 EUR/MWh   +0.61 EUR/MWh
  Sharpe            1.85            1.12            1.54
  Max Drawdown      -1,200.00 EUR   -6,400.00 EUR   -3,100.00 EUR
  Profit Factor     2.40            1.65            1.90
  Win Rate          68.2%           54.1%           61.3%
  Trades            124             312             186

  Best by PnL: aggressive (+14,890.20 EUR)
  Best risk-adjusted: conservative (Sharpe 1.85)
```

### 4. Comparison HTML Report

Extend the report generation from task 03 with comparison-specific
content. The report should be self-contained HTML, same as single-algo
reports.

**Charts:**

1. **Overlaid equity curves** - all algos on the same chart with
   different colours. Legend shows algo names. This is the hero chart
   of the report.
2. **Drawdown comparison** - stacked or overlaid drawdown chart.
3. **Daily PnL comparison** - grouped bar chart (each day has N bars,
   one per algo).
4. **Cumulative VWAP edge** - overlaid line chart.

**Tables:**

1. **Metrics comparison** - one column per algo, one row per metric.
   Highlight the best value in each row.
2. **Per-algo trade summaries** - collapsible sections for each algo
   with their top 10 best/worst trades.

**Styling:**

Use the Phase Nexa palette. Assign each algo a distinct colour from
the accent palette:

- Algo 1: Cyber Cyan (#00E5FF)
- Algo 2: Pulse Violet (#8A6CFF)
- Algo 3: Neon Gold (#FFD700)
- Algo 4: Mint (#4DFFC3)

If more than 4 algos, cycle colours. Background, cards, fonts follow
the same styling guidelines as single-algo reports.

### 5. Update CLI

Add `--compare` mode to `nexa run`:

```bash
# Compare multiple algo files
nexa run --compare \
    conservative:algos/conservative.py \
    aggressive:algos/aggressive.py \
    mean_revert:algos/mean_revert.py \
    --exchange nordpool \
    --start 2026-03-01 \
    --end 2026-03-31 \
    --products NO1_DA \
    --data-dir ./data \
    --capital 100000 \
    --output reports/comparison.html
```

The format is `display_name:algo_path`. If no display name is given
(just a path), use the filename without extension as the display name.

```bash
# Short form (names inferred from filenames)
nexa run --compare \
    algos/conservative.py \
    algos/aggressive.py \
    --exchange nordpool \
    --start 2026-03-01 --end 2026-03-31
```

Limit to 8 algos maximum. More than that makes the comparison report
unreadable and memory usage unreasonable.

Signals and models are shared across all algos. If an algo subscribes
to a signal that another doesn't use, that's fine (the unused algo
simply never calls `ctx.get_signal()` for it).

### 6. Programmatic API

```python
from nexa_backtest import SharedReplayEngine
from algos.conservative import ConservativeAlgo
from algos.aggressive import AggressiveAlgo

comparison = SharedReplayEngine(
    algos={
        "conservative": ConservativeAlgo(),
        "aggressive": AggressiveAlgo(),
    },
    exchange="nordpool",
    start=date(2026, 3, 1),
    end=date(2026, 3, 31),
    products=["NO1_DA"],
    initial_capital=100_000,
).run()

print(comparison.summary())
comparison.to_html("reports/comparison.html")

# Access individual results
for name, result in comparison.results.items():
    print(f"{name}: {result.total_pnl} EUR")
```

### 7. Memory Accounting

Add memory tracking so customers can see the benefit of shared
replay vs running separate backtests:

```python
class ComparisonResult:
    # ...
    peak_memory_bytes: int
    estimated_separate_memory_bytes: int  # What it would have been

    # In summary():
    # Memory: 712 MB (saved ~1,420 MB vs separate backtests)
```

Use `SlidingWindow.memory_usage_bytes` (from task 04) for the data
layer, plus per-algo context overhead estimated from position and
order counts.

---

## Test Fixture Algos

Create three simple algos for testing comparisons:

```text
tests/fixtures/comparison/
    always_buy.py         # Buys every MTU at market price
    threshold_low.py      # Buys when price < forecast - 3
    threshold_high.py     # Buys when price < forecast - 8
```

These are intentionally simple. The point is testing the shared
replay machinery, not the trading logic.

---

## Tests

1. **SharedReplayEngine - basic**: run 2 algos against DA fixture.
   Verify both produce valid BacktestResults with different PnL
   (since they trade differently).

2. **Lockstep ordering**: run 2 algos, verify they both see the
   same market data at the same timestamps. Log the timestamps
   each algo processes and compare.

3. **Isolation**: run 2 algos. Algo A places an order. Verify Algo B's
   `get_orderbook()` does NOT show Algo A's order. Each algo must
   see only the historical book, not each other's orders.

4. **Same algo, same result**: run the same algo instance twice
   (registered under two names). Verify identical PnL, fills,
   equity curves. This is the strongest isolation test.

5. **Mixed API levels**: run a SimpleAlgo and an @algo in the same
   SharedReplayEngine. Verify both complete and produce valid results.

6. **ComparisonResult.ranking()**: verify ranking by total_pnl, by
   sharpe_ratio, by max_drawdown. Verify best/worst properties.

7. **Comparison summary**: verify the text summary includes all
   algos and highlights the best performer.

8. **Comparison HTML report**: verify the HTML contains chart
   containers for overlaid equity curves and the metrics comparison
   table. Parse HTML, check structure.

9. **CLI --compare**: use CliRunner to test with 2 algo files.
   Verify output is generated. Test with display names and without
   (inferred from filename).

10. **CLI --compare limit**: attempt with 9 algos, verify error
    message about 8 maximum.

11. **Memory tracking**: run SharedReplayEngine with 2 algos.
    Verify peak_memory_bytes is reported and
    estimated_separate_memory_bytes is roughly 2x the data layer.

12. **IDC shared replay**: run 2 algos against IDC fixture data.
    Verify the SlidingWindow is loaded once (check memory_usage_bytes
    is not doubled).

---

## What NOT to build

- Parameter sweeps (same algo, different params via config). This
  needs a parameter config format which is a separate design problem.
  Defer to a later task.
- Statistical significance testing between algo results (paired
  t-test, bootstrap confidence intervals). Useful but not essential.
- Live comparison (running multiple algos in paper/live mode
  simultaneously). Shared replay is backtest-only; paper/live algos
  run independently.
- Algo tournament / bracket format
- Automated algo selection ("pick the best one for me")

---

## Acceptance criteria

1. `make ci` passes
2. `SharedReplayEngine` runs multiple algos against the same data
   with one copy in memory
3. Algos are fully isolated (no order/position leakage between them)
4. The comparison HTML report shows overlaid equity curves and a
   metrics comparison table
5. `nexa run --compare` works from the CLI with 2-8 algo files
6. ComparisonResult provides ranking, best, worst, per-algo access
7. Memory tracking shows the saving vs separate backtests
8. All new types have type hints and frozen Pydantic models where
   appropriate
9. All new public API has Google-style docstrings
