# Task 03: Equity Curve, Advanced Metrics, and HTML Report

## Goal

Upgrade the analysis layer so customers get a proper picture of their
algo's performance over time, not just a final PnL number. Add an equity
curve that tracks capital through each MTU, compute Sharpe ratio and max
drawdown from it, and produce an HTML report with charts that can be
opened in a browser or shared with a team.

This builds on tasks 01 and 02. The DA engine, signals, and CLI remain
unchanged. We're improving what happens after `engine.run()` returns.

---

## What to build

### 1. Equity curve tracking in `engines/backtest.py`

The `BacktestEngine` currently produces a flat list of fills and a final
PnL. It needs to track capital over time so we can compute time-series
metrics.

After each auction period is processed (fills applied, positions updated),
record a snapshot:

```python
@dataclass(frozen=True)
class EquitySnapshot:
    """Capital state at a point in time."""
    timestamp: datetime
    realised_pnl: Decimal       # Cumulative realised PnL to this point
    unrealised_pnl: Decimal     # Mark-to-market of open positions
    total_equity: Decimal       # initial_capital + realised + unrealised
    cash: Decimal               # Cash after settled trades
    net_position_mw: Decimal    # Total net MW across all products
```

The engine builds a list of `EquitySnapshot` entries, one per MTU that
had trading activity. These are stored on `BacktestResult`.

For DA-only backtesting, "unrealised PnL" is always zero because DA
auctions settle immediately at the clearing price (there's no holding
period). The field exists for IDC compatibility in later stages where
positions are held across multiple MTUs.

### 2. Update `BacktestResult`

Extend the result object:

```python
class BacktestResult:
    # Existing from task 01
    trades: list[Fill]
    positions: dict[str, Position]
    total_pnl: Decimal
    vwap_edge: Decimal
    vwap_edge_pct: Decimal
    total_volume_mw: Decimal
    win_rate: Decimal
    trade_count: int

    # New in task 03
    equity_curve: list[EquitySnapshot]
    initial_capital: Decimal
    sharpe_ratio: Decimal | None     # None if < 2 data points
    max_drawdown: Decimal            # Absolute EUR
    max_drawdown_pct: Decimal        # As percentage of peak equity
    profit_factor: Decimal | None    # None if no losing trades
    avg_trade_pnl: Decimal
    best_trade: Fill | None
    worst_trade: Fill | None
    start_date: date
    end_date: date
    duration_days: int

    def summary(self) -> str: ...
    def equity_curve_df(self) -> pd.DataFrame: ...
    def trades_df(self) -> pd.DataFrame: ...
    def daily_pnl_df(self) -> pd.DataFrame: ...
    def to_html(self, path: str) -> None: ...
    def to_json(self, path: str) -> None: ...
    def to_parquet(self, path: str) -> None: ...
```

The `equity_curve_df()`, `trades_df()`, and `daily_pnl_df()` methods
return pandas DataFrames. These require pandas as a dependency. If
pandas is not installed, raise an `ImportError` with a message suggesting
`pip install nexa-backtest[pandas]`.

### 3. Advanced metrics in `analysis/metrics.py`

Add the following calculations. All operate on the equity curve and/or
trade list, not on raw market data.

**Sharpe ratio:**

- Calculate returns from the equity curve (percentage change between
  consecutive snapshots)
- Sharpe = mean(returns) / std(returns) * sqrt(annualisation_factor)
- Annualisation factor: for 15-minute MTUs, there are 96 per day and
  roughly 35,040 per year. But trading doesn't happen on all MTUs, so
  use the actual number of return observations and scale to annualised.
- Use `sqrt(252 * periods_per_day)` where periods_per_day is derived
  from the actual average frequency of equity snapshots.
- Return `None` if fewer than 2 equity snapshots (can't compute std).

**Max drawdown:**

- Track the running peak of total_equity.
- Drawdown at each point = peak - current equity.
- Max drawdown = largest drawdown observed.
- Max drawdown percentage = max drawdown / peak equity at that point.

**Profit factor:**

- Sum of all profitable trades / abs(sum of all losing trades).
- Return `None` if there are no losing trades (infinite profit factor).

**Average trade PnL:**

- Total PnL / number of trades.

**Best and worst trade:**

- The fill with the highest and lowest individual PnL respectively.
- Individual trade PnL: for a buy fill, PnL is determined when the
  position is closed. For DA-only (where we buy at auction and the
  "value" is the clearing price), each trade's PnL is simply:
  (clearing_price - bid_price) * volume for buys.
- Keep this simple for now. Full realised PnL attribution across
  multiple fills comes with IDC in a later stage.

### 4. Daily PnL aggregation

Add a helper that aggregates the equity curve into daily buckets:

```python
def daily_pnl(self) -> list[DailyPnL]:
    """Aggregate PnL by calendar day."""

@dataclass(frozen=True)
class DailyPnL:
    date: date
    pnl: Decimal
    volume_mw: Decimal
    trade_count: int
    vwap_edge: Decimal
```

This is used by both the text summary and the HTML report.

### 5. Update `summary()` output

The text summary should now include the new metrics:

```text
Backtest Results: 2026-03-01 to 2026-03-31 (31 days)
Exchange: Nord Pool | Products: NO1_DA

  Total PnL:        +12,340.50 EUR
  vs VWAP:           +0.65 EUR/MWh (+3.2%)
  Sharpe Ratio:      1.42
  Max Drawdown:     -4,200.00 EUR (-3.8%)
  Profit Factor:     1.85

  Trades:            186
  Win Rate:          62.4%
  Avg Trade PnL:    +66.35 EUR
  Best Trade:       +840.00 EUR (NO1_DA 2026-03-14 08:00)
  Worst Trade:      -320.00 EUR (NO1_DA 2026-03-07 17:45)

  Total Volume:      3,240 MW
  Initial Capital:   100,000.00 EUR
  Final Equity:      112,340.50 EUR
```

### 6. HTML report generation in `analysis/report.py`

Generate a self-contained HTML file with embedded charts. The report
must work when opened directly in a browser (no external dependencies,
no server needed). Use inline CSS and inline JavaScript.

**Charts (use Plotly.js via CDN, or inline a lightweight charting lib):**

1. **Equity curve** - Line chart showing total_equity over time. Include
   a horizontal line at initial_capital for reference.
2. **Drawdown chart** - Area chart showing drawdown below the equity
   curve. Negative values, filled red.
3. **Daily PnL bar chart** - Green bars for positive days, red bars for
   negative days.
4. **VWAP edge over time** - Line chart showing cumulative VWAP edge.
5. **Trade distribution by hour** - Bar chart showing trade count or
   volume by hour of day (helps identify when the algo is most active).

**Tables:**

1. **Summary metrics** - The same numbers as the text summary, in a
   formatted table.
2. **Top 10 best trades** - Date, product, side, price, volume, PnL.
3. **Top 10 worst trades** - Same columns.
4. **Daily breakdown** - Date, PnL, volume, trade count, VWAP edge.

**Styling:**

Use the Phase Nexa colour palette from the styling guidelines:

- Background: Quantum Blue (#0E2A47)
- Cards/panels: Nebula (#122F4D)
- Positive values / profit: Mint (#4DFFC3)
- Negative values / loss: Electric Coral (#FF6B6B)
- Accent / highlights: Cyber Cyan (#00E5FF)
- Primary action / headers: Pulse Violet (#8A6CFF)
- High-value numbers: Neon Gold (#FFD700)
- Body text: white/light grey
- Data font: monospace (JetBrains Mono from Google Fonts CDN, with
  Courier New fallback)
- Interface font: Inter from Google Fonts CDN, with sans-serif fallback

The report should feel professional. Dense but readable. Follow the
styling guidelines: "the interface recedes, the data advances."

```python
# Usage
result = engine.run()
result.to_html("reports/march_2026.html")
```

### 7. JSON and Parquet export

**JSON export** (`to_json`):
Serialize the BacktestResult to JSON. Equity curve as an array of
objects. Trades as an array. Metrics at the top level. Decimal values
serialized as strings to preserve precision.

**Parquet export** (`to_parquet`):
Write the equity curve and trade list as Parquet files in a directory:

```text
output_dir/
  metadata.json       # Metrics, config, date range
  equity_curve.parquet
  trades.parquet
  daily_pnl.parquet
```

### 8. Update CLI

Add `--output` flag to `nexa run`:

```bash
# Print summary to stdout (default, no change)
nexa run my_algo.py --exchange nordpool --start 2026-03-01 --end 2026-03-31

# Generate HTML report
nexa run my_algo.py --exchange nordpool --start 2026-03-01 --end 2026-03-31 \
    --output reports/march.html

# Generate JSON
nexa run my_algo.py --exchange nordpool --start 2026-03-01 --end 2026-03-31 \
    --output results/march.json

# Output format inferred from file extension (.html, .json, .parquet)
```

The summary is always printed to stdout regardless of `--output`.

---

## Tests

1. **Equity curve tracking**: run a backtest with known fixture data
   and a deterministic algo. Verify the equity curve has the expected
   number of snapshots, and the final total_equity matches
   initial_capital + total_pnl.

2. **Sharpe ratio**: construct an equity curve with known returns and
   verify the Sharpe calculation matches a hand-calculated expected
   value. Test edge case: fewer than 2 snapshots returns None.

3. **Max drawdown**: construct an equity curve that goes up, then
   down, then up again. Verify max drawdown and max drawdown percentage
   match expected values. Test edge case: monotonically increasing
   equity has zero drawdown.

4. **Profit factor**: test with known winning and losing trades. Test
   edge case: no losing trades returns None.

5. **Daily PnL aggregation**: verify trades spanning midnight are
   assigned to the correct day.

6. **HTML report**: generate a report and verify it's valid HTML (parse
   with html.parser, no exceptions). Verify it contains the expected
   chart containers and metric values. Don't test visual rendering.

7. **JSON export**: round-trip test. Export to JSON, read it back,
   verify all metrics match. Verify Decimal precision is preserved.

8. **Parquet export**: verify the directory structure and that each
   Parquet file has the expected schema.

9. **CLI --output**: test with CliRunner that each output format
   is created.

---

## What NOT to build

- Interactive charts (the HTML is static, no server)
- PDF report generation
- Comparison reports (algo A vs algo B)
- Parameter sensitivity / sweep analysis
- Any IDC changes
- Any changes to the signal system
- Any changes to the matching engine
- Built-in signal providers

---

## Acceptance criteria

1. `make ci` passes
2. `BacktestResult` includes equity curve, Sharpe, drawdown, and
   profit factor
3. `result.summary()` prints all new metrics
4. `result.to_html()` produces a self-contained HTML report that
   opens in a browser with charts and tables using the Phase Nexa
   colour palette
5. `result.to_json()` and `result.to_parquet()` produce correct
   output
6. `nexa run --output report.html` generates the HTML report
7. All new types have type hints and frozen Pydantic models where
   appropriate
8. All new public API has Google-style docstrings
9. The notebooks/backtester_walkthrough.ipynb should still run and make sense, don't add any of the new stuff that isn't crucial to the basics tutorial (the purpose of this notebook)
