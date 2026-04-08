# Task 04: IDC Continuous Matching and Windowed Replay

## Goal

Add intraday continuous (IDC) backtesting. This is where the real
complexity lives: orders match against a historical order book using
price-time priority, data volumes are ~1000x larger than DA, and the
engine needs windowed replay to stay within memory limits.

After this task, a customer can backtest IDC strategies against
historical order book data for a single exchange (Nord Pool). They
get the same PnL, VWAP, and HTML report output from task 03.

This is the hardest task so far. Take it slow.

---

## Context: How IDC Differs from DA

In DA auctions, all bids are submitted before gate closure, and
everyone gets the same clearing price. Simple.

In IDC continuous trading, orders are submitted throughout the day
and matched immediately when a counterparty exists. It works like a
stock exchange order book:

1. Your algo places a buy order at 52.40 EUR/MWh for 5 MW
2. If someone is already selling at 52.40 or below, you get filled
   immediately at their price (price-time priority)
3. If not, your order rests in the book until someone matches or you
   cancel it
4. Multiple products trade simultaneously (each 15-min delivery
   period is a separate product, e.g., NO1-QH-0900, NO1-QH-0915)
5. Gate closure happens per product (you can't trade a product
   after its gate closes)

The backtest replays this using historical order book event data.

---

## What to build

### 1. `data/window.py` - Sliding Window Manager

The core data loading mechanism for IDC. Reads Parquet row groups on
demand as the simulated clock advances.

```python
class DataManifest:
    """Index of all Parquet files and their row group timestamp ranges.
    Built once at startup, kept in memory (~KB). Knows where every
    chunk of data lives on disk without loading any actual data."""

    def __init__(self, data_dir: str, zone: str) -> None: ...
    def row_groups_for_range(
        self, start: datetime, end: datetime
    ) -> list[RowGroupRef]: ...

@dataclass(frozen=True)
class RowGroupRef:
    """Pointer to a specific row group in a specific Parquet file."""
    file_path: str
    row_group_index: int
    min_timestamp: datetime
    max_timestamp: datetime
    num_rows: int

class SlidingWindow:
    """Maintains an in-memory window of IDC data. Loads row groups
    from Parquet files via PyArrow as the clock advances."""

    def __init__(
        self,
        manifest: DataManifest,
        lookback: timedelta = timedelta(hours=4),
        lookahead: timedelta = timedelta(hours=1),
    ) -> None: ...

    def advance_to(self, timestamp: datetime) -> None:
        """Move the window forward. Load new row groups that fall
        within [timestamp - lookback, timestamp + lookahead].
        Evict row groups that fall entirely before timestamp - lookback.
        This is called between MTU boundaries, never mid-processing."""

    def events_between(
        self, start: datetime, end: datetime
    ) -> Iterator[MarketEvent]:
        """Yield market events within the given time range, sorted
        by timestamp. Used by the backtest engine to feed events
        to the matching engine."""

    @property
    def memory_usage_bytes(self) -> int:
        """Current memory usage of loaded data. For monitoring."""
```

Key implementation details:

- Use `pyarrow.parquet.ParquetFile` to read individual row groups
  without loading the entire file.
- Row group metadata (min/max timestamps) is read from Parquet
  footer metadata, not by scanning the data.
- Loaded row groups are held as PyArrow Tables (zero-copy to numpy
  for event iteration).
- Eviction is by row group: when all events in a row group are
  older than `timestamp - lookback`, drop the reference and let
  GC reclaim the memory.

### 2. `data/schema.py` - IDC Schemas

Add Parquet schemas for IDC data alongside the existing DA schema:

```python
IDC_EVENTS_SCHEMA = pa.schema([
    ("timestamp", pa.timestamp("ns", tz="UTC")),
    ("event_type", pa.string()),      # "new", "modify", "cancel", "trade"
    ("order_id", pa.string()),
    ("zone", pa.string()),
    ("product_id", pa.string()),      # e.g., "NO1-QH-0900"
    ("side", pa.string()),            # "buy" or "sell"
    ("price_eur_mwh", pa.float64()),
    ("volume_mw", pa.float64()),
    ("remaining_mw", pa.float64()),
])

IDC_TRADES_SCHEMA = pa.schema([
    ("timestamp", pa.timestamp("ns", tz="UTC")),
    ("trade_id", pa.string()),
    ("zone", pa.string()),
    ("product_id", pa.string()),
    ("price_eur_mwh", pa.float64()),
    ("volume_mw", pa.float64()),
    ("aggressor_side", pa.string()),  # nullable
])
```

### 3. `engines/matching.py` - IDC Continuous Matching Engine

Add `ContinuousMatchingEngine` alongside the existing
`AuctionMatchingEngine`. This is the most complex piece.

**How the historical replay matching works:**

The algo's orders are matched against the **historical order book
activity**, not a reconstructed live book. The approach:

1. The engine replays historical events (new orders, modifications,
   cancellations, trades) in timestamp order
2. When the algo places an order, the engine checks if any
   historical order on the opposite side would have matched
3. "Would have matched" means: the algo's buy price >= the
   historical ask price (or algo's sell price <= historical bid
   price), and the historical order was still active at that
   timestamp
4. If matched, generate a fill at the historical order's price
   (the algo is the aggressor, so it crosses to the resting
   order's price)
5. If not matched, the algo's order rests until either a matching
   historical event arrives or the algo cancels it

```python
class ContinuousMatchingEngine:
    """Simulates IDC continuous matching against historical data."""

    def __init__(self) -> None:
        # Active historical orders (rebuilt from events)
        self._historical_book: dict[str, OrderBook] = {}
        # Algo's resting orders
        self._algo_orders: dict[str, Order] = {}

    def process_historical_event(self, event: MarketEvent) -> list[Fill]:
        """Process a historical market event. If the event creates
        a new resting order that matches one of the algo's orders,
        generate a fill."""

    def place_algo_order(self, order: Order) -> list[Fill]:
        """Place an algo order. Check if it matches any historical
        resting orders. If yes, fill immediately. If no, add to
        algo's resting orders."""

    def cancel_algo_order(self, order_id: str) -> CancelResult: ...

    def check_gate_closures(self, current_time: datetime) -> list[str]:
        """Return product_ids whose gate has closed. Cancel any
        algo orders for those products."""
```

**Important assumptions and limitations:**

- **Price-taker assumption still applies.** The algo's orders do not
  affect the historical order flow. If a historical trade happened at
  53.10 and the algo had a resting buy at 53.10, we assume both fills
  happened (the historical trade still occurs, the algo also gets
  filled). In reality, one of them would have consumed the liquidity.
  For small algo volumes relative to market volume, this is acceptable.
  Document this limitation clearly.

- **aggressor_side handling.** When aggressor_side is available in the
  trade data, use it to determine whether the algo's resting order
  would have been hit. When it's not available (e.g., EPEX data),
  fall back to: if a historical trade occurred at a price that matches
  the algo's resting order, assume the algo would also have been
  filled at that price. This is less accurate but better than nothing.

- **No partial fill simulation from historical data.** If a historical
  order was for 10 MW and the algo wants 3 MW, the algo gets 3 MW
  filled and the remaining 7 MW of the historical order is still
  available. We don't track whether other historical participants
  consumed that remaining volume (we can't know that without full
  L3 book reconstruction, which is out of scope).

### 4. Event types

Add new event types for the IDC event stream:

```python
@dataclass(frozen=True)
class MarketEvent:
    """Base for all market events."""
    timestamp: datetime
    product_id: str

@dataclass(frozen=True)
class MarketDataUpdate(MarketEvent):
    """A change to the order book."""
    event_type: str  # "new", "modify", "cancel"
    order_id: str
    side: str
    price_eur_mwh: Decimal
    volume_mw: Decimal
    remaining_mw: Decimal

@dataclass(frozen=True)
class HistoricalTrade(MarketEvent):
    """A trade that occurred in the historical data."""
    trade_id: str
    price_eur_mwh: Decimal
    volume_mw: Decimal
    aggressor_side: str | None

@dataclass(frozen=True)
class GateClosureWarning(MarketEvent):
    """Warning that gate closure is approaching."""
    remaining: timedelta

@dataclass(frozen=True)
class GateClosureEvent(MarketEvent):
    """Gate has closed for this product."""
    pass
```

### 5. Update `engines/backtest.py`

The `BacktestEngine` needs to handle both DA and IDC modes. The mode
is determined by the products requested:

- Products ending in `_DA` use the auction matching engine (existing)
- Products matching the pattern `{zone}-QH-{time}` or similar IDC
  product patterns use the continuous matching engine (new)
- Mixed DA + IDC products in a single backtest are supported (the
  engine runs both matching engines in parallel, advancing the
  shared clock)

For IDC mode:

```python
# Simplified IDC backtest loop (per MTU boundary):
#
# 1. Advance the SlidingWindow to the current time
# 2. Get all historical events for this MTU period
# 3. For each event (in timestamp order):
#    a. Feed to ContinuousMatchingEngine.process_historical_event()
#    b. If the event triggers a fill on an algo order, call
#       algo.on_fill()
#    c. Feed to the algo as a MarketDataUpdate event (if using
#       the low-level @algo API) or trigger on_bar (if using
#       SimpleAlgo)
# 4. Check gate closures, cancel expired algo orders
# 5. Record equity snapshot
```

### 6. Update `algo.py` - SimpleAlgo IDC Hooks

Add hooks relevant to IDC:

```python
class SimpleAlgo:
    # Existing hooks (unchanged)
    def on_setup(self, ctx: TradingContext) -> None: ...
    def on_auction_open(self, ctx: TradingContext, auction: AuctionInfo) -> None: ...
    def on_fill(self, ctx: TradingContext, fill: Fill) -> None: ...
    def on_gate_closure(self, ctx: TradingContext, product_id: str) -> None: ...
    def on_teardown(self, ctx: TradingContext) -> None: ...
    def on_signal(self, ctx: TradingContext, name: str, value: SignalValue) -> None: ...

    # New hooks for IDC
    def on_bar(self, ctx: TradingContext) -> None:
        """Called at each MTU boundary (every 15 minutes).
        Use for periodic strategy evaluation."""

    def on_cancel(self, ctx: TradingContext, order_id: str, reason: str) -> None:
        """Called when an order is cancelled (by the algo or by
        gate closure)."""

    def on_error(self, ctx: TradingContext, error: Exception) -> None:
        """Called when an error occurs during processing."""
```

### 7. Update `context.py` - Order Book Access

The TradingContext protocol has stubs for order book methods from
task 01. Implement them in the backtest context:

```python
class TradingContext(Protocol):
    # These were stubs, now implemented:
    def get_orderbook(self, product_id: str) -> OrderBook: ...
    def get_best_bid(self, product_id: str) -> PriceLevel | None: ...
    def get_best_ask(self, product_id: str) -> PriceLevel | None: ...
```

The order book the algo sees is the **historical** order book state
at the current simulated time, not a book that includes the algo's
own orders. The algo's resting orders are tracked separately in the
matching engine.

Add the `OrderBook` type to `types.py`:

```python
@dataclass(frozen=True)
class OrderBook:
    product_id: str
    timestamp: datetime
    bids: list[PriceLevel]   # Sorted best (highest) first
    asks: list[PriceLevel]   # Sorted best (lowest) first

    @property
    def best_bid(self) -> PriceLevel | None: ...

    @property
    def best_ask(self) -> PriceLevel | None: ...

    @property
    def spread(self) -> Decimal | None:
        """Best ask - best bid. None if either side is empty."""

    @property
    def mid_price(self) -> Decimal | None:
        """(best_bid + best_ask) / 2. None if either side is empty."""
```

### 8. Update `exchanges/nordpool.py`

Add Nord Pool IDC configuration to the existing adapter:

- IDC product naming convention (e.g., `NO1-QH-0900`)
- Gate closure rules for IDC products (typically 30 minutes before
  delivery start for Nordics, but varies)
- Price and volume limits for IDC
- Set `supports_continuous_trading = True` in capabilities

### 9. Data partitioning and fixture generation

Create IDC test fixtures. Since real IDC data is huge, generate a
small synthetic dataset for testing:

- 1 day of IDC activity for NO1
- 3 products: NO1-QH-0800, NO1-QH-0815, NO1-QH-0830
- ~50 events per product per hour (new orders, cancels, trades)
- Realistic price movement (random walk around 50 EUR/MWh with
  +/- 5 EUR range)
- Include trades with and without aggressor_side to test both paths

Place at: `tests/fixtures/nordpool/idc_events/NO1_2026_03.parquet`

Create the fixture generator in `tests/generate_idc_fixtures.py` with
a deterministic seed.

Keep the fixture small (hundreds of events, not millions). The goal
is to test correctness, not performance. Performance testing with
realistic volumes can be a separate benchmark suite later.

### 10. Update CLI

Add `--market-type` flag or infer from product names:

```bash
# DA (existing, no change)
nexa run my_algo.py --exchange nordpool --products NO1_DA

# IDC
nexa run my_algo.py --exchange nordpool --products NO1-QH

# When products contain "-QH-", use IDC mode.
# When products end in "_DA", use DA mode.
# Mixed: both engines run, shared clock.
```

The `--products` flag for IDC can accept a zone prefix (e.g., `NO1-QH`)
which means "all quarter-hour products for this zone." The engine
discovers available products from the data files.

---

## How the `@algo` Low-Level API Works with IDC

This task does NOT implement the `@algo` decorator and async event
stream (that's a later task). But the architecture must support it.
The IDC backtest loop should be structured so that events can be
dispatched either to SimpleAlgo hooks or to an async event stream
without changing the engine logic:

```python
# Inside the engine, dispatch is abstract:
for event in events_for_this_mtu:
    fills = matching_engine.process_historical_event(event)
    for fill in fills:
        dispatcher.on_fill(ctx, fill)  # SimpleAlgo hook or event queue
    dispatcher.on_market_data(ctx, event)  # SimpleAlgo no-op or event queue
```

The `dispatcher` abstraction is the seam. For now it calls SimpleAlgo
hooks. Later it can push to an async queue for the `@algo` API.

---

## Tests

1. **SlidingWindow**: create a small Parquet file with known row
   groups. Advance the window, verify correct row groups are loaded
   and old ones are evicted. Verify memory_usage_bytes decreases
   after eviction.

2. **DataManifest**: build a manifest from fixture files, query for
   row groups in a time range, verify correct results.

3. **ContinuousMatchingEngine - algo buy hits historical ask**:
   historical ask at 53.10. Algo places buy at 53.50. Verify fill
   at 53.10 (resting order's price, not algo's price).

4. **ContinuousMatchingEngine - algo order rests, later matched**:
   algo places buy at 50.00. No historical match yet. Later, a
   historical sell at 49.80 arrives. Verify fill at 49.80.

5. **ContinuousMatchingEngine - no match**: algo places buy at 45.00.
   Historical asks are all above 50.00. Verify no fill.

6. **ContinuousMatchingEngine - gate closure**: algo has a resting
   order. Gate closes for that product. Verify the order is cancelled
   and on_cancel is called.

7. **ContinuousMatchingEngine - aggressor_side missing**: verify the
   fallback matching logic works when aggressor_side is None.

8. **End-to-end IDC backtest**: write a simple IDC algo that places
   buy orders below the current best ask. Run against the synthetic
   fixture. Verify fills, PnL, and VWAP are all computed correctly.
   Verify the equity curve has snapshots at MTU boundaries.

9. **Mixed DA + IDC**: (stretch, can be deferred) run a backtest with
   both DA and IDC products. Verify both matching engines run and
   results are combined.

10. **Window memory**: verify that peak memory stays bounded. Load
    fixtures, advance through the full day, check that
    memory_usage_bytes never exceeds expected limits.

---

## Performance Expectations

This task is about correctness, not speed. With the small synthetic
fixture (~150 events total), the IDC backtest should complete in
under a second. Real-world performance with production-scale data
(millions of events) is a concern for a separate benchmarking task
once the correctness tests pass.

The windowed replay architecture must be correct though, even if we're
not testing it at scale. Loading the entire fixture into memory would
"work" but would mask bugs in the window management.

---

## What NOT to build

- `@algo` decorator and async event stream (later task)
- EPEX SPOT or EEX adapters (later task)
- Built-in signal providers (later task)
- Shared data replay for multi-algo runs (later task)
- Portfolio-level NOP aggregation (later task, but Position.net_mw
  per product already covers per-product NOP)
- Performance benchmarking suite
- Full L3 order book reconstruction
- Market impact modelling

---

## Acceptance criteria

1. `make ci` passes
2. A customer can write a SimpleAlgo that uses on_bar, get_orderbook,
   get_best_bid/ask, and place_order to trade IDC products
3. Orders match against historical data using price-time priority
4. The SlidingWindow loads and evicts row groups correctly
5. Gate closure cancels resting orders and fires on_gate_closure
   and on_cancel hooks
6. PnL, VWAP, and the HTML report (from task 03) work correctly
   with IDC trades
7. The engine handles missing aggressor_side gracefully
8. Peak memory stays bounded regardless of replay period length
   (verified by SlidingWindow.memory_usage_bytes in tests)
9. All new types have type hints and frozen Pydantic models where
   appropriate
10. All new public API has Google-style docstrings
