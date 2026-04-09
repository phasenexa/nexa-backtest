# Task 05: Low-Level Algo API, EPEX SPOT Adapter, and Portfolio NOP

## Goal

Three things that complete stage 2:

1. The `@algo` decorator with async event streams, giving quants full
   control over the event loop
2. The EPEX SPOT exchange adapter, proving the exchange abstraction
   actually works with a second exchange
3. Portfolio-level NOP tracking across products for the same delivery
   period

After this task, the backtester supports two exchanges, two API levels,
and gives IDC traders a proper view of their net exposure.

---

## What to build

### 1. `@algo` Decorator and Async Event Stream

The low-level API gives the algo direct access to the event stream
instead of routing through SimpleAlgo hooks. The algo is an async
generator that receives events and acts on them.

**In `algo.py`:**

```python
def algo(
    name: str,
    version: str,
) -> Callable:
    """Decorator that marks an async function as a trading algo.

    The decorated function receives a TradingContext and must
    consume events via ctx.events().

    Usage:
        @algo(name="spread_scalper", version="1.0.0")
        async def run(ctx: TradingContext) -> None:
            async for event in ctx.events():
                match event:
                    case MarketDataUpdate(...):
                        ...
    """
```

The decorator should:

- Record the algo's name and version as metadata
- Validate that the decorated function is async
- Validate that it accepts exactly one argument (the TradingContext)
- Make the function compatible with the `BacktestEngine` (the engine
  currently expects a SimpleAlgo instance; it needs to also accept
  a decorated async function)

**In `context.py`, add to TradingContext:**

```python
class TradingContext(Protocol):
    # Existing methods unchanged...

    async def events(self) -> AsyncIterator[MarketEvent]:
        """Yields market events in timestamp order.
        Only available in the low-level @algo API.
        Calling this from a SimpleAlgo raises AlgoError."""
        ...
```

**Event types available in the stream:**

The algo receives all of these through `ctx.events()`:

| Event | When |
|-------|------|
| `MarketDataUpdate` | Order book change (new/modify/cancel) |
| `HistoricalTrade` | A trade occurred in the historical data |
| `GateClosureWarning` | N minutes before gate closes (configurable) |
| `GateClosureEvent` | Gate has closed for a product |
| `SignalUpdate` | A subscribed signal has a new value |
| `BarEvent` | MTU boundary reached (periodic tick) |
| `FillEvent` | One of the algo's orders was filled |
| `CancelEvent` | One of the algo's orders was cancelled |

```python
@dataclass(frozen=True)
class SignalUpdate(MarketEvent):
    """A signal has updated."""
    name: str
    value: SignalValue

@dataclass(frozen=True)
class BarEvent(MarketEvent):
    """MTU boundary tick."""
    mtu: MTU

@dataclass(frozen=True)
class FillEvent(MarketEvent):
    """Algo order was filled."""
    fill: Fill

@dataclass(frozen=True)
class CancelEvent(MarketEvent):
    """Algo order was cancelled."""
    order_id: str
    reason: str  # "gate_closure", "user_cancel", "exchange_cancel"
```

**How the engine drives the async algo:**

The backtest engine is synchronous (simulated clock, no real async
I/O). The async interface is a convenience for the algo author, not
a requirement of the runtime. Internally, the engine uses an
`asyncio.Queue` to push events and runs the algo coroutine with
`asyncio.run()` or a controlled event loop:

```python
# Simplified engine loop for @algo:
#
# 1. Create an asyncio.Queue for events
# 2. Start the algo coroutine (it blocks on queue.get())
# 3. For each historical event:
#    a. Push to queue
#    b. Let the algo process it (run until it awaits again)
#    c. Collect any orders the algo placed
#    d. Run matching
#    e. Push fill/cancel events back to queue
# 4. When replay is done, push a sentinel to stop the algo
```

The key constraint: the algo must `await` between processing events.
It cannot spin in a tight loop without yielding. This ensures the
engine maintains control of the clock.

### 2. Update `engines/backtest.py` - Unified Dispatch

The engine needs to support both SimpleAlgo and @algo functions.
Refactor the dispatch layer:

```python
class AlgoDispatcher(Protocol):
    """Abstraction over SimpleAlgo hooks vs async event stream."""
    def on_setup(self, ctx: TradingContext) -> None: ...
    def on_event(self, ctx: TradingContext, event: MarketEvent) -> None: ...
    def on_teardown(self, ctx: TradingContext) -> None: ...
    def get_pending_orders(self) -> list[Order]: ...

class SimpleAlgoDispatcher:
    """Routes events to SimpleAlgo hooks."""

class AsyncAlgoDispatcher:
    """Pushes events to async queue, collects orders."""
```

The engine works with `AlgoDispatcher` and doesn't know which API
the algo uses.

### 3. EPEX SPOT Exchange Adapter

Create `exchanges/epex_spot.py`. This proves the exchange abstraction
works with more than one exchange and exposes any assumptions baked
into the Nord Pool adapter.

**EPEX SPOT specifics:**

- **Product naming**: different convention from Nord Pool. EPEX uses
  delivery area codes (e.g., `DE-LU`, `FR`, `AT`) rather than Nord
  Pool's zone codes (`NO1`, `SE3`).
- **Gate closure**: EPEX SPOT IDC typically closes 30 minutes before
  delivery (same as Nord Pool for most products, but the rules differ
  for cross-border products).
- **Price limits**: EPEX has different min/max price limits than
  Nord Pool. Currently -500 to +4,000 EUR/MWh for most products.
- **Volume limits**: minimum order size differs.
- **Auction vs continuous**: EPEX runs both intraday auctions (IDA)
  and continuous trading. The adapter should declare capabilities for
  both.
- **Data format differences**: EPEX's historical data export format
  uses different column names (see the EPEX schema from the original
  conversation: `OrderId, InitialId, ParentId, Side, Product,
  DeliveryStart, DeliveryEnd, CreationTime, DeliveryArea,
  ExecutionRestriction, UserDefinedBlock, LinkedBasketId, RevisionNo,
  ActionCode, TransactionTime, ValidityTime, Price, Currency,
  Quantity, QuantityUnit, Volume, VolumeUnit`). The data loader needs
  an EPEX-specific parser that normalises this to the standard IDC
  events schema.

**Capabilities:**

```python
EPEX_SPOT_CAPABILITIES = ExchangeCapabilities(
    exchange_name="epex_spot",
    supports_block_bids=True,       # In auctions
    supports_linked_orders=False,   # Not in continuous
    supports_exclusive_groups=False,
    supports_flexi_orders=False,
    supports_curtailable_blocks=False,
    supports_continuous_trading=True,
    supports_auction_trading=True,
    min_volume_mw=Decimal("0.1"),
    max_price_eur_mwh=Decimal("4000"),
    min_price_eur_mwh=Decimal("-500"),
    mtu_duration=timedelta(minutes=15),
    gate_closure_rules={...},
)
```

**Data normalisation:**

Create `data/parsers/epex.py` (or add to `data/loader.py`) that reads
EPEX's native CSV/Parquet export format and converts to the standard
IDC events schema. Key mappings:

| EPEX field | Standard field | Notes |
|------------|---------------|-------|
| CreationTime | timestamp | Parse as UTC |
| ActionCode | event_type | Map: A01=new, A02=modify, A03=cancel |
| OrderId | order_id | |
| Side | side | Map: BUY=buy, SELL=sell |
| Product + DeliveryStart | product_id | Construct from area + time |
| DeliveryArea | zone | |
| Price | price_eur_mwh | |
| Quantity | volume_mw | |
| Quantity (remaining) | remaining_mw | Inferred from RevisionNo |

`aggressor_side` is not directly available in EPEX data. It can
sometimes be inferred from `ActionCode` and `TransactionTime` ordering,
but this is fragile. Default to `None` and let the matching engine use
its fallback logic (from task 04).

### 4. Portfolio-Level NOP

Add a method to `TradingContext` that aggregates net positions across
products for the same delivery period:

```python
class TradingContext(Protocol):
    # Existing...
    def get_position(self, product_id: str) -> Position: ...
    def get_all_positions(self) -> dict[str, Position]: ...

    # New:
    def get_delivery_position(self, delivery_start: datetime) -> DeliveryPosition: ...
    def get_all_delivery_positions(self) -> dict[datetime, DeliveryPosition]: ...

@dataclass(frozen=True)
class DeliveryPosition:
    """Net position for a specific delivery period, aggregated
    across all products covering that period."""
    delivery_start: datetime
    delivery_end: datetime
    net_mw: Decimal
    positions: list[Position]  # Contributing per-product positions
```

This matters for IDC because a trader might have positions in the same
delivery period from different trading sessions or order types. The
portfolio NOP tells them their actual exposure for physical delivery.

For DA-only backtests, `get_delivery_position` returns the same as
`get_position` (one product per delivery period). It only diverges
when multiple products map to the same delivery window.

### 5. NOP at Gate Closure Tracking

Add NOP at gate closure as a metric in the analysis:

```python
@dataclass(frozen=True)
class GateClosureSnapshot:
    """Position state when gate closed for a product."""
    product_id: str
    gate_closure_time: datetime
    delivery_start: datetime
    net_mw: Decimal

class BacktestResult:
    # Existing fields...

    # New:
    gate_closure_positions: list[GateClosureSnapshot]
    avg_gate_closure_nop_mw: Decimal  # Average absolute NOP at gate
    max_gate_closure_nop_mw: Decimal  # Largest absolute NOP at gate
```

Large NOP at gate closure suggests the algo isn't managing exposure
well. Include these metrics in the summary output and the HTML report.

Add a section to the HTML report: "Gate Closure Exposure" showing a
chart of NOP at gate closure over time.

---

## Tests

1. **@algo decorator**: verify it rejects non-async functions, rejects
   functions with wrong argument count, records name/version metadata.

2. **Async event stream**: write a simple @algo that counts events and
   places an order after seeing 5 MarketDataUpdates. Run against IDC
   fixture. Verify it receives the expected event types in order.

3. **SimpleAlgo and @algo equivalence**: write the same strategy as
   both a SimpleAlgo and an @algo. Run both against the same fixture.
   Verify identical fills and PnL.

4. **AlgoDispatcher**: verify the engine correctly dispatches to both
   SimpleAlgoDispatcher and AsyncAlgoDispatcher without knowing which
   one it's using.

5. **EPEX SPOT capabilities**: verify the capabilities dataclass is
   correctly populated and differs from Nord Pool where expected.

6. **EPEX data normalisation**: create a small EPEX-format CSV fixture.
   Parse it through the normaliser. Verify the output matches the
   standard IDC events schema. Verify aggressor_side is None.

7. **EPEX end-to-end**: run an IDC backtest against EPEX SPOT with a
   simple algo. Verify fills, PnL, and report generation work.

8. **Same algo, two exchanges**: run the same algo against both Nord
   Pool and EPEX SPOT fixtures. Verify both complete without errors
   and produce valid results. (The PnL will differ because the
   market data differs, but both must be structurally correct.)

9. **DeliveryPosition aggregation**: create positions in multiple
   products covering the same delivery period. Verify
   get_delivery_position returns the correct net.

10. **Gate closure NOP**: run an IDC backtest where the algo
    deliberately holds a position through gate closure. Verify
    the GateClosureSnapshot is recorded with the correct NOP.

11. **NOP metrics in report**: verify the HTML report includes the
    gate closure exposure section.

---

## What NOT to build

- EEX adapter (can follow the same pattern later if needed)
- Built-in signal providers (DA price, wind, etc.)
- Validation pipeline (`nexa validate`)
- ML model registry
- Multi-algo shared data replay
- Paper/live trading engines
- Code compilation

---

## Acceptance criteria

1. `make ci` passes
2. The `@algo` API works end-to-end: decorator, async event stream,
   order placement, fills, PnL
3. A SimpleAlgo and an equivalent @algo produce identical results
   against the same data
4. The EPEX SPOT adapter loads EPEX-format data, normalises it, and
   runs a backtest correctly
5. The same algo can run against both Nord Pool and EPEX SPOT by
   changing the exchange parameter
6. `ctx.get_delivery_position()` correctly aggregates NOP across
   products
7. Gate closure NOP is tracked and included in metrics, summary,
   and HTML report
8. All new types have type hints and frozen Pydantic models where
   appropriate
9. All new public API has Google-style docstrings
