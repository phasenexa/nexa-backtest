# Task 01: Core Types, TradingContext Protocol, and DA Matching

## Goal

Establish the foundational types and protocols that everything else builds on.
By the end of this task, the library has a working `TradingContext` protocol,
core domain types, a simulated clock, and a DA auction matching engine.

Nothing runs end-to-end yet. There is no `BacktestEngine`, no CLI, no PnL
analysis. This task builds the building blocks that Task 02 will wire together.

## Why this scope

If you give Claude Code the full backtester in one go, it makes a mess. This
task is deliberately small: types, a protocol, a clock, and a matcher. Each
piece is independently testable with no complex integration. The test suite
should be comprehensive here because every subsequent task depends on these
types being correct.

## What to build

### 1. `src/nexa_backtest/types.py`

Core domain types. All must be immutable (frozen dataclasses or Pydantic models
with `frozen=True`). Use `decimal.Decimal` for prices and volumes.

Types needed:

- `MTU` - a 15-minute market time unit (start, end, zone)
- `PriceLevel` - price + volume at a single level (bid or ask)
- `OrderBook` - best bid, best ask, timestamp, product_id
- `Order` - with class methods: `Order.buy(...)`, `Order.sell(...)`,
  `Order.market(...)`, `Order.block_bid(...)`
- `OrderResult` - order_id, status (accepted/rejected/filled), fill details
- `CancelResult` - order_id, status (cancelled/not_found)
- `Fill` - trade execution: order_id, product_id, price, volume, timestamp, side
- `Position` - net_mw, avg_entry_price, unrealised_pnl for a single product
- `Side` - enum: BUY, SELL
- `OrderStatus` - enum: PENDING, ACCEPTED, PARTIALLY_FILLED, FILLED, CANCELLED, REJECTED
- `AuctionInfo` - product_id, auction_type (DA/IDA), gate_closure_time, zone

Keep it clean. No business logic in the types. They are data containers.

### 2. `src/nexa_backtest/exceptions.py`

Exception hierarchy:

- `NexaBacktestError` (base)
  - `AlgoError` - errors raised by user algo code
  - `DataError` - data loading/schema issues
  - `ExchangeError` - exchange adapter issues
  - `ValidationError` - pre-run validation failures
  - `UnsupportedFeatureError(ValidationError)` - algo uses unsupported exchange feature
  - `MatchingError` - matching engine issues
  - `SignalError` - signal provider issues

### 3. `src/nexa_backtest/context.py`

The `TradingContext` protocol. This is the interface your algo sees. It must be
a `typing.Protocol` (not an ABC) so that mypy can structurally check compliance.

Methods (all documented with Google-style docstrings):

```python
class TradingContext(Protocol):
    def now(self) -> datetime: ...
    def time_to_gate_closure(self, product_id: str) -> timedelta: ...
    def current_mtu(self) -> MTU: ...

    def get_orderbook(self, product_id: str) -> OrderBook: ...
    def get_best_bid(self, product_id: str) -> PriceLevel | None: ...
    def get_best_ask(self, product_id: str) -> PriceLevel | None: ...
    def get_last_price(self, product_id: str) -> Decimal | None: ...
    def get_vwap(self, product_id: str) -> Decimal | None: ...

    def place_order(self, order: Order) -> OrderResult: ...
    def cancel_order(self, order_id: str) -> CancelResult: ...
    def modify_order(self, order_id: str, **changes: Any) -> OrderResult: ...

    def get_position(self, product_id: str) -> Position: ...
    def get_all_positions(self) -> dict[str, Position]: ...
    def get_unrealised_pnl(self) -> Decimal: ...

    def get_signal(self, name: str) -> SignalValue: ...
    def get_signal_history(self, name: str, lookback: int) -> list[SignalValue]: ...

    def predict(self, model_name: str, features: dict[str, Any]) -> Any: ...

    def log(self, message: str, level: str = "info") -> None: ...
```

Also define `SignalValue` here (timestamp + value + name).

### 4. `src/nexa_backtest/engines/clock.py`

Two clock implementations:

- `SimulatedClock` - advances instantly between timestamps. Used by backtest engine.
  Methods: `now()`, `advance_to(timestamp)`, `peek_next()`.
- `RealtimeClock` - returns actual wall-clock time. Used by paper/live engines.
  Methods: `now()`.

Both implement a `Clock` protocol.

### 5. `src/nexa_backtest/engines/matching.py`

DA auction matching engine only (IDC continuous comes in Task 02).

`DAAuctionMatcher`:
- Takes a historical clearing price for a product/MTU
- Receives an order from the algo
- If buy order price >= clearing price: fill at clearing price
- If sell order price <= clearing price: fill at clearing price
- Otherwise: reject
- All fills are at the clearing price (uniform price auction)
- Returns `Fill` or updates `OrderResult` status to REJECTED

This is the price-taker assumption. The algo's order does not affect the
clearing price.

### 6. `src/nexa_backtest/exchanges/base.py`

- `ExchangeCapabilities` frozen dataclass with all capability flags
  (supports_block_bids, supports_linked_orders, etc., min/max volume,
  min/max price, mtu_duration, gate_closure_rules)
- `ExchangeAdapter` protocol (capabilities property, get_products,
  get_orderbook, submit_order, cancel_order)

Do NOT implement Nord Pool / EPEX SPOT / EEX adapters yet. Just the
protocol and capabilities type.

### 7. Tests

Comprehensive tests for everything above:

- `tests/test_types.py` - construction, immutability, Order class methods,
  Decimal precision, serialisation round-trips
- `tests/test_context.py` - verify Protocol is structurally checkable
  (a mock class that implements it should pass mypy)
- `tests/test_engines/test_clock.py` - SimulatedClock advance, ordering,
  peek; RealtimeClock returns sane values
- `tests/test_engines/test_matching.py` - DA matcher with various scenarios:
  buy at clearing price (fill), buy below (reject), sell at clearing price
  (fill), sell above (reject), exact price match (fill), zero-volume
  rejection, multiple orders same product

### 8. Scaffolding

- `src/nexa_backtest/__init__.py` with version and key public exports
- `src/nexa_backtest/_version.py`
- `tests/conftest.py` with common fixtures (sample MTUs, orders, clearing prices)
- `pyproject.toml` with dependencies, ruff config, mypy config
- `Makefile` with targets: install, lint, typecheck, test, ci
- `.gitignore`

## Out of scope for this task

- BacktestEngine (Task 02)
- SimpleAlgo / @algo decorator (Task 02)
- PnL / VWAP analysis (Task 02)
- CLI (Task 02)
- Data loading / Parquet (Task 02)
- Signals (Task 03)
- IDC continuous matching (Task 03)
- Exchange adapters (Task 03)
- ML models (Task 04)
- Validation pipeline (Task 04)
- Paper/Live engines (later)
- Code protection (later)

## Acceptance criteria

- `make ci` passes (ruff + mypy strict + pytest)
- All types are immutable and use Decimal for prices/volumes
- TradingContext is a Protocol, not an ABC
- DA matcher correctly handles fill/reject for buy and sell orders
- Test coverage > 80% for all new code
- No naive datetimes anywhere
- Google-style docstrings on all public API
