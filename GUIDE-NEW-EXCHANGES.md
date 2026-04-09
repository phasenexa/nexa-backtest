# Adding a New Exchange to nexa-backtest

This guide explains exactly what to create, implement, and test when adding
support for a new exchange. It is based on the Nord Pool and EPEX SPOT
adapters. Follow every step in order.

---

## Overview

An exchange adapter in nexa-backtest is a stateless configuration object. It
declares what the exchange supports and validates orders against exchange rules.
Actual order matching is performed by the engine's internal matching engines
(`DAAuctionMatcher`, `ContinuousMatchingEngine`), not by the adapter.

You need to touch four areas:

1. **Exchange adapter** — new file in `src/nexa_backtest/exchanges/`
2. **Data parser** — new file in `src/nexa_backtest/data/parsers/` (if the
   exchange has a non-standard data export format)
3. **Public exports** — wire in to `src/nexa_backtest/__init__.py`
4. **Tests** — new files in `tests/test_exchanges/` and `tests/test_data/`

---

## Step 1 — Gather exchange facts

Before writing any code, answer these questions:

| Question | Nord Pool | EPEX SPOT |
|---|---|---|
| Short `exchange_id` string | `"nordpool"` | `"epex_spot"` |
| Geographic identifier | bidding zone (`NO1`) | delivery area (`DE-LU`) |
| Product naming format | `{zone}-QH-{HHMM}` | `{area}-QH-{HHMM}` |
| MTU duration (minutes) | 15 | 15 |
| IDC gate closure (min before delivery) | 30 | 30 |
| DA/IDA gate closure offset | 1 hour | 1 hour |
| Min price (EUR/MWh) | -500 | -500 |
| Max price (EUR/MWh) | 3,000 | 4,000 |
| Min volume (MW) | 0.1 | 0.1 |
| Max volume (MW) | None (uncapped) | None |
| Supports block bids | Yes | Yes |
| Supports linked orders | No | No |
| Supports market orders | No | No |
| Supports partial fills | Yes | Yes |
| Supports continuous (IDC) trading | Yes | Yes |
| Supports auction (DA/IDA) trading | No (DA only via matching) | Yes |

All values end up in `ExchangeCapabilities` and in the constants at the top of
the adapter module.

---

## Step 2 — Create the adapter file

Create `src/nexa_backtest/exchanges/{exchange_name}.py`.

Use this template, substituting all `{…}` placeholders:

```python
"""{ Exchange Full Name } exchange adapter.

Implements the :class:`~nexa_backtest.exchanges.base.ExchangeAdapter` protocol
for { Exchange } markets.

{ Brief description of what makes this exchange different from Nord Pool. }

Supported { zones / areas } (non-exhaustive)::

    { CODE1 }    # Description
    { CODE2 }    # Description
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from nexa_backtest.exchanges.base import ExchangeAdapter, ExchangeCapabilities
from nexa_backtest.types import CancelResult, Order, OrderBook, OrderResult

# Gate closure offsets
{EXCHANGE}_IDC_GATE_CLOSURE = timedelta(minutes={N})
{EXCHANGE}_DA_GATE_CLOSURE = timedelta(hours={N})

# Price limits (EUR/MWh)
{EXCHANGE}_MIN_PRICE = Decimal("{min}")
{EXCHANGE}_MAX_PRICE = Decimal("{max}")

# Volume limits (MW)
{EXCHANGE}_MIN_VOLUME = Decimal("0.1")


class {ExchangeName}Adapter:
    """Exchange adapter for { Exchange } (DA + IDC).

    This is a stateless configuration object.  The backtest engine uses it to
    query capabilities; actual order matching is performed by the engine-
    internal matching engines, not by this adapter.

    Attributes:
        { zone_or_area }: { Zone / Area } identifier, e.g. ``"{ EXAMPLE }"``.
        supports_continuous_trading: Always ``True``.
        supports_auction_trading: ``True`` if the exchange runs auctions.
    """

    def __init__(self, { zone_or_area }: str) -> None:
        self._{ zone_or_area } = { zone_or_area }
        self._capabilities = ExchangeCapabilities(
            exchange_id="{exchange_id}",
            supports_block_bids={True|False},
            supports_linked_orders={True|False},
            supports_market_orders={True|False},
            supports_partial_fills={True|False},
            min_volume_mw={EXCHANGE}_MIN_VOLUME,
            max_volume_mw=None,
            min_price_eur_mwh={EXCHANGE}_MIN_PRICE,
            max_price_eur_mwh={EXCHANGE}_MAX_PRICE,
            mtu_duration_minutes=15,
            gate_closure_minutes_before_delivery={N},
        )
        self.supports_continuous_trading = True
        self.supports_auction_trading = {True|False}

    @property
    def capabilities(self) -> ExchangeCapabilities:
        """Declare { Exchange } capabilities.

        Returns:
            Frozen :class:`~nexa_backtest.exchanges.base.ExchangeCapabilities`.
        """
        return self._capabilities

    def get_products(self) -> list[str]:
        """Return all 96 quarter-hourly product IDs for this { zone / area }.

        Products use the format ``{ {zone_or_area}-QH-HHMM }``.

        Returns:
            List of 96 product identifiers (00:00-23:45 UTC).
        """
        products: list[str] = []
        for hour in range(24):
            for minute in (0, 15, 30, 45):
                products.append(
                    f"{self._{ zone_or_area }}-QH-{hour:02d}{minute:02d}"
                )
        return products

    def get_orderbook(self, product_id: str) -> OrderBook:
        """Return an empty order book (maintained by the matching engine).

        Args:
            product_id: Exchange product identifier.

        Returns:
            Empty :class:`~nexa_backtest.types.OrderBook`.
        """
        return OrderBook(
            product_id=product_id,
            bids=[],
            asks=[],
            timestamp=datetime.now(tz=UTC),
        )

    def submit_order(self, order: Order) -> OrderResult:
        """Order submission is handled by the engine, not the adapter.

        Raises:
            NotImplementedError: Always.  Use ``ctx.place_order()`` instead.
        """
        raise NotImplementedError(
            "submit_order is not implemented on { ExchangeName }Adapter. "
            "Use ctx.place_order() inside the algo."
        )

    def cancel_order(self, order_id: str) -> CancelResult:
        """Order cancellation is handled by the engine, not the adapter.

        Raises:
            NotImplementedError: Always.  Use ``ctx.cancel_order()`` instead.
        """
        raise NotImplementedError(
            "cancel_order is not implemented on { ExchangeName }Adapter. "
            "Use ctx.cancel_order() inside the algo."
        )

    def gate_closure_offset(self, product_type: str = "IDC") -> timedelta:
        """Return the gate closure offset for a given product type.

        Args:
            product_type: ``"DA"``, ``"IDA"``, or ``"IDC"``.

        Returns:
            Time before delivery start that gate closes.
        """
        if product_type in ("DA", "IDA"):
            return {EXCHANGE}_DA_GATE_CLOSURE
        return {EXCHANGE}_IDC_GATE_CLOSURE

    def validate_order(self, order: Order) -> str | None:
        """Validate an order against { Exchange } exchange rules.

        Args:
            order: The order to validate.

        Returns:
            Error message string if invalid, or ``None`` if valid.
        """
        if order.volume_mw < self._capabilities.min_volume_mw:
            return (
                f"Volume {order.volume_mw} MW below { Exchange } minimum "
                f"{self._capabilities.min_volume_mw} MW."
            )
        if order.price_eur_mwh is not None:
            if order.price_eur_mwh < self._capabilities.min_price_eur_mwh:
                return (
                    f"Price {order.price_eur_mwh} EUR/MWh below { Exchange } minimum "
                    f"{self._capabilities.min_price_eur_mwh} EUR/MWh."
                )
            if order.price_eur_mwh > self._capabilities.max_price_eur_mwh:
                return (
                    f"Price {order.price_eur_mwh} EUR/MWh above { Exchange } maximum "
                    f"{self._capabilities.max_price_eur_mwh} EUR/MWh."
                )
        if order.price_eur_mwh is None and not self._capabilities.supports_market_orders:
            return "{ Exchange } does not support market orders."
        return None


# Satisfy ExchangeAdapter protocol at type-check time.
def _check_adapter_protocol(adapter: ExchangeAdapter) -> None:  # pragma: no cover
    """Compile-time check that { ExchangeName }Adapter satisfies ExchangeAdapter."""


_check_adapter_protocol({ ExchangeName }Adapter.__new__({ ExchangeName }Adapter))
```

### Protocol compliance check

The last three lines — `_check_adapter_protocol` — are critical. mypy validates
the function call at import time and will error if any required protocol method
is missing or has the wrong signature. Do not omit them.

---

## Step 3 — Create the data parser (if needed)

If the exchange's historical data export uses non-standard column names or
action codes, add a parser module. Skip this step only if the exchange exports
data that already matches the standard IDC events schema exactly.

Create `src/nexa_backtest/data/parsers/{exchange_name}.py`:

```python
"""{ Exchange } data normaliser.

Reads { Exchange } native { CSV / Parquet } export format and converts it to the
standard IDC events schema used throughout nexa-backtest.

{ Exchange } export columns
--------------------------
{ List the actual column names from the exchange's data export here }

Standard IDC events schema (output)
------------------------------------
``timestamp`` (datetime64[ns, UTC]), ``event_type`` (str), ``order_id`` (str),
``zone`` (str), ``product_id`` (str), ``side`` (str), ``price_eur_mwh``
(float64), ``volume_mw`` (float64), ``remaining_mw`` (float64),
``aggressor_side`` (str, nullable), ``trade_id`` (str, nullable)

ActionCode mapping:
    { source code } -> { "new" | "modify" | "cancel" | "trade" }
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

from nexa_backtest.exceptions import DataError

_ACTION_CODE_MAP: dict[str, str] = {
    "{ SOURCE_CODE }": "new",
    "{ SOURCE_CODE }": "modify",
    "{ SOURCE_CODE }": "cancel",
}

_REQUIRED_COLUMNS = frozenset(
    {
        "{ ColumnA }",
        "{ ColumnB }",
        # ... all columns you will read
    }
)


def parse_{exchange}_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise a { Exchange } DataFrame to the standard IDC events schema.

    Args:
        df: Raw { Exchange } export DataFrame.

    Returns:
        Normalised DataFrame with columns matching the standard IDC events schema.

    Raises:
        :class:`~nexa_backtest.exceptions.DataError`: If required columns are missing.
    """
    import pandas as pd

    missing = _REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise DataError(
            f"{ Exchange } data is missing required columns: {sorted(missing)}. "
            f"Available columns: {sorted(df.columns.tolist())}"
        )

    result = pd.DataFrame()
    result["timestamp"] = pd.to_datetime(df["{ TimestampColumn }"], utc=True)
    result["event_type"] = df["{ ActionColumn }"].map(_ACTION_CODE_MAP).fillna("new")
    result["order_id"] = df["{ OrderIdColumn }"].astype(str)
    result["zone"] = df["{ AreaColumn }"].astype(str)
    result["product_id"] = [
        _build_product_id(str(area), ds)
        for area, ds in zip(df["{ AreaColumn }"], pd.to_datetime(df["{ DeliveryStart }"], utc=True), strict=False)
    ]
    result["side"] = df["{ SideColumn }"].str.lower()
    result["price_eur_mwh"] = pd.to_numeric(df["{ PriceColumn }"], errors="coerce").astype("float64")
    result["volume_mw"] = pd.to_numeric(df["{ VolumeColumn }"], errors="coerce").astype("float64")
    result["remaining_mw"] = result["volume_mw"]  # approximate if not available
    result["aggressor_side"] = None  # set to None if not exported
    result["trade_id"] = None
    result = result.sort_values("timestamp").reset_index(drop=True)
    return result


def parse_{exchange}_csv(path: Path | str) -> pd.DataFrame:
    """Load and normalise a { Exchange } CSV export file.

    Args:
        path: Path to the CSV file.

    Returns:
        Normalised DataFrame with the standard IDC events schema.

    Raises:
        :class:`~nexa_backtest.exceptions.DataError`: If the file cannot be read.
    """
    import pandas as pd

    try:
        raw = pd.read_csv(path)
    except Exception as exc:
        raise DataError(f"Failed to read { Exchange } CSV file '{path}': {exc}") from exc

    return parse_{exchange}_df(raw)
```

### Key rules for parsers

- **`aggressor_side` must be `None` if not exported.** Never guess or infer it;
  let the matching engine use its own heuristic. The EPEX SPOT format is an
  example of this.
- **`remaining_mw` must be approximated from the order volume** if the export
  does not include it explicitly. Document this in the module docstring.
- **Always sort by `timestamp` ascending** at the end of `parse_*_df`.
- **`product_id` must match the adapter's naming format.** For `{area}-QH-HHMM`
  adapters, use a `_build_product_id(area, delivery_start)` helper that formats
  the HHMM from the delivery start UTC timestamp.
- **Never load pandas at module import time.** Import it inside the function body
  (`import pandas as pd`) so the module loads without pandas installed.

---

## Step 4 — Export from `__init__.py`

Add the adapter (and parser if applicable) to the public API.

In `src/nexa_backtest/__init__.py`:

```python
# In the import block (keep alphabetical within each group):
from nexa_backtest.exchanges.{exchange_name} import {ExchangeName}Adapter

# In __all__ (keep alphabetical):
"{ExchangeName}Adapter",
```

The data parser does **not** need to go in `__init__.py`. Parsers are imported
directly by users who need them:

```python
from nexa_backtest.data.parsers.{exchange_name} import parse_{exchange}_df
```

---

## Step 5 — Write tests

### Adapter tests

Create `tests/test_exchanges/test_{exchange_name}.py`. Cover every field and
behaviour. The EPEX SPOT test file (`tests/test_exchanges/test_epex_spot.py`)
is the reference.

Minimum test cases:

```python
# Capabilities
def test_{exchange}_exchange_id(adapter) -> None: ...
def test_{exchange}_supports_block_bids(adapter) -> None: ...
def test_{exchange}_does_not_support_linked_orders(adapter) -> None: ...
def test_{exchange}_does_not_support_market_orders(adapter) -> None: ...
def test_{exchange}_supports_partial_fills(adapter) -> None: ...
def test_{exchange}_min_price(adapter) -> None: ...
def test_{exchange}_max_price(adapter) -> None: ...
def test_{exchange}_mtu_duration(adapter) -> None: ...
def test_{exchange}_min_volume(adapter) -> None: ...
def test_{exchange}_gate_closure_minutes(adapter) -> None: ...

# Differentiation from Nord Pool
def test_{exchange}_max_price_differs_from_nordpool() -> None: ...
def test_{exchange}_exchange_id_differs_from_nordpool() -> None: ...

# Products
def test_{exchange}_products_returns_96_qh(adapter) -> None: ...
def test_{exchange}_products_use_area_prefix(adapter) -> None: ...
def test_{exchange}_products_include_midnight(adapter) -> None: ...
def test_{exchange}_products_include_last_quarter(adapter) -> None: ...
def test_{exchange}_different_area() -> None: ...

# Order book
def test_{exchange}_orderbook_is_empty(adapter) -> None: ...

# Order validation
def test_{exchange}_validate_order_valid(adapter) -> None: ...
def test_{exchange}_validate_order_below_min_volume(adapter) -> None: ...
def test_{exchange}_validate_order_price_too_low(adapter) -> None: ...
def test_{exchange}_validate_order_price_too_high(adapter) -> None: ...
def test_{exchange}_validate_order_at_max_price(adapter) -> None: ...
def test_{exchange}_no_market_orders(adapter) -> None: ...
def test_{exchange}_submit_order_raises(adapter) -> None: ...
def test_{exchange}_cancel_order_raises(adapter) -> None: ...

# Gate closure
def test_{exchange}_idc_gate_closure_offset(adapter) -> None: ...
def test_{exchange}_da_gate_closure_offset(adapter) -> None: ...
```

Use a `@pytest.fixture` for the adapter to avoid repeating construction:

```python
@pytest.fixture
def adapter() -> {ExchangeName}Adapter:
    return {ExchangeName}Adapter("{ EXAMPLE_ZONE }")
```

### Parser tests

Create `tests/test_data/test_{exchange_name}_parser.py`. The EPEX SPOT parser
tests (`tests/test_data/test_epex_parser.py`) are the reference.

Minimum test cases:

```python
def test_parse_returns_standard_columns(minimal_df) -> None: ...
def test_parse_timestamps_are_utc(minimal_df) -> None: ...
def test_parse_action_code_new_maps_to_new(minimal_df) -> None: ...
def test_parse_action_code_modify_maps_to_modify(minimal_df) -> None: ...
def test_parse_action_code_cancel_maps_to_cancel(minimal_df) -> None: ...
def test_parse_unknown_action_code_defaults_to_new(minimal_df) -> None: ...
def test_parse_order_id_is_string(minimal_df) -> None: ...
def test_parse_side_is_lowercase(minimal_df) -> None: ...
def test_parse_product_id_format(minimal_df) -> None: ...
def test_parse_zone_matches_area(minimal_df) -> None: ...
def test_parse_price_is_float64(minimal_df) -> None: ...
def test_parse_volume_is_float64(minimal_df) -> None: ...
def test_parse_remaining_mw_equals_volume_mw(minimal_df) -> None: ...
def test_parse_aggressor_side_is_none(minimal_df) -> None: ...
def test_parse_sorted_by_timestamp(unsorted_df) -> None: ...
def test_parse_raises_on_missing_columns() -> None: ...
def test_parse_csv_raises_on_bad_path() -> None: ...
```

Build a `@pytest.fixture` that returns the minimum valid raw DataFrame:

```python
@pytest.fixture
def minimal_df() -> pd.DataFrame:
    return pd.DataFrame({
        "{ TimestampColumn }": ["2026-01-01T08:00:00Z"],
        "{ ActionColumn }": ["{ A01 }"],
        "{ OrderIdColumn }": ["ORD-001"],
        "{ AreaColumn }": ["{ DE-LU }"],
        "{ DeliveryStart }": ["2026-01-01T09:00:00Z"],
        "{ SideColumn }": ["BUY"],
        "{ PriceColumn }": [50.0],
        "{ VolumeColumn }": [1.0],
    })
```

---

## Step 6 — Verify

Run the full CI suite:

```bash
make ci
```

This runs lint (`ruff`), type checking (`mypy --strict`), all tests, and
notebook execution. All must pass before the exchange is considered done.

Specifically check:

1. `mypy` catches the protocol compliance check at the bottom of the adapter
   file — if any method signature is wrong, mypy will error here.
2. All 96 products are returned by `get_products()`.
3. `validate_order` returns `None` for valid orders and a non-`None` string for
   every invalid case.
4. `submit_order` and `cancel_order` both raise `NotImplementedError`.
5. The parser's output columns exactly match the standard IDC events schema.

---

## What you do NOT need to touch

- `engines/backtest.py` — the engine uses the `ExchangeAdapter` protocol; it
  does not need to know about specific exchange classes.
- `engines/matching.py` — the matching logic is exchange-agnostic.
- `context.py` or `types.py` — no new types are needed for a new exchange.
- Any existing exchange adapter — they are independent.

The only coupling point between the engine and an exchange is the `exchange`
string argument to `BacktestEngine`. That string is used to look up the correct
data directory, not to select matching logic. If the new exchange needs a new
data directory layout, update `data/loader.py` accordingly.

---

## Common mistakes

**Wrong product ID format.** Product IDs must match exactly what the engine
passes to `validate_order`, `get_orderbook`, and what appears in the historical
data. If the parser generates `DE-LU-QH-900` but the adapter generates
`DE-LU-QH-0900`, orders will never match. Always use `f"{hour:02d}{minute:02d}"`.

**Forgetting `from __future__ import annotations`.** Without this, `str | None`
return types fail on Python 3.9. All modules in this codebase use it.

**Importing pandas at module level.** Keep `import pandas as pd` inside the
function body. Pandas is an optional dependency; importing it at module level
breaks users who have not installed it.

**Using naive datetimes.** All timestamps must be timezone-aware. In parsers,
always pass `utc=True` to `pd.to_datetime`. In the adapter's `get_orderbook`,
always pass `tz=UTC` to `datetime.now`.

**Guessing `aggressor_side`.** If the exchange does not export it, set it to
`None`. The matching engine has a fallback heuristic. Inferring it incorrectly
is worse than not having it.

**Missing the protocol compliance check.** The `_check_adapter_protocol` call at
the bottom of the adapter file is how mypy catches protocol mismatches at
import time without needing a test. Always include it.

**Skipping the `gate_closure_offset` method.** Even if you only use the default
value from `ExchangeCapabilities.gate_closure_minutes_before_delivery`, add the
method. It allows callers to query per-product-type offsets (DA vs IDC) without
knowing the exchange's internal constants.
