"""Tests for portfolio-level NOP (DeliveryPosition aggregation) — task 05."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from nexa_backtest.engines.backtest import _BacktestContext
from nexa_backtest.engines.clock import SimulatedClock
from nexa_backtest.signals.registry import SignalRegistry
from nexa_backtest.types import Fill, Side


def _make_ctx(
    products: dict[str, datetime] | None = None,
) -> _BacktestContext:
    """Create a minimal _BacktestContext with optional product delivery starts."""
    clock = SimulatedClock(
        initial_time=datetime(2026, 3, 1, 9, 0, tzinfo=UTC),
    )
    ctx = _BacktestContext(clock=clock, signal_registry=SignalRegistry())
    if products:
        ctx._product_delivery_starts = products
    return ctx


def _record_fill(ctx: _BacktestContext, product_id: str, side: Side, volume: Decimal) -> None:
    """Helper to record a fill in the context's position tracker."""
    fill = Fill(
        order_id="order-1",
        product_id=product_id,
        price=Decimal("50"),
        volume=volume,
        timestamp=datetime(2026, 3, 1, 8, 0, tzinfo=UTC),
        side=side,
    )
    ctx._record_fill(fill)


# ---------------------------------------------------------------------------
# get_delivery_position
# ---------------------------------------------------------------------------


def test_delivery_position_zero_when_no_fills() -> None:
    """Returns zero position for a delivery start with no trades."""
    ds = datetime(2026, 3, 1, 9, 0, tzinfo=UTC)
    ctx = _make_ctx({"NO1-QH-0900": ds})
    dp = ctx.get_delivery_position(ds)
    assert dp.net_mw == Decimal("0")
    assert dp.delivery_start == ds
    assert dp.delivery_end == ds + timedelta(minutes=15)
    assert dp.positions == ()


def test_delivery_position_single_product_long() -> None:
    """Single product with a long position aggregates correctly."""
    ds = datetime(2026, 3, 1, 9, 0, tzinfo=UTC)
    ctx = _make_ctx({"NO1-QH-0900": ds})
    _record_fill(ctx, "NO1-QH-0900", Side.BUY, Decimal("5"))

    dp = ctx.get_delivery_position(ds)
    assert dp.net_mw == Decimal("5")
    assert len(dp.positions) == 1


def test_delivery_position_single_product_short() -> None:
    """Single product with a short position has negative net_mw."""
    ds = datetime(2026, 3, 1, 9, 0, tzinfo=UTC)
    ctx = _make_ctx({"NO1-QH-0900": ds})
    _record_fill(ctx, "NO1-QH-0900", Side.SELL, Decimal("3"))

    dp = ctx.get_delivery_position(ds)
    assert dp.net_mw == Decimal("-3")


def test_delivery_position_aggregates_multiple_products() -> None:
    """Two products mapped to the same delivery start must sum their positions."""
    ds = datetime(2026, 3, 1, 9, 0, tzinfo=UTC)
    # Two hypothetical products covering the same 09:00 window.
    ctx = _make_ctx({"PROD-A-QH-0900": ds, "PROD-B-QH-0900": ds})

    _record_fill(ctx, "PROD-A-QH-0900", Side.BUY, Decimal("5"))
    _record_fill(ctx, "PROD-B-QH-0900", Side.SELL, Decimal("2"))

    dp = ctx.get_delivery_position(ds)
    assert dp.net_mw == Decimal("3")  # 5 - 2
    assert len(dp.positions) == 2


def test_delivery_position_excludes_other_periods() -> None:
    """Position in a different delivery period must not bleed into the query."""
    ds_0900 = datetime(2026, 3, 1, 9, 0, tzinfo=UTC)
    ds_0915 = datetime(2026, 3, 1, 9, 15, tzinfo=UTC)
    ctx = _make_ctx({"NO1-QH-0900": ds_0900, "NO1-QH-0915": ds_0915})

    _record_fill(ctx, "NO1-QH-0900", Side.BUY, Decimal("10"))
    _record_fill(ctx, "NO1-QH-0915", Side.BUY, Decimal("7"))

    dp_0900 = ctx.get_delivery_position(ds_0900)
    assert dp_0900.net_mw == Decimal("10")

    dp_0915 = ctx.get_delivery_position(ds_0915)
    assert dp_0915.net_mw == Decimal("7")


def test_delivery_position_unknown_start_returns_zero() -> None:
    """Querying a delivery start with no mapped products returns zero position."""
    ctx = _make_ctx()
    unknown_ds = datetime(2026, 3, 1, 15, 0, tzinfo=UTC)
    dp = ctx.get_delivery_position(unknown_ds)
    assert dp.net_mw == Decimal("0")
    assert dp.positions == ()


# ---------------------------------------------------------------------------
# get_all_delivery_positions
# ---------------------------------------------------------------------------


def test_get_all_delivery_positions_empty_when_no_fills() -> None:
    ds = datetime(2026, 3, 1, 9, 0, tzinfo=UTC)
    ctx = _make_ctx({"NO1-QH-0900": ds})
    result = ctx.get_all_delivery_positions()
    assert result == {}


def test_get_all_delivery_positions_returns_nonzero_only() -> None:
    ds_0900 = datetime(2026, 3, 1, 9, 0, tzinfo=UTC)
    ds_0915 = datetime(2026, 3, 1, 9, 15, tzinfo=UTC)
    ctx = _make_ctx({"NO1-QH-0900": ds_0900, "NO1-QH-0915": ds_0915})

    # Only trade in 0900.
    _record_fill(ctx, "NO1-QH-0900", Side.BUY, Decimal("5"))

    result = ctx.get_all_delivery_positions()
    assert ds_0900 in result
    assert ds_0915 not in result
    assert result[ds_0900].net_mw == Decimal("5")


def test_get_all_delivery_positions_multiple_periods() -> None:
    ds_0900 = datetime(2026, 3, 1, 9, 0, tzinfo=UTC)
    ds_0915 = datetime(2026, 3, 1, 9, 15, tzinfo=UTC)
    ctx = _make_ctx({"NO1-QH-0900": ds_0900, "NO1-QH-0915": ds_0915})

    _record_fill(ctx, "NO1-QH-0900", Side.BUY, Decimal("5"))
    _record_fill(ctx, "NO1-QH-0915", Side.SELL, Decimal("3"))

    result = ctx.get_all_delivery_positions()
    assert len(result) == 2
    assert result[ds_0900].net_mw == Decimal("5")
    assert result[ds_0915].net_mw == Decimal("-3")
