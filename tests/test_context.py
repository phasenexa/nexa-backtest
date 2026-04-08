"""Tests for TradingContext protocol structural compliance."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

from nexa_backtest.context import SignalValue, TradingContext
from nexa_backtest.types import (
    MTU,
    CancelResult,
    Order,
    OrderBook,
    OrderResult,
    OrderStatus,
    Position,
    PriceLevel,
)


class _MockContext:
    """A minimal concrete implementation of TradingContext for testing.

    This class satisfies the protocol structurally. If mypy is run with
    --strict it will verify that this class implements all required methods
    with the correct signatures.
    """

    def now(self) -> datetime:
        return datetime(2026, 1, 15, 12, 0, tzinfo=UTC)

    def time_to_gate_closure(self, product_id: str) -> timedelta:
        return timedelta(hours=2)

    def current_mtu(self) -> MTU:
        return MTU(
            start=datetime(2026, 1, 15, 12, 0, tzinfo=UTC),
            end=datetime(2026, 1, 15, 12, 15, tzinfo=UTC),
            zone="NO1",
        )

    def get_orderbook(self, product_id: str) -> OrderBook:
        return OrderBook(
            product_id=product_id,
            bids=[PriceLevel(price=Decimal("49"), volume=Decimal("10"))],
            asks=[PriceLevel(price=Decimal("51"), volume=Decimal("10"))],
            timestamp=self.now(),
        )

    def get_best_bid(self, product_id: str) -> PriceLevel | None:
        return PriceLevel(price=Decimal("49"), volume=Decimal("10"))

    def get_best_ask(self, product_id: str) -> PriceLevel | None:
        return PriceLevel(price=Decimal("51"), volume=Decimal("10"))

    def get_last_price(self, product_id: str) -> Decimal | None:
        return Decimal("50.00")

    def get_vwap(self, product_id: str) -> Decimal | None:
        return Decimal("50.10")

    def place_order(self, order: Order) -> OrderResult:
        return OrderResult(order_id=order.order_id, status=OrderStatus.ACCEPTED)

    def cancel_order(self, order_id: str) -> CancelResult:
        return CancelResult(order_id=order_id, status="cancelled")

    def modify_order(self, order_id: str, **changes: Any) -> OrderResult:
        return OrderResult(order_id=order_id, status=OrderStatus.ACCEPTED)

    def get_position(self, product_id: str) -> Position:
        return Position(
            product_id=product_id,
            net_mw=Decimal("0"),
            avg_entry_price=Decimal("0"),
            unrealised_pnl=Decimal("0"),
        )

    def get_all_positions(self) -> dict[str, Position]:
        return {}

    def get_unrealised_pnl(self) -> Decimal:
        return Decimal("0")

    def get_signal(self, name: str) -> SignalValue:
        return SignalValue(name=name, timestamp=self.now(), value=42.0)

    def get_signal_history(self, name: str, lookback: int) -> list[SignalValue]:
        return [
            SignalValue(name=name, timestamp=self.now(), value=float(i)) for i in range(lookback)
        ]

    def predict(self, model_name: str, features: dict[str, Any]) -> Any:
        return 0.0

    def log(self, message: str, level: str = "info") -> None:
        pass


def _accepts_context(ctx: TradingContext) -> None:
    """Type-level assertion that _MockContext satisfies TradingContext."""
    _ = ctx.now()


class TestTradingContextProtocol:
    """Verify that a compliant implementation satisfies the protocol at runtime."""

    def test_mock_satisfies_protocol(self) -> None:
        ctx = _MockContext()
        # If mypy is run with --strict, this will catch protocol non-compliance
        _accepts_context(ctx)

    def test_now_returns_aware_datetime(self) -> None:
        ctx = _MockContext()
        assert ctx.now().tzinfo is not None

    def test_time_to_gate_closure(self) -> None:
        ctx = _MockContext()
        td = ctx.time_to_gate_closure("NO1-QH-0900")
        assert isinstance(td, timedelta)
        assert td.total_seconds() > 0

    def test_current_mtu(self) -> None:
        ctx = _MockContext()
        mtu = ctx.current_mtu()
        assert isinstance(mtu, MTU)
        assert mtu.start < mtu.end

    def test_get_orderbook(self) -> None:
        ctx = _MockContext()
        ob = ctx.get_orderbook("NO1-QH-0900")
        assert ob.product_id == "NO1-QH-0900"
        assert ob.best_bid is not None
        assert ob.best_ask is not None

    def test_get_best_bid(self) -> None:
        ctx = _MockContext()
        bid = ctx.get_best_bid("NO1-QH-0900")
        assert bid is not None
        assert isinstance(bid.price, Decimal)

    def test_get_best_ask(self) -> None:
        ctx = _MockContext()
        ask = ctx.get_best_ask("NO1-QH-0900")
        assert ask is not None
        assert isinstance(ask.price, Decimal)

    def test_get_last_price(self) -> None:
        ctx = _MockContext()
        price = ctx.get_last_price("NO1-QH-0900")
        assert price is not None
        assert isinstance(price, Decimal)

    def test_get_vwap(self) -> None:
        ctx = _MockContext()
        vwap = ctx.get_vwap("NO1-QH-0900")
        assert vwap is not None

    def test_place_order(self) -> None:
        ctx = _MockContext()
        order = Order.buy("NO1-QH-0900", volume_mw=10, price_eur_mwh=55)
        result = ctx.place_order(order)
        assert result.order_id == order.order_id

    def test_cancel_order(self) -> None:
        ctx = _MockContext()
        result = ctx.cancel_order("some-id")
        assert result.status == "cancelled"

    def test_modify_order(self) -> None:
        ctx = _MockContext()
        result = ctx.modify_order("some-id", price_eur_mwh=Decimal("48"))
        assert result.order_id == "some-id"

    def test_get_position(self) -> None:
        ctx = _MockContext()
        pos = ctx.get_position("NO1-QH-0900")
        assert isinstance(pos, Position)
        assert pos.product_id == "NO1-QH-0900"

    def test_get_all_positions(self) -> None:
        ctx = _MockContext()
        positions = ctx.get_all_positions()
        assert isinstance(positions, dict)

    def test_get_unrealised_pnl(self) -> None:
        ctx = _MockContext()
        pnl = ctx.get_unrealised_pnl()
        assert isinstance(pnl, Decimal)

    def test_get_signal(self) -> None:
        ctx = _MockContext()
        signal = ctx.get_signal("wind_forecast")
        assert isinstance(signal, SignalValue)
        assert signal.name == "wind_forecast"

    def test_get_signal_history(self) -> None:
        ctx = _MockContext()
        history = ctx.get_signal_history("wind_forecast", lookback=3)
        assert len(history) == 3

    def test_predict(self) -> None:
        ctx = _MockContext()
        result = ctx.predict("price_model", {"wind": 12.5, "load": 40000.0})
        assert result == 0.0

    def test_log(self) -> None:
        ctx = _MockContext()
        ctx.log("test message")
        ctx.log("warning message", level="warning")


class TestSignalValue:
    def test_construction(self) -> None:
        ts = datetime(2026, 1, 15, 12, 0, tzinfo=UTC)
        sv = SignalValue(name="wind", timestamp=ts, value=14500.0)
        assert sv.name == "wind"
        assert sv.value == 14500.0
        assert sv.timestamp.tzinfo is not None

    def test_immutable(self) -> None:
        import pytest
        from pydantic import ValidationError

        ts = datetime(2026, 1, 15, tzinfo=UTC)
        sv = SignalValue(name="x", timestamp=ts, value=1.0)
        with pytest.raises(ValidationError):
            sv.value = 2.0  # type: ignore[misc]
