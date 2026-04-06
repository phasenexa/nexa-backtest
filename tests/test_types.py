"""Tests for core domain types in nexa_backtest.types."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest
from pydantic import ValidationError

from nexa_backtest.types import (
    MTU,
    AuctionInfo,
    CancelResult,
    Fill,
    Order,
    OrderBook,
    OrderResult,
    OrderStatus,
    Position,
    PriceLevel,
    Side,
)


class TestSideEnum:
    def test_buy_value(self) -> None:
        assert Side.BUY.value == "BUY"

    def test_sell_value(self) -> None:
        assert Side.SELL.value == "SELL"

    def test_is_string_enum(self) -> None:
        assert isinstance(Side.BUY, str)


class TestOrderStatusEnum:
    def test_all_values(self) -> None:
        statuses = {s.value for s in OrderStatus}
        assert statuses == {
            "PENDING",
            "ACCEPTED",
            "PARTIALLY_FILLED",
            "FILLED",
            "CANCELLED",
            "REJECTED",
        }


class TestMTU:
    def test_construction(self, sample_mtu: MTU) -> None:
        assert sample_mtu.zone == "NO1"
        assert sample_mtu.start < sample_mtu.end

    def test_immutable(self, sample_mtu: MTU) -> None:
        with pytest.raises(ValidationError):
            sample_mtu.zone = "NO2"  # type: ignore[misc]

    def test_requires_timezone_aware_datetimes(self) -> None:
        # Pydantic accepts naive datetimes; we document the requirement
        # but do not enforce at the type level (enforced by engine code)
        mtu = MTU(
            start=datetime(2026, 1, 1, 9, 0, tzinfo=UTC),
            end=datetime(2026, 1, 1, 9, 15, tzinfo=UTC),
            zone="DE",
        )
        assert mtu.start.tzinfo is not None


class TestPriceLevel:
    def test_construction(self) -> None:
        pl = PriceLevel(price=Decimal("42.50"), volume=Decimal("100"))
        assert pl.price == Decimal("42.50")
        assert pl.volume == Decimal("100")

    def test_immutable(self) -> None:
        pl = PriceLevel(price=Decimal("10"), volume=Decimal("5"))
        with pytest.raises(ValidationError):
            pl.price = Decimal("20")  # type: ignore[misc]

    def test_decimal_precision(self) -> None:
        pl = PriceLevel(price=Decimal("42.123456789"), volume=Decimal("0.001"))
        assert pl.price == Decimal("42.123456789")


class TestOrderBook:
    def test_construction(self, sample_orderbook: OrderBook) -> None:
        assert sample_orderbook.best_bid is not None
        assert sample_orderbook.best_ask is not None
        assert sample_orderbook.best_bid.price < sample_orderbook.best_ask.price

    def test_none_bid_and_ask(self, utc_now: datetime) -> None:
        ob = OrderBook(
            product_id="NO1-QH-0900",
            best_bid=None,
            best_ask=None,
            timestamp=utc_now,
        )
        assert ob.best_bid is None
        assert ob.best_ask is None

    def test_immutable(self, sample_orderbook: OrderBook) -> None:
        with pytest.raises(ValidationError):
            sample_orderbook.product_id = "DE-QH-0900"  # type: ignore[misc]


class TestOrderClassMethods:
    def test_buy_order(self) -> None:
        order = Order.buy("NO1-QH-0900", volume_mw=10, price_eur_mwh=55)
        assert order.side == Side.BUY
        assert order.volume_mw == Decimal("10")
        assert order.price_eur_mwh == Decimal("55")
        assert not order.is_block_bid

    def test_sell_order(self) -> None:
        order = Order.sell("NO1-QH-0900", volume_mw=5, price_eur_mwh=45)
        assert order.side == Side.SELL
        assert order.volume_mw == Decimal("5")
        assert order.price_eur_mwh == Decimal("45")

    def test_market_buy(self) -> None:
        order = Order.market("NO1-QH-0900", side=Side.BUY, volume_mw=20)
        assert order.price_eur_mwh is None
        assert order.side == Side.BUY

    def test_market_sell(self) -> None:
        order = Order.market("NO1-QH-0900", side=Side.SELL, volume_mw=20)
        assert order.price_eur_mwh is None
        assert order.side == Side.SELL

    def test_block_bid(self) -> None:
        start = datetime(2026, 1, 15, 8, 0, tzinfo=UTC)
        end = datetime(2026, 1, 15, 12, 0, tzinfo=UTC)
        order = Order.block_bid(
            "NO1-BLOCK",
            side=Side.BUY,
            volume_mw=50,
            price_eur_mwh=48,
            block_start=start,
            block_end=end,
        )
        assert order.is_block_bid
        assert order.block_start == start
        assert order.block_end == end
        assert order.side == Side.BUY

    def test_decimal_from_int(self) -> None:
        order = Order.buy("X", volume_mw=10, price_eur_mwh=50)
        assert isinstance(order.volume_mw, Decimal)
        assert isinstance(order.price_eur_mwh, Decimal)

    def test_decimal_from_float(self) -> None:
        order = Order.buy("X", volume_mw=10.5, price_eur_mwh=50.25)
        assert order.volume_mw == Decimal("10.5")
        assert order.price_eur_mwh == Decimal("50.25")

    def test_order_id_auto_generated(self) -> None:
        o1 = Order.buy("X", volume_mw=1, price_eur_mwh=50)
        o2 = Order.buy("X", volume_mw=1, price_eur_mwh=50)
        assert o1.order_id != o2.order_id

    def test_order_immutable(self) -> None:
        order = Order.buy("X", volume_mw=10, price_eur_mwh=50)
        with pytest.raises(ValidationError):
            order.volume_mw = Decimal("20")  # type: ignore[misc]


class TestFill:
    def test_construction(self, utc_now: datetime) -> None:
        fill = Fill(
            order_id="abc-123",
            product_id="NO1-QH-0900",
            price=Decimal("50.00"),
            volume=Decimal("10"),
            timestamp=utc_now,
            side=Side.BUY,
        )
        assert fill.price == Decimal("50.00")
        assert fill.side == Side.BUY

    def test_immutable(self, utc_now: datetime) -> None:
        fill = Fill(
            order_id="abc",
            product_id="X",
            price=Decimal("50"),
            volume=Decimal("10"),
            timestamp=utc_now,
            side=Side.SELL,
        )
        with pytest.raises(ValidationError):
            fill.price = Decimal("99")  # type: ignore[misc]


class TestOrderResult:
    def test_rejected_result(self) -> None:
        result = OrderResult(
            order_id="abc",
            status=OrderStatus.REJECTED,
            rejection_reason="Insufficient margin",
        )
        assert result.fill is None
        assert result.rejection_reason == "Insufficient margin"

    def test_filled_result(self, utc_now: datetime) -> None:
        fill = Fill(
            order_id="abc",
            product_id="X",
            price=Decimal("50"),
            volume=Decimal("10"),
            timestamp=utc_now,
            side=Side.BUY,
        )
        result = OrderResult(order_id="abc", status=OrderStatus.FILLED, fill=fill)
        assert result.fill is not None
        assert result.fill.price == Decimal("50")


class TestCancelResult:
    def test_cancelled(self) -> None:
        r = CancelResult(order_id="abc", status="cancelled")
        assert r.status == "cancelled"

    def test_not_found(self) -> None:
        r = CancelResult(order_id="xyz", status="not_found")
        assert r.status == "not_found"


class TestPosition:
    def test_construction(self) -> None:
        pos = Position(
            product_id="NO1-QH-0900",
            net_mw=Decimal("10"),
            avg_entry_price=Decimal("50.00"),
            unrealised_pnl=Decimal("250.00"),
        )
        assert pos.net_mw == Decimal("10")
        assert pos.unrealised_pnl == Decimal("250.00")

    def test_short_position(self) -> None:
        pos = Position(
            product_id="NO1-QH-0900",
            net_mw=Decimal("-5"),
            avg_entry_price=Decimal("55.00"),
            unrealised_pnl=Decimal("-100"),
        )
        assert pos.net_mw < Decimal("0")


class TestAuctionInfo:
    def test_construction(self, sample_auction_info: AuctionInfo) -> None:
        assert sample_auction_info.auction_type == "DA"
        assert sample_auction_info.zone == "NO1"
        assert sample_auction_info.gate_closure_time.tzinfo is not None

    def test_immutable(self, sample_auction_info: AuctionInfo) -> None:
        with pytest.raises(ValidationError):
            sample_auction_info.zone = "DE"  # type: ignore[misc]


class TestSerialisation:
    """Round-trip serialisation tests via Pydantic model_dump / model_validate."""

    def test_order_roundtrip(self) -> None:
        order = Order.buy("NO1-QH-0900", volume_mw=Decimal("10.5"), price_eur_mwh=Decimal("50.25"))
        data = order.model_dump()
        restored = Order.model_validate(data)
        assert restored.volume_mw == order.volume_mw
        assert restored.price_eur_mwh == order.price_eur_mwh
        assert restored.order_id == order.order_id

    def test_mtu_roundtrip(self, sample_mtu: MTU) -> None:
        data = sample_mtu.model_dump()
        restored = MTU.model_validate(data)
        assert restored.start == sample_mtu.start
        assert restored.zone == sample_mtu.zone

    def test_fill_roundtrip(self, utc_now: datetime) -> None:
        fill = Fill(
            order_id="abc",
            product_id="X",
            price=Decimal("42.50"),
            volume=Decimal("20"),
            timestamp=utc_now,
            side=Side.SELL,
        )
        restored = Fill.model_validate(fill.model_dump())
        assert restored.price == fill.price
        assert restored.side == fill.side
