"""Tests for the DA auction matching engine."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from nexa_backtest.engines.matching import DAAuctionMatcher
from nexa_backtest.types import Order, OrderStatus, Side

CLEARING = Decimal("50.00")
TS = datetime(2026, 1, 15, 12, 0, tzinfo=UTC)


@pytest.fixture
def matcher() -> DAAuctionMatcher:
    return DAAuctionMatcher(clearing_price=CLEARING, fill_timestamp=TS)


class TestDAAuctionMatcherConstruction:
    def test_clearing_price_stored(self) -> None:
        m = DAAuctionMatcher(clearing_price=Decimal("42.50"), fill_timestamp=TS)
        assert m.clearing_price == Decimal("42.50")

    def test_naive_fill_timestamp_raises(self) -> None:
        naive = datetime(2026, 1, 15, 12, 0)
        with pytest.raises(ValueError, match="timezone-aware"):
            DAAuctionMatcher(clearing_price=CLEARING, fill_timestamp=naive)

    def test_default_fill_timestamp(self) -> None:
        m = DAAuctionMatcher(clearing_price=CLEARING)
        assert m._fill_timestamp.tzinfo is not None


class TestBuyOrderMatching:
    def test_buy_above_clearing_fills(self, matcher: DAAuctionMatcher) -> None:
        order = Order.buy("NO1-QH-0900", volume_mw=10, price_eur_mwh=55)
        result = matcher.match(order)
        assert result.status == OrderStatus.FILLED
        assert result.fill is not None
        assert result.fill.price == CLEARING

    def test_buy_at_clearing_price_fills(self, matcher: DAAuctionMatcher) -> None:
        order = Order.buy("NO1-QH-0900", volume_mw=10, price_eur_mwh=50)
        result = matcher.match(order)
        assert result.status == OrderStatus.FILLED

    def test_buy_below_clearing_rejects(self, matcher: DAAuctionMatcher) -> None:
        order = Order.buy("NO1-QH-0900", volume_mw=10, price_eur_mwh=45)
        result = matcher.match(order)
        assert result.status == OrderStatus.REJECTED
        assert result.fill is None
        assert result.rejection_reason is not None

    def test_buy_fill_at_clearing_not_order_price(self, matcher: DAAuctionMatcher) -> None:
        """Fill should be at clearing price, not at order limit."""
        order = Order.buy("NO1-QH-0900", volume_mw=10, price_eur_mwh=99)
        result = matcher.match(order)
        assert result.fill is not None
        assert result.fill.price == CLEARING

    def test_buy_fill_volume_matches_order(self, matcher: DAAuctionMatcher) -> None:
        order = Order.buy("NO1-QH-0900", volume_mw=Decimal("7.5"), price_eur_mwh=55)
        result = matcher.match(order)
        assert result.fill is not None
        assert result.fill.volume == Decimal("7.5")

    def test_buy_fill_side_is_buy(self, matcher: DAAuctionMatcher) -> None:
        order = Order.buy("NO1-QH-0900", volume_mw=10, price_eur_mwh=55)
        result = matcher.match(order)
        assert result.fill is not None
        assert result.fill.side == Side.BUY


class TestSellOrderMatching:
    def test_sell_below_clearing_fills(self, matcher: DAAuctionMatcher) -> None:
        order = Order.sell("NO1-QH-0900", volume_mw=10, price_eur_mwh=45)
        result = matcher.match(order)
        assert result.status == OrderStatus.FILLED
        assert result.fill is not None
        assert result.fill.price == CLEARING

    def test_sell_at_clearing_price_fills(self, matcher: DAAuctionMatcher) -> None:
        order = Order.sell("NO1-QH-0900", volume_mw=10, price_eur_mwh=50)
        result = matcher.match(order)
        assert result.status == OrderStatus.FILLED

    def test_sell_above_clearing_rejects(self, matcher: DAAuctionMatcher) -> None:
        order = Order.sell("NO1-QH-0900", volume_mw=10, price_eur_mwh=55)
        result = matcher.match(order)
        assert result.status == OrderStatus.REJECTED
        assert result.fill is None

    def test_sell_fill_at_clearing_not_order_price(self, matcher: DAAuctionMatcher) -> None:
        order = Order.sell("NO1-QH-0900", volume_mw=10, price_eur_mwh=1)
        result = matcher.match(order)
        assert result.fill is not None
        assert result.fill.price == CLEARING

    def test_sell_fill_side_is_sell(self, matcher: DAAuctionMatcher) -> None:
        order = Order.sell("NO1-QH-0900", volume_mw=10, price_eur_mwh=45)
        result = matcher.match(order)
        assert result.fill is not None
        assert result.fill.side == Side.SELL


class TestZeroVolumeRejection:
    def test_zero_volume_buy_rejected(self, matcher: DAAuctionMatcher) -> None:
        order = Order(
            product_id="NO1-QH-0900",
            side=Side.BUY,
            volume_mw=Decimal("0"),
            price_eur_mwh=Decimal("55"),
        )
        result = matcher.match(order)
        assert result.status == OrderStatus.REJECTED
        assert "positive" in (result.rejection_reason or "").lower()

    def test_negative_volume_rejected(self, matcher: DAAuctionMatcher) -> None:
        order = Order(
            product_id="NO1-QH-0900",
            side=Side.BUY,
            volume_mw=Decimal("-5"),
            price_eur_mwh=Decimal("55"),
        )
        result = matcher.match(order)
        assert result.status == OrderStatus.REJECTED


class TestMarketOrders:
    def test_market_buy_always_fills(self, matcher: DAAuctionMatcher) -> None:
        order = Order.market("NO1-QH-0900", side=Side.BUY, volume_mw=10)
        result = matcher.match(order)
        assert result.status == OrderStatus.FILLED
        assert result.fill is not None
        assert result.fill.price == CLEARING

    def test_market_sell_always_fills(self, matcher: DAAuctionMatcher) -> None:
        order = Order.market("NO1-QH-0900", side=Side.SELL, volume_mw=5)
        result = matcher.match(order)
        assert result.status == OrderStatus.FILLED


class TestExactPriceMatch:
    def test_buy_exactly_at_clearing_fills(self, matcher: DAAuctionMatcher) -> None:
        order = Order.buy("NO1-QH-0900", volume_mw=10, price_eur_mwh=Decimal("50.00"))
        result = matcher.match(order)
        assert result.status == OrderStatus.FILLED

    def test_sell_exactly_at_clearing_fills(self, matcher: DAAuctionMatcher) -> None:
        order = Order.sell("NO1-QH-0900", volume_mw=10, price_eur_mwh=Decimal("50.00"))
        result = matcher.match(order)
        assert result.status == OrderStatus.FILLED

    def test_buy_one_cent_below_clearing_rejects(self) -> None:
        matcher = DAAuctionMatcher(clearing_price=Decimal("50.00"), fill_timestamp=TS)
        order = Order.buy("NO1-QH-0900", volume_mw=10, price_eur_mwh=Decimal("49.99"))
        result = matcher.match(order)
        assert result.status == OrderStatus.REJECTED

    def test_sell_one_cent_above_clearing_rejects(self) -> None:
        matcher = DAAuctionMatcher(clearing_price=Decimal("50.00"), fill_timestamp=TS)
        order = Order.sell("NO1-QH-0900", volume_mw=10, price_eur_mwh=Decimal("50.01"))
        result = matcher.match(order)
        assert result.status == OrderStatus.REJECTED


class TestFillMetadata:
    def test_fill_timestamp_matches_matcher(self, matcher: DAAuctionMatcher) -> None:
        order = Order.buy("NO1-QH-0900", volume_mw=10, price_eur_mwh=55)
        result = matcher.match(order)
        assert result.fill is not None
        assert result.fill.timestamp == TS

    def test_fill_order_id_matches(self, matcher: DAAuctionMatcher) -> None:
        order = Order.buy("NO1-QH-0900", volume_mw=10, price_eur_mwh=55)
        result = matcher.match(order)
        assert result.fill is not None
        assert result.fill.order_id == order.order_id

    def test_fill_product_id_matches(self, matcher: DAAuctionMatcher) -> None:
        order = Order.buy("NO1-QH-1500", volume_mw=10, price_eur_mwh=55)
        result = matcher.match(order)
        assert result.fill is not None
        assert result.fill.product_id == "NO1-QH-1500"


class TestMultipleOrders:
    def test_multiple_orders_same_product(self) -> None:
        matcher = DAAuctionMatcher(clearing_price=Decimal("50.00"), fill_timestamp=TS)
        orders = [
            Order.buy("NO1-QH-0900", volume_mw=10, price_eur_mwh=55),
            Order.buy("NO1-QH-0900", volume_mw=5, price_eur_mwh=45),  # reject
            Order.sell("NO1-QH-0900", volume_mw=8, price_eur_mwh=45),
        ]
        results = matcher.match_many(orders)
        assert len(results) == 3
        assert results[0].status == OrderStatus.FILLED
        assert results[1].status == OrderStatus.REJECTED
        assert results[2].status == OrderStatus.FILLED

    def test_clearing_price_unchanged_after_multiple_orders(self) -> None:
        """Price-taker assumption: clearing price must not change."""
        matcher = DAAuctionMatcher(clearing_price=Decimal("50.00"), fill_timestamp=TS)
        for _ in range(10):
            order = Order.buy("X", volume_mw=1000, price_eur_mwh=99)
            result = matcher.match(order)
            assert result.fill is not None
            assert result.fill.price == Decimal("50.00")

    def test_different_products_independent(self) -> None:
        matcher = DAAuctionMatcher(clearing_price=Decimal("50.00"), fill_timestamp=TS)
        o1 = Order.buy("NO1-QH-0900", volume_mw=10, price_eur_mwh=55)
        o2 = Order.buy("NO1-QH-0915", volume_mw=10, price_eur_mwh=55)
        r1 = matcher.match(o1)
        r2 = matcher.match(o2)
        assert r1.status == OrderStatus.FILLED
        assert r2.status == OrderStatus.FILLED
        assert r1.fill is not None
        assert r2.fill is not None
        assert r1.fill.product_id == "NO1-QH-0900"
        assert r2.fill.product_id == "NO1-QH-0915"
