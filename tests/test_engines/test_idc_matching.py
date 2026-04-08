"""Tests for ContinuousMatchingEngine (IDC price-time priority matching)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from nexa_backtest.engines.matching import ContinuousMatchingEngine
from nexa_backtest.types import (
    HistoricalTrade,
    MarketDataUpdate,
    Order,
    Side,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TS = datetime(2026, 3, 1, 9, 0, tzinfo=UTC)
PRODUCT = "NO1-QH-0900"


def _ts(offset_seconds: int = 0) -> datetime:
    return TS + timedelta(seconds=offset_seconds)


def _new_order(
    product_id: str = PRODUCT,
    side: str = "sell",
    price: float = 53.10,
    volume: float = 10.0,
    order_id: str = "hist-1",
) -> MarketDataUpdate:
    return MarketDataUpdate(
        timestamp=_ts(),
        product_id=product_id,
        event_type="new",
        order_id=order_id,
        side=side,
        price_eur_mwh=Decimal(str(price)),
        volume_mw=Decimal(str(volume)),
        remaining_mw=Decimal(str(volume)),
    )


def _cancel_event(
    order_id: str = "hist-1", side: str = "sell", price: float = 53.10
) -> MarketDataUpdate:
    return MarketDataUpdate(
        timestamp=_ts(10),
        product_id=PRODUCT,
        event_type="cancel",
        order_id=order_id,
        side=side,
        price_eur_mwh=Decimal(str(price)),
        volume_mw=Decimal("10"),
        remaining_mw=Decimal("0"),
    )


def _trade_event(
    price: float = 53.10,
    volume: float = 5.0,
    aggressor_side: str | None = "buy",
    trade_id: str = "trade-1",
) -> HistoricalTrade:
    return HistoricalTrade(
        timestamp=_ts(5),
        product_id=PRODUCT,
        trade_id=trade_id,
        price_eur_mwh=Decimal(str(price)),
        volume_mw=Decimal(str(volume)),
        aggressor_side=aggressor_side,
    )


# ---------------------------------------------------------------------------
# Test: algo buy hits historical ask immediately
# ---------------------------------------------------------------------------


class TestAlgoBuyHitsHistoricalAsk:
    def test_fill_at_ask_price_not_algo_price(self) -> None:
        """Aggressor algo buy at 53.50 fills at historical ask 53.10."""
        engine = ContinuousMatchingEngine()
        engine.process_historical_event(_new_order(side="sell", price=53.10, volume=10.0))

        algo_order = Order.buy(PRODUCT, volume_mw=5, price_eur_mwh=Decimal("53.50"))
        fills = engine.place_algo_order(algo_order, _ts(1))

        assert len(fills) == 1
        fill = fills[0]
        assert fill.price == Decimal("53.10")
        assert fill.volume == Decimal("5")
        assert fill.side == Side.BUY

    def test_no_resting_order_after_full_fill(self) -> None:
        engine = ContinuousMatchingEngine()
        engine.process_historical_event(_new_order(side="sell", price=50.0, volume=10.0))

        algo_order = Order.buy(PRODUCT, volume_mw=10, price_eur_mwh=Decimal("55"))
        engine.place_algo_order(algo_order, _ts(1))

        assert algo_order.order_id not in engine._algo_states

    def test_partial_fill_leaves_resting_order(self) -> None:
        """Historical ask has 3 MW; algo wants 10 MW → rest 7 MW."""
        engine = ContinuousMatchingEngine()
        engine.process_historical_event(_new_order(side="sell", price=50.0, volume=3.0))

        algo_order = Order.buy(PRODUCT, volume_mw=10, price_eur_mwh=Decimal("55"))
        fills = engine.place_algo_order(algo_order, _ts(1))

        assert len(fills) == 1
        assert fills[0].volume == Decimal("3")
        assert algo_order.order_id in engine._algo_states
        assert engine._algo_states[algo_order.order_id].remaining_mw == Decimal("7")

    def test_algo_price_below_ask_no_fill(self) -> None:
        engine = ContinuousMatchingEngine()
        engine.process_historical_event(_new_order(side="sell", price=53.10))

        algo_order = Order.buy(PRODUCT, volume_mw=5, price_eur_mwh=Decimal("45.00"))
        fills = engine.place_algo_order(algo_order, _ts(1))

        assert len(fills) == 0
        assert algo_order.order_id in engine._algo_states


# ---------------------------------------------------------------------------
# Test: resting algo order matched by later historical event
# ---------------------------------------------------------------------------


class TestRestingAlgoOrderMatchedLater:
    def test_resting_buy_filled_by_new_hist_sell(self) -> None:
        """Algo places buy at 50.00 with no match. Later a historical sell
        at 49.80 arrives — algo fill at 49.80 (resting order's price)."""
        engine = ContinuousMatchingEngine()

        # No matching asks in the book initially
        algo_order = Order.buy(PRODUCT, volume_mw=5, price_eur_mwh=Decimal("50.00"))
        fills_immediate = engine.place_algo_order(algo_order, _ts(0))
        assert len(fills_immediate) == 0

        # Later, a historical sell at 49.80 enters the book
        hist_sell = MarketDataUpdate(
            timestamp=_ts(30),
            product_id=PRODUCT,
            event_type="new",
            order_id="hist-late",
            side="sell",
            price_eur_mwh=Decimal("49.80"),
            volume_mw=Decimal("10"),
            remaining_mw=Decimal("10"),
        )
        later_fills = engine.process_historical_event(hist_sell)

        assert len(later_fills) == 1
        assert later_fills[0].price == Decimal("49.80")
        assert later_fills[0].side == Side.BUY
        assert algo_order.order_id not in engine._algo_states  # fully filled

    def test_resting_sell_filled_by_new_hist_buy(self) -> None:
        engine = ContinuousMatchingEngine()

        algo_order = Order.sell(PRODUCT, volume_mw=5, price_eur_mwh=Decimal("50.00"))
        engine.place_algo_order(algo_order, _ts(0))

        hist_buy = MarketDataUpdate(
            timestamp=_ts(30),
            product_id=PRODUCT,
            event_type="new",
            order_id="hist-buy",
            side="buy",
            price_eur_mwh=Decimal("51.00"),
            volume_mw=Decimal("10"),
            remaining_mw=Decimal("10"),
        )
        fills = engine.process_historical_event(hist_buy)

        assert len(fills) == 1
        assert fills[0].price == Decimal("51.00")
        assert fills[0].side == Side.SELL


# ---------------------------------------------------------------------------
# Test: no match
# ---------------------------------------------------------------------------


class TestNoMatch:
    def test_algo_buy_far_below_all_asks(self) -> None:
        engine = ContinuousMatchingEngine()
        engine.process_historical_event(_new_order(side="sell", price=53.10))
        engine.process_historical_event(_new_order(side="sell", price=55.00, order_id="hist-2"))

        algo_order = Order.buy(PRODUCT, volume_mw=5, price_eur_mwh=Decimal("45.00"))
        fills = engine.place_algo_order(algo_order, _ts(1))

        assert len(fills) == 0
        assert algo_order.order_id in engine._algo_states

    def test_algo_sell_far_above_all_bids(self) -> None:
        engine = ContinuousMatchingEngine()
        engine.process_historical_event(_new_order(side="buy", price=48.00, order_id="bid-1"))

        algo_order = Order.sell(PRODUCT, volume_mw=5, price_eur_mwh=Decimal("60.00"))
        fills = engine.place_algo_order(algo_order, _ts(1))

        assert len(fills) == 0


# ---------------------------------------------------------------------------
# Test: gate closure cancels resting orders
# ---------------------------------------------------------------------------


class TestGateClosure:
    def test_gate_closure_cancels_resting_order(self) -> None:
        engine = ContinuousMatchingEngine()
        gate_time = _ts(100)
        engine.set_gate_closure(PRODUCT, gate_time)

        algo_order = Order.buy(PRODUCT, volume_mw=5, price_eur_mwh=Decimal("50.00"))
        engine.place_algo_order(algo_order, _ts(0))
        assert algo_order.order_id in engine._algo_states

        # Before gate closure — nothing cancelled
        closed = engine.check_gate_closures(_ts(99))
        assert PRODUCT not in closed
        assert algo_order.order_id in engine._algo_states

        # At gate closure — order cancelled
        closed = engine.check_gate_closures(gate_time)
        assert PRODUCT in closed
        assert algo_order.order_id not in engine._algo_states

    def test_gate_closure_does_not_affect_other_products(self) -> None:
        engine = ContinuousMatchingEngine()
        other_product = "NO1-QH-0915"
        gate_time = _ts(10)
        engine.set_gate_closure(PRODUCT, gate_time)

        order_other = Order.buy(other_product, volume_mw=5, price_eur_mwh=Decimal("50"))
        engine.place_algo_order(order_other, _ts(0))

        engine.check_gate_closures(gate_time)
        assert order_other.order_id in engine._algo_states


# ---------------------------------------------------------------------------
# Test: aggressor_side handling
# ---------------------------------------------------------------------------


class TestAggressorSideHandling:
    def test_aggressor_sell_hits_resting_buy(self) -> None:
        """Trade with aggressor_side=sell → resting buyer was filled."""
        engine = ContinuousMatchingEngine()

        algo_buy = Order.buy(PRODUCT, volume_mw=5, price_eur_mwh=Decimal("53.10"))
        engine.place_algo_order(algo_buy, _ts(0))

        trade = _trade_event(price=53.10, volume=5.0, aggressor_side="sell")
        fills = engine.process_historical_event(trade)

        assert len(fills) == 1
        assert fills[0].price == Decimal("53.10")
        assert fills[0].side == Side.BUY

    def test_aggressor_buy_hits_resting_sell(self) -> None:
        engine = ContinuousMatchingEngine()

        algo_sell = Order.sell(PRODUCT, volume_mw=5, price_eur_mwh=Decimal("50.00"))
        engine.place_algo_order(algo_sell, _ts(0))

        trade = _trade_event(price=50.00, volume=5.0, aggressor_side="buy")
        fills = engine.process_historical_event(trade)

        assert len(fills) == 1
        assert fills[0].side == Side.SELL

    def test_aggressor_sell_does_not_hit_resting_sell(self) -> None:
        """Trade with aggressor_side=sell should not hit another resting sell."""
        engine = ContinuousMatchingEngine()

        algo_sell = Order.sell(PRODUCT, volume_mw=5, price_eur_mwh=Decimal("50.00"))
        engine.place_algo_order(algo_sell, _ts(0))

        trade = _trade_event(price=50.00, volume=5.0, aggressor_side="sell")
        fills = engine.process_historical_event(trade)
        assert len(fills) == 0

    def test_missing_aggressor_side_fallback(self) -> None:
        """Without aggressor_side, any resting order at the trade price is filled."""
        engine = ContinuousMatchingEngine()

        # Resting buy at exactly the trade price
        algo_buy = Order.buy(PRODUCT, volume_mw=5, price_eur_mwh=Decimal("50.00"))
        engine.place_algo_order(algo_buy, _ts(0))

        trade = _trade_event(price=50.00, volume=5.0, aggressor_side=None)
        fills = engine.process_historical_event(trade)

        assert len(fills) == 1

    def test_missing_aggressor_side_no_match_if_price_wrong(self) -> None:
        """Fallback: resting buy at 48.00 should not fill on trade at 50.00."""
        engine = ContinuousMatchingEngine()

        algo_buy = Order.buy(PRODUCT, volume_mw=5, price_eur_mwh=Decimal("48.00"))
        engine.place_algo_order(algo_buy, _ts(0))

        trade = _trade_event(price=50.00, volume=5.0, aggressor_side=None)
        fills = engine.process_historical_event(trade)

        assert len(fills) == 0


# ---------------------------------------------------------------------------
# Test: cancel_algo_order
# ---------------------------------------------------------------------------


class TestCancelAlgoOrder:
    def test_cancel_resting_order(self) -> None:
        engine = ContinuousMatchingEngine()
        order = Order.buy(PRODUCT, volume_mw=5, price_eur_mwh=Decimal("50"))
        engine.place_algo_order(order, _ts(0))
        assert order.order_id in engine._algo_states

        result = engine.cancel_algo_order(order.order_id)
        assert result.status == "cancelled"
        assert order.order_id not in engine._algo_states

    def test_cancel_unknown_order_returns_not_found(self) -> None:
        engine = ContinuousMatchingEngine()
        result = engine.cancel_algo_order("nonexistent-id")
        assert result.status == "not_found"


# ---------------------------------------------------------------------------
# Test: get_orderbook reflects historical book state
# ---------------------------------------------------------------------------


class TestGetOrderBook:
    def test_empty_book_initially(self) -> None:
        engine = ContinuousMatchingEngine()
        book = engine.get_orderbook(PRODUCT, _ts())
        assert book.best_bid is None
        assert book.best_ask is None

    def test_book_updated_after_new_order(self) -> None:
        engine = ContinuousMatchingEngine()
        engine.process_historical_event(_new_order(side="sell", price=53.10, order_id="ask-1"))
        engine.process_historical_event(_new_order(side="buy", price=49.50, order_id="bid-1"))

        book = engine.get_orderbook(PRODUCT, _ts())
        assert book.best_ask is not None
        assert book.best_bid is not None
        assert book.best_ask.price == Decimal("53.10")
        assert book.best_bid.price == Decimal("49.50")

    def test_book_spread_and_mid(self) -> None:
        engine = ContinuousMatchingEngine()
        engine.process_historical_event(_new_order(side="sell", price=53.10, order_id="ask-1"))
        engine.process_historical_event(_new_order(side="buy", price=49.50, order_id="bid-1"))

        book = engine.get_orderbook(PRODUCT, _ts())
        assert book.spread == Decimal("53.10") - Decimal("49.50")
        assert book.mid_price is not None

    def test_cancelled_order_removed_from_book(self) -> None:
        engine = ContinuousMatchingEngine()
        engine.process_historical_event(_new_order(side="sell", price=53.10, order_id="ask-1"))
        engine.process_historical_event(_cancel_event(order_id="ask-1", side="sell", price=53.10))

        book = engine.get_orderbook(PRODUCT, _ts())
        assert book.best_ask is None


# ---------------------------------------------------------------------------
# Test: process_historical_event modify / trade
# ---------------------------------------------------------------------------


class TestProcessHistoricalEventModifyTrade:
    def test_modify_updates_price(self) -> None:
        engine = ContinuousMatchingEngine()
        engine.process_historical_event(_new_order(side="sell", price=55.0, order_id="o1"))

        modify = MarketDataUpdate(
            timestamp=_ts(5),
            product_id=PRODUCT,
            event_type="modify",
            order_id="o1",
            side="sell",
            price_eur_mwh=Decimal("53.00"),
            volume_mw=Decimal("10"),
            remaining_mw=Decimal("10"),
        )
        engine.process_historical_event(modify)

        book = engine.get_orderbook(PRODUCT, _ts(5))
        assert book.best_ask is not None
        assert book.best_ask.price == Decimal("53.00")

    def test_trade_updates_remaining(self) -> None:
        engine = ContinuousMatchingEngine()
        engine.process_historical_event(
            _new_order(side="sell", price=53.10, order_id="o1", volume=10.0)
        )

        trade_update = MarketDataUpdate(
            timestamp=_ts(5),
            product_id=PRODUCT,
            event_type="trade",
            order_id="o1",
            side="sell",
            price_eur_mwh=Decimal("53.10"),
            volume_mw=Decimal("3"),
            remaining_mw=Decimal("7"),
        )
        engine.process_historical_event(trade_update)

        assert engine._hist_orders["o1"].remaining_mw == Decimal("7")

    def test_trade_with_zero_remaining_removes_order(self) -> None:
        engine = ContinuousMatchingEngine()
        engine.process_historical_event(_new_order(side="sell", price=53.10, order_id="o1"))

        trade_update = MarketDataUpdate(
            timestamp=_ts(5),
            product_id=PRODUCT,
            event_type="trade",
            order_id="o1",
            side="sell",
            price_eur_mwh=Decimal("53.10"),
            volume_mw=Decimal("10"),
            remaining_mw=Decimal("0"),
        )
        engine.process_historical_event(trade_update)
        assert "o1" not in engine._hist_orders


# ---------------------------------------------------------------------------
# Test: multiple products
# ---------------------------------------------------------------------------


class TestMultipleProducts:
    def test_orders_isolated_by_product(self) -> None:
        engine = ContinuousMatchingEngine()
        p1, p2 = "NO1-QH-0900", "NO1-QH-0915"

        ask_p1 = _new_order(product_id=p1, side="sell", price=50.0, order_id="ask-p1")
        engine.process_historical_event(ask_p1)

        algo_buy_p2 = Order.buy(p2, volume_mw=5, price_eur_mwh=Decimal("55"))
        fills = engine.place_algo_order(algo_buy_p2, _ts(1))

        # Ask on p1 should not match a buy on p2
        assert len(fills) == 0
