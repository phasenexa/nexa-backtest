"""Matching engines for simulated order execution.

This module provides both the DA auction matching engine and the IDC
continuous matching engine.

The price-taker assumption applies throughout: the algo's orders do not
affect the historical clearing prices or order flow.  This is realistic for
most participants in European power markets.

**IDC price-taker caveat**: when the algo has a resting order and a
historical trade occurs at a matching price, *both* the historical trade and
the algo fill are assumed to happen independently.  In reality one of them
would have consumed the liquidity.  For small algo volumes relative to
market volume this is acceptable.  Document this limitation when building
production strategies.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal

from nexa_backtest.types import (
    CancelResult,
    Fill,
    HistoricalTrade,
    MarketDataUpdate,
    MarketEvent,
    Order,
    OrderBook,
    OrderResult,
    OrderStatus,
    PriceLevel,
    Side,
)


class DAAuctionMatcher:
    """Simulates matching for a day-ahead price auction.

    Uses the price-taker assumption: the algo's bid volume does not affect
    the auction clearing price. The clearing price is taken from historical
    data.

    Matching rules:
    - Buy order: filled if ``order.price_eur_mwh >= clearing_price``.
    - Sell order: filled if ``order.price_eur_mwh <= clearing_price``.
    - All fills occur at the clearing price (uniform price auction).
    - Orders with zero volume are rejected.
    - Market orders (no price limit) are always filled.

    Args:
        clearing_price: Historical clearing price in EUR/MWh for the
            product/MTU being matched.
        fill_timestamp: Timezone-aware timestamp to stamp on resulting fills.
            Defaults to the current UTC time if not provided.

    Example:
        >>> from decimal import Decimal
        >>> from datetime import datetime, timezone
        >>> from nexa_backtest.types import Order
        >>> ts = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
        >>> matcher = DAAuctionMatcher(clearing_price=Decimal("50.00"), fill_timestamp=ts)
        >>> order = Order.buy("NO1-H-12", volume_mw=10, price_eur_mwh=55)
        >>> result = matcher.match(order)
        >>> result.status.value
        'FILLED'
    """

    def __init__(
        self,
        clearing_price: Decimal,
        fill_timestamp: datetime | None = None,
    ) -> None:
        if fill_timestamp is not None and fill_timestamp.tzinfo is None:
            raise ValueError("fill_timestamp must be timezone-aware")
        self._clearing_price = clearing_price
        self._fill_timestamp = fill_timestamp or datetime.now(tz=UTC)

    @property
    def clearing_price(self) -> Decimal:
        """Historical clearing price used for matching.

        Returns:
            Clearing price in EUR/MWh.
        """
        return self._clearing_price

    def match(self, order: Order) -> OrderResult:
        """Attempt to match an order against the auction clearing price.

        Args:
            order: The order to match.

        Returns:
            ``OrderResult`` with status ``FILLED`` (and fill details) or
            ``REJECTED`` (with rejection reason).

        Raises:
            MatchingError: If the order is structurally invalid in a way that
                indicates a programming error rather than a business rejection
                (e.g., negative volume).
        """
        if order.volume_mw <= Decimal("0"):
            return OrderResult(
                order_id=order.order_id,
                status=OrderStatus.REJECTED,
                rejection_reason="Volume must be positive",
            )

        filled = self._is_filled(order)

        if not filled:
            return OrderResult(
                order_id=order.order_id,
                status=OrderStatus.REJECTED,
                rejection_reason=(
                    f"Order price {order.price_eur_mwh} does not meet "
                    f"clearing price {self._clearing_price}"
                ),
            )

        fill = Fill(
            order_id=order.order_id,
            product_id=order.product_id,
            price=self._clearing_price,
            volume=order.volume_mw,
            timestamp=self._fill_timestamp,
            side=order.side,
        )
        return OrderResult(
            order_id=order.order_id,
            status=OrderStatus.FILLED,
            fill=fill,
        )

    def _is_filled(self, order: Order) -> bool:
        """Determine whether an order should be filled.

        Args:
            order: The order to evaluate.

        Returns:
            ``True`` if the order meets the clearing price condition.
        """
        # Market orders always fill
        if order.price_eur_mwh is None:
            return True

        if order.side == Side.BUY:
            return order.price_eur_mwh >= self._clearing_price
        else:
            return order.price_eur_mwh <= self._clearing_price

    def match_many(self, orders: list[Order]) -> list[OrderResult]:
        """Match multiple orders against the same clearing price.

        Each order is matched independently. The clearing price is not
        affected by other orders in the batch (price-taker assumption).

        Args:
            orders: List of orders to match.

        Returns:
            List of results in the same order as the input.
        """
        return [self.match(order) for order in orders]


# ---------------------------------------------------------------------------
# IDC continuous matching engine
# ---------------------------------------------------------------------------


@dataclass
class _HistoricalOrder:
    """Internal representation of a resting historical order."""

    order_id: str
    product_id: str
    side: str  # "buy" or "sell"
    price: Decimal
    remaining_mw: Decimal


@dataclass
class _AlgoOrderState:
    """Tracks remaining volume for a resting algo order."""

    order: Order
    remaining_mw: Decimal
    placed_at: datetime


class ContinuousMatchingEngine:
    """Simulates IDC continuous matching against historical order book data.

    Replays historical market events (new orders, modifications, cancellations,
    trades) in timestamp order.  When the algo places an order the engine
    checks for immediate matches against the current historical book.  If not
    immediately matched, the order rests and is checked against each
    subsequent historical event.

    **Price-taker assumption**: the algo's orders do not affect the historical
    order flow.  Both a historical trade and an algo fill at the same price
    are assumed to happen independently.  For small algo volumes this is
    acceptable.

    **No partial fill accounting for historical liquidity**: if a historical
    order has 10 MW and the algo fills 3 MW against it, the historical order
    is still shown as having 10 MW available.  We do not track how other
    historical participants consumed the remaining volume.

    **aggressor_side handling**: when a :class:`~nexa_backtest.types.HistoricalTrade`
    event has ``aggressor_side=None`` (common for EPEX SPOT data), the engine
    falls back to: if a resting algo order would have matched at the trade
    price, assume it was filled.
    """

    def __init__(self) -> None:
        # Historical book state maintained by replaying events
        self._hist_orders: dict[str, _HistoricalOrder] = {}
        # Product → set of order_ids on each side (for fast lookup)
        self._hist_buy_ids: dict[str, set[str]] = defaultdict(set)
        self._hist_sell_ids: dict[str, set[str]] = defaultdict(set)

        # Algo's resting orders (placed but not yet fully filled or cancelled)
        self._algo_states: dict[str, _AlgoOrderState] = {}

        # Last trade price per product (for get_last_price on context)
        self._last_trade_price: dict[str, Decimal] = {}

        # Gate closure times per product (set by the engine before each MTU)
        self._gate_closures: dict[str, datetime] = {}

    # ------------------------------------------------------------------
    # Historical event processing
    # ------------------------------------------------------------------

    def process_historical_event(self, event: MarketEvent) -> list[Fill]:
        """Process a single historical market event.

        Updates the internal historical order book state and checks whether
        any resting algo orders would have been matched.

        Args:
            event: A :class:`~nexa_backtest.types.MarketDataUpdate` or
                :class:`~nexa_backtest.types.HistoricalTrade`.

        Returns:
            List of :class:`~nexa_backtest.types.Fill` objects for any algo
            orders that were matched by this event.
        """
        if isinstance(event, MarketDataUpdate):
            return self._process_market_data_update(event)
        if isinstance(event, HistoricalTrade):
            return self._process_historical_trade(event)
        return []

    def _process_market_data_update(self, event: MarketDataUpdate) -> list[Fill]:
        et = event.event_type

        if et == "new":
            hist = _HistoricalOrder(
                order_id=event.order_id,
                product_id=event.product_id,
                side=event.side,
                price=event.price_eur_mwh,
                remaining_mw=event.remaining_mw,
            )
            self._hist_orders[event.order_id] = hist
            if event.side == "buy":
                self._hist_buy_ids[event.product_id].add(event.order_id)
            else:
                self._hist_sell_ids[event.product_id].add(event.order_id)
            # A new historical order may immediately match resting algo orders
            return self._check_algo_orders_vs_new_hist(hist, event.timestamp)

        if et in ("modify", "trade") and event.order_id in self._hist_orders:
            hist = self._hist_orders[event.order_id]
            updated = _HistoricalOrder(
                order_id=hist.order_id,
                product_id=hist.product_id,
                side=hist.side,
                price=event.price_eur_mwh,
                remaining_mw=event.remaining_mw,
            )
            self._hist_orders[event.order_id] = updated
            if event.remaining_mw <= Decimal("0"):
                self._remove_hist_order(event.order_id, hist.product_id, hist.side)

        if et == "cancel" and event.order_id in self._hist_orders:
            hist = self._hist_orders[event.order_id]
            self._remove_hist_order(event.order_id, hist.product_id, hist.side)

        return []

    def _process_historical_trade(self, event: HistoricalTrade) -> list[Fill]:
        self._last_trade_price[event.product_id] = event.price_eur_mwh
        return self._check_algo_orders_vs_trade(event)

    # ------------------------------------------------------------------
    # Algo order management
    # ------------------------------------------------------------------

    def place_algo_order(self, order: Order, timestamp: datetime) -> list[Fill]:
        """Place an algo order and attempt immediate matching.

        Checks the current historical book for any resting orders that would
        match.  Fills the algo order at the historical resting order's price.
        Any unfilled remainder rests in ``_algo_states`` for future matching.

        Args:
            order: The algo's order.
            timestamp: Current simulated time (used to timestamp fills).

        Returns:
            List of :class:`~nexa_backtest.types.Fill` for immediately matched
            volume.  If empty the full order is resting.
        """
        if order.volume_mw <= Decimal("0"):
            return []

        fills: list[Fill] = []
        remaining = order.volume_mw

        if order.side == Side.BUY:
            # Match against historical asks (lowest price first)
            candidates = self._sorted_hist_asks(order.product_id)
            for hist in candidates:
                if order.price_eur_mwh is not None and hist.price > order.price_eur_mwh:
                    break
                fill_vol = min(remaining, hist.remaining_mw)
                fills.append(
                    Fill(
                        order_id=order.order_id,
                        product_id=order.product_id,
                        price=hist.price,
                        volume=fill_vol,
                        timestamp=timestamp,
                        side=Side.BUY,
                    )
                )
                remaining -= fill_vol
                if remaining <= Decimal("0"):
                    break
        else:
            # Match against historical bids (highest price first)
            candidates = self._sorted_hist_bids(order.product_id)
            for hist in candidates:
                if order.price_eur_mwh is not None and hist.price < order.price_eur_mwh:
                    break
                fill_vol = min(remaining, hist.remaining_mw)
                fills.append(
                    Fill(
                        order_id=order.order_id,
                        product_id=order.product_id,
                        price=hist.price,
                        volume=fill_vol,
                        timestamp=timestamp,
                        side=Side.SELL,
                    )
                )
                remaining -= fill_vol
                if remaining <= Decimal("0"):
                    break

        if remaining > Decimal("0"):
            self._algo_states[order.order_id] = _AlgoOrderState(
                order=order,
                remaining_mw=remaining,
                placed_at=timestamp,
            )

        return fills

    def cancel_algo_order(self, order_id: str) -> CancelResult:
        """Cancel a resting algo order.

        Args:
            order_id: ID of the order to cancel.

        Returns:
            :class:`~nexa_backtest.types.CancelResult` with status
            ``"cancelled"`` or ``"not_found"``.
        """
        if order_id in self._algo_states:
            del self._algo_states[order_id]
            return CancelResult(order_id=order_id, status="cancelled")
        return CancelResult(order_id=order_id, status="not_found")

    def check_gate_closures(self, current_time: datetime) -> list[str]:
        """Cancel resting algo orders for products whose gate has closed.

        Args:
            current_time: Current simulated time.

        Returns:
            List of ``product_id`` strings whose gates closed (and whose
            algo orders were cancelled).
        """
        closed: list[str] = []
        for product_id, closure_time in list(self._gate_closures.items()):
            if current_time >= closure_time:
                closed.append(product_id)
                # Cancel all resting algo orders for this product
                for order_id in [
                    oid
                    for oid, state in self._algo_states.items()
                    if state.order.product_id == product_id
                ]:
                    del self._algo_states[order_id]
                del self._gate_closures[product_id]
        return closed

    def set_gate_closure(self, product_id: str, closure_time: datetime) -> None:
        """Register the gate closure time for a product.

        Args:
            product_id: Exchange product identifier.
            closure_time: Timezone-aware gate closure timestamp.
        """
        self._gate_closures[product_id] = closure_time

    # ------------------------------------------------------------------
    # Order book access
    # ------------------------------------------------------------------

    def get_orderbook(self, product_id: str, timestamp: datetime) -> OrderBook:
        """Return the current historical order book snapshot for a product.

        The algo sees the historical order book state, *not* a book that
        includes the algo's own resting orders.

        Args:
            product_id: Exchange product identifier.
            timestamp: Current simulated time (stamped on the snapshot).

        Returns:
            :class:`~nexa_backtest.types.OrderBook` with all resting
            historical bids and asks.
        """
        bids = sorted(
            [
                PriceLevel(price=h.price, volume=h.remaining_mw)
                for oid in self._hist_buy_ids.get(product_id, set())
                if (h := self._hist_orders.get(oid)) is not None
            ],
            key=lambda pl: pl.price,
            reverse=True,  # best (highest) first
        )
        asks = sorted(
            [
                PriceLevel(price=h.price, volume=h.remaining_mw)
                for oid in self._hist_sell_ids.get(product_id, set())
                if (h := self._hist_orders.get(oid)) is not None
            ],
            key=lambda pl: pl.price,  # best (lowest) first
        )
        return OrderBook(
            product_id=product_id,
            bids=bids,
            asks=asks,
            timestamp=timestamp,
        )

    def get_last_trade_price(self, product_id: str) -> Decimal | None:
        """Return the last historical trade price for a product.

        Args:
            product_id: Exchange product identifier.

        Returns:
            Last trade price in EUR/MWh, or ``None`` if no trades yet.
        """
        return self._last_trade_price.get(product_id)

    def get_resting_algo_order_ids(self) -> list[str]:
        """Return IDs of all resting algo orders.

        Returns:
            List of order IDs currently resting in the matching engine.
        """
        return list(self._algo_states.keys())

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _remove_hist_order(self, order_id: str, product_id: str, side: str) -> None:
        self._hist_orders.pop(order_id, None)
        if side == "buy":
            self._hist_buy_ids[product_id].discard(order_id)
        else:
            self._hist_sell_ids[product_id].discard(order_id)

    def _sorted_hist_asks(self, product_id: str) -> list[_HistoricalOrder]:
        """Return active historical asks sorted by price ascending (best first)."""
        asks = [
            h
            for oid in self._hist_sell_ids.get(product_id, set())
            if (h := self._hist_orders.get(oid)) is not None and h.remaining_mw > Decimal("0")
        ]
        return sorted(asks, key=lambda h: h.price)

    def _sorted_hist_bids(self, product_id: str) -> list[_HistoricalOrder]:
        """Return active historical bids sorted by price descending (best first)."""
        bids = [
            h
            for oid in self._hist_buy_ids.get(product_id, set())
            if (h := self._hist_orders.get(oid)) is not None and h.remaining_mw > Decimal("0")
        ]
        return sorted(bids, key=lambda h: h.price, reverse=True)

    def _check_algo_orders_vs_new_hist(
        self, hist: _HistoricalOrder, timestamp: datetime
    ) -> list[Fill]:
        """Check if a newly arrived historical order matches any resting algo orders."""
        fills: list[Fill] = []
        for order_id, state in list(self._algo_states.items()):
            order = state.order
            if order.product_id != hist.product_id:
                continue
            if (
                order.side == Side.BUY
                and hist.side == "sell"
                and (order.price_eur_mwh is None or order.price_eur_mwh >= hist.price)
            ):
                fill_vol = min(state.remaining_mw, hist.remaining_mw)
                fills.append(
                    Fill(
                        order_id=order.order_id,
                        product_id=order.product_id,
                        price=hist.price,
                        volume=fill_vol,
                        timestamp=timestamp,
                        side=Side.BUY,
                    )
                )
                state.remaining_mw -= fill_vol
                if state.remaining_mw <= Decimal("0"):
                    del self._algo_states[order_id]
            elif (
                order.side == Side.SELL
                and hist.side == "buy"
                and (order.price_eur_mwh is None or order.price_eur_mwh <= hist.price)
            ):
                fill_vol = min(state.remaining_mw, hist.remaining_mw)
                fills.append(
                    Fill(
                        order_id=order.order_id,
                        product_id=order.product_id,
                        price=hist.price,
                        volume=fill_vol,
                        timestamp=timestamp,
                        side=Side.SELL,
                    )
                )
                state.remaining_mw -= fill_vol
                if state.remaining_mw <= Decimal("0"):
                    del self._algo_states[order_id]
        return fills

    def _check_algo_orders_vs_trade(self, trade: HistoricalTrade) -> list[Fill]:
        """Check if a historical trade would have hit any resting algo orders."""
        fills: list[Fill] = []
        for order_id, state in list(self._algo_states.items()):
            order = state.order
            if order.product_id != trade.product_id:
                continue

            hit = self._trade_hits_algo_order(order, trade)
            if hit:
                fill_vol = min(state.remaining_mw, trade.volume_mw)
                fills.append(
                    Fill(
                        order_id=order.order_id,
                        product_id=order.product_id,
                        price=trade.price_eur_mwh,
                        volume=fill_vol,
                        timestamp=trade.timestamp,
                        side=order.side,
                    )
                )
                state.remaining_mw -= fill_vol
                if state.remaining_mw <= Decimal("0"):
                    del self._algo_states[order_id]
        return fills

    @staticmethod
    def _trade_hits_algo_order(order: Order, trade: HistoricalTrade) -> bool:
        """Determine whether a historical trade would have hit a resting algo order.

        Uses ``aggressor_side`` when available.  Falls back to: if the trade
        price crosses the algo order's price, assume the order was filled.
        """
        aggressor = trade.aggressor_side
        price = trade.price_eur_mwh

        if order.side == Side.BUY:
            # Algo is resting buyer — hit when aggressive seller crossed down to our price
            algo_price_ok = order.price_eur_mwh is None or price <= order.price_eur_mwh
            if not algo_price_ok:
                return False
            if aggressor is None:
                return True  # fallback: trade at our price → assume hit
            return aggressor == "sell"

        # Algo is resting seller — hit when aggressive buyer crossed up to our price
        algo_price_ok = order.price_eur_mwh is None or price >= order.price_eur_mwh
        if not algo_price_ok:
            return False
        if aggressor is None:
            return True  # fallback
        return aggressor == "buy"
