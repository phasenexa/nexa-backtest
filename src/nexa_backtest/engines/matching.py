"""Matching engines for simulated order execution.

This module provides the DA auction matching engine. The IDC continuous
matching engine (price-time priority) will be added in Task 02.

The price-taker assumption applies throughout: the algo's orders do not
affect the clearing price. This is realistic for most participants in
European power markets.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from nexa_backtest.types import Fill, Order, OrderResult, OrderStatus, Side


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
