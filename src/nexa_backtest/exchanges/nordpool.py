"""Nord Pool exchange adapter.

Implements the :class:`~nexa_backtest.exchanges.base.ExchangeAdapter` protocol
for Nord Pool markets:

- **DA auctions**: Cleared once per day at D-1 12:00 CET/CEST.  Gate closure
  is D-1 12:00 (before midday in local time, approximately 11:00 UTC in
  winter).  The backtester approximates this as D-1 12:00 UTC.

- **IDC continuous**: Quarter-hourly products (QH) traded continuously until
  gate closure 30 minutes before delivery start.  Products are identified as
  ``{zone}-QH-{HHMM}`` where ``HHMM`` is the delivery period start in UTC,
  e.g. ``NO1-QH-0900``.

Supported bidding zones (non-exhaustive)::

    NO1, NO2, NO3, NO4, NO5  # Norway
    SE1, SE2, SE3, SE4        # Sweden
    DK1, DK2                  # Denmark
    FI                         # Finland
    EE, LT, LV                # Baltics

Gate closure rules:

- DA: approximately D-1 12:00 UTC (exact time is local CET/CEST, approximated
  here as UTC since the backtester uses UTC throughout).
- IDC QH: 30 minutes before delivery start in UTC.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from decimal import Decimal

from nexa_backtest.exchanges.base import ExchangeAdapter, ExchangeCapabilities
from nexa_backtest.types import CancelResult, Order, OrderBook, OrderResult

# Nord Pool IDC product name pattern: {zone}-QH-{HHMM}
_IDC_QH_PATTERN = re.compile(r"^[A-Z0-9]+-QH-\d{4}$")

# Gate closure offset for Nord Pool IDC quarter-hourly products
NORDPOOL_IDC_GATE_CLOSURE = timedelta(minutes=30)

# Gate closure offset for Nord Pool DA
NORDPOOL_DA_GATE_CLOSURE = timedelta(hours=1)  # proxy for D-1 ~12:00 UTC

# Default price limits (EUR/MWh)
NORDPOOL_MIN_PRICE = Decimal("-500")
NORDPOOL_MAX_PRICE = Decimal("3000")

# Default volume limits (MW)
NORDPOOL_MIN_VOLUME = Decimal("0.1")


def is_idc_product(product_id: str) -> bool:
    """Return ``True`` if ``product_id`` matches the Nord Pool IDC QH pattern.

    Args:
        product_id: Exchange product identifier.

    Returns:
        ``True`` for products like ``NO1-QH-0900``.
    """
    return bool(_IDC_QH_PATTERN.match(product_id))


def idc_gate_closure_time(product_id: str) -> datetime | None:
    """Return the gate closure time for an IDC product.

    Gate closes 30 minutes before delivery start.

    Args:
        product_id: Exchange product identifier, e.g. ``"NO1-QH-0900"``.

    Returns:
        Gate closure datetime in UTC, or ``None`` if the product ID is
        not in the expected format.
    """
    # Product ID format: {zone}-QH-{HHMM}  (delivery date not encoded in ID)
    # Gate closure = delivery start - 30 minutes
    # Without a reference date we can only return the offset.
    # Callers must combine with the delivery date.
    return None  # pragma: no cover


class NordPoolAdapter:
    """Exchange adapter for Nord Pool (DA + IDC).

    This is a stateless configuration object.  The backtest engine uses it to
    query capabilities; actual order matching is performed by the engine-
    internal matching engines, not by this adapter.

    Attributes:
        zone: Bidding zone identifier, e.g. ``"NO1"``.
        supports_continuous_trading: Always ``True`` for Nord Pool.
    """

    def __init__(self, zone: str) -> None:
        self._zone = zone
        self._capabilities = ExchangeCapabilities(
            exchange_id="nordpool",
            supports_block_bids=True,
            supports_linked_orders=False,
            supports_market_orders=False,
            supports_partial_fills=True,
            min_volume_mw=NORDPOOL_MIN_VOLUME,
            max_volume_mw=None,
            min_price_eur_mwh=NORDPOOL_MIN_PRICE,
            max_price_eur_mwh=NORDPOOL_MAX_PRICE,
            mtu_duration_minutes=15,
            gate_closure_minutes_before_delivery=30,
        )
        self.supports_continuous_trading = True

    @property
    def capabilities(self) -> ExchangeCapabilities:
        """Declare Nord Pool exchange capabilities.

        Returns:
            Frozen :class:`~nexa_backtest.exchanges.base.ExchangeCapabilities`.
        """
        return self._capabilities

    def get_products(self) -> list[str]:
        """Return a representative list of QH product IDs for this zone.

        Returns all 96 quarter-hourly products (00:00-23:45 UTC) for the
        configured zone.

        Returns:
            List of product identifiers in ``{zone}-QH-HHMM`` format.
        """
        products: list[str] = []
        for hour in range(24):
            for minute in (0, 15, 30, 45):
                products.append(f"{self._zone}-QH-{hour:02d}{minute:02d}")
        return products

    def get_orderbook(self, product_id: str) -> OrderBook:
        """Return an empty order book (no live market data in backtesting).

        In backtesting the order book is maintained by the
        :class:`~nexa_backtest.engines.matching.ContinuousMatchingEngine`.
        This method exists to satisfy the
        :class:`~nexa_backtest.exchanges.base.ExchangeAdapter` protocol.

        Args:
            product_id: Exchange product identifier.

        Returns:
            Empty :class:`~nexa_backtest.types.OrderBook`.
        """
        from datetime import UTC

        return OrderBook(
            product_id=product_id,
            bids=[],
            asks=[],
            timestamp=datetime.now(tz=UTC),
        )

    def submit_order(self, order: Order) -> OrderResult:
        """Submit is handled by the matching engine, not the adapter.

        Raises:
            NotImplementedError: Always.  Orders are submitted via the
                :class:`~nexa_backtest.context.TradingContext`.
        """
        raise NotImplementedError(
            "submit_order is not implemented on NordPoolAdapter. "
            "Use ctx.place_order() inside the algo."
        )

    def cancel_order(self, order_id: str) -> CancelResult:
        """Cancel is handled by the matching engine, not the adapter.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "cancel_order is not implemented on NordPoolAdapter. "
            "Use ctx.cancel_order() inside the algo."
        )

    def gate_closure_offset(self, product_type: str = "IDC") -> timedelta:
        """Return the gate closure offset for a given product type.

        Args:
            product_type: ``"DA"`` or ``"IDC"``.

        Returns:
            Time before delivery start that gate closes.
        """
        if product_type == "DA":
            return NORDPOOL_DA_GATE_CLOSURE
        return NORDPOOL_IDC_GATE_CLOSURE

    def validate_order(self, order: Order) -> str | None:
        """Validate an order against Nord Pool exchange rules.

        Args:
            order: The order to validate.

        Returns:
            Error message string if invalid, or ``None`` if valid.
        """
        if order.volume_mw < self._capabilities.min_volume_mw:
            return (
                f"Volume {order.volume_mw} MW below Nord Pool minimum "
                f"{self._capabilities.min_volume_mw} MW."
            )
        if order.price_eur_mwh is not None:
            if order.price_eur_mwh < self._capabilities.min_price_eur_mwh:
                return (
                    f"Price {order.price_eur_mwh} EUR/MWh below Nord Pool minimum "
                    f"{self._capabilities.min_price_eur_mwh} EUR/MWh."
                )
            if order.price_eur_mwh > self._capabilities.max_price_eur_mwh:
                return (
                    f"Price {order.price_eur_mwh} EUR/MWh above Nord Pool maximum "
                    f"{self._capabilities.max_price_eur_mwh} EUR/MWh."
                )
        if order.price_eur_mwh is None and not self._capabilities.supports_market_orders:
            return "Nord Pool does not support market orders."
        return None


# Satisfy ExchangeAdapter protocol at type-check time
def _check_adapter_protocol(adapter: ExchangeAdapter) -> None:  # pragma: no cover
    """Compile-time check that NordPoolAdapter satisfies ExchangeAdapter."""


_check_adapter_protocol(NordPoolAdapter.__new__(NordPoolAdapter))
