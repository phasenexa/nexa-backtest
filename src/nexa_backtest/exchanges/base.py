"""Exchange adapter protocol and capability declarations.

Each supported exchange implements the ``ExchangeAdapter`` protocol and
declares an ``ExchangeCapabilities`` instance describing what features it
supports. The validation pipeline uses this to detect unsupported feature
usage before a backtest run.

Concrete exchange adapters (Nord Pool, EPEX SPOT, EEX) are implemented in
separate modules within this package.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Protocol

from pydantic import BaseModel, ConfigDict

from nexa_backtest.types import CancelResult, Order, OrderBook, OrderResult


class ExchangeCapabilities(BaseModel):
    """Declares what features a specific exchange supports.

    Used by the validation pipeline to detect feature usage at validate-time
    rather than at runtime.

    Attributes:
        exchange_id: Short identifier, e.g. ``"nordpool"``.
        supports_block_bids: Whether block bids spanning multiple MTUs are
            accepted.
        supports_linked_orders: Whether linked/iceberg order types are
            accepted.
        supports_market_orders: Whether market (no limit price) orders are
            accepted.
        supports_partial_fills: Whether orders can be partially filled.
        min_volume_mw: Minimum order volume in MW. Reject smaller orders.
        max_volume_mw: Maximum order volume in MW, or ``None`` if uncapped.
        min_price_eur_mwh: Minimum allowed price in EUR/MWh, e.g. ``-500``.
        max_price_eur_mwh: Maximum allowed price in EUR/MWh, e.g. ``3000``.
        mtu_duration_minutes: Duration of a single MTU in minutes (15 or 60).
        gate_closure_minutes_before_delivery: Minutes before delivery start
            that gate closes. Exchange- and product-type-specific.
    """

    model_config = ConfigDict(frozen=True)

    exchange_id: str
    supports_block_bids: bool = False
    supports_linked_orders: bool = False
    supports_market_orders: bool = False
    supports_partial_fills: bool = False
    min_volume_mw: Decimal = Decimal("0.1")
    max_volume_mw: Decimal | None = None
    min_price_eur_mwh: Decimal = Decimal("-500")
    max_price_eur_mwh: Decimal = Decimal("3000")
    mtu_duration_minutes: int = 15
    gate_closure_minutes_before_delivery: int = 60


class ExchangeAdapter(Protocol):
    """Interface that all exchange adapters must implement.

    Each exchange has its own adapter that translates between the generic
    ``nexa-backtest`` domain types and exchange-specific rules. The adapter
    also declares its ``ExchangeCapabilities`` so the validation pipeline can
    perform feature compatibility checks.
    """

    @property
    def capabilities(self) -> ExchangeCapabilities:
        """Declare the capabilities of this exchange.

        Returns:
            Frozen ``ExchangeCapabilities`` instance.
        """
        ...

    def get_products(self) -> list[str]:
        """Return the list of tradeable product IDs for the current session.

        Returns:
            List of product identifiers, e.g. ``["NO1-QH-0900", "NO1-QH-0915"]``.
        """
        ...

    def get_orderbook(self, product_id: str) -> OrderBook:
        """Return the current order book snapshot for a product.

        Args:
            product_id: Exchange product identifier.

        Returns:
            Order book snapshot with best bid and ask.
        """
        ...

    def submit_order(self, order: Order) -> OrderResult:
        """Submit an order to the exchange.

        Args:
            order: The order to submit.

        Returns:
            Result with the initial order status. The order may be pending,
            accepted, or immediately filled depending on the exchange and
            market conditions.
        """
        ...

    def cancel_order(self, order_id: str) -> CancelResult:
        """Cancel a previously submitted order.

        Args:
            order_id: ID of the order to cancel.

        Returns:
            Result with status ``"cancelled"`` or ``"not_found"``.
        """
        ...
