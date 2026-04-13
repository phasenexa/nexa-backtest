"""Exchange adapter protocol and capability declarations.

Each supported exchange implements the ``ExchangeAdapter`` protocol and
declares an ``ExchangeCapabilities`` instance describing what features it
supports. The validation pipeline uses this to detect unsupported feature
usage before a backtest run.

Concrete exchange adapters (Nord Pool, EPEX SPOT, EEX) are implemented in
separate modules within this package and inherit from
:class:`_BaseExchangeAdapter` for the shared implementation.
"""

from __future__ import annotations

from datetime import UTC, datetime
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


class _BaseExchangeAdapter:
    """Shared implementation for exchange adapters.

    Provides the boilerplate that all adapters need: 96-product QH listing,
    empty order book, stubbed submit/cancel (the engine handles those), and
    capability-based order validation.

    Subclasses must call ``super().__init__`` and set any exchange-specific
    public attributes (e.g. ``supports_auction_trading``) in their own
    ``__init__``.
    """

    def __init__(self, area_id: str, capabilities: ExchangeCapabilities, display_name: str) -> None:
        self._area_id = area_id
        self._capabilities = capabilities
        self._display_name = display_name

    @property
    def capabilities(self) -> ExchangeCapabilities:
        """Declare the capabilities of this exchange.

        Returns:
            Frozen :class:`ExchangeCapabilities` instance.
        """
        return self._capabilities

    def get_products(self) -> list[str]:
        """Return all 96 quarter-hourly product IDs for the configured area.

        Returns:
            List of 96 product identifiers in ``{area_id}-QH-HHMM`` format,
            covering 00:00-23:45 UTC.
        """
        products: list[str] = []
        for hour in range(24):
            for minute in (0, 15, 30, 45):
                products.append(f"{self._area_id}-QH-{hour:02d}{minute:02d}")
        return products

    def get_orderbook(self, product_id: str) -> OrderBook:
        """Return an empty order book.

        In backtesting the order book is maintained by the matching engine.
        This method exists to satisfy the :class:`ExchangeAdapter` protocol.

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
            NotImplementedError: Always. Use ``ctx.place_order()`` instead.
        """
        raise NotImplementedError(
            f"submit_order is not implemented on {type(self).__name__}. "
            "Use ctx.place_order() inside the algo."
        )

    def cancel_order(self, order_id: str) -> CancelResult:
        """Order cancellation is handled by the engine, not the adapter.

        Raises:
            NotImplementedError: Always. Use ``ctx.cancel_order()`` instead.
        """
        raise NotImplementedError(
            f"cancel_order is not implemented on {type(self).__name__}. "
            "Use ctx.cancel_order() inside the algo."
        )

    def validate_order(self, order: Order) -> str | None:
        """Validate an order against the exchange's capability limits.

        Checks volume and price against :attr:`capabilities`, and rejects
        market orders when ``supports_market_orders`` is ``False``.

        Args:
            order: The order to validate.

        Returns:
            Error message string if invalid, or ``None`` if valid.
        """
        caps = self._capabilities
        if order.volume_mw < caps.min_volume_mw:
            return (
                f"Volume {order.volume_mw} MW below {self._display_name} minimum "
                f"{caps.min_volume_mw} MW."
            )
        if order.price_eur_mwh is not None:
            if order.price_eur_mwh < caps.min_price_eur_mwh:
                return (
                    f"Price {order.price_eur_mwh} EUR/MWh below {self._display_name} minimum "
                    f"{caps.min_price_eur_mwh} EUR/MWh."
                )
            if order.price_eur_mwh > caps.max_price_eur_mwh:
                return (
                    f"Price {order.price_eur_mwh} EUR/MWh above {self._display_name} maximum "
                    f"{caps.max_price_eur_mwh} EUR/MWh."
                )
        if order.price_eur_mwh is None and not caps.supports_market_orders:
            return f"{self._display_name} does not support market orders."
        return None


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
