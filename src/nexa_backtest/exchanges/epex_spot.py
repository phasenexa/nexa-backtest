"""EPEX SPOT exchange adapter.

Implements the :class:`~nexa_backtest.exchanges.base.ExchangeAdapter` protocol
for EPEX SPOT markets, proving the exchange abstraction works across multiple
exchanges.

EPEX SPOT runs both intraday continuous (IDC) and intraday auction (IDA)
trading for European power markets.  Key differences from Nord Pool:

- **Product naming**: delivery area codes (``DE-LU``, ``FR``, ``AT``) rather
  than Nord Pool zone codes (``NO1``, ``SE3``).
- **Price limits**: ``-500`` to ``+4,000`` EUR/MWh (Nord Pool caps at 3,000).
- **Gate closure**: 30 minutes before delivery for most IDC products; may
  differ for cross-border products.
- **Data format**: EPEX historical exports use different column names — see
  :func:`~nexa_backtest.data.parsers.epex.parse_epex_df` for normalisation.

Supported EPEX areas (non-exhaustive)::

    DE-LU    # Germany / Luxembourg
    FR       # France
    AT       # Austria
    BE       # Belgium
    NL       # Netherlands
    CH       # Switzerland
    GB       # Great Britain
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from nexa_backtest.exchanges.base import ExchangeAdapter, ExchangeCapabilities
from nexa_backtest.types import CancelResult, Order, OrderBook, OrderResult

# EPEX SPOT IDC gate closure offset (most products, not cross-border)
EPEX_IDC_GATE_CLOSURE = timedelta(minutes=30)

# EPEX SPOT DA/IDA gate closure offset
EPEX_DA_GATE_CLOSURE = timedelta(hours=1)

# EPEX SPOT price limits (EUR/MWh)
EPEX_MIN_PRICE = Decimal("-500")
EPEX_MAX_PRICE = Decimal("4000")

# EPEX SPOT minimum order volume (MW)
EPEX_MIN_VOLUME = Decimal("0.1")


class EpexSpotAdapter:
    """Exchange adapter for EPEX SPOT (IDC and IDA).

    This is a stateless configuration object.  The backtest engine uses it to
    query capabilities; actual order matching is performed by the engine-
    internal matching engines, not by this adapter.

    Attributes:
        area: Delivery area identifier, e.g. ``"DE-LU"``, ``"FR"``.
        supports_continuous_trading: Always ``True``.
        supports_auction_trading: Always ``True`` (EPEX runs IDC + IDA).
    """

    def __init__(self, area: str) -> None:
        self._area = area
        self._capabilities = ExchangeCapabilities(
            exchange_id="epex_spot",
            supports_block_bids=True,
            supports_linked_orders=False,
            supports_market_orders=False,
            supports_partial_fills=True,
            min_volume_mw=EPEX_MIN_VOLUME,
            max_volume_mw=None,
            min_price_eur_mwh=EPEX_MIN_PRICE,
            max_price_eur_mwh=EPEX_MAX_PRICE,
            mtu_duration_minutes=15,
            gate_closure_minutes_before_delivery=30,
        )
        self.supports_continuous_trading = True
        self.supports_auction_trading = True

    @property
    def capabilities(self) -> ExchangeCapabilities:
        """Declare EPEX SPOT exchange capabilities.

        Returns:
            Frozen :class:`~nexa_backtest.exchanges.base.ExchangeCapabilities`.
        """
        return self._capabilities

    def get_products(self) -> list[str]:
        """Return all 96 quarter-hourly product IDs for this delivery area.

        Products use the format ``{area}-QH-{HHMM}``, e.g.
        ``DE-LU-QH-0900``.

        Returns:
            List of 96 product identifiers (00:00-23:45 UTC).
        """
        products: list[str] = []
        for hour in range(24):
            for minute in (0, 15, 30, 45):
                products.append(f"{self._area}-QH-{hour:02d}{minute:02d}")
        return products

    def get_orderbook(self, product_id: str) -> OrderBook:
        """Return an empty order book (maintained by the matching engine).

        In backtesting the order book state is managed by the
        :class:`~nexa_backtest.engines.matching.ContinuousMatchingEngine`.

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
            NotImplementedError: Always.  Use ``ctx.place_order()`` instead.
        """
        raise NotImplementedError(
            "submit_order is not implemented on EpexSpotAdapter. "
            "Use ctx.place_order() inside the algo."
        )

    def cancel_order(self, order_id: str) -> CancelResult:
        """Order cancellation is handled by the engine, not the adapter.

        Raises:
            NotImplementedError: Always.  Use ``ctx.cancel_order()`` instead.
        """
        raise NotImplementedError(
            "cancel_order is not implemented on EpexSpotAdapter. "
            "Use ctx.cancel_order() inside the algo."
        )

    def gate_closure_offset(self, product_type: str = "IDC") -> timedelta:
        """Return the gate closure offset for a given product type.

        Args:
            product_type: ``"DA"``, ``"IDA"``, or ``"IDC"``.

        Returns:
            Time before delivery start that gate closes.
        """
        if product_type in ("DA", "IDA"):
            return EPEX_DA_GATE_CLOSURE
        return EPEX_IDC_GATE_CLOSURE

    def validate_order(self, order: Order) -> str | None:
        """Validate an order against EPEX SPOT exchange rules.

        Args:
            order: The order to validate.

        Returns:
            Error message string if invalid, or ``None`` if valid.
        """
        if order.volume_mw < self._capabilities.min_volume_mw:
            return (
                f"Volume {order.volume_mw} MW below EPEX SPOT minimum "
                f"{self._capabilities.min_volume_mw} MW."
            )
        if order.price_eur_mwh is not None:
            if order.price_eur_mwh < self._capabilities.min_price_eur_mwh:
                return (
                    f"Price {order.price_eur_mwh} EUR/MWh below EPEX SPOT minimum "
                    f"{self._capabilities.min_price_eur_mwh} EUR/MWh."
                )
            if order.price_eur_mwh > self._capabilities.max_price_eur_mwh:
                return (
                    f"Price {order.price_eur_mwh} EUR/MWh above EPEX SPOT maximum "
                    f"{self._capabilities.max_price_eur_mwh} EUR/MWh."
                )
        if order.price_eur_mwh is None and not self._capabilities.supports_market_orders:
            return "EPEX SPOT does not support market orders."
        return None


# Satisfy ExchangeAdapter protocol at type-check time.
def _check_adapter_protocol(adapter: ExchangeAdapter) -> None:  # pragma: no cover
    """Compile-time check that EpexSpotAdapter satisfies ExchangeAdapter."""


_check_adapter_protocol(EpexSpotAdapter.__new__(EpexSpotAdapter))
