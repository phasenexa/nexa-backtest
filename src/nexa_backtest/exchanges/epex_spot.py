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

from datetime import timedelta
from decimal import Decimal

from nexa_backtest.exchanges.base import ExchangeAdapter, ExchangeCapabilities, _BaseExchangeAdapter

# EPEX SPOT IDC gate closure offset (most products, not cross-border)
EPEX_IDC_GATE_CLOSURE = timedelta(minutes=30)

# EPEX SPOT DA/IDA gate closure offset
EPEX_DA_GATE_CLOSURE = timedelta(hours=1)

# EPEX SPOT price limits (EUR/MWh)
EPEX_MIN_PRICE = Decimal("-500")
EPEX_MAX_PRICE = Decimal("4000")

# EPEX SPOT minimum order volume (MW)
EPEX_MIN_VOLUME = Decimal("0.1")


class EpexSpotAdapter(_BaseExchangeAdapter):
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
        capabilities = ExchangeCapabilities(
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
        super().__init__(area, capabilities, "EPEX SPOT")
        self.supports_continuous_trading = True
        self.supports_auction_trading = True

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


# Satisfy ExchangeAdapter protocol at type-check time.
def _check_adapter_protocol(adapter: ExchangeAdapter) -> None:  # pragma: no cover
    """Compile-time check that EpexSpotAdapter satisfies ExchangeAdapter."""


_check_adapter_protocol(EpexSpotAdapter.__new__(EpexSpotAdapter))
