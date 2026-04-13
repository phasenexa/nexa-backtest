"""Fixture algo: buys when price is below forecast minus a low threshold."""

from __future__ import annotations

from decimal import Decimal

from nexa_backtest.algo import SimpleAlgo
from nexa_backtest.context import TradingContext
from nexa_backtest.exceptions import SignalError
from nexa_backtest.types import AuctionInfo, Order


class ThresholdLowAlgo(SimpleAlgo):
    """Buy when clearing price < forecast - 3 EUR/MWh.

    Uses the ``price_forecast`` signal.  More permissive than
    :class:`ThresholdHighAlgo` so it places more orders and fills more.
    """

    def on_setup(self, ctx: TradingContext) -> None:
        """Subscribe to the price forecast signal."""
        self.subscribe_signal("price_forecast")
        self._threshold = Decimal("3")

    def on_auction_open(self, ctx: TradingContext, auction: AuctionInfo) -> None:
        """Place a buy order if bid price >= 3 EUR/MWh below forecast."""
        try:
            signal = ctx.get_signal("price_forecast")
        except SignalError:
            return

        bid_price = Decimal(str(signal.value)) - self._threshold
        ctx.place_order(
            Order.buy(
                product_id=auction.product_id,
                volume_mw=Decimal("10"),
                price_eur_mwh=bid_price,
            )
        )
