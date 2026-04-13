"""Fixture algo: buys only when price is well below forecast (high threshold)."""

from __future__ import annotations

from decimal import Decimal

from nexa_backtest.algo import SimpleAlgo
from nexa_backtest.context import TradingContext
from nexa_backtest.exceptions import SignalError
from nexa_backtest.types import AuctionInfo, Order


class ThresholdHighAlgo(SimpleAlgo):
    """Buy when clearing price < forecast - 8 EUR/MWh.

    Uses the ``price_forecast`` signal.  More selective than
    :class:`ThresholdLowAlgo` so it places fewer orders but with a higher
    expected alpha per fill.
    """

    def on_setup(self, ctx: TradingContext) -> None:
        """Subscribe to the price forecast signal."""
        self.subscribe_signal("price_forecast")
        self._threshold = Decimal("8")

    def on_auction_open(self, ctx: TradingContext, auction: AuctionInfo) -> None:
        """Place a buy order if bid price >= 8 EUR/MWh below forecast."""
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
