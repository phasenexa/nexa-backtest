"""Interface compliance: uses signal without subscribing."""

from __future__ import annotations

from decimal import Decimal

from nexa_backtest.algo import SimpleAlgo
from nexa_backtest.context import TradingContext
from nexa_backtest.types import AuctionInfo, Order


class UnsubscribedSignalAlgo(SimpleAlgo):
    def on_setup(self, ctx: TradingContext) -> None:
        # Forgot to call self.subscribe_signal("price_forecast")
        pass

    def on_auction_open(self, ctx: TradingContext, auction: AuctionInfo) -> None:
        signal = ctx.get_signal("price_forecast")  # Not subscribed!
        ctx.place_order(
            Order.buy(
                product_id=auction.product_id,
                volume_mw=Decimal("10"),
                price_eur_mwh=Decimal(str(signal.value)),
            )
        )
