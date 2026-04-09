"""Valid SimpleAlgo: passes all validation steps."""

from __future__ import annotations

from decimal import Decimal

from nexa_backtest.algo import SimpleAlgo
from nexa_backtest.context import TradingContext
from nexa_backtest.types import AuctionInfo, Fill, Order


class ValidSimpleAlgo(SimpleAlgo):
    def on_setup(self, ctx: TradingContext) -> None:
        self.subscribe_signal("price_forecast")
        self.volume: Decimal = Decimal("10")

    def on_auction_open(self, ctx: TradingContext, auction: AuctionInfo) -> None:
        ctx.place_order(
            Order.buy(
                product_id=auction.product_id,
                volume_mw=self.volume,
                price_eur_mwh=Decimal("50"),
            )
        )

    def on_fill(self, ctx: TradingContext, fill: Fill) -> None:
        ctx.log(f"Filled {fill.volume} MW @ {fill.price}")
