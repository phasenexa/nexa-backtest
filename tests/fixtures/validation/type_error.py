"""Type error: uses float where Decimal is expected."""

from __future__ import annotations

from nexa_backtest.algo import SimpleAlgo
from nexa_backtest.context import TradingContext
from nexa_backtest.types import AuctionInfo, Order


class TypeErrorAlgo(SimpleAlgo):
    def on_setup(self, ctx: TradingContext) -> None:
        self.volume: float = 10.0  # float, not Decimal

    def on_auction_open(self, ctx: TradingContext, auction: AuctionInfo) -> None:
        ctx.place_order(
            Order.buy(
                product_id=auction.product_id,
                volume_mw=self.volume,  # type: ignore[arg-type]  # intentional
                price_eur_mwh=self.volume,  # type: ignore[arg-type]  # intentional
            )
        )
