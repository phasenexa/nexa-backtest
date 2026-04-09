"""Resource safety violation: uses time.sleep()."""

from __future__ import annotations

import time
from decimal import Decimal

from nexa_backtest.algo import SimpleAlgo
from nexa_backtest.context import TradingContext
from nexa_backtest.types import AuctionInfo, Order


class SleepAlgo(SimpleAlgo):
    def on_auction_open(self, ctx: TradingContext, auction: AuctionInfo) -> None:
        time.sleep(1)  # Pauses the real clock — wrong in backtest.
        ctx.place_order(
            Order.buy(
                product_id=auction.product_id,
                volume_mw=Decimal("10"),
                price_eur_mwh=Decimal("50"),
            )
        )
