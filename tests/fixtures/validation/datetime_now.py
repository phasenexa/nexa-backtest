"""Resource safety violation: uses datetime.now() instead of ctx.now()."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from nexa_backtest.algo import SimpleAlgo
from nexa_backtest.context import TradingContext
from nexa_backtest.types import AuctionInfo, Order


class DatetimeNowAlgo(SimpleAlgo):
    def on_auction_open(self, ctx: TradingContext, auction: AuctionInfo) -> None:
        now = datetime.now()  # Wrong: wall-clock time, not simulated time.
        ctx.log(f"Wall clock time: {now}")
        ctx.place_order(
            Order.buy(
                product_id=auction.product_id,
                volume_mw=Decimal("10"),
                price_eur_mwh=Decimal("50"),
            )
        )
