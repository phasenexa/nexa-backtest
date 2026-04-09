"""Uses Order.block_bid() — unsupported on EPEX SPOT continuous."""

from __future__ import annotations

from decimal import Decimal

from nexa_backtest.algo import SimpleAlgo
from nexa_backtest.context import TradingContext
from nexa_backtest.types import AuctionInfo, Order


class BlockBidAlgo(SimpleAlgo):
    def on_auction_open(self, ctx: TradingContext, auction: AuctionInfo) -> None:
        ctx.place_order(
            Order.block_bid(
                product_ids=[auction.product_id],
                volume_mw=Decimal("10"),
                price_eur_mwh=Decimal("50"),
            )
        )
