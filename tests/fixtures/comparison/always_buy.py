"""Fixture algo: always buys at market price for every auction product."""

from __future__ import annotations

from decimal import Decimal

from nexa_backtest.algo import SimpleAlgo
from nexa_backtest.context import TradingContext
from nexa_backtest.types import AuctionInfo, Order


class AlwaysBuyAlgo(SimpleAlgo):
    """Places a buy order at a very high price for every auction product.

    Used in comparison tests to exercise the shared replay machinery.
    The price is set high enough that the order always fills.
    """

    def on_auction_open(self, ctx: TradingContext, auction: AuctionInfo) -> None:
        """Place a buy order that always fills."""
        ctx.place_order(
            Order.buy(
                product_id=auction.product_id,
                volume_mw=Decimal("10"),
                price_eur_mwh=Decimal("999"),
            )
        )
