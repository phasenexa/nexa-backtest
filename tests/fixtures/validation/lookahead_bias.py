"""Look-ahead bias: uses df.shift(-1) and large get_signal_history lookback."""

from __future__ import annotations

from decimal import Decimal

import pandas as pd

from nexa_backtest.algo import SimpleAlgo
from nexa_backtest.context import TradingContext
from nexa_backtest.types import AuctionInfo, Order


class LookaheadAlgo(SimpleAlgo):
    def on_setup(self, ctx: TradingContext) -> None:
        self.subscribe_signal("price_forecast")

    def on_auction_open(self, ctx: TradingContext, auction: AuctionInfo) -> None:
        # Deliberately look-ahead: negative shift reads future data.
        df = pd.DataFrame({"price": [10.0, 20.0, 30.0]})
        future_price = df["price"].shift(-1)

        # Large lookback — may exceed signal publication offset.
        history = ctx.get_signal_history("price_forecast", lookback=96)  # noqa: F841

        if future_price.iloc[0] > 20:
            ctx.place_order(
                Order.buy(
                    product_id=auction.product_id,
                    volume_mw=Decimal("10"),
                    price_eur_mwh=Decimal("50"),
                )
            )
