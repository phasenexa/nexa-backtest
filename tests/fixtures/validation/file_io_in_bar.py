"""File I/O in on_bar — should warn (hot-path call)."""

from __future__ import annotations

from decimal import Decimal

from nexa_backtest.algo import SimpleAlgo
from nexa_backtest.context import TradingContext
from nexa_backtest.types import Order


class FileIoInBarAlgo(SimpleAlgo):
    def on_bar(self, ctx: TradingContext) -> None:
        # Opening a file on every tick is wasteful and suspicious.
        with open("prices.csv") as f:
            data = f.read()
        ctx.log(f"Read {len(data)} bytes")
        ctx.place_order(
            Order.buy(
                product_id="NO1-QH-0900",
                volume_mw=Decimal("5"),
                price_eur_mwh=Decimal("45"),
            )
        )
