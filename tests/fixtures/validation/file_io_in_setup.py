"""File I/O only in on_setup — this is fine (not a hot-path call)."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

from nexa_backtest.algo import SimpleAlgo
from nexa_backtest.context import TradingContext
from nexa_backtest.types import AuctionInfo, Order


class FileIoInSetupAlgo(SimpleAlgo):
    def on_setup(self, ctx: TradingContext) -> None:
        # Reading config in setup is acceptable — runs once.
        config_path = Path("config.json")
        if config_path.exists():
            self.config = config_path.read_text()
        else:
            self.config = "{}"
        self.volume: Decimal = Decimal("10")

    def on_auction_open(self, ctx: TradingContext, auction: AuctionInfo) -> None:
        ctx.place_order(
            Order.buy(
                product_id=auction.product_id,
                volume_mw=self.volume,
                price_eur_mwh=Decimal("50"),
            )
        )
