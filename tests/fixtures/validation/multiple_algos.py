"""Interface compliance: two SimpleAlgo subclasses in one file — ambiguous."""

from __future__ import annotations

from nexa_backtest.algo import SimpleAlgo
from nexa_backtest.context import TradingContext
from nexa_backtest.types import AuctionInfo


class AlgoOne(SimpleAlgo):
    def on_auction_open(self, ctx: TradingContext, auction: AuctionInfo) -> None:
        pass


class AlgoTwo(SimpleAlgo):
    def on_auction_open(self, ctx: TradingContext, auction: AuctionInfo) -> None:
        pass
