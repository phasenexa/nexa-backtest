"""Example IDC algo: buys at the best ask for each active product.

Demonstrates the on_bar, on_fill, and on_gate_closure hooks.

Run via::

    nexa run examples/simple_idc_algo.py \\
        --exchange nordpool \\
        --start 2026-03-01 \\
        --end 2026-03-31 \\
        --products NO1-QH \\
        --data-dir tests/fixtures/nordpool \\
        --capital 100000
"""

from nexa_backtest import SimpleAlgo, Order
from nexa_backtest.context import TradingContext
from nexa_backtest.types import Fill


class BestAskBuyerAlgo(SimpleAlgo):
    """Buys 1 MW at the best ask price every bar, for each IDC product.

    This is an aggressive strategy — it always crosses the spread to fill
    immediately. It doesn't care about price; it just wants to trade.
    In practice this is a poor strategy (you always pay the ask), but it
    demonstrates the IDC mechanics clearly.
    """

    # Products to trade — the three QH products available in the fixture
    PRODUCTS = ["NO1-QH-0800", "NO1-QH-0815", "NO1-QH-0830"]

    def on_setup(self, ctx: TradingContext) -> None:
        self.fill_count = 0
        self.gate_closures_seen: list[str] = []
        ctx.log("BestAskBuyerAlgo started")

    def on_bar(self, ctx: TradingContext) -> None:
        """Called every 15 minutes — evaluate and trade."""
        for product_id in self.PRODUCTS:
            # Don't place orders when gate is closing in < 5 minutes
            ttg = ctx.time_to_gate_closure(product_id)
            if ttg.total_seconds() < 300:
                continue

            # Check current best ask for this product
            ask = ctx.get_best_ask(product_id)
            if ask is None:
                # No sellers in the book yet — skip
                continue

            # Place a buy 1 EUR above the ask to guarantee an immediate fill.
            # (In practice you'd bid exactly at the ask, but adding 1 EUR
            # makes the match absolutely certain even if the book updates.)
            order = Order.buy(
                product_id=product_id,
                volume_mw=1.0,
                price_eur_mwh=float(ask.price) + 1.0,
            )
            ctx.place_order(order)

    def on_fill(self, ctx: TradingContext, fill: Fill) -> None:
        self.fill_count += 1

    def on_gate_closure(self, ctx: TradingContext, product_id: str) -> None:
        """Called when trading closes for a product — no more orders allowed."""
        self.gate_closures_seen.append(product_id)
        ctx.log(f"Gate closed for {product_id} — {len(self.gate_closures_seen)} products closed so far")

    def on_teardown(self, ctx: TradingContext) -> None:
        ctx.log(f"Backtest done. Total fills: {self.fill_count}")
        ctx.log(f"Gate closures seen: {self.gate_closures_seen}")
