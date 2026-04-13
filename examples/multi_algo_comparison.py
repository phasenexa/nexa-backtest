"""Multi-algo comparison example: run three strategies and compare results.

Demonstrates SharedReplayEngine: three strategies are run against the same
historical DA data in a single pass. Market data is loaded once and shared;
each strategy gets its own isolated TradingContext.

Run via::

    python examples/multi_algo_comparison.py

Or via the CLI::

    nexa compare \\
        "conservative:examples/multi_algo_comparison.py" \\
        "aggressive:examples/multi_algo_comparison.py" \\
        --exchange nordpool \\
        --start 2026-03-01 \\
        --end 2026-03-31 \\
        --products NO1_DA \\
        --data-dir tests/fixtures \\
        --output /tmp/comparison.html

Strategies compared:
  - conservative: buys only when forecast is >= 8 EUR/MWh above clearing price
  - moderate:     buys when forecast is >= 5 EUR/MWh above clearing price
  - aggressive:   buys whenever the forecast is above clearing (no threshold)

The aggressive strategy fills more often but has lower average alpha per fill.
The conservative strategy fills less but with higher confidence in each trade.
"""

from __future__ import annotations

import sys
from datetime import date
from decimal import Decimal
from pathlib import Path

from nexa_backtest.algo import SimpleAlgo
from nexa_backtest.context import TradingContext
from nexa_backtest.engines.shared import SharedReplayEngine
from nexa_backtest.exceptions import SignalError
from nexa_backtest.types import AuctionInfo, Fill, Order


class ConservativeAlgo(SimpleAlgo):
    """Buy only when forecast is >= 8 EUR/MWh above clearing price.

    Places fewer orders but each fill has a higher expected alpha.
    Suitable for risk-averse desks that prefer quality over quantity.
    """

    def on_setup(self, ctx: TradingContext) -> None:
        """Subscribe to price forecast and set a high threshold."""
        self.subscribe_signal("price_forecast")
        self.threshold: Decimal = Decimal("8.0")
        self.volume_mw: Decimal = Decimal("10")

    def on_auction_open(self, ctx: TradingContext, auction: AuctionInfo) -> None:
        """Place a buy order only when forecast indicates strong undervaluation."""
        try:
            signal = ctx.get_signal("price_forecast")
        except SignalError:
            return

        bid = Decimal(str(signal.value)) - self.threshold
        ctx.place_order(
            Order.buy(
                product_id=auction.product_id,
                volume_mw=self.volume_mw,
                price_eur_mwh=bid,
            )
        )

    def on_fill(self, ctx: TradingContext, fill: Fill) -> None:
        """Log each fill."""
        print(f"  [conservative] Fill: {fill.product_id}  {float(fill.price):.2f} EUR/MWh")


class ModerateAlgo(SimpleAlgo):
    """Buy when forecast is >= 5 EUR/MWh above clearing price.

    A balanced approach: more fills than the conservative strategy,
    fewer than the aggressive strategy.
    """

    def on_setup(self, ctx: TradingContext) -> None:
        """Subscribe to price forecast and set a moderate threshold."""
        self.subscribe_signal("price_forecast")
        self.threshold: Decimal = Decimal("5.0")
        self.volume_mw: Decimal = Decimal("10")

    def on_auction_open(self, ctx: TradingContext, auction: AuctionInfo) -> None:
        """Place a buy order when forecast indicates moderate undervaluation."""
        try:
            signal = ctx.get_signal("price_forecast")
        except SignalError:
            return

        bid = Decimal(str(signal.value)) - self.threshold
        ctx.place_order(
            Order.buy(
                product_id=auction.product_id,
                volume_mw=self.volume_mw,
                price_eur_mwh=bid,
            )
        )


class AggressiveAlgo(SimpleAlgo):
    """Buy whenever forecast is above clearing price (no meaningful threshold).

    Maximises fill count at the cost of lower average alpha per trade.
    Suitable for desks focused on volume and market participation.
    """

    def on_setup(self, ctx: TradingContext) -> None:
        """Subscribe to price forecast and set a minimal threshold."""
        self.subscribe_signal("price_forecast")
        self.threshold: Decimal = Decimal("0.1")  # Near-zero threshold
        self.volume_mw: Decimal = Decimal("10")

    def on_auction_open(self, ctx: TradingContext, auction: AuctionInfo) -> None:
        """Place a buy order whenever the forecast is above clearing price."""
        try:
            signal = ctx.get_signal("price_forecast")
        except SignalError:
            return

        bid = Decimal(str(signal.value)) - self.threshold
        ctx.place_order(
            Order.buy(
                product_id=auction.product_id,
                volume_mw=self.volume_mw,
                price_eur_mwh=bid,
            )
        )


def run_comparison() -> None:
    """Run all three strategies and print the comparison report."""
    data_dir = Path("tests/fixtures")

    if not (data_dir / "da_prices.parquet").exists():
        print("ERROR: tests/fixtures/da_prices.parquet not found.")
        print("Run: python tests/fixtures/generate.py")
        sys.exit(1)

    print("Running multi-algo comparison...")
    print("Strategies: conservative (threshold=8), moderate (threshold=5), aggressive (threshold=0.1)")
    print()

    engine = SharedReplayEngine(
        algos={
            "conservative": ConservativeAlgo(),
            "moderate": ModerateAlgo(),
            "aggressive": AggressiveAlgo(),
        },
        exchange="nordpool",
        start=date(2026, 3, 1),
        end=date(2026, 3, 31),
        products=["NO1_DA"],
        data_dir=data_dir,
        initial_capital=Decimal("100000"),
    )

    comparison = engine.run()

    print(comparison.summary())

    ranked = comparison.ranking("total_pnl")
    print(f"\nRanking by Total PnL: {' > '.join(ranked)}")

    ranked_sharpe = comparison.ranking("sharpe_ratio")
    print(f"Ranking by Sharpe:    {' > '.join(ranked_sharpe)}")

    # Write an HTML comparison report
    report_path = "/tmp/nexa_comparison_report.html"
    comparison.to_html(report_path)
    print(f"\nHTML comparison report written to: {report_path}")

    # Write a JSON export
    json_path = "/tmp/nexa_comparison_results.json"
    comparison.to_json(json_path)
    print(f"JSON export written to:             {json_path}")


if __name__ == "__main__":
    run_comparison()
