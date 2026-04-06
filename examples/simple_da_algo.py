"""Example DA algo: buy when the clearing price is below a price forecast.

Demonstrates the signal system: the algo subscribes to a price forecast CSV
and places buy orders only when the forecast is meaningfully above the
current clearing price, suggesting the market is undervaluing the delivery
period.

Run via::

    nexa run examples/simple_da_algo.py \\
        --exchange nordpool \\
        --start 2026-03-01 \\
        --end 2026-03-31 \\
        --products NO1_DA \\
        --data-dir tests/fixtures \\
        --capital 100000
"""

from __future__ import annotations

from decimal import Decimal

from nexa_backtest.algo import SimpleAlgo
from nexa_backtest.context import TradingContext
from nexa_backtest.exceptions import SignalError
from nexa_backtest.types import AuctionInfo, Fill, Order


class ForecastAlgo(SimpleAlgo):
    """Buy when the DA clearing price is below the price forecast minus a threshold.

    The algo subscribes to a ``price_forecast`` signal (loaded from
    ``{data_dir}/signals/price_forecast.csv``) and places a buy order for
    each delivery product where the forecast indicates the price will be at
    least ``threshold`` EUR/MWh above the clearing price.

    Because we fill at the clearing price (price-taker assumption), the order
    is placed at ``forecast - threshold``. If that bid is >= the clearing
    price, we fill. This means the algo selects delivery periods it expects to
    be cheap relative to the forecast, accumulating positive VWAP alpha.
    """

    def on_setup(self, ctx: TradingContext) -> None:
        """Subscribe to the price forecast signal and set the bid threshold."""
        self.subscribe_signal("price_forecast")
        self.threshold: Decimal = Decimal("5.0")
        self.volume_mw: Decimal = Decimal("10")

    def on_auction_open(self, ctx: TradingContext, auction: AuctionInfo) -> None:
        """Place a buy order when the forecast exceeds clearing + threshold."""
        try:
            signal = ctx.get_signal("price_forecast")
        except SignalError:
            # No forecast available yet for this period — skip
            return

        forecast = Decimal(str(signal.value))
        bid_price = forecast - self.threshold

        ctx.place_order(
            Order.buy(
                product_id=auction.product_id,
                volume_mw=self.volume_mw,
                price_eur_mwh=bid_price,
            )
        )

    def on_fill(self, ctx: TradingContext, fill: Fill) -> None:
        """Log each fill for visibility."""
        ctx.log(
            f"FILL {fill.side} {fill.volume} MW @ {fill.price} EUR/MWh "
            f"[{fill.product_id}]"
        )

    def on_teardown(self, ctx: TradingContext) -> None:
        """Log the final unrealised PnL."""
        ctx.log(f"Backtest complete. Unrealised PnL: {ctx.get_unrealised_pnl()} EUR")
