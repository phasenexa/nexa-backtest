"""SimpleAlgo base class for the hook-based trading algorithm interface.

Subclass :class:`SimpleAlgo` and override the hooks you need. All hooks are
no-ops by default. The algo must never import engine-specific classes; all
market interaction goes through :class:`~nexa_backtest.context.TradingContext`.
"""

from __future__ import annotations

from nexa_backtest.context import SignalValue, TradingContext
from nexa_backtest.types import AuctionInfo, Fill


class SimpleAlgo:
    """Base class for simple hook-based trading algorithms.

    Override the lifecycle hooks to implement your strategy. Hooks are called
    by the engine in the order: :meth:`on_setup`, then auction hooks per
    product, then :meth:`on_teardown`.

    Example::

        class MyAlgo(SimpleAlgo):
            def on_setup(self, ctx: TradingContext) -> None:
                self.subscribe_signal("wind_forecast")

            def on_auction_open(
                self, ctx: TradingContext, auction: AuctionInfo
            ) -> None:
                signal = ctx.get_signal("wind_forecast")
                if signal.value > 1000:
                    ctx.place_order(
                        Order.buy(
                            product_id=auction.product_id,
                            volume_mw=10,
                            price_eur_mwh=45.0,
                        )
                    )

    The algo is engine-agnostic: the same code runs under the backtest,
    paper, and live engines without modification.
    """

    def __init__(self) -> None:
        self._subscribed_signals: list[str] = []

    def subscribe_signal(self, name: str) -> None:
        """Register interest in a named signal.

        Call this from :meth:`on_setup`. The engine will ensure the signal
        provider is registered before the first auction hook fires.

        Args:
            name: Signal name to subscribe to, e.g. ``"price_forecast"``.
        """
        if name not in self._subscribed_signals:
            self._subscribed_signals.append(name)

    def on_setup(self, ctx: TradingContext) -> None:
        """Called once before the backtest begins.

        Use this hook to subscribe to signals and initialise any persistent
        state (thresholds, counters, etc.).

        Args:
            ctx: The trading context.
        """

    def on_auction_open(self, ctx: TradingContext, auction: AuctionInfo) -> None:
        """Called once per auction product when the auction opens.

        Place orders by calling
        :meth:`~nexa_backtest.context.TradingContext.place_order`. Any order
        placed here will be matched against the historical clearing price for
        ``auction.product_id``.

        Args:
            ctx: The trading context.
            auction: Metadata about the auction product opening.
        """

    def on_fill(self, ctx: TradingContext, fill: Fill) -> None:
        """Called immediately after one of the algo's orders is filled.

        Args:
            ctx: The trading context.
            fill: Details of the fill, including price, volume, and side.
        """

    def on_signal(self, ctx: TradingContext, name: str, value: SignalValue) -> None:
        """Called when a subscribed signal updates.

        For DA backtesting this is called once per delivery day, before any
        :meth:`on_auction_open` calls for that day. Signals can also be
        polled directly at any time via
        :meth:`~nexa_backtest.context.TradingContext.get_signal`.

        Args:
            ctx: The trading context.
            name: Signal name matching the name passed to
                :meth:`subscribe_signal`.
            value: The latest signal value visible at the current simulated
                time.
        """

    def on_teardown(self, ctx: TradingContext) -> None:
        """Called once after the backtest ends.

        Use this hook to log final positions or perform any cleanup.

        Args:
            ctx: The trading context.
        """
