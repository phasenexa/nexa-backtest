"""SimpleAlgo base class and the ``@algo`` decorator for trading algorithms.

Two APIs are provided:

- **SimpleAlgo**: subclass and override lifecycle hooks.  Best for simple
  strategies and beginners.
- **@algo decorator**: marks an async function as a trading algo that
  consumes the full event stream via ``ctx.events()``.  Best for complex
  strategies that need fine-grained control over the event loop.

Both APIs use the same :class:`~nexa_backtest.context.TradingContext`
protocol and are engine-agnostic.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Coroutine
from typing import Any

from nexa_backtest.context import SignalValue, TradingContext
from nexa_backtest.exceptions import AlgoError
from nexa_backtest.types import AuctionInfo, Fill


def algo(
    name: str,
    version: str,
) -> Callable[[Callable[..., Coroutine[Any, Any, None]]], Callable[..., Coroutine[Any, Any, None]]]:
    """Decorator that marks an async function as a trading algo.

    The decorated function receives a :class:`~nexa_backtest.context.TradingContext`
    and must consume events via ``ctx.events()``.  The engine drives the algo
    by pushing market events onto an internal async queue; the algo processes
    them one at a time.

    Args:
        name: Human-readable algo name, recorded in the
            :class:`~nexa_backtest.analysis.metrics.BacktestResult`.
        version: Semantic version string, e.g. ``"1.0.0"``.

    Returns:
        A decorator that validates and annotates the decorated function.

    Raises:
        :class:`~nexa_backtest.exceptions.AlgoError`: If the decorated
            function is not ``async``, or does not accept exactly one argument
            (the ``TradingContext``).

    Example::

        @algo(name="spread_scalper", version="1.0.0")
        async def run(ctx: TradingContext) -> None:
            async for event in ctx.events():
                match event:
                    case MarketDataUpdate():
                        ...
                    case BarEvent():
                        ctx.place_order(Order.buy(...))
    """

    def decorator(
        fn: Callable[..., Coroutine[Any, Any, None]],
    ) -> Callable[..., Coroutine[Any, Any, None]]:
        if not inspect.iscoroutinefunction(fn):
            raise AlgoError(
                f"@algo decorated function must be async. "
                f"Got: {fn.__name__!r} (not a coroutine function)."
            )

        sig = inspect.signature(fn)
        if len(sig.parameters) != 1:
            raise AlgoError(
                f"@algo decorated function must accept exactly one argument "
                f"(the TradingContext). "
                f"Got: {fn.__name__!r} with {len(sig.parameters)} parameters."
            )

        # Attach metadata so the engine can identify @algo functions.
        fn._is_algo = True  # type: ignore[attr-defined]
        fn._algo_name = name  # type: ignore[attr-defined]
        fn._algo_version = version  # type: ignore[attr-defined]
        return fn

    return decorator


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

    def on_bar(self, ctx: TradingContext) -> None:
        """Called at each MTU boundary (every 15 minutes) during IDC replay.

        Use this hook for periodic strategy evaluation in intraday continuous
        markets.  Orders placed here are matched against the current state of
        the historical order book before the next bar is processed.

        Args:
            ctx: The trading context.
        """

    def on_cancel(self, ctx: TradingContext, order_id: str, reason: str) -> None:
        """Called when one of the algo's orders is cancelled.

        This is fired both when the algo explicitly cancels an order *and*
        when the engine cancels it automatically (e.g. gate closure).

        Args:
            ctx: The trading context.
            order_id: ID of the cancelled order.
            reason: Human-readable reason for the cancellation, e.g.
                ``"gate_closure"`` or ``"algo_cancel"``.
        """

    def on_gate_closure(self, ctx: TradingContext, product_id: str) -> None:
        """Called when gate closes for an IDC product.

        Any resting algo orders for ``product_id`` have already been cancelled
        before this hook fires.

        Args:
            ctx: The trading context.
            product_id: Exchange product identifier whose gate just closed.
        """

    def on_error(self, ctx: TradingContext, error: Exception) -> None:
        """Called when an unhandled error occurs during event processing.

        The default implementation re-raises the exception.  Override to
        implement custom error handling or logging.

        Args:
            ctx: The trading context.
            error: The exception that was raised.
        """
        raise error
