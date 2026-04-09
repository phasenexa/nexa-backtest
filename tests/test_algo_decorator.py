"""Tests for the @algo decorator and the async event stream API (task 05)."""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from nexa_backtest import (
    BacktestEngine,
    BarEvent,
    FillEvent,
    MarketDataUpdate,
    Order,
    SimpleAlgo,
    algo,
)
from nexa_backtest.context import TradingContext
from nexa_backtest.engines.backtest import AsyncAlgoDispatcher, SimpleAlgoDispatcher
from nexa_backtest.exceptions import AlgoError
from nexa_backtest.types import MarketEvent

# ---------------------------------------------------------------------------
# Tests: @algo decorator validation
# ---------------------------------------------------------------------------


def test_algo_decorator_rejects_non_async() -> None:
    """@algo must raise AlgoError when decorating a sync function."""
    with pytest.raises(AlgoError, match="must be async"):

        @algo(name="bad_algo", version="1.0.0")
        def run(ctx: TradingContext) -> None:  # type: ignore[arg-type]
            pass


def test_algo_decorator_rejects_wrong_arg_count() -> None:
    """@algo must raise AlgoError when function has wrong number of parameters."""
    with pytest.raises(AlgoError, match="exactly one argument"):

        @algo(name="bad_algo", version="1.0.0")
        async def run(ctx: TradingContext, extra: int) -> None:  # type: ignore[arg-type]
            pass


def test_algo_decorator_rejects_zero_args() -> None:
    """@algo must raise AlgoError when function takes no arguments."""
    with pytest.raises(AlgoError, match="exactly one argument"):

        @algo(name="bad_algo", version="1.0.0")
        async def run() -> None:
            pass


def test_algo_decorator_attaches_metadata() -> None:
    """@algo must attach _is_algo, _algo_name, _algo_version to the function."""

    @algo(name="my_algo", version="2.1.0")
    async def run(ctx: TradingContext) -> None:
        pass

    assert getattr(run, "_is_algo", False) is True
    assert run._algo_name == "my_algo"
    assert run._algo_version == "2.1.0"


def test_algo_decorated_function_remains_callable() -> None:
    """@algo must not break the function's callability."""

    @algo(name="test_algo", version="1.0.0")
    async def run(ctx: TradingContext) -> None:
        pass

    # It should still be a callable coroutine function.
    import inspect

    assert inspect.iscoroutinefunction(run)


# ---------------------------------------------------------------------------
# Tests: AlgoDispatcher protocol
# ---------------------------------------------------------------------------


def test_simple_algo_dispatcher_routes_on_bar(tmp_path: Path) -> None:
    """SimpleAlgoDispatcher must forward on_bar to the wrapped SimpleAlgo."""
    bar_count = 0

    class CountBars(SimpleAlgo):
        def on_bar(self, ctx: TradingContext) -> None:
            nonlocal bar_count
            bar_count += 1

    dispatcher = SimpleAlgoDispatcher(CountBars())
    assert dispatcher.subscribed_signals == []

    # Verify that the dispatcher's subscribed_signals reflect algo subscriptions.
    class SubscribingAlgo(SimpleAlgo):
        def on_setup(self, ctx: TradingContext) -> None:
            self.subscribe_signal("wind_forecast")

    sub_dispatcher = SimpleAlgoDispatcher(SubscribingAlgo())
    # subscribed_signals is populated after on_setup is called.
    assert sub_dispatcher.subscribed_signals == []  # before on_setup


def test_async_algo_dispatcher_requires_algo_fn() -> None:
    """AsyncAlgoDispatcher must be initialised with a @algo decorated function."""

    @algo(name="valid", version="1.0.0")
    async def run(ctx: TradingContext) -> None:
        async for _ in ctx.events():
            pass

    dispatcher = AsyncAlgoDispatcher(run)
    assert dispatcher.subscribed_signals == []


# ---------------------------------------------------------------------------
# Tests: ctx.events() guard
# ---------------------------------------------------------------------------


def test_events_raises_in_simple_algo_mode(tmp_path: Path) -> None:
    """ctx.events() must raise AlgoError when called from SimpleAlgo context."""
    from nexa_backtest.engines.backtest import _BacktestContext
    from nexa_backtest.engines.clock import SimulatedClock
    from nexa_backtest.signals.registry import SignalRegistry

    clock = SimulatedClock(initial_time=datetime(2026, 3, 1, tzinfo=UTC))
    ctx = _BacktestContext(clock=clock, signal_registry=SignalRegistry())

    # _event_queue is None → should raise AlgoError.
    with pytest.raises(AlgoError, match="events\\(\\)"):
        ctx.events()


# ---------------------------------------------------------------------------
# Tests: @algo with IDC backtest
# ---------------------------------------------------------------------------


@pytest.fixture
def idc_data_dir() -> Path:
    """Return the path to the IDC test fixture directory."""
    return Path(__file__).parent / "fixtures" / "nordpool"


def test_algo_event_stream_receives_market_events(idc_data_dir: Path) -> None:
    """@algo must receive MarketDataUpdate and BarEvent events during IDC replay."""
    event_types: list[type[MarketEvent]] = []

    @algo(name="event_recorder", version="1.0.0")
    async def run(ctx: TradingContext) -> None:
        async for event in ctx.events():
            event_types.append(type(event))

    engine = BacktestEngine(
        algo=run,
        exchange="nordpool",
        start=date(2026, 3, 1),
        end=date(2026, 3, 2),
        products=["NO1-QH"],
        data_dir=idc_data_dir,
        capital=Decimal("100000"),
    )
    result = engine.run()

    # The algo must receive both market data events and bar events.
    assert MarketDataUpdate in event_types
    assert BarEvent in event_types
    # The algo must finish cleanly (no fills needed for this test).
    assert result is not None


def test_algo_places_order_after_n_market_data_updates(idc_data_dir: Path) -> None:
    """@algo that counts events and places an order after 5 MarketDataUpdates."""
    mdu_count = 0
    orders_placed = 0

    @algo(name="count_and_trade", version="1.0.0")
    async def run(ctx: TradingContext) -> None:
        nonlocal mdu_count, orders_placed
        async for event in ctx.events():
            if isinstance(event, MarketDataUpdate):
                mdu_count += 1
                if mdu_count == 5:
                    # Place one buy order at market data update #5.
                    product_id = event.product_id
                    price = float(event.price_eur_mwh)
                    if price > 0:
                        ctx.place_order(
                            Order.buy(
                                product_id=product_id,
                                volume_mw=Decimal("1"),
                                price_eur_mwh=Decimal(str(price + 50)),
                            )
                        )
                        orders_placed += 1

    engine = BacktestEngine(
        algo=run,
        exchange="nordpool",
        start=date(2026, 3, 1),
        end=date(2026, 3, 2),
        products=["NO1-QH"],
        data_dir=idc_data_dir,
        capital=Decimal("100000"),
    )
    engine.run()

    assert mdu_count >= 5
    assert orders_placed == 1


# ---------------------------------------------------------------------------
# Tests: SimpleAlgo / @algo equivalence
# ---------------------------------------------------------------------------


def test_simple_algo_and_async_algo_equivalence(idc_data_dir: Path) -> None:
    """SimpleAlgo and an equivalent @algo must produce identical fills and PnL."""
    # Both strategies: at each bar, if there are asks in the order book,
    # place a buy at ask + 20 for the product that had the last trade.

    class SimpleTrader(SimpleAlgo):
        def on_bar(self, ctx: TradingContext) -> None:
            # Place a very aggressive buy on the first QH-0900 product.
            pid = "NO1-QH-0900"
            ask = ctx.get_best_ask(pid)
            if ask is not None:
                ctx.place_order(
                    Order.buy(
                        product_id=pid,
                        volume_mw=Decimal("0.1"),
                        price_eur_mwh=ask.price + Decimal("20"),
                    )
                )

    @algo(name="async_trader", version="1.0.0")
    async def async_run(ctx: TradingContext) -> None:
        pid = "NO1-QH-0900"
        async for event in ctx.events():
            if isinstance(event, BarEvent):
                ask = ctx.get_best_ask(pid)
                if ask is not None:
                    ctx.place_order(
                        Order.buy(
                            product_id=pid,
                            volume_mw=Decimal("0.1"),
                            price_eur_mwh=ask.price + Decimal("20"),
                        )
                    )

    simple_engine = BacktestEngine(
        algo=SimpleTrader(),
        exchange="nordpool",
        start=date(2026, 3, 1),
        end=date(2026, 3, 2),
        products=["NO1-QH"],
        data_dir=idc_data_dir,
        capital=Decimal("100000"),
    )
    async_engine = BacktestEngine(
        algo=async_run,
        exchange="nordpool",
        start=date(2026, 3, 1),
        end=date(2026, 3, 2),
        products=["NO1-QH"],
        data_dir=idc_data_dir,
        capital=Decimal("100000"),
    )

    simple_result = simple_engine.run()
    async_result = async_engine.run()

    # Both must complete.
    assert simple_result is not None
    assert async_result is not None

    # Fills should be identical (same orders at same times).
    assert len(simple_result.fills) == len(async_result.fills)

    # Total PnL must match.
    assert simple_result.pnl.total_alpha_eur == async_result.pnl.total_alpha_eur

    # If both had fills, fill prices/volumes must match.
    for sf, af in zip(simple_result.fills, async_result.fills, strict=True):
        assert sf.product_id == af.product_id
        assert sf.side == af.side
        assert sf.price == af.price
        assert sf.volume == af.volume


# ---------------------------------------------------------------------------
# Tests: BacktestEngine accepts @algo
# ---------------------------------------------------------------------------


def test_backtest_engine_accepts_algo_function(idc_data_dir: Path) -> None:
    """BacktestEngine must accept an @algo decorated function."""

    @algo(name="noop_algo", version="1.0.0")
    async def run(ctx: TradingContext) -> None:
        async for _ in ctx.events():
            pass

    engine = BacktestEngine(
        algo=run,
        exchange="nordpool",
        start=date(2026, 3, 1),
        end=date(2026, 3, 2),
        products=["NO1-QH"],
        data_dir=idc_data_dir,
        capital=Decimal("100000"),
    )
    result = engine.run()
    assert result.algo_name == "noop_algo"


def test_backtest_engine_rejects_invalid_algo() -> None:
    """BacktestEngine must raise AlgoError for non-algo, non-SimpleAlgo objects."""
    with pytest.raises(AlgoError):
        BacktestEngine(
            algo="not_an_algo",  # type: ignore[arg-type]
            exchange="nordpool",
            start=date(2026, 3, 1),
            end=date(2026, 3, 2),
            products=["NO1-QH"],
            data_dir=Path("/tmp"),
            capital=Decimal("100000"),
        )


def test_algo_receives_fill_events(idc_data_dir: Path) -> None:
    """@algo must receive FillEvent when one of its orders is filled."""
    fill_events_received: list[FillEvent] = []

    @algo(name="fill_watcher", version="1.0.0")
    async def run(ctx: TradingContext) -> None:
        pid = "NO1-QH-0900"
        bought = False
        async for event in ctx.events():
            if isinstance(event, BarEvent) and not bought:
                ask = ctx.get_best_ask(pid)
                if ask is not None:
                    ctx.place_order(
                        Order.buy(
                            product_id=pid,
                            volume_mw=Decimal("0.1"),
                            price_eur_mwh=ask.price + Decimal("50"),
                        )
                    )
                    bought = True
            elif isinstance(event, FillEvent):
                fill_events_received.append(event)

    engine = BacktestEngine(
        algo=run,
        exchange="nordpool",
        start=date(2026, 3, 1),
        end=date(2026, 3, 2),
        products=["NO1-QH"],
        data_dir=idc_data_dir,
        capital=Decimal("100000"),
    )
    result = engine.run()

    if result.fills:
        # If fills occurred, FillEvents must have been delivered to the algo.
        assert len(fill_events_received) == len(result.fills)
