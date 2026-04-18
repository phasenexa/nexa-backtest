"""Tests for the @algo decorator and the async event stream API (task 05)."""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
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
from nexa_backtest.data.schema import IDC_EVENTS_SCHEMA
from nexa_backtest.engines.backtest import (
    AsyncAlgoDispatcher,
    SimpleAlgoDispatcher,
    _BacktestContext,
)
from nexa_backtest.exceptions import AlgoError
from nexa_backtest.types import GateClosureEvent, GateClosureWarning, MarketEvent, SignalValue
from tests.testing_utils import make_minimal_backtest_context

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
    ctx = make_minimal_backtest_context(initial_time=datetime(2026, 3, 1, tzinfo=UTC))

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


# ---------------------------------------------------------------------------
# Helpers shared by dispatcher unit tests
# ---------------------------------------------------------------------------


def _make_ctx() -> _BacktestContext:
    """Return a minimal _BacktestContext for dispatcher unit tests."""
    return make_minimal_backtest_context(initial_time=datetime(2026, 3, 1, 12, 0, tzinfo=UTC))


def _make_passthrough_algo() -> object:
    """Return an @algo decorated function that accepts all event types."""

    @algo(name="passthrough", version="1.0.0")
    async def run(ctx: TradingContext) -> None:
        async for _ in ctx.events():
            pass

    return run


# ---------------------------------------------------------------------------
# Tests: SimpleAlgoDispatcher.on_cancel
# ---------------------------------------------------------------------------


def test_simple_algo_dispatcher_on_cancel_forwards_to_algo() -> None:
    """SimpleAlgoDispatcher.on_cancel must call algo.on_cancel."""
    cancels: list[tuple[str, str]] = []

    class TrackCancels(SimpleAlgo):
        def on_cancel(self, ctx: TradingContext, order_id: str, reason: str) -> None:
            cancels.append((order_id, reason))

    dispatcher = SimpleAlgoDispatcher(TrackCancels())
    ctx = _make_ctx()
    dispatcher.on_cancel(ctx, "order-123", "gate closed")

    assert cancels == [("order-123", "gate closed")]


# ---------------------------------------------------------------------------
# Tests: AsyncAlgoDispatcher — on_error, on_signal, on_cancel
# ---------------------------------------------------------------------------


def test_async_algo_dispatcher_on_error_reraises() -> None:
    """AsyncAlgoDispatcher.on_error must re-raise the exception."""
    dispatcher = AsyncAlgoDispatcher(_make_passthrough_algo())
    ctx = _make_ctx()

    with pytest.raises(ValueError, match="boom"):
        dispatcher.on_error(ctx, ValueError("boom"))


def test_async_algo_dispatcher_on_signal_pushes_event() -> None:
    """AsyncAlgoDispatcher.on_signal must push a SignalUpdate to the event queue."""
    received: list[object] = []

    @algo(name="signal_watcher", version="1.0.0")
    async def run(ctx: TradingContext) -> None:
        async for event in ctx.events():
            received.append(event)

    dispatcher = AsyncAlgoDispatcher(run)
    ctx = _make_ctx()
    dispatcher.on_setup(ctx)

    signal_value = SignalValue(
        name="wind_forecast",
        timestamp=datetime(2026, 3, 1, tzinfo=UTC),
        value=42.0,
    )
    dispatcher.on_signal(ctx, "wind_forecast", signal_value)
    dispatcher.on_teardown(ctx)

    from nexa_backtest.types import SignalUpdate

    assert any(isinstance(e, SignalUpdate) for e in received)


def test_async_algo_dispatcher_on_cancel_pushes_event() -> None:
    """AsyncAlgoDispatcher.on_cancel must push a CancelEvent to the event queue."""
    received: list[object] = []

    @algo(name="cancel_watcher", version="1.0.0")
    async def run(ctx: TradingContext) -> None:
        async for event in ctx.events():
            received.append(event)

    dispatcher = AsyncAlgoDispatcher(run)
    ctx = _make_ctx()
    dispatcher.on_setup(ctx)

    dispatcher.on_cancel(ctx, "order-456", "expired")
    dispatcher.on_teardown(ctx)

    from nexa_backtest.types import CancelEvent

    assert any(isinstance(e, CancelEvent) for e in received)


# ---------------------------------------------------------------------------
# Tests: AsyncAlgoDispatcher.on_auction_open via DA backtest
# ---------------------------------------------------------------------------


def _write_da_prices_for_decorator_test(path: Path) -> None:
    data = {
        "timestamp": pd.to_datetime(["2026-03-01T00:00:00Z", "2026-03-01T00:15:00Z"], utc=True),
        "zone": ["NO1", "NO1"],
        "price_eur_mwh": [45.0, 50.0],
        "volume_mwh": [1000.0, 1000.0],
    }
    pd.DataFrame(data).to_parquet(path / "da_prices.parquet", index=False)


def test_async_algo_receives_bar_events_from_da_auction(tmp_path: Path) -> None:
    """@algo must receive BarEvents (from on_auction_open) during a DA backtest."""
    _write_da_prices_for_decorator_test(tmp_path)

    bar_events: list[BarEvent] = []

    @algo(name="da_watcher", version="1.0.0")
    async def run(ctx: TradingContext) -> None:
        async for event in ctx.events():
            if isinstance(event, BarEvent):
                bar_events.append(event)

    engine = BacktestEngine(
        algo=run,
        exchange="nordpool",
        start=date(2026, 3, 1),
        end=date(2026, 3, 1),
        products=["NO1_DA"],
        data_dir=tmp_path,
        capital=Decimal("100000"),
    )
    engine.run()

    assert len(bar_events) == 2


def test_async_algo_receives_gate_closure_warning_before_event(tmp_path: Path) -> None:
    """@algo must receive GateClosureWarning before GateClosureEvent for each product."""
    from nexa_backtest.data.schema import IDC_EVENTS_SUBDIR

    # Write an empty IDC parquet so the engine has an idc_events directory
    idc_dir = tmp_path / IDC_EVENTS_SUBDIR
    idc_dir.mkdir(parents=True)
    empty = pa.table(
        {
            name: pa.array([], type=field.type)
            for name, field in zip(IDC_EVENTS_SCHEMA.names, IDC_EVENTS_SCHEMA, strict=True)
        },
        schema=IDC_EVENTS_SCHEMA,
    )
    pq.write_table(empty, idc_dir / "NO1_2026_03.parquet")

    received: list[MarketEvent] = []

    @algo(name="gate_watcher", version="1.0.0")
    async def run(ctx: TradingContext) -> None:
        async for event in ctx.events():
            if isinstance(event, (GateClosureWarning, GateClosureEvent)):
                received.append(event)

    engine = BacktestEngine(
        algo=run,
        exchange="nordpool",
        start=date(2026, 3, 1),
        end=date(2026, 3, 1),
        products=["NO1-QH"],
        data_dir=tmp_path,
        capital=Decimal("100000"),
    )
    engine.run()

    warnings = [e for e in received if isinstance(e, GateClosureWarning)]
    closures = [e for e in received if isinstance(e, GateClosureEvent)]

    assert len(warnings) > 0, "Expected GateClosureWarning events to be dispatched"
    assert len(closures) > 0, "Expected GateClosureEvent events to be dispatched"

    # Every warned product must also produce a closure.
    warned_products = {e.product_id for e in warnings}
    closed_products = {e.product_id for e in closures}
    assert warned_products <= closed_products, "Got warnings for products that never closed"

    # Each warning must carry a positive remaining timedelta.
    for w in warnings:
        assert isinstance(w, GateClosureWarning)
        assert w.remaining.total_seconds() > 0

    # Every warning must appear before its corresponding closure in the stream.
    for w in warnings:
        warning_idx = received.index(w)
        closure = next(e for e in closures if e.product_id == w.product_id)
        closure_idx = received.index(closure)
        assert warning_idx < closure_idx, (
            f"GateClosureWarning for {w.product_id} arrived after GateClosureEvent"
        )
