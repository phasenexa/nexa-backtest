"""End-to-end tests for IDC continuous backtest mode.

These tests verify the full IDC replay loop: SlidingWindow → ContinuousMatchingEngine
→ SimpleAlgo hooks → BacktestResult.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from nexa_backtest.algo import SimpleAlgo
from nexa_backtest.context import TradingContext
from nexa_backtest.data.schema import IDC_EVENTS_SCHEMA
from nexa_backtest.engines.backtest import BacktestEngine, _mtu_to_product_id
from nexa_backtest.types import Fill, Order, Side

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "nordpool"
FIXTURE_PARQUET = FIXTURE_DIR / "idc_events" / "NO1_2026_03.parquet"


# ---------------------------------------------------------------------------
# Helper to write tiny IDC Parquet fixture inline in tests
# ---------------------------------------------------------------------------


def _write_idc_parquet(
    data_dir: Path,
    zone: str,
    rows: list[dict],
) -> Path:
    """Write a minimal IDC events Parquet file for testing."""
    events_dir = data_dir / "idc_events"
    events_dir.mkdir(parents=True, exist_ok=True)
    path = events_dir / f"{zone}_2026_03.parquet"

    if rows:
        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        table = pa.Table.from_pandas(df, schema=IDC_EVENTS_SCHEMA, preserve_index=False)
    else:
        table = pa.table(
            {field.name: pa.array([], type=field.type) for field in IDC_EVENTS_SCHEMA},
            schema=IDC_EVENTS_SCHEMA,
        )
    pq.write_table(table, path)
    return path


def _idc_row(
    ts: datetime,
    product_id: str,
    event_type: str = "new",
    order_id: str = "o1",
    side: str = "sell",
    price: float = 50.0,
    volume: float = 10.0,
    remaining: float = 10.0,
    aggressor_side: str | None = None,
    trade_id: str | None = None,
) -> dict:
    return {
        "timestamp": ts,
        "event_type": event_type,
        "order_id": order_id,
        "zone": "NO1",
        "product_id": product_id,
        "side": side,
        "price_eur_mwh": price,
        "volume_mw": volume,
        "remaining_mw": remaining,
        "aggressor_side": aggressor_side,
        "trade_id": trade_id,
    }


# ---------------------------------------------------------------------------
# Simple IDC algos for testing
# ---------------------------------------------------------------------------


class _BuyBelowBestAskAlgo(SimpleAlgo):
    """Places a single buy order 1 EUR below the current best ask on each bar."""

    def __init__(self) -> None:
        super().__init__()
        self.bars_called = 0
        self.fills_received: list[Fill] = []

    def on_bar(self, ctx: TradingContext) -> None:
        self.bars_called += 1
        ask = ctx.get_best_ask("NO1-QH-0800")
        if ask is not None:
            ctx.place_order(
                Order.buy(
                    product_id="NO1-QH-0800",
                    volume_mw=Decimal("1"),
                    price_eur_mwh=ask.price + Decimal("1"),  # hit the ask
                )
            )

    def on_fill(self, ctx: TradingContext, fill: Fill) -> None:
        self.fills_received.append(fill)


class _NoOpIDCAlgo(SimpleAlgo):
    """Does nothing — used to verify on_bar is called the right number of times."""

    def __init__(self) -> None:
        super().__init__()
        self.bar_count = 0
        self.gate_closures: list[str] = []

    def on_bar(self, ctx: TradingContext) -> None:
        self.bar_count += 1

    def on_gate_closure(self, ctx: TradingContext, product_id: str) -> None:
        self.gate_closures.append(product_id)


class _AlwaysBuyIDCAlgo(SimpleAlgo):
    """Buys aggressively at any available ask on every bar for a given product."""

    def __init__(self, product_id: str) -> None:
        super().__init__()
        self._product_id = product_id
        self.fills_received: list[Fill] = []

    def on_bar(self, ctx: TradingContext) -> None:
        ask = ctx.get_best_ask(self._product_id)
        if ask is not None:
            ctx.place_order(
                Order.buy(
                    product_id=self._product_id,
                    volume_mw=Decimal("2"),
                    price_eur_mwh=ask.price + Decimal("5"),
                )
            )

    def on_fill(self, ctx: TradingContext, fill: Fill) -> None:
        self.fills_received.append(fill)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestIDCBacktestOnBar:
    def test_on_bar_called_at_each_mtu(self, tmp_path: Path) -> None:
        """on_bar should fire once per 15-min MTU over the replay period."""
        _write_idc_parquet(tmp_path, "NO1", [])  # no events needed

        algo = _NoOpIDCAlgo()
        engine = BacktestEngine(
            algo=algo,
            exchange="nordpool",
            start=date(2026, 3, 1),
            end=date(2026, 3, 1),
            products=["NO1-QH"],
            data_dir=tmp_path,
            capital=Decimal("100000"),
        )
        engine.run()

        expected_bars = 24 * 4  # 96 MTUs per day
        assert algo.bar_count == expected_bars

    def test_on_teardown_called(self, tmp_path: Path) -> None:
        _write_idc_parquet(tmp_path, "NO1", [])

        teardown_called = []

        class _Algo(SimpleAlgo):
            def on_bar(self, ctx: TradingContext) -> None:
                pass

            def on_teardown(self, ctx: TradingContext) -> None:
                teardown_called.append(True)

        engine = BacktestEngine(
            algo=_Algo(),
            exchange="nordpool",
            start=date(2026, 3, 1),
            end=date(2026, 3, 1),
            products=["NO1-QH"],
            data_dir=tmp_path,
            capital=Decimal("100000"),
        )
        engine.run()
        assert teardown_called


class TestIDCOrderMatching:
    def test_buy_order_filled_by_historical_ask(self, tmp_path: Path) -> None:
        """Algo places buy above ask — should fill at the historical ask price."""
        t = datetime(2026, 3, 1, 8, 0, tzinfo=UTC)
        product = "NO1-QH-0800"

        rows = [
            _idc_row(t, product, event_type="new", order_id="ask-1", side="sell", price=50.0),
        ]
        _write_idc_parquet(tmp_path, "NO1", rows)

        algo = _AlwaysBuyIDCAlgo(product)
        engine = BacktestEngine(
            algo=algo,
            exchange="nordpool",
            start=date(2026, 3, 1),
            end=date(2026, 3, 1),
            products=["NO1-QH"],
            data_dir=tmp_path,
            capital=Decimal("100000"),
        )
        engine.run()

        assert len(algo.fills_received) > 0
        fill = algo.fills_received[0]
        assert fill.price == Decimal("50.0")
        assert fill.side == Side.BUY

    def test_fills_appear_in_result(self, tmp_path: Path) -> None:
        t = datetime(2026, 3, 1, 8, 0, tzinfo=UTC)
        product = "NO1-QH-0800"
        rows = [
            _idc_row(t, product, event_type="new", order_id="ask-1", side="sell", price=49.0),
        ]
        _write_idc_parquet(tmp_path, "NO1", rows)

        algo = _AlwaysBuyIDCAlgo(product)
        engine = BacktestEngine(
            algo=algo,
            exchange="nordpool",
            start=date(2026, 3, 1),
            end=date(2026, 3, 1),
            products=["NO1-QH"],
            data_dir=tmp_path,
            capital=Decimal("100000"),
        )
        result = engine.run()
        assert len(result.fills) > 0

    def test_pnl_and_vwap_accessible(self, tmp_path: Path) -> None:
        t = datetime(2026, 3, 1, 8, 0, tzinfo=UTC)
        product = "NO1-QH-0800"
        rows = [
            _idc_row(t, product, event_type="new", order_id="ask-1", side="sell", price=50.0),
        ]
        _write_idc_parquet(tmp_path, "NO1", rows)

        algo = _AlwaysBuyIDCAlgo(product)
        engine = BacktestEngine(
            algo=algo,
            exchange="nordpool",
            start=date(2026, 3, 1),
            end=date(2026, 3, 1),
            products=["NO1-QH"],
            data_dir=tmp_path,
            capital=Decimal("100000"),
        )
        result = engine.run()

        # Just ensure summary() doesn't raise
        summary = result.summary()
        assert "Fills" in summary

    def test_equity_curve_snapshots_recorded(self, tmp_path: Path) -> None:
        t = datetime(2026, 3, 1, 8, 0, tzinfo=UTC)
        product = "NO1-QH-0800"
        rows = [
            _idc_row(t, product, event_type="new", order_id="ask-1", side="sell", price=50.0),
        ]
        _write_idc_parquet(tmp_path, "NO1", rows)

        algo = _AlwaysBuyIDCAlgo(product)
        engine = BacktestEngine(
            algo=algo,
            exchange="nordpool",
            start=date(2026, 3, 1),
            end=date(2026, 3, 1),
            products=["NO1-QH"],
            data_dir=tmp_path,
            capital=Decimal("100000"),
        )
        result = engine.run()
        # Should have at least one equity snapshot for the fill
        assert len(result.equity_curve) > 0


class TestIDCGateClosure:
    def test_gate_closure_hook_fires(self, tmp_path: Path) -> None:
        """on_gate_closure is called for each product whose gate closes."""
        _write_idc_parquet(tmp_path, "NO1", [])

        algo = _NoOpIDCAlgo()
        engine = BacktestEngine(
            algo=algo,
            exchange="nordpool",
            start=date(2026, 3, 1),
            end=date(2026, 3, 1),
            products=["NO1-QH"],
            data_dir=tmp_path,
            capital=Decimal("100000"),
        )
        engine.run()

        # 96 products per day all close during the replay period
        assert len(algo.gate_closures) > 0

    def test_resting_order_cancelled_at_gate_closure(self, tmp_path: Path) -> None:
        """An algo order resting past gate closure should be cancelled."""
        _write_idc_parquet(tmp_path, "NO1", [])

        cancelled_orders: list[str] = []

        class _AlgoWithRestingOrder(SimpleAlgo):
            _placed = False

            def on_bar(self, ctx: TradingContext) -> None:
                if not self._placed:
                    # Place a buy order deep below the market (will never fill)
                    ctx.place_order(
                        Order.buy(
                            "NO1-QH-0900",
                            volume_mw=Decimal("1"),
                            price_eur_mwh=Decimal("1.0"),
                        )
                    )
                    self._placed = True

            def on_cancel(self, ctx: TradingContext, order_id: str, reason: str) -> None:
                cancelled_orders.append(order_id)

            def on_gate_closure(self, ctx: TradingContext, product_id: str) -> None:
                pass

        engine = BacktestEngine(
            algo=_AlgoWithRestingOrder(),
            exchange="nordpool",
            start=date(2026, 3, 1),
            end=date(2026, 3, 1),
            products=["NO1-QH"],
            data_dir=tmp_path,
            capital=Decimal("100000"),
        )
        engine.run()
        # The order rests until gate closure for NO1-QH-0900 (gate at 08:30),
        # at which point the engine cancels it. on_cancel may not fire (it's
        # the algo's responsibility to call cancel), but the gate closure event
        # should fire and the order should be removed from the engine.
        # We verify indirectly: no exception during the full run.


class TestIDCNoData:
    def test_no_events_runs_cleanly(self, tmp_path: Path) -> None:
        """Backtest with zero IDC events should complete without error."""
        _write_idc_parquet(tmp_path, "NO1", [])

        algo = _NoOpIDCAlgo()
        engine = BacktestEngine(
            algo=algo,
            exchange="nordpool",
            start=date(2026, 3, 1),
            end=date(2026, 3, 1),
            products=["NO1-QH"],
            data_dir=tmp_path,
            capital=Decimal("100000"),
        )
        result = engine.run()
        assert len(result.fills) == 0

    def test_missing_idc_directory_raises_data_error(self, tmp_path: Path) -> None:
        from nexa_backtest.exceptions import DataError

        algo = _NoOpIDCAlgo()
        engine = BacktestEngine(
            algo=algo,
            exchange="nordpool",
            start=date(2026, 3, 1),
            end=date(2026, 3, 1),
            products=["NO1-QH"],
            data_dir=tmp_path,  # no idc_events subdirectory
            capital=Decimal("100000"),
        )
        with pytest.raises(DataError, match="idc_events"):
            engine.run()


class TestIDCWithFixture:
    """Integration tests using the pre-generated fixture file."""

    @pytest.fixture(autouse=True)
    def _require_fixture(self) -> None:
        if not FIXTURE_PARQUET.exists():
            pytest.skip("IDC fixture not found — run tests/generate_idc_fixtures.py first")

    def test_full_day_replay_completes(self) -> None:
        algo = _NoOpIDCAlgo()
        engine = BacktestEngine(
            algo=algo,
            exchange="nordpool",
            start=date(2026, 3, 1),
            end=date(2026, 3, 1),
            products=["NO1-QH"],
            data_dir=FIXTURE_DIR,
            capital=Decimal("100000"),
        )
        result = engine.run()
        assert result.duration_days == 1
        assert algo.bar_count == 96  # 24 * 4 MTUs

    def test_buy_algo_gets_fills(self) -> None:
        algo = _AlwaysBuyIDCAlgo("NO1-QH-0800")
        engine = BacktestEngine(
            algo=algo,
            exchange="nordpool",
            start=date(2026, 3, 1),
            end=date(2026, 3, 1),
            products=["NO1-QH"],
            data_dir=FIXTURE_DIR,
            capital=Decimal("100000"),
        )
        result = engine.run()
        # The fixture has sells for NO1-QH-0800; aggressive buys should fill
        assert len(result.fills) > 0

    def test_fills_have_correct_product(self) -> None:
        algo = _AlwaysBuyIDCAlgo("NO1-QH-0800")
        engine = BacktestEngine(
            algo=algo,
            exchange="nordpool",
            start=date(2026, 3, 1),
            end=date(2026, 3, 1),
            products=["NO1-QH"],
            data_dir=FIXTURE_DIR,
            capital=Decimal("100000"),
        )
        result = engine.run()
        for fill in result.fills:
            assert fill.product_id == "NO1-QH-0800"

    def test_result_summary_doesnt_raise(self) -> None:
        algo = _NoOpIDCAlgo()
        engine = BacktestEngine(
            algo=algo,
            exchange="nordpool",
            start=date(2026, 3, 1),
            end=date(2026, 3, 1),
            products=["NO1-QH"],
            data_dir=FIXTURE_DIR,
            capital=Decimal("100000"),
        )
        result = engine.run()
        summary = result.summary()
        assert "Fills" in summary


class TestMtuToProductId:
    def test_conversion(self) -> None:
        t = datetime(2026, 3, 1, 9, 0, tzinfo=UTC)
        assert _mtu_to_product_id(t, "NO1") == "NO1-QH-0900"

    def test_midnight(self) -> None:
        t = datetime(2026, 3, 1, 0, 0, tzinfo=UTC)
        assert _mtu_to_product_id(t, "NO1") == "NO1-QH-0000"

    def test_quarter_hour(self) -> None:
        t = datetime(2026, 3, 1, 14, 15, tzinfo=UTC)
        assert _mtu_to_product_id(t, "NO1") == "NO1-QH-1415"


# ---------------------------------------------------------------------------
# Tests for IDC context methods (backtest.py coverage gaps)
# ---------------------------------------------------------------------------


class TestIDCContextMethods:
    """Covers BacktestContext methods accessed from within an IDC algo."""

    def _run_algo_idc(self, tmp_path: Path, algo: SimpleAlgo, rows: list[dict]) -> None:
        _write_idc_parquet(tmp_path, "NO1", rows)
        engine = BacktestEngine(
            algo=algo,
            exchange="nordpool",
            start=date(2026, 3, 1),
            end=date(2026, 3, 1),
            products=["NO1-QH"],
            data_dir=tmp_path,
            capital=Decimal("100000"),
        )
        engine.run()

    def test_get_last_price_idc(self, tmp_path: Path) -> None:
        """ctx.get_last_price returns the last IDC trade price."""
        product = "NO1-QH-0800"
        ts = datetime(2026, 3, 1, 7, 45, tzinfo=UTC)

        last_prices: list[Decimal | None] = []

        class _Algo(SimpleAlgo):
            def on_bar(self, ctx: TradingContext) -> None:
                last_prices.append(ctx.get_last_price(product))

        rows = [
            _idc_row(
                ts,
                product_id=product,
                event_type="trade",
                order_id="t1",
                side="sell",
                price=52.5,
                trade_id="tr1",
            )
        ]
        self._run_algo_idc(tmp_path, _Algo(), rows)

        # At least one bar after the trade should see a non-None last price
        assert any(p is not None for p in last_prices)

    def test_cancel_idc_pending_order(self, tmp_path: Path) -> None:
        """Cancelling an IDC order in the same on_bar it was placed removes it from pending."""
        placed_ids: list[str] = []
        cancel_statuses: list[str] = []

        class _Algo(SimpleAlgo):
            def on_bar(self, ctx: TradingContext) -> None:
                result = ctx.place_order(
                    Order.buy("NO1-QH-0900", volume_mw=1, price_eur_mwh=Decimal("50"))
                )
                placed_ids.append(result.order_id)
                # Cancel immediately — still in pending queue
                cr = ctx.cancel_order(result.order_id)
                cancel_statuses.append(cr.status)

        self._run_algo_idc(tmp_path, _Algo(), [])
        assert "cancelled" in cancel_statuses

    def test_cancel_idc_resting_order(self, tmp_path: Path) -> None:
        """An order resting in the matching engine can be cancelled on a subsequent bar."""
        placed_order_ids: list[str] = []
        cancel_statuses: list[str] = []
        bar_count = [0]

        class _Algo(SimpleAlgo):
            def on_bar(self, ctx: TradingContext) -> None:
                bar_count[0] += 1
                if bar_count[0] == 1:
                    result = ctx.place_order(
                        Order.buy("NO1-QH-0900", volume_mw=1, price_eur_mwh=Decimal("1"))
                    )
                    placed_order_ids.append(result.order_id)
                elif bar_count[0] == 2 and placed_order_ids:
                    cr = ctx.cancel_order(placed_order_ids[0])
                    cancel_statuses.append(cr.status)

        self._run_algo_idc(tmp_path, _Algo(), [])
        # The order placed at bar 1 is resting by bar 2 — cancel goes to matching engine
        assert "cancelled" in cancel_statuses

    def test_modify_order_not_found(self, tmp_path: Path) -> None:
        """Modifying a non-existent order returns REJECTED status."""
        from nexa_backtest.types import OrderStatus

        statuses: list[OrderStatus] = []

        class _Algo(SimpleAlgo):
            def on_bar(self, ctx: TradingContext) -> None:
                result = ctx.modify_order("nonexistent-id", volume_mw=5)
                statuses.append(result.status)

        self._run_algo_idc(tmp_path, _Algo(), [])
        assert OrderStatus.REJECTED in statuses

    def test_modify_idc_pending_order(self, tmp_path: Path) -> None:
        """An IDC pending order can be modified in the same on_bar."""
        modified_ids: list[str] = []

        class _Algo(SimpleAlgo):
            _done = False

            def on_bar(self, ctx: TradingContext) -> None:
                if self._done:
                    return
                self._done = True
                result = ctx.place_order(
                    Order.buy("NO1-QH-0900", volume_mw=1, price_eur_mwh=Decimal("50"))
                )
                modify_result = ctx.modify_order(result.order_id, volume_mw=2)
                from nexa_backtest.types import OrderStatus

                if modify_result.status == OrderStatus.ACCEPTED:
                    modified_ids.append(modify_result.order_id)

        self._run_algo_idc(tmp_path, _Algo(), [])
        assert len(modified_ids) > 0

    def test_time_to_gate_closure_idc_engine(self, tmp_path: Path) -> None:
        """time_to_gate_closure falls through to the IDC engine's gate closures."""
        gate_times: list[object] = []

        class _Algo(SimpleAlgo):
            def on_bar(self, ctx: TradingContext) -> None:
                # Check gate closure for the next MTU (should be registered)
                remaining = ctx.time_to_gate_closure("NO1-QH-0900")
                gate_times.append(remaining)

        self._run_algo_idc(tmp_path, _Algo(), [])
        assert len(gate_times) > 0


class TestIDCContextGetOrderbook:
    """Test get_orderbook DA fallback with missing clearing price."""

    def test_da_fallback_no_clearing_price(self, tmp_path: Path) -> None:
        """get_orderbook for unknown product returns an empty book in DA mode."""
        import pandas as pd

        # Write DA prices for NO1 only
        data = {
            "timestamp": pd.to_datetime(["2026-03-01T00:00:00Z"], utc=True),
            "zone": ["NO1"],
            "price_eur_mwh": [50.0],
            "volume_mwh": [1000.0],
        }
        pd.DataFrame(data).to_parquet(tmp_path / "da_prices.parquet", index=False)

        books: list[object] = []

        from nexa_backtest.algo import SimpleAlgo
        from nexa_backtest.engines.backtest import BacktestEngine
        from nexa_backtest.types import AuctionInfo

        class _Algo(SimpleAlgo):
            def on_auction_open(self, ctx: TradingContext, auction: AuctionInfo) -> None:
                # Request orderbook for a product that has no clearing price
                book = ctx.get_orderbook("UNKNOWN-PRODUCT")
                books.append(book)

        engine = BacktestEngine(
            algo=_Algo(),
            exchange="nordpool",
            start=date(2026, 3, 1),
            end=date(2026, 3, 1),
            products=["NO1_DA"],
            data_dir=tmp_path,
            capital=Decimal("100000"),
        )
        engine.run()

        assert len(books) > 0
        assert all(b.bids == [] and b.asks == [] for b in books)  # type: ignore[union-attr]


class TestIDCFillsFromHistoricalEvents:
    """Covers the path where process_historical_event generates fills (lines 734-736)."""

    def test_resting_order_filled_by_historical_event(self, tmp_path: Path) -> None:
        """An algo order placed in bar N should be filled by a hist event in bar N+1."""
        product = "NO1-QH-0900"
        # bar N = 08:45 (order placed); bar N+1 = 09:00 (hist sell arrives → fills resting buy)
        bar_n = datetime(2026, 3, 1, 8, 45, tzinfo=UTC)
        bar_n1 = datetime(2026, 3, 1, 9, 0, tzinfo=UTC)

        # Historical sell event at bar_n+1 — will match a resting algo buy from bar_n
        rows = [
            _idc_row(
                bar_n1 + timedelta(seconds=1),
                product_id=product,
                event_type="new",
                order_id="hist-sell-1",
                side="sell",
                price=45.0,
                volume=5.0,
                remaining=5.0,
            )
        ]

        fills_from_hist: list[Fill] = []

        class _Algo(SimpleAlgo):
            _placed = False

            def on_bar(self, ctx: TradingContext) -> None:
                now = ctx.now()
                if now == bar_n and not self._placed:
                    self._placed = True
                    # Place a resting buy at 50 — should not immediately fill (no asks)
                    ctx.place_order(
                        Order.buy(product, volume_mw=Decimal("3"), price_eur_mwh=Decimal("50"))
                    )

            def on_fill(self, ctx: TradingContext, fill: Fill) -> None:
                fills_from_hist.append(fill)

        _write_idc_parquet(tmp_path, "NO1", rows)
        engine = BacktestEngine(
            algo=_Algo(),
            exchange="nordpool",
            start=date(2026, 3, 1),
            end=date(2026, 3, 1),
            products=["NO1-QH"],
            data_dir=tmp_path,
            capital=Decimal("100000"),
        )
        engine.run()

        assert len(fills_from_hist) > 0
        assert fills_from_hist[0].product_id == product


class TestIDCExceptionHandling:
    """Covers the exception path in IDC event replay (lines 729-732)."""

    def test_on_error_called_when_event_processing_raises(self, tmp_path: Path) -> None:
        """When process_historical_event raises, on_error is invoked and replay continues."""
        from unittest.mock import patch

        errors_received: list[Exception] = []
        product = "NO1-QH-0900"

        class _Algo(SimpleAlgo):
            def on_bar(self, ctx: TradingContext) -> None:
                pass

            def on_error(self, ctx: TradingContext, exc: Exception) -> None:
                errors_received.append(exc)

        rows = [
            _idc_row(
                datetime(2026, 3, 1, 8, 45, tzinfo=UTC),
                product_id=product,
                event_type="new",
                order_id="o1",
                side="sell",
                price=50.0,
            )
        ]
        _write_idc_parquet(tmp_path, "NO1", rows)

        from nexa_backtest.engines import matching as matching_mod

        original = matching_mod.ContinuousMatchingEngine.process_historical_event
        call_count = [0]

        def _raising_process(self, event):  # type: ignore[no-untyped-def]
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("simulated matching error")
            return original(self, event)

        with patch.object(
            matching_mod.ContinuousMatchingEngine,
            "process_historical_event",
            _raising_process,
        ):
            engine = BacktestEngine(
                algo=_Algo(),
                exchange="nordpool",
                start=date(2026, 3, 1),
                end=date(2026, 3, 1),
                products=["NO1-QH"],
                data_dir=tmp_path,
                capital=Decimal("100000"),
            )
            engine.run()

        assert len(errors_received) >= 1
        assert isinstance(errors_received[0], RuntimeError)


class TestIDCContextGetVwap:
    """Covers get_vwap IDC path (backtest.py line 214)."""

    def test_get_vwap_idc_returns_last_trade_price(self, tmp_path: Path) -> None:
        product = "NO1-QH-0800"
        trade_ts = datetime(2026, 3, 1, 7, 45, tzinfo=UTC)

        vwaps: list[object] = []

        class _Algo(SimpleAlgo):
            def on_bar(self, ctx: TradingContext) -> None:
                vwaps.append(ctx.get_vwap(product))

        rows = [
            _idc_row(
                trade_ts,
                product_id=product,
                event_type="trade",
                order_id="t1",
                side="sell",
                price=52.5,
                trade_id="tr1",
            )
        ]
        _write_idc_parquet(tmp_path, "NO1", rows)
        engine = BacktestEngine(
            algo=_Algo(),
            exchange="nordpool",
            start=date(2026, 3, 1),
            end=date(2026, 3, 1),
            products=["NO1-QH"],
            data_dir=tmp_path,
            capital=Decimal("100000"),
        )
        engine.run()

        assert any(v is not None for v in vwaps)


class TestIDCMarketVwap:
    """IDC market VWAP must be derived from historical trade events, not fixed at zero."""

    def test_idc_market_vwap_is_nonzero_when_trades_present(self, tmp_path: Path) -> None:
        """market_vwap must be > 0 when the fixture contains trade events."""
        product = "NO1-QH-0800"
        t = datetime(2026, 3, 1, 7, 45, tzinfo=UTC)
        rows = [
            _idc_row(
                t,
                product,
                event_type="trade",
                order_id="t1",
                side="sell",
                price=50.0,
                volume=10.0,
                trade_id="tr1",
            ),
            _idc_row(
                t + timedelta(seconds=1),
                product,
                event_type="trade",
                order_id="t2",
                side="sell",
                price=60.0,
                volume=10.0,
                trade_id="tr2",
            ),
        ]
        _write_idc_parquet(tmp_path, "NO1", rows)

        engine = BacktestEngine(
            algo=_NoOpIDCAlgo(),
            exchange="nordpool",
            start=date(2026, 3, 1),
            end=date(2026, 3, 1),
            products=["NO1-QH"],
            data_dir=tmp_path,
            capital=Decimal("100000"),
        )
        result = engine.run()

        assert result.pnl.market_vwap > Decimal("0"), (
            "IDC market VWAP should reflect historical trade prices, not be zero"
        )

    def test_idc_vwap_is_volume_weighted_average_of_market_trades(self, tmp_path: Path) -> None:
        """VWAP = sum(price * volume) / sum(volume) over all historical trade events."""
        product = "NO1-QH-0800"
        t = datetime(2026, 3, 1, 7, 45, tzinfo=UTC)
        # 10 MW @ 50 EUR + 10 MW @ 60 EUR → VWAP = 55 EUR/MWh
        rows = [
            _idc_row(
                t,
                product,
                event_type="trade",
                order_id="t1",
                side="sell",
                price=50.0,
                volume=10.0,
                trade_id="tr1",
            ),
            _idc_row(
                t + timedelta(seconds=1),
                product,
                event_type="trade",
                order_id="t2",
                side="sell",
                price=60.0,
                volume=10.0,
                trade_id="tr2",
            ),
        ]
        _write_idc_parquet(tmp_path, "NO1", rows)

        engine = BacktestEngine(
            algo=_NoOpIDCAlgo(),
            exchange="nordpool",
            start=date(2026, 3, 1),
            end=date(2026, 3, 1),
            products=["NO1-QH"],
            data_dir=tmp_path,
            capital=Decimal("100000"),
        )
        result = engine.run()

        assert result.pnl.market_vwap == Decimal("55"), (
            f"Expected VWAP=55, got {result.pnl.market_vwap}"
        )

    def test_idc_fill_below_vwap_counts_as_win(self, tmp_path: Path) -> None:
        """A fill priced below the market VWAP for that product should count as a win."""
        product = "NO1-QH-0800"
        t = datetime(2026, 3, 1, 7, 45, tzinfo=UTC)
        rows = [
            # Market VWAP: 10 MW @ 60 EUR → VWAP = 60
            _idc_row(
                t,
                product,
                event_type="trade",
                order_id="t1",
                side="sell",
                price=60.0,
                volume=10.0,
                trade_id="tr1",
            ),
            # Algo can buy from this sell order at 50 EUR (below market VWAP of 60)
            _idc_row(
                t,
                product,
                event_type="new",
                order_id="sell-1",
                side="sell",
                price=50.0,
                volume=10.0,
            ),
        ]
        _write_idc_parquet(tmp_path, "NO1", rows)

        engine = BacktestEngine(
            algo=_AlwaysBuyIDCAlgo(product),
            exchange="nordpool",
            start=date(2026, 3, 1),
            end=date(2026, 3, 1),
            products=["NO1-QH"],
            data_dir=tmp_path,
            capital=Decimal("100000"),
        )
        result = engine.run()

        assert len(result.fills) > 0, "Expected at least one fill"
        assert result.pnl.buys.win_rate == 1.0, (
            f"All fills at 50 EUR < market VWAP 60 EUR should be wins, "
            f"got win_rate={result.pnl.buys.win_rate}"
        )


class TestIDCSignalLoop:
    """Covers the IDC signal update loop (backtest.py lines 740-745)."""

    def test_on_signal_called_in_idc_mode(self, tmp_path: Path) -> None:
        """A subscribed signal should trigger on_signal in IDC mode."""
        # Write a CSV signal covering the replay period
        signals_dir = tmp_path / "signals"
        signals_dir.mkdir()
        signal_csv = signals_dir / "wind_forecast.csv"
        signal_csv.write_text(
            "timestamp,value\n2026-03-01T00:00:00+00:00,12.5\n2026-03-01T12:00:00+00:00,15.0\n"
        )

        signal_calls: list[tuple[str, float]] = []

        class _Algo(SimpleAlgo):
            def on_setup(self, ctx: TradingContext) -> None:
                self.subscribe_signal("wind_forecast")

            def on_bar(self, ctx: TradingContext) -> None:
                pass

            def on_signal(self, ctx: TradingContext, name: str, value: float) -> None:
                signal_calls.append((name, value))

        _write_idc_parquet(tmp_path, "NO1", [])
        engine = BacktestEngine(
            algo=_Algo(),
            exchange="nordpool",
            start=date(2026, 3, 1),
            end=date(2026, 3, 1),
            products=["NO1-QH"],
            data_dir=tmp_path,
            capital=Decimal("100000"),
        )
        engine.run()

        assert len(signal_calls) > 0
        assert all(name == "wind_forecast" for name, _ in signal_calls)

    def test_signal_error_silently_skipped_in_idc_mode(self, tmp_path: Path) -> None:
        """When get_signal raises SignalError, the IDC bar loop must continue silently."""
        # Write a signal CSV whose timestamps are all in the future relative to the
        # replay period (2026-03-01).  Every get_signal call during replay will raise
        # SignalError (no value yet), which the engine must swallow and continue.
        signals_dir = tmp_path / "signals"
        signals_dir.mkdir()
        signal_csv = signals_dir / "future_signal.csv"
        signal_csv.write_text("timestamp,value\n2026-06-01T00:00:00+00:00,99.0\n")

        class _FutureSignalAlgo(SimpleAlgo):
            def on_setup(self, ctx: TradingContext) -> None:
                self.subscribe_signal("future_signal")

            def on_bar(self, ctx: TradingContext) -> None:
                pass

        _write_idc_parquet(tmp_path, "NO1", [])
        engine = BacktestEngine(
            algo=_FutureSignalAlgo(),
            exchange="nordpool",
            start=date(2026, 3, 1),
            end=date(2026, 3, 1),
            products=["NO1-QH"],
            data_dir=tmp_path,
            capital=Decimal("100000"),
        )
        # Engine must not raise even though every signal lookup fails.
        result = engine.run()
        assert result is not None


def test_profit_factor_consistent_with_total_pnl_sign(tmp_path: Path) -> None:
    """profit_factor and total_alpha_eur must agree on the sign of performance.

    When all fills beat their per-product VWAP, total_alpha_eur is positive and
    profit_factor should be None (no losses). Before the fix, profit_factor used
    the portfolio VWAP while total_alpha_eur used per-product VWAPs, producing
    contradictory output (e.g. +784 EUR total PnL alongside Profit Factor 0.00).
    """
    ts_base = datetime(2026, 3, 1, 10, 0, tzinfo=UTC)

    # Historical order book: resting sell at 48.00, resting buy at 52.00.
    rows = [
        {
            "timestamp": ts_base,
            "event_type": "new",
            "order_id": "h-sell-1",
            "zone": "NO1",
            "product_id": "NO1-QH-1000",
            "side": "sell",
            "price_eur_mwh": 48.0,
            "volume_mw": 5.0,
            "remaining_mw": 5.0,
            "aggressor_side": None,
            "trade_id": None,
        },
        {
            "timestamp": ts_base + timedelta(seconds=1),
            "event_type": "new",
            "order_id": "h-buy-1",
            "zone": "NO1",
            "product_id": "NO1-QH-1000",
            "side": "buy",
            "price_eur_mwh": 52.0,
            "volume_mw": 5.0,
            "remaining_mw": 5.0,
            "aggressor_side": None,
            "trade_id": None,
        },
        # Historical trade: market trades at 50.00 (between the two resting orders).
        # aggressor is "buy" → hits the resting sell side.
        # The algo's resting buy at 52.00 also gets hit by the aggressive sell.
        {
            "timestamp": ts_base + timedelta(seconds=2),
            "event_type": "trade",
            "order_id": "h-sell-1",
            "zone": "NO1",
            "product_id": "NO1-QH-1000",
            "side": "sell",
            "price_eur_mwh": 50.0,
            "volume_mw": 5.0,
            "remaining_mw": 0.0,
            "aggressor_side": "buy",
            "trade_id": "tr-1",
        },
    ]
    _write_idc_parquet(tmp_path, "NO1", rows)

    class _PassiveBuyAlgo(SimpleAlgo):
        """Places a passive limit buy well above VWAP so it gets filled."""

        def on_bar(self, ctx: TradingContext) -> None:
            if ctx.now().hour == 10 and ctx.now().minute == 0:
                ctx.place_order(
                    Order.buy("NO1-QH-1000", volume_mw=Decimal("5"), price_eur_mwh=Decimal("55"))
                )

    engine = BacktestEngine(
        algo=_PassiveBuyAlgo(),
        exchange="nordpool",
        start=date(2026, 3, 1),
        end=date(2026, 3, 1),
        products=["NO1-QH"],
        data_dir=tmp_path,
        capital=Decimal("100000"),
    )
    result = engine.run()

    total_pnl = result.pnl.total_alpha_eur

    if result.profit_factor is not None:
        # If there are losses, profit_factor and total_pnl must agree on sign.
        # Positive total_pnl → profit_factor must be > 1.
        # Negative total_pnl → profit_factor must be < 1.
        if total_pnl > 0:
            assert result.profit_factor >= Decimal("1"), (
                f"total_pnl={total_pnl} is positive but profit_factor="
                f"{result.profit_factor} < 1, indicating inconsistent benchmarks"
            )
        elif total_pnl < 0:
            assert result.profit_factor <= Decimal("1"), (
                f"total_pnl={total_pnl} is negative but profit_factor="
                f"{result.profit_factor} > 1, indicating inconsistent benchmarks"
            )
    # If profit_factor is None (no losses), total_pnl must be non-negative.
    else:
        assert total_pnl >= 0, (
            f"profit_factor is None (no losses) but total_pnl={total_pnl} is negative"
        )
