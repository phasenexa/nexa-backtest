"""Integration tests for the BacktestEngine with signals.

Tests verify:
- Algo that uses a signal makes signal-informed trading decisions
- Signal value influences fills (algo skips products above threshold)
- Look-ahead bias: signal with publication_offset does not expose future values
- BacktestResult.summary() runs without error
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest

from nexa_backtest.algo import SimpleAlgo
from nexa_backtest.analysis.metrics import BacktestResult
from nexa_backtest.context import TradingContext
from nexa_backtest.engines.backtest import BacktestEngine, _BacktestContext
from nexa_backtest.engines.clock import SimulatedClock
from nexa_backtest.exceptions import DataError, SignalError
from nexa_backtest.signals.csv_loader import CsvSignalProvider
from nexa_backtest.signals.registry import SignalRegistry
from nexa_backtest.types import AuctionInfo, Fill, Order, OrderStatus, Side

# ---------------------------------------------------------------------------
# Helpers: fixture data generation
# ---------------------------------------------------------------------------


def _write_da_prices(path: Path, rows: list[tuple[str, str, float, float]]) -> None:
    """Write a minimal da_prices.parquet from (product_id, timestamp_str, price, vol) rows."""
    data = {
        "timestamp": pd.to_datetime([r[1] for r in rows], utc=True),
        "zone": [r[0] for r in rows],
        "price_eur_mwh": [r[2] for r in rows],
        "volume_mwh": [r[3] for r in rows],
    }
    pd.DataFrame(data).to_parquet(path / "da_prices.parquet", index=False)


def _write_signal_csv(path: Path, name: str, rows: list[tuple[str, float]]) -> None:
    """Write a signal CSV to {path}/signals/{name}.csv."""
    signals_dir = path / "signals"
    signals_dir.mkdir(exist_ok=True)
    content = "timestamp,value\n" + "".join(f"{ts},{val}\n" for ts, val in rows)
    (signals_dir / f"{name}.csv").write_text(content)


# ---------------------------------------------------------------------------
# Simple passthrough algo (no signals)
# ---------------------------------------------------------------------------


class _AlwaysBuyAlgo(SimpleAlgo):
    """Places a buy order at clearing price for every product."""

    def on_auction_open(self, ctx: TradingContext, auction: AuctionInfo) -> None:
        ctx.place_order(
            Order.buy(
                product_id=auction.product_id,
                volume_mw=Decimal("10"),
                price_eur_mwh=Decimal("999"),  # always fills
            )
        )


class _AlwaysSellAlgo(SimpleAlgo):
    """Places a sell order at clearing price for every product."""

    def on_auction_open(self, ctx: TradingContext, auction: AuctionInfo) -> None:
        ctx.place_order(
            Order.sell(
                product_id=auction.product_id,
                volume_mw=Decimal("10"),
                price_eur_mwh=Decimal("0"),  # always fills
            )
        )


class _NoOpAlgo(SimpleAlgo):
    """Does nothing."""


# ---------------------------------------------------------------------------
# Tests: basic engine operation
# ---------------------------------------------------------------------------


class TestBacktestEngineBasic:
    @pytest.fixture
    def data_dir(self, tmp_path: Path) -> Path:
        _write_da_prices(
            tmp_path,
            [
                ("NO1", "2026-03-01T00:00:00Z", 45.0, 1000.0),
                ("NO1", "2026-03-01T00:15:00Z", 50.0, 1000.0),
                ("NO1", "2026-03-01T00:30:00Z", 40.0, 1000.0),
            ],
        )
        return tmp_path

    def _engine(self, algo: SimpleAlgo, data_dir: Path) -> BacktestEngine:
        return BacktestEngine(
            algo=algo,
            exchange="nordpool",
            start=date(2026, 3, 1),
            end=date(2026, 3, 1),
            products=["NO1_DA"],
            data_dir=data_dir,
            capital=Decimal("100000"),
        )

    def test_no_op_algo_produces_no_fills(self, data_dir: Path) -> None:
        result = self._engine(_NoOpAlgo(), data_dir).run()
        assert len(result.fills) == 0

    def test_always_buy_produces_fills(self, data_dir: Path) -> None:
        result = self._engine(_AlwaysBuyAlgo(), data_dir).run()
        assert len(result.fills) == 3

    def test_fills_are_at_clearing_price(self, data_dir: Path) -> None:
        result = self._engine(_AlwaysBuyAlgo(), data_dir).run()
        prices = {float(f.price) for f in result.fills}
        assert prices == {45.0, 50.0, 40.0}

    def test_result_is_backtest_result(self, data_dir: Path) -> None:
        result = self._engine(_NoOpAlgo(), data_dir).run()
        assert isinstance(result, BacktestResult)

    def test_summary_runs_without_error(self, data_dir: Path) -> None:
        result = self._engine(_AlwaysBuyAlgo(), data_dir).run()
        summary = result.summary()
        assert "nordpool" in summary

    def test_missing_data_file_raises_data_error(self, tmp_path: Path) -> None:
        with pytest.raises(DataError, match="da_prices"):
            self._engine(_NoOpAlgo(), tmp_path).run()

    def test_invalid_product_spec_raises_data_error(self, data_dir: Path) -> None:
        engine = BacktestEngine(
            algo=_NoOpAlgo(),
            exchange="nordpool",
            start=date(2026, 3, 1),
            end=date(2026, 3, 1),
            products=["BADFORMAT"],
            data_dir=data_dir,
            capital=Decimal("100000"),
        )
        with pytest.raises(DataError):
            engine.run()


# ---------------------------------------------------------------------------
# Tests: signal-driven trading decisions
# ---------------------------------------------------------------------------


class _SignalAlgo(SimpleAlgo):
    """Buys when forecast > clearing + threshold; never buys otherwise."""

    threshold: Decimal = Decimal("5.0")

    def on_setup(self, ctx: TradingContext) -> None:
        self.subscribe_signal("price_forecast")
        self.threshold = Decimal("5.0")

    def on_auction_open(self, ctx: TradingContext, auction: AuctionInfo) -> None:
        try:
            signal = ctx.get_signal("price_forecast")
        except SignalError:
            return
        bid = Decimal(str(signal.value)) - self.threshold
        ctx.place_order(
            Order.buy(
                product_id=auction.product_id,
                volume_mw=Decimal("10"),
                price_eur_mwh=bid,
            )
        )


class TestSignalDrivenTrading:
    @pytest.fixture
    def data_dir(self, tmp_path: Path) -> Path:
        # Three products with clearing prices 30, 50, 40
        _write_da_prices(
            tmp_path,
            [
                ("NO1", "2026-03-01T00:00:00Z", 30.0, 1000.0),
                ("NO1", "2026-03-01T00:15:00Z", 50.0, 1000.0),
                ("NO1", "2026-03-01T00:30:00Z", 40.0, 1000.0),
            ],
        )
        # Forecast: 60 for all periods, available at auction time (D-1 12:00)
        # publication_offset=36h -> at D-1 12:00 we can see T <= D+1 00:00
        # (all day-D products visible)
        # Auction time = 2026-02-28T12:00Z
        # Visible: T <= 2026-02-28T12:00 + 36h = 2026-03-01T24:00Z → all visible
        _write_signal_csv(
            tmp_path,
            "price_forecast",
            [
                ("2026-03-01T00:00:00+00:00", 60.0),
                ("2026-03-01T00:15:00+00:00", 60.0),
                ("2026-03-01T00:30:00+00:00", 60.0),
            ],
        )
        return tmp_path

    def _engine(self, data_dir: Path, algo: SimpleAlgo | None = None) -> BacktestEngine:
        return BacktestEngine(
            algo=algo or _SignalAlgo(),
            exchange="nordpool",
            start=date(2026, 3, 1),
            end=date(2026, 3, 1),
            products=["NO1_DA"],
            data_dir=data_dir,
            capital=Decimal("100000"),
            signals=[
                CsvSignalProvider(
                    name="price_forecast",
                    path=data_dir / "signals" / "price_forecast.csv",
                    unit="EUR/MWh",
                    description="Test forecast",
                    publication_offset=timedelta(hours=36),
                )
            ],
        )

    def test_algo_fills_when_forecast_above_threshold(self, data_dir: Path) -> None:
        # forecast=60, threshold=5 -> bid=55. clearing prices: 30, 50, 40.
        # Bids 55 >= 30, 55 >= 50, 55 >= 40 → all three fill
        result = self._engine(data_dir).run()
        assert len(result.fills) == 3

    def test_algo_skips_when_bid_below_clearing(self, data_dir: Path) -> None:
        """When forecast is barely above clearing, orders with bid < clearing reject."""
        # forecast=45, threshold=5 -> bid=40. clearing prices: 30, 50, 40.
        # bid 40 >= 30 → fill; bid 40 < 50 → reject; bid 40 >= 40 → fill (at boundary)
        _write_signal_csv(
            data_dir,
            "price_forecast",
            [
                ("2026-03-01T00:00:00+00:00", 45.0),
                ("2026-03-01T00:15:00+00:00", 45.0),
                ("2026-03-01T00:30:00+00:00", 45.0),
            ],
        )
        result = self._engine(data_dir).run()
        # 50 EUR/MWh product should not fill (bid=40 < clearing=50)
        fill_prices = {float(f.price) for f in result.fills}
        assert 50.0 not in fill_prices

    def test_signal_value_influences_number_of_fills(self, data_dir: Path) -> None:
        """Verify fills differ when forecast is high vs low."""
        # high forecast: all fill
        result_high = self._engine(data_dir).run()

        # low forecast: bid=5 → almost nothing fills
        _write_signal_csv(
            data_dir,
            "price_forecast",
            [
                ("2026-03-01T00:00:00+00:00", 10.0),
                ("2026-03-01T00:15:00+00:00", 10.0),
                ("2026-03-01T00:30:00+00:00", 10.0),
            ],
        )
        result_low = self._engine(data_dir).run()
        assert len(result_high.fills) > len(result_low.fills)

    def test_subscribed_signal_auto_discovered_from_csv(self, tmp_path: Path) -> None:
        """Engine auto-loads signal CSV from data_dir/signals/{name}.csv."""
        _write_da_prices(
            tmp_path,
            [("NO1", "2026-03-01T00:00:00Z", 45.0, 1000.0)],
        )
        # Auction time for 2026-03-01 delivery = 2026-02-28T12:00Z.
        # Auto-discovered provider has no publication_offset → values visible
        # at their timestamp. Use a timestamp before auction time so it's
        # visible without an offset.
        _write_signal_csv(
            tmp_path,
            "price_forecast",
            [("2026-02-28T00:00:00+00:00", 99.0)],
        )
        engine = BacktestEngine(
            algo=_SignalAlgo(),
            exchange="nordpool",
            start=date(2026, 3, 1),
            end=date(2026, 3, 1),
            products=["NO1_DA"],
            data_dir=tmp_path,
            capital=Decimal("100000"),
        )
        result = engine.run()
        # bid = 99 - 5 = 94, clearing = 45 → fill
        assert len(result.fills) == 1

    def test_missing_signal_csv_raises_data_error(self, tmp_path: Path) -> None:
        _write_da_prices(
            tmp_path,
            [("NO1", "2026-03-01T00:00:00Z", 45.0, 1000.0)],
        )
        # No signals/ dir → DataError
        engine = BacktestEngine(
            algo=_SignalAlgo(),
            exchange="nordpool",
            start=date(2026, 3, 1),
            end=date(2026, 3, 1),
            products=["NO1_DA"],
            data_dir=tmp_path,
            capital=Decimal("100000"),
        )
        with pytest.raises(DataError, match="price_forecast"):
            engine.run()


# ---------------------------------------------------------------------------
# Tests: look-ahead bias prevention
# ---------------------------------------------------------------------------


class TestLookAheadBias:
    """Verify publication_offset prevents future values from being visible."""

    def test_value_not_visible_before_publication_time(self, tmp_path: Path) -> None:
        """With offset=0, values are only visible at or after their timestamp."""
        _write_da_prices(
            tmp_path,
            [("NO1", "2026-03-01T00:00:00Z", 45.0, 1000.0)],
        )
        # Auction time = 2026-02-28T12:00Z (D-1 12:00 UTC for delivery 2026-03-01)
        # Signal row: timestamp=2026-03-01T01:00Z, offset=0h
        # At auction time 2026-02-28T12:00: only rows where ts <= 2026-02-28T12:00 visible
        # 2026-03-01T01:00 > 2026-02-28T12:00 → NOT visible → SignalError → no fill
        _write_signal_csv(
            tmp_path,
            "price_forecast",
            [("2026-03-01T01:00:00+00:00", 99.0)],
        )
        provider = CsvSignalProvider(
            name="price_forecast",
            path=tmp_path / "signals" / "price_forecast.csv",
            unit="EUR/MWh",
            description="",
            publication_offset=timedelta(0),  # no offset
        )
        engine = BacktestEngine(
            algo=_SignalAlgo(),
            exchange="nordpool",
            start=date(2026, 3, 1),
            end=date(2026, 3, 1),
            products=["NO1_DA"],
            data_dir=tmp_path,
            capital=Decimal("100000"),
            signals=[provider],
        )
        result = engine.run()
        # Signal not visible at auction time → SignalError caught → no orders placed
        assert len(result.fills) == 0

    def test_value_visible_with_sufficient_offset(self, tmp_path: Path) -> None:
        """With a large offset, the same value becomes visible."""
        _write_da_prices(
            tmp_path,
            [("NO1", "2026-03-01T00:00:00Z", 45.0, 1000.0)],
        )
        # Auction time = 2026-02-28T12:00Z
        # Signal: timestamp=2026-03-01T01:00Z, offset=36h
        # Visible when: ts <= auction_time + 36h = 2026-02-28T12:00 + 36h = 2026-03-02T00:00
        # 2026-03-01T01:00 <= 2026-03-02T00:00 ✓ → visible
        _write_signal_csv(
            tmp_path,
            "price_forecast",
            [("2026-03-01T01:00:00+00:00", 99.0)],
        )
        provider = CsvSignalProvider(
            name="price_forecast",
            path=tmp_path / "signals" / "price_forecast.csv",
            unit="EUR/MWh",
            description="",
            publication_offset=timedelta(hours=36),
        )
        engine = BacktestEngine(
            algo=_SignalAlgo(),
            exchange="nordpool",
            start=date(2026, 3, 1),
            end=date(2026, 3, 1),
            products=["NO1_DA"],
            data_dir=tmp_path,
            capital=Decimal("100000"),
            signals=[provider],
        )
        result = engine.run()
        # bid = 99 - 5 = 94 >= clearing 45 → fill
        assert len(result.fills) == 1


# ---------------------------------------------------------------------------
# Helpers for _BacktestContext unit tests
# ---------------------------------------------------------------------------


def _make_context(
    initial_time: datetime | None = None,
) -> _BacktestContext:
    t = initial_time or datetime(2026, 3, 1, 12, 0, tzinfo=UTC)
    clock = SimulatedClock(initial_time=t)
    registry = SignalRegistry()
    return _BacktestContext(clock=clock, signal_registry=registry)


def _make_fill(side: Side, price: float, product_id: str = "P1", volume: float = 10.0) -> Fill:
    return Fill(
        order_id="o1",
        product_id=product_id,
        side=side,
        price=Decimal(str(price)),
        volume=Decimal(str(volume)),
        timestamp=datetime(2026, 3, 1, 12, 0, tzinfo=UTC),
    )


# ---------------------------------------------------------------------------
# Tests: _BacktestContext methods
# ---------------------------------------------------------------------------


class TestBacktestContextTime:
    def test_now_returns_clock_time(self) -> None:
        t = datetime(2026, 3, 15, 10, 0, tzinfo=UTC)
        ctx = _make_context(initial_time=t)
        assert ctx.now() == t

    def test_time_to_gate_closure_known_product(self) -> None:
        t = datetime(2026, 3, 1, 12, 0, tzinfo=UTC)
        ctx = _make_context(initial_time=t)
        closure = t + timedelta(hours=2)
        ctx._gate_closures["P1"] = closure
        remaining = ctx.time_to_gate_closure("P1")
        assert remaining == timedelta(hours=2)

    def test_time_to_gate_closure_unknown_product_returns_zero(self) -> None:
        ctx = _make_context()
        assert ctx.time_to_gate_closure("UNKNOWN") == timedelta(0)

    def test_time_to_gate_closure_past_returns_zero(self) -> None:
        t = datetime(2026, 3, 1, 12, 0, tzinfo=UTC)
        ctx = _make_context(initial_time=t)
        ctx._gate_closures["P1"] = t - timedelta(hours=1)
        assert ctx.time_to_gate_closure("P1") == timedelta(0)

    def test_current_mtu_rounds_to_15min_slot(self) -> None:
        t = datetime(2026, 3, 1, 9, 37, 45, tzinfo=UTC)
        ctx = _make_context(initial_time=t)
        mtu = ctx.current_mtu()
        assert mtu.start.minute == 30
        assert mtu.end.minute == 45


class TestBacktestContextMarketData:
    def test_get_orderbook_known_product(self) -> None:
        ctx = _make_context()
        ctx._clearing_prices["P1"] = Decimal("50")
        ob = ctx.get_orderbook("P1")
        assert ob.best_bid is not None
        assert ob.best_ask is not None
        assert ob.best_bid.price == Decimal("50")

    def test_get_orderbook_unknown_product_returns_empty(self) -> None:
        ctx = _make_context()
        ob = ctx.get_orderbook("UNKNOWN")
        assert ob.best_bid is None
        assert ob.best_ask is None

    def test_get_best_bid_returns_price_level(self) -> None:
        ctx = _make_context()
        ctx._clearing_prices["P1"] = Decimal("45")
        bid = ctx.get_best_bid("P1")
        assert bid is not None
        assert bid.price == Decimal("45")

    def test_get_best_ask_returns_price_level(self) -> None:
        ctx = _make_context()
        ctx._clearing_prices["P1"] = Decimal("45")
        ask = ctx.get_best_ask("P1")
        assert ask is not None
        assert ask.price == Decimal("45")

    def test_get_last_price_known(self) -> None:
        ctx = _make_context()
        ctx._clearing_prices["P1"] = Decimal("55")
        assert ctx.get_last_price("P1") == Decimal("55")

    def test_get_last_price_unknown_returns_none(self) -> None:
        ctx = _make_context()
        assert ctx.get_last_price("UNKNOWN") is None

    def test_get_vwap_known(self) -> None:
        ctx = _make_context()
        ctx._clearing_prices["P1"] = Decimal("48")
        assert ctx.get_vwap("P1") == Decimal("48")

    def test_get_vwap_unknown_returns_none(self) -> None:
        ctx = _make_context()
        assert ctx.get_vwap("UNKNOWN") is None


class TestBacktestContextOrderManagement:
    def test_cancel_order_success(self) -> None:
        ctx = _make_context()
        order = Order.buy(product_id="P1", volume_mw=Decimal("10"), price_eur_mwh=Decimal("50"))
        ctx.place_order(order)
        result = ctx.cancel_order(order.order_id)
        assert result.status == "cancelled"
        assert order.order_id not in ctx._pending_orders

    def test_cancel_order_not_found(self) -> None:
        ctx = _make_context()
        result = ctx.cancel_order("nonexistent-id")
        assert result.status == "not_found"

    def test_modify_order_not_found_returns_rejected(self) -> None:
        ctx = _make_context()
        result = ctx.modify_order("nonexistent-id", price_eur_mwh=Decimal("50"))
        assert result.status == OrderStatus.REJECTED
        assert "not found" in (result.rejection_reason or "")

    def test_modify_order_success(self) -> None:
        ctx = _make_context()
        order = Order.buy(product_id="P1", volume_mw=Decimal("10"), price_eur_mwh=Decimal("50"))
        ctx.place_order(order)
        result = ctx.modify_order(order.order_id, price_eur_mwh=Decimal("55"))
        assert result.status == OrderStatus.ACCEPTED
        # Old order gone, new order present
        assert order.order_id not in ctx._pending_orders
        new_order = ctx._pending_orders[result.order_id]
        assert new_order.price_eur_mwh == Decimal("55")

    def test_modify_order_invalid_data_returns_rejected(self) -> None:
        ctx = _make_context()
        order = Order.buy(product_id="P1", volume_mw=Decimal("10"), price_eur_mwh=Decimal("50"))
        ctx.place_order(order)
        # Invalid Side enum value causes pydantic validation failure
        result = ctx.modify_order(order.order_id, side="INVALID_SIDE")
        assert result.status == OrderStatus.REJECTED


class TestBacktestContextPositions:
    def test_get_position_no_fills_returns_zero(self) -> None:
        ctx = _make_context()
        pos = ctx.get_position("P1")
        assert pos.net_mw == Decimal("0")

    def test_get_position_after_buy(self) -> None:
        ctx = _make_context()
        fill = _make_fill(Side.BUY, 50.0, "P1", 10.0)
        ctx._record_fill(fill)
        pos = ctx.get_position("P1")
        assert pos.net_mw == Decimal("10")

    def test_get_position_after_sell(self) -> None:
        ctx = _make_context()
        fill = _make_fill(Side.SELL, 50.0, "P1", 10.0)
        ctx._record_fill(fill)
        pos = ctx.get_position("P1")
        assert pos.net_mw == Decimal("-10")

    def test_get_position_net_zero_after_round_trip(self) -> None:
        ctx = _make_context()
        ctx._record_fill(_make_fill(Side.BUY, 50.0, "P1", 10.0))
        ctx._record_fill(_make_fill(Side.SELL, 50.0, "P1", 10.0))
        pos = ctx.get_position("P1")
        assert pos.net_mw == Decimal("0")
        assert pos.avg_entry_price == Decimal("0")

    def test_get_all_positions_excludes_zero_positions(self) -> None:
        ctx = _make_context()
        ctx._record_fill(_make_fill(Side.BUY, 50.0, "P1"))
        ctx._record_fill(_make_fill(Side.SELL, 50.0, "P1"))  # nets to zero
        ctx._record_fill(_make_fill(Side.BUY, 45.0, "P2"))
        positions = ctx.get_all_positions()
        assert "P1" not in positions
        assert "P2" in positions

    def test_get_unrealised_pnl_with_open_position(self) -> None:
        ctx = _make_context()
        ctx._clearing_prices["P1"] = Decimal("55")
        ctx._record_fill(_make_fill(Side.BUY, 50.0, "P1", 10.0))
        pnl = ctx.get_unrealised_pnl()
        assert pnl == Decimal("50")  # (55 - 50) * 10


class TestBacktestContextMisc:
    def test_predict_raises_not_implemented(self) -> None:
        ctx = _make_context()
        with pytest.raises(NotImplementedError, match="ML model"):
            ctx.predict("some_model", {})

    def test_log_does_not_raise(self) -> None:
        ctx = _make_context()
        ctx.log("test message", level="info")
        ctx.log("warning message", level="warning")

    def test_get_signal_history_returns_list(self, tmp_path: Path) -> None:
        _write_signal_csv(
            tmp_path,
            "test_signal",
            [
                ("2026-03-01T00:00:00+00:00", 10.0),
                ("2026-03-01T01:00:00+00:00", 20.0),
                ("2026-03-01T02:00:00+00:00", 30.0),
            ],
        )
        provider = CsvSignalProvider(
            name="test_signal",
            path=tmp_path / "signals" / "test_signal.csv",
            unit="EUR/MWh",
            description="",
        )
        t = datetime(2026, 3, 1, 12, 0, tzinfo=UTC)
        clock = SimulatedClock(initial_time=t)
        registry = SignalRegistry()
        registry.register(provider)
        ctx = _BacktestContext(clock=clock, signal_registry=registry)
        history = ctx.get_signal_history("test_signal", 2)
        assert len(history) == 2


# ---------------------------------------------------------------------------
# Tests: BacktestEngine edge cases
# ---------------------------------------------------------------------------


class TestBacktestEngineEdgeCases:
    def test_no_products_raises_data_error(self, tmp_path: Path) -> None:
        _write_da_prices(
            tmp_path,
            [("NO1", "2026-03-01T00:00:00Z", 45.0, 1000.0)],
        )
        engine = BacktestEngine(
            algo=_NoOpAlgo(),
            exchange="nordpool",
            start=date(2026, 3, 1),
            end=date(2026, 3, 1),
            products=[],
            data_dir=tmp_path,
            capital=Decimal("100000"),
        )
        with pytest.raises(DataError, match="No products specified"):
            engine.run()

    def test_order_for_unknown_product_is_skipped(self, tmp_path: Path) -> None:
        """An order placed for a product not in clearing prices is silently ignored."""
        _write_da_prices(
            tmp_path,
            [("NO1", "2026-03-01T00:00:00Z", 45.0, 1000.0)],
        )

        class _WrongProductAlgo(SimpleAlgo):
            def on_auction_open(self, ctx: TradingContext, auction: AuctionInfo) -> None:
                ctx.place_order(
                    Order.buy(
                        product_id="NONEXISTENT_PRODUCT",
                        volume_mw=Decimal("10"),
                        price_eur_mwh=Decimal("999"),
                    )
                )

        engine = BacktestEngine(
            algo=_WrongProductAlgo(),
            exchange="nordpool",
            start=date(2026, 3, 1),
            end=date(2026, 3, 1),
            products=["NO1_DA"],
            data_dir=tmp_path,
            capital=Decimal("100000"),
        )
        result = engine.run()
        assert len(result.fills) == 0

    def test_always_sell_produces_fills_and_summary(self, tmp_path: Path) -> None:
        _write_da_prices(
            tmp_path,
            [
                ("NO1", "2026-03-01T00:00:00Z", 45.0, 1000.0),
                ("NO1", "2026-03-01T00:15:00Z", 50.0, 1000.0),
            ],
        )
        engine = BacktestEngine(
            algo=_AlwaysSellAlgo(),
            exchange="nordpool",
            start=date(2026, 3, 1),
            end=date(2026, 3, 1),
            products=["NO1_DA"],
            data_dir=tmp_path,
            capital=Decimal("100000"),
        )
        result = engine.run()
        assert len(result.fills) == 2
        summary = result.summary()
        assert "Sells" in summary
