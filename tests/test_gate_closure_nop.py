"""Tests for gate closure NOP tracking and HTML report section — task 05."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from nexa_backtest import BacktestEngine, Order, SimpleAlgo
from nexa_backtest.context import TradingContext
from nexa_backtest.types import GateClosureSnapshot


@pytest.fixture
def idc_data_dir() -> Path:
    return Path(__file__).parent / "fixtures" / "nordpool"


# ---------------------------------------------------------------------------
# Gate closure snapshot recording
# ---------------------------------------------------------------------------


def test_gate_closure_snapshots_are_empty_for_no_fill_algo(idc_data_dir: Path) -> None:
    """A no-op algo should have no gate closure positions (zero NOP is omitted)."""

    class NoopAlgo(SimpleAlgo):
        pass

    engine = BacktestEngine(
        algo=NoopAlgo(),
        exchange="nordpool",
        start=date(2026, 3, 1),
        end=date(2026, 3, 2),
        products=["NO1-QH"],
        data_dir=idc_data_dir,
        capital=Decimal("100000"),
    )
    result = engine.run()
    # All NOPs at closure should be zero — result should still have snapshots
    # for all closed products, but with net_mw == 0.
    # BacktestResult only includes nonzero ones in avg/max metrics.
    assert result.avg_gate_closure_nop_mw == Decimal("0")
    assert result.max_gate_closure_nop_mw == Decimal("0")


def test_gate_closure_nop_tracked_for_held_position(idc_data_dir: Path) -> None:
    """An algo holding a position through gate closure must produce a snapshot."""

    class HoldPosition(SimpleAlgo):
        def __init__(self) -> None:
            super().__init__()
            self._bought = False

        def on_bar(self, ctx: TradingContext) -> None:
            if not self._bought:
                pid = "NO1-QH-0900"
                ask = ctx.get_best_ask(pid)
                if ask is not None:
                    ctx.place_order(
                        Order.buy(
                            product_id=pid,
                            volume_mw=Decimal("1"),
                            price_eur_mwh=ask.price + Decimal("50"),
                        )
                    )
                    self._bought = True

        def on_fill(self, ctx: TradingContext, fill: object) -> None:
            pass  # intentionally hold the position through gate

    engine = BacktestEngine(
        algo=HoldPosition(),
        exchange="nordpool",
        start=date(2026, 3, 1),
        end=date(2026, 3, 2),
        products=["NO1-QH"],
        data_dir=idc_data_dir,
        capital=Decimal("100000"),
    )
    result = engine.run()

    if result.fills:
        # Fills occurred → there must be nonzero NOP snapshots.
        nonzero_snapshots = [s for s in result.gate_closure_positions if s.net_mw != 0]
        assert len(nonzero_snapshots) > 0
        assert result.max_gate_closure_nop_mw > Decimal("0")
        assert result.avg_gate_closure_nop_mw > Decimal("0")

        # Each snapshot must have a product_id, gate_closure_time, and delivery_start.
        for snap in nonzero_snapshots:
            assert isinstance(snap, GateClosureSnapshot)
            assert snap.product_id != ""
            assert snap.gate_closure_time.tzinfo is not None
            assert snap.delivery_start.tzinfo is not None


def test_gate_closure_nop_in_backtest_result_fields(idc_data_dir: Path) -> None:
    """BacktestResult must expose gate_closure_positions, avg, and max fields."""

    class NoopAlgo(SimpleAlgo):
        pass

    engine = BacktestEngine(
        algo=NoopAlgo(),
        exchange="nordpool",
        start=date(2026, 3, 1),
        end=date(2026, 3, 2),
        products=["NO1-QH"],
        data_dir=idc_data_dir,
        capital=Decimal("100000"),
    )
    result = engine.run()

    # Fields must exist and have appropriate types.
    assert isinstance(result.gate_closure_positions, tuple)
    assert isinstance(result.avg_gate_closure_nop_mw, Decimal)
    assert isinstance(result.max_gate_closure_nop_mw, Decimal)


# ---------------------------------------------------------------------------
# Summary output
# ---------------------------------------------------------------------------


def test_gate_closure_section_in_summary_when_position_held(idc_data_dir: Path) -> None:
    """summary() must include Gate Closure NOP section when NOP > 0."""

    class BuyAndHold(SimpleAlgo):
        def __init__(self) -> None:
            super().__init__()
            self._done = False

        def on_bar(self, ctx: TradingContext) -> None:
            if not self._done:
                pid = "NO1-QH-0900"
                ask = ctx.get_best_ask(pid)
                if ask is not None:
                    ctx.place_order(
                        Order.buy(
                            product_id=pid,
                            volume_mw=Decimal("1"),
                            price_eur_mwh=ask.price + Decimal("50"),
                        )
                    )
                    self._done = True

    engine = BacktestEngine(
        algo=BuyAndHold(),
        exchange="nordpool",
        start=date(2026, 3, 1),
        end=date(2026, 3, 2),
        products=["NO1-QH"],
        data_dir=idc_data_dir,
        capital=Decimal("100000"),
    )
    result = engine.run()

    if result.fills:
        summary = result.summary()
        assert "Gate Closure NOP" in summary


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------


def test_html_report_includes_gate_closure_section(idc_data_dir: Path) -> None:
    """HTML report must contain the Gate Closure Exposure section."""

    class NoopAlgo(SimpleAlgo):
        pass

    engine = BacktestEngine(
        algo=NoopAlgo(),
        exchange="nordpool",
        start=date(2026, 3, 1),
        end=date(2026, 3, 2),
        products=["NO1-QH"],
        data_dir=idc_data_dir,
        capital=Decimal("100000"),
    )
    result = engine.run()

    from nexa_backtest.analysis.report import generate_html_report

    html = generate_html_report(result)
    assert "Gate Closure Exposure" in html
    assert "chart-gate-nop" in html


def test_html_report_gate_closure_stats(idc_data_dir: Path) -> None:
    """HTML report gate closure card must show avg and max NOP metrics."""

    class NoopAlgo(SimpleAlgo):
        pass

    engine = BacktestEngine(
        algo=NoopAlgo(),
        exchange="nordpool",
        start=date(2026, 3, 1),
        end=date(2026, 3, 2),
        products=["NO1-QH"],
        data_dir=idc_data_dir,
        capital=Decimal("100000"),
    )
    result = engine.run()

    from nexa_backtest.analysis.report import generate_html_report

    html = generate_html_report(result)
    assert "Avg NOP at Closure" in html
    assert "Max NOP at Closure" in html
