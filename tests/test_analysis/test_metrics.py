"""Tests for analysis/metrics.py: BacktestResult.summary()."""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal

from nexa_backtest.analysis.metrics import BacktestResult
from nexa_backtest.analysis.pnl import PnlSummary, SideSummary
from nexa_backtest.types import Fill, Side


def _empty_side() -> SideSummary:
    return SideSummary(
        count=0,
        volume_mwh=Decimal("0"),
        avg_price=Decimal("0"),
        vwap_alpha=Decimal("0"),
        total_alpha_eur=Decimal("0"),
        win_rate=0.0,
    )


def _side(
    count: int,
    volume: float,
    avg_price: float,
    vwap_alpha: float,
    total_alpha: float,
    win_rate: float,
) -> SideSummary:
    return SideSummary(
        count=count,
        volume_mwh=Decimal(str(volume)),
        avg_price=Decimal(str(avg_price)),
        vwap_alpha=Decimal(str(vwap_alpha)),
        total_alpha_eur=Decimal(str(total_alpha)),
        win_rate=win_rate,
    )


def _fill(side: Side, price: float) -> Fill:
    return Fill(
        order_id="o1",
        product_id="P1",
        side=side,
        price=Decimal(str(price)),
        volume=Decimal("10"),
        timestamp=datetime(2026, 3, 1, 12, 0, tzinfo=UTC),
    )


def _result(buys: SideSummary, sells: SideSummary, fills: tuple[Fill, ...] = ()) -> BacktestResult:
    pnl = PnlSummary(
        market_vwap=Decimal("50"),
        buys=buys,
        sells=sells,
        total_alpha_eur=buys.total_alpha_eur + sells.total_alpha_eur,
    )
    return BacktestResult(
        algo_name="TestAlgo",
        exchange="nordpool",
        start=date(2026, 3, 1),
        end=date(2026, 3, 31),
        fills=fills,
        pnl=pnl,
    )


class TestBacktestResultSummary:
    def test_summary_with_only_buys(self) -> None:
        buys = _side(3, 30.0, 45.0, -5.0, 150.0, 1.0)
        result = _result(buys=buys, sells=_empty_side())
        summary = result.summary()
        assert "Buys" in summary
        assert "Sells" not in summary

    def test_summary_with_only_sells(self) -> None:
        sells = _side(2, 20.0, 55.0, -5.0, 100.0, 1.0)
        result = _result(buys=_empty_side(), sells=sells)
        summary = result.summary()
        assert "Sells" in summary
        assert "Buys" not in summary

    def test_summary_with_both_buys_and_sells(self) -> None:
        buys = _side(2, 20.0, 45.0, -5.0, 100.0, 1.0)
        sells = _side(2, 20.0, 55.0, -5.0, 100.0, 1.0)
        result = _result(buys=buys, sells=sells)
        summary = result.summary()
        assert "Buys" in summary
        assert "Sells" in summary

    def test_summary_no_fills_message(self) -> None:
        result = _result(buys=_empty_side(), sells=_empty_side())
        summary = result.summary()
        assert "No fills recorded" in summary

    def test_summary_contains_algo_name(self) -> None:
        result = _result(buys=_empty_side(), sells=_empty_side())
        assert "TestAlgo" in result.summary()

    def test_summary_contains_exchange(self) -> None:
        result = _result(buys=_empty_side(), sells=_empty_side())
        assert "nordpool" in result.summary()

    def test_summary_sells_above_vwap_note(self) -> None:
        # vwap_alpha <= 0 means sold above VWAP (good)
        sells = _side(1, 10.0, 55.0, -5.0, 50.0, 1.0)
        result = _result(buys=_empty_side(), sells=sells)
        summary = result.summary()
        assert "sold above VWAP" in summary

    def test_summary_sells_below_vwap_note(self) -> None:
        # vwap_alpha > 0 means sold below VWAP (bad)
        sells = _side(1, 10.0, 45.0, 5.0, -50.0, 0.0)
        result = _result(buys=_empty_side(), sells=sells)
        summary = result.summary()
        assert "sold below VWAP" in summary
