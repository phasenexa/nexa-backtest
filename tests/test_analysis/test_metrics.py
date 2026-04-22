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


class TestComputeFillPnl:
    """Tests for compute_fill_pnl with and without per-product VWAPs."""

    def test_buy_below_portfolio_vwap_positive(self) -> None:
        from nexa_backtest.analysis.metrics import compute_fill_pnl

        fill = _fill(Side.BUY, 45.0)
        # bought at 45, portfolio VWAP 50 → +5 * 10 = +50
        assert compute_fill_pnl(fill, Decimal("50")) == Decimal("50")

    def test_sell_above_portfolio_vwap_positive(self) -> None:
        from nexa_backtest.analysis.metrics import compute_fill_pnl

        fill = _fill(Side.SELL, 55.0)
        # sold at 55, portfolio VWAP 50 → +5 * 10 = +50
        assert compute_fill_pnl(fill, Decimal("50")) == Decimal("50")

    def test_buy_uses_product_vwap_when_provided(self) -> None:
        from nexa_backtest.analysis.metrics import compute_fill_pnl

        fill = _fill(Side.BUY, 48.0)
        product_vwaps = {"P1": Decimal("52.0")}
        # product VWAP 52, bought at 48 → +4 * 10 = +40
        result = compute_fill_pnl(fill, Decimal("50"), product_vwaps)
        assert result == Decimal("40")

    def test_buy_falls_back_to_portfolio_vwap_for_unknown_product(self) -> None:
        from nexa_backtest.analysis.metrics import compute_fill_pnl

        fill = _fill(Side.BUY, 48.0)
        product_vwaps = {"OTHER": Decimal("52.0")}
        # product not in map, falls back to portfolio VWAP 50 → +2 * 10 = +20
        result = compute_fill_pnl(fill, Decimal("50"), product_vwaps)
        assert result == Decimal("20")

    def test_sell_uses_product_vwap_when_provided(self) -> None:
        from nexa_backtest.analysis.metrics import compute_fill_pnl

        fill = _fill(Side.SELL, 54.0)
        product_vwaps = {"P1": Decimal("48.0")}
        # product VWAP 48, sold at 54 → +6 * 10 = +60
        result = compute_fill_pnl(fill, Decimal("50"), product_vwaps)
        assert result == Decimal("60")


class TestComputeProfitFactor:
    """Tests for compute_profit_factor with and without per-product VWAPs."""

    def test_no_losses_returns_none(self) -> None:
        from nexa_backtest.analysis.metrics import compute_profit_factor

        fills = [_fill(Side.BUY, 45.0), _fill(Side.SELL, 55.0)]
        # Both beats VWAP of 50 → no losses → None
        assert compute_profit_factor(fills, Decimal("50")) is None

    def test_all_losses_returns_zero(self) -> None:
        from nexa_backtest.analysis.metrics import compute_profit_factor

        fills = [_fill(Side.BUY, 60.0)]  # bought above VWAP → loss
        result = compute_profit_factor(fills, Decimal("50"))
        assert result == Decimal("0")

    def test_gains_and_losses_ratio(self) -> None:
        from nexa_backtest.analysis.metrics import compute_profit_factor

        gain_fill = _fill(Side.BUY, 40.0)  # +10 * 10 = +100
        loss_fill = _fill(Side.BUY, 60.0)  # -10 * 10 = -100
        result = compute_profit_factor([gain_fill, loss_fill], Decimal("50"))
        assert result == Decimal("1")

    def test_uses_product_vwaps_consistently(self) -> None:
        from nexa_backtest.analysis.metrics import compute_profit_factor

        # product VWAP for P1 is 60; fill buys at 55 → gain of 5*10=50
        # portfolio VWAP is 50; same fill would show a loss of 5*10=50
        fill = _fill(Side.BUY, 55.0)
        product_vwaps = {"P1": Decimal("60.0")}

        portfolio_result = compute_profit_factor([fill], Decimal("50"))
        assert portfolio_result == Decimal("0")  # loss vs portfolio VWAP

        product_result = compute_profit_factor([fill], Decimal("50"), product_vwaps)
        assert product_result is None  # gain vs product VWAP → no losses


class TestComputeTimeInDrawdown:
    """Tests for compute_time_in_drawdown."""

    def test_empty_snapshots_returns_zero(self) -> None:
        from datetime import timedelta

        from nexa_backtest.analysis.metrics import compute_time_in_drawdown

        assert compute_time_in_drawdown([]) == timedelta(0)

    def test_single_snapshot_returns_zero(self) -> None:
        from datetime import timedelta

        from nexa_backtest.analysis.metrics import compute_time_in_drawdown
        from nexa_backtest.types import EquitySnapshot

        snap = EquitySnapshot(
            timestamp=datetime(2026, 3, 1, 10, 0, tzinfo=UTC),
            realised_pnl=Decimal("100"),
            unrealised_pnl=Decimal("0"),
            total_equity=Decimal("1100"),
            cash=Decimal("1100"),
            net_position_mw=Decimal("0"),
        )
        assert compute_time_in_drawdown([snap]) == timedelta(0)

    def test_monotonically_increasing_no_drawdown(self) -> None:
        from datetime import timedelta

        from nexa_backtest.analysis.metrics import compute_time_in_drawdown
        from nexa_backtest.types import EquitySnapshot

        snaps = [
            EquitySnapshot(
                timestamp=datetime(2026, 3, 1, h, 0, tzinfo=UTC),
                realised_pnl=Decimal(str(h * 10)),
                unrealised_pnl=Decimal("0"),
                total_equity=Decimal(str(1000 + h * 10)),
                cash=Decimal(str(1000 + h * 10)),
                net_position_mw=Decimal("0"),
            )
            for h in range(10, 15)
        ]
        assert compute_time_in_drawdown(snaps) == timedelta(0)

    def test_drawdown_period_measured(self) -> None:
        from datetime import timedelta

        from nexa_backtest.analysis.metrics import compute_time_in_drawdown
        from nexa_backtest.types import EquitySnapshot

        def _snap(hour: int, equity: int) -> EquitySnapshot:
            return EquitySnapshot(
                timestamp=datetime(2026, 3, 1, hour, 0, tzinfo=UTC),
                realised_pnl=Decimal(str(equity - 1000)),
                unrealised_pnl=Decimal("0"),
                total_equity=Decimal(str(equity)),
                cash=Decimal(str(equity)),
                net_position_mw=Decimal("0"),
            )

        # Peak at h=10 (1100), dips at h=11 (1050), h=12 (1080), recovers h=13 (1100)
        snaps = [
            _snap(10, 1100),  # peak
            _snap(11, 1050),  # below peak → drawdown starts
            _snap(12, 1080),  # still below peak
            _snap(13, 1100),  # recovered to peak
        ]
        # Drawdown time: h11→h12 (1h) + h12→h13 (1h) = 2h
        assert compute_time_in_drawdown(snaps) == timedelta(hours=2)
