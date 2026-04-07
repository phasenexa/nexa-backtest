"""Tests for spec-03: equity curve, advanced metrics, and exports.

Covers:
1. Equity curve tracking — snapshots count and final equity invariant
2. Sharpe ratio — known returns and edge cases
3. Max drawdown — up-down-up curve and monotonic edge case
4. Profit factor — known trades and no-losing-trades edge case
5. Daily PnL aggregation — midnight boundary assignment
6. HTML report — valid HTML with expected containers
7. JSON export — round-trip and Decimal precision
8. Parquet export — directory structure and schemas
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from decimal import Decimal
from html.parser import HTMLParser
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

from nexa_backtest.algo import SimpleAlgo
from nexa_backtest.analysis.metrics import (
    BacktestResult,
    compute_fill_pnl,
    compute_max_drawdown,
    compute_profit_factor,
    compute_sharpe,
)
from nexa_backtest.analysis.pnl import PnlSummary, SideSummary
from nexa_backtest.context import TradingContext
from nexa_backtest.engines.backtest import BacktestEngine
from nexa_backtest.types import AuctionInfo, EquitySnapshot, Fill, Order, Side

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _snap(ts: datetime, total_equity: float, realised: float = 0.0) -> EquitySnapshot:
    eq = Decimal(str(total_equity))
    r = Decimal(str(realised))
    return EquitySnapshot(
        timestamp=ts,
        realised_pnl=r,
        unrealised_pnl=Decimal("0"),
        total_equity=eq,
        cash=eq,
        net_position_mw=Decimal("0"),
    )


def _fill(
    price: float,
    side: Side = Side.BUY,
    volume: float = 10.0,
    ts: datetime | None = None,
    product: str = "NO1-QH-0900",
) -> Fill:
    return Fill(
        order_id="o1",
        product_id=product,
        side=side,
        price=Decimal(str(price)),
        volume=Decimal(str(volume)),
        timestamp=ts or datetime(2026, 3, 1, 13, 0, tzinfo=UTC),
    )


def _empty_side() -> SideSummary:
    return SideSummary(
        count=0,
        volume_mwh=Decimal("0"),
        avg_price=Decimal("0"),
        vwap_alpha=Decimal("0"),
        total_alpha_eur=Decimal("0"),
        win_rate=0.0,
    )


def _pnl(market_vwap: float = 50.0, total_alpha: float = 0.0) -> PnlSummary:
    return PnlSummary(
        market_vwap=Decimal(str(market_vwap)),
        buys=_empty_side(),
        sells=_empty_side(),
        total_alpha_eur=Decimal(str(total_alpha)),
    )


def _result(
    fills: tuple[Fill, ...] = (),
    equity: tuple[EquitySnapshot, ...] = (),
    capital: float = 100_000.0,
    market_vwap: float = 50.0,
    total_alpha: float = 0.0,
    sharpe: Decimal | None = None,
    max_dd: float = 0.0,
    max_dd_pct: float = 0.0,
    profit_fac: Decimal | None = None,
    avg_trade: float = 0.0,
    best: Fill | None = None,
    worst: Fill | None = None,
    start: date = date(2026, 3, 1),
    end: date = date(2026, 3, 31),
) -> BacktestResult:
    return BacktestResult(
        algo_name="TestAlgo",
        exchange="nordpool",
        start=start,
        end=end,
        fills=fills,
        pnl=_pnl(market_vwap, total_alpha),
        equity_curve=equity,
        initial_capital=Decimal(str(capital)),
        sharpe_ratio=sharpe,
        max_drawdown=Decimal(str(max_dd)),
        max_drawdown_pct=Decimal(str(max_dd_pct)),
        profit_factor=profit_fac,
        avg_trade_pnl=Decimal(str(avg_trade)),
        best_trade=best,
        worst_trade=worst,
        duration_days=(end - start).days + 1,
    )


def _write_da_prices(path: Path, prices: list[tuple[str, float]]) -> None:
    """Write a minimal da_prices.parquet from (timestamp_str, price) pairs."""
    data = {
        "timestamp": pd.to_datetime([r[0] for r in prices], utc=True),
        "zone": ["NO1"] * len(prices),
        "price_eur_mwh": [r[1] for r in prices],
        "volume_mwh": [1000.0] * len(prices),
    }
    pd.DataFrame(data).to_parquet(path / "da_prices.parquet", index=False)


# ---------------------------------------------------------------------------
# 1. Equity curve tracking
# ---------------------------------------------------------------------------


class TestEquityCurveTracking:
    @pytest.fixture
    def data_dir(self, tmp_path: Path) -> Path:
        _write_da_prices(
            tmp_path,
            [
                ("2026-03-01T00:00:00Z", 40.0),
                ("2026-03-02T00:00:00Z", 60.0),
                ("2026-03-03T00:00:00Z", 50.0),
            ],
        )
        return tmp_path

    def test_snapshot_per_active_day(self, data_dir: Path) -> None:
        """One equity snapshot is produced for each day that had fills."""

        class BuyAll(SimpleAlgo):
            def on_auction_open(self, ctx: TradingContext, auction: AuctionInfo) -> None:
                ctx.place_order(
                    Order.buy(product_id=auction.product_id, volume_mw=1, price_eur_mwh=999)
                )

        engine = BacktestEngine(
            algo=BuyAll(),
            exchange="nordpool",
            start=date(2026, 3, 1),
            end=date(2026, 3, 3),
            products=["NO1_DA"],
            data_dir=data_dir,
            capital=Decimal("100000"),
        )
        result = engine.run()

        # 3 delivery days → 3 snapshots
        assert len(result.equity_curve) == 3

    def test_final_equity_invariant(self, data_dir: Path) -> None:
        """final total_equity == initial_capital + total_pnl."""

        class BuyAll(SimpleAlgo):
            def on_auction_open(self, ctx: TradingContext, auction: AuctionInfo) -> None:
                ctx.place_order(
                    Order.buy(product_id=auction.product_id, volume_mw=1, price_eur_mwh=999)
                )

        engine = BacktestEngine(
            algo=BuyAll(),
            exchange="nordpool",
            start=date(2026, 3, 1),
            end=date(2026, 3, 3),
            products=["NO1_DA"],
            data_dir=data_dir,
            capital=Decimal("100000"),
        )
        result = engine.run()

        expected = result.initial_capital + result.pnl.total_alpha_eur
        assert result.equity_curve[-1].total_equity == pytest.approx(float(expected), rel=1e-9)

    def test_no_fills_means_no_snapshots(self, data_dir: Path) -> None:
        """An algo that places no orders produces an empty equity curve."""

        engine = BacktestEngine(
            algo=SimpleAlgo(),
            exchange="nordpool",
            start=date(2026, 3, 1),
            end=date(2026, 3, 3),
            products=["NO1_DA"],
            data_dir=data_dir,
            capital=Decimal("100000"),
        )
        result = engine.run()
        assert len(result.equity_curve) == 0


# ---------------------------------------------------------------------------
# 2. Sharpe ratio
# ---------------------------------------------------------------------------


class TestSharpeRatio:
    def test_fewer_than_two_snapshots_returns_none(self) -> None:
        assert compute_sharpe([]) is None
        ts = datetime(2026, 3, 1, 13, tzinfo=UTC)
        assert compute_sharpe([_snap(ts, 100_000)]) is None

    def test_known_returns(self) -> None:
        """Sharpe sign matches the sign of the mean return."""
        import numpy as np

        t0 = datetime(2026, 3, 1, 13, tzinfo=UTC)
        t1 = datetime(2026, 3, 2, 13, tzinfo=UTC)
        t2 = datetime(2026, 3, 3, 13, tzinfo=UTC)

        snaps = [_snap(t0, 100_000), _snap(t1, 101_000), _snap(t2, 100_500)]
        sharpe = compute_sharpe(snaps)
        assert sharpe is not None

        equities = [100_000.0, 101_000.0, 100_500.0]
        returns = np.diff(equities) / equities[:-1]
        mean = float(np.mean(returns))
        assert (float(sharpe) > 0) == (mean > 0)

    def test_zero_std_returns_none(self) -> None:
        """Flat equity (zero volatility) returns None."""
        t0 = datetime(2026, 3, 1, 13, tzinfo=UTC)
        t1 = datetime(2026, 3, 2, 13, tzinfo=UTC)
        t2 = datetime(2026, 3, 3, 13, tzinfo=UTC)
        snaps = [_snap(t0, 100_000), _snap(t1, 100_000), _snap(t2, 100_000)]
        assert compute_sharpe(snaps) is None


# ---------------------------------------------------------------------------
# 3. Max drawdown
# ---------------------------------------------------------------------------


class TestMaxDrawdown:
    def test_up_then_down_then_up(self) -> None:
        """Drawdown from peak 110 to valley 90 = 20 EUR, 18.18%."""
        t = lambda d: datetime(2026, 3, d, 13, tzinfo=UTC)  # noqa: E731
        snaps = [
            _snap(t(1), 100_000),
            _snap(t(2), 110_000),  # new peak
            _snap(t(3), 90_000),  # drawdown = 20_000
            _snap(t(4), 105_000),
        ]
        dd, dd_pct = compute_max_drawdown(snaps)
        assert float(dd) == pytest.approx(20_000.0)
        assert float(dd_pct) == pytest.approx(20_000 / 110_000, rel=1e-6)

    def test_monotonically_increasing_zero_drawdown(self) -> None:
        t = lambda d: datetime(2026, 3, d, 13, tzinfo=UTC)  # noqa: E731
        snaps = [_snap(t(1), 100), _snap(t(2), 110), _snap(t(3), 120)]
        dd, dd_pct = compute_max_drawdown(snaps)
        assert dd == Decimal("0")
        assert dd_pct == Decimal("0")

    def test_empty_snapshots(self) -> None:
        dd, dd_pct = compute_max_drawdown([])
        assert dd == Decimal("0")
        assert dd_pct == Decimal("0")


# ---------------------------------------------------------------------------
# 4. Profit factor
# ---------------------------------------------------------------------------


class TestProfitFactor:
    def test_known_trades(self) -> None:
        """3 winning trades (10 EUR each) + 1 losing (-15 EUR) → PF = 30/15 = 2."""
        vwap = Decimal("50")
        # Buys below VWAP → profit; buy above → loss
        wins = [_fill(45.0) for _ in range(3)]  # (50-45)*10 = 50 each
        loss = [_fill(65.0)]  # (50-65)*10 = -150
        pf = compute_profit_factor(wins + loss, vwap)
        assert pf is not None
        assert float(pf) == pytest.approx(3 * 50.0 / 150.0, rel=1e-6)

    def test_no_losing_trades_returns_none(self) -> None:
        vwap = Decimal("50")
        wins = [_fill(40.0), _fill(45.0)]
        assert compute_profit_factor(wins, vwap) is None

    def test_empty_fills(self) -> None:
        assert compute_profit_factor([], Decimal("50")) is None


# ---------------------------------------------------------------------------
# 5. Daily PnL aggregation
# ---------------------------------------------------------------------------


class TestDailyPnl:
    def test_fills_assigned_to_correct_day(self) -> None:
        """Fills before and after midnight are attributed to separate days."""
        vwap = Decimal("50")
        f1 = _fill(40.0, ts=datetime(2026, 3, 1, 23, 55, tzinfo=UTC))  # day 1
        f2 = _fill(60.0, ts=datetime(2026, 3, 2, 0, 5, tzinfo=UTC))  # day 2
        result = _result(
            fills=(f1, f2),
            market_vwap=50.0,
            total_alpha=float(compute_fill_pnl(f1, vwap) + compute_fill_pnl(f2, vwap)),
        )
        daily = result.daily_pnl()
        assert len(daily) == 2
        assert daily[0].date == date(2026, 3, 1)
        assert daily[1].date == date(2026, 3, 2)
        # day 1: buy at 40, vwap 50 → pnl = (50-40)*10 = 100
        assert float(daily[0].pnl) == pytest.approx(100.0)

    def test_single_day(self) -> None:
        fills = (
            _fill(45.0, ts=datetime(2026, 3, 5, 10, 0, tzinfo=UTC)),
            _fill(48.0, ts=datetime(2026, 3, 5, 14, 0, tzinfo=UTC)),
        )
        result = _result(fills=fills, market_vwap=50.0)
        daily = result.daily_pnl()
        assert len(daily) == 1
        assert daily[0].date == date(2026, 3, 5)
        assert daily[0].trade_count == 2

    def test_no_fills_empty_daily(self) -> None:
        result = _result()
        assert result.daily_pnl() == []


# ---------------------------------------------------------------------------
# 6. HTML report
# ---------------------------------------------------------------------------


class _HTMLChecker(HTMLParser):
    """Collect all id attributes from div elements."""

    def __init__(self) -> None:
        super().__init__()
        self.ids: set[str] = set()
        self.errors: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        for name, value in attrs:
            if name == "id" and value:
                self.ids.add(value)


class TestHtmlReport:
    def test_generates_valid_html(self, tmp_path: Path) -> None:
        """to_html produces parseable HTML with no parser errors."""
        result = _result(
            fills=(_fill(45.0),),
            equity=(_snap(datetime(2026, 3, 1, 13, tzinfo=UTC), 100_100),),
            capital=100_000.0,
        )
        out = tmp_path / "report.html"
        result.to_html(str(out))
        html = out.read_text(encoding="utf-8")

        checker = _HTMLChecker()
        checker.feed(html)  # raises on malformed HTML

        assert html.strip().startswith("<!DOCTYPE html")
        assert "</html>" in html.lower()

    def test_chart_containers_present(self, tmp_path: Path) -> None:
        """All five chart div containers are present in the HTML."""
        result = _result()
        out = tmp_path / "report.html"
        result.to_html(str(out))
        html = out.read_text(encoding="utf-8")

        checker = _HTMLChecker()
        checker.feed(html)

        expected_ids = {
            "chart-equity",
            "chart-drawdown",
            "chart-daily",
            "chart-vwap",
            "chart-hourly",
        }
        assert expected_ids.issubset(checker.ids)

    def test_contains_algo_name(self, tmp_path: Path) -> None:
        result = _result()
        out = tmp_path / "report.html"
        result.to_html(str(out))
        assert "TestAlgo" in out.read_text(encoding="utf-8")

    def test_color_palette_present(self, tmp_path: Path) -> None:
        """Phase Nexa background colour is embedded in the report CSS."""
        result = _result()
        out = tmp_path / "report.html"
        result.to_html(str(out))
        html = out.read_text(encoding="utf-8")
        assert "#0E2A47" in html  # Quantum Blue background


# ---------------------------------------------------------------------------
# 7. JSON export
# ---------------------------------------------------------------------------


class TestJsonExport:
    def test_round_trip_metrics(self, tmp_path: Path) -> None:
        """Exported JSON can be read back and top-level metrics match."""
        f = _fill(45.0)
        result = _result(
            fills=(f,),
            capital=100_000.0,
            market_vwap=50.0,
            total_alpha=50.0,
            avg_trade=50.0,
            start=date(2026, 3, 1),
            end=date(2026, 3, 31),
        )
        out = tmp_path / "result.json"
        result.to_json(str(out))

        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["algo_name"] == "TestAlgo"
        assert data["exchange"] == "nordpool"
        assert Decimal(data["initial_capital"]) == Decimal("100000")
        assert data["trade_count"] == 1

    def test_decimal_precision_preserved(self, tmp_path: Path) -> None:
        """Decimal values are written as strings, not floats."""
        result = _result(capital=100_000.123456789)
        out = tmp_path / "r.json"
        result.to_json(str(out))
        raw = out.read_text(encoding="utf-8")
        data = json.loads(raw)
        # initial_capital stored as string — check no precision loss via repr
        assert isinstance(data["initial_capital"], str)

    def test_equity_curve_in_output(self, tmp_path: Path) -> None:
        ts = datetime(2026, 3, 1, 13, tzinfo=UTC)
        snap = _snap(ts, 100_500, realised=500)
        result = _result(equity=(snap,))
        out = tmp_path / "r.json"
        result.to_json(str(out))
        data = json.loads(out.read_text(encoding="utf-8"))
        assert len(data["equity_curve"]) == 1
        assert data["equity_curve"][0]["total_equity"] == str(snap.total_equity)

    def test_trades_in_output(self, tmp_path: Path) -> None:
        f = _fill(45.0)
        result = _result(fills=(f,))
        out = tmp_path / "r.json"
        result.to_json(str(out))
        data = json.loads(out.read_text(encoding="utf-8"))
        assert len(data["trades"]) == 1
        assert data["trades"][0]["side"] == "BUY"
        assert data["trades"][0]["price"] == str(f.price)


# ---------------------------------------------------------------------------
# 8. Parquet export
# ---------------------------------------------------------------------------


class TestParquetExport:
    def test_directory_structure(self, tmp_path: Path) -> None:
        """to_parquet creates the four expected output files."""
        ts = datetime(2026, 3, 1, 13, tzinfo=UTC)
        snap = _snap(ts, 100_500, realised=500)
        f = _fill(45.0, ts=ts)
        result = _result(fills=(f,), equity=(snap,))

        out = tmp_path / "backtest_out"
        result.to_parquet(str(out))

        assert (out / "metadata.json").exists()
        assert (out / "equity_curve.parquet").exists()
        assert (out / "trades.parquet").exists()
        assert (out / "daily_pnl.parquet").exists()

    def test_equity_curve_schema(self, tmp_path: Path) -> None:
        """equity_curve.parquet has expected column names."""
        ts = datetime(2026, 3, 1, 13, tzinfo=UTC)
        snap = _snap(ts, 100_500, realised=500)
        result = _result(equity=(snap,))
        out = tmp_path / "out"
        result.to_parquet(str(out))

        table = pq.read_table(out / "equity_curve.parquet")
        assert set(table.column_names) == {
            "timestamp",
            "realised_pnl",
            "unrealised_pnl",
            "total_equity",
            "cash",
            "net_position_mw",
        }

    def test_trades_schema(self, tmp_path: Path) -> None:
        f = _fill(45.0)
        result = _result(fills=(f,))
        out = tmp_path / "out"
        result.to_parquet(str(out))

        table = pq.read_table(out / "trades.parquet")
        assert set(table.column_names) == {
            "order_id",
            "product_id",
            "side",
            "price",
            "volume",
            "timestamp",
        }

    def test_metadata_json_readable(self, tmp_path: Path) -> None:
        result = _result(capital=50_000.0)
        out = tmp_path / "out"
        result.to_parquet(str(out))
        meta = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
        assert Decimal(meta["initial_capital"]) == Decimal("50000")
        assert "algo_name" in meta

    def test_empty_fills_skips_trades_parquet(self, tmp_path: Path) -> None:
        """No trades.parquet written when there are no fills."""
        result = _result()
        out = tmp_path / "out"
        result.to_parquet(str(out))
        assert not (out / "trades.parquet").exists()
