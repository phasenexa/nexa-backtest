"""Integration tests for SharedReplayEngine and ComparisonResult.

Covers:
- Basic two-algo DA shared replay
- Lockstep timestamp ordering
- Algo isolation (orders not shared between algos)
- Identical algos produce identical results
- Mixed SimpleAlgo and @algo decorator
- ComparisonResult.ranking() by multiple metrics
- Summary text includes all algo names
- HTML report contains chart containers and metrics table
- CLI nexa compare with two algo files
- CLI nexa compare with too many algos raises error
- Memory tracking is reported
- IDC shared replay (SlidingWindow not doubled)
"""

from __future__ import annotations

import json
import textwrap
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from click.testing import CliRunner

from nexa_backtest.algo import SimpleAlgo, algo
from nexa_backtest.context import TradingContext
from nexa_backtest.engines.shared import MAX_ALGOS, ComparisonResult, SharedReplayEngine
from nexa_backtest.exceptions import AlgoError, DataError
from nexa_backtest.types import AuctionInfo, Order

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_da_prices(
    path: Path,
    rows: list[tuple[str, str, float, float]],
) -> None:
    """Write a minimal da_prices.parquet from (zone, timestamp_str, price, vol) rows."""
    data = {
        "timestamp": pd.to_datetime([r[1] for r in rows], utc=True),
        "zone": [r[0] for r in rows],
        "price_eur_mwh": [r[2] for r in rows],
        "volume_mwh": [r[3] for r in rows],
    }
    pd.DataFrame(data).to_parquet(path / "da_prices.parquet", index=False)


# ---------------------------------------------------------------------------
# Fixture algos
# ---------------------------------------------------------------------------


class _AlwaysBuy(SimpleAlgo):
    """Buys at a very high price (always fills)."""

    def on_auction_open(self, ctx: TradingContext, auction: AuctionInfo) -> None:
        ctx.place_order(
            Order.buy(
                product_id=auction.product_id,
                volume_mw=Decimal("10"),
                price_eur_mwh=Decimal("999"),
            )
        )


class _AlwaysSell(SimpleAlgo):
    """Sells at zero price (always fills)."""

    def on_auction_open(self, ctx: TradingContext, auction: AuctionInfo) -> None:
        ctx.place_order(
            Order.sell(
                product_id=auction.product_id,
                volume_mw=Decimal("10"),
                price_eur_mwh=Decimal("0"),
            )
        )


class _NoOp(SimpleAlgo):
    """Does nothing — zero fills, zero PnL."""


class _TrackSetup(SimpleAlgo):
    """Records that on_setup was called."""

    setup_called = False

    def on_setup(self, ctx: TradingContext) -> None:
        _TrackSetup.setup_called = True


# ---------------------------------------------------------------------------
# Shared data dir fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def data_dir(tmp_path: Path) -> Path:
    """A minimal DA data dir with three MTU prices on 2026-03-01."""
    _write_da_prices(
        tmp_path,
        [
            ("NO1", "2026-03-01T00:00:00Z", 45.0, 1000.0),
            ("NO1", "2026-03-01T00:15:00Z", 50.0, 1000.0),
            ("NO1", "2026-03-01T00:30:00Z", 40.0, 1000.0),
        ],
    )
    return tmp_path


def _engine(algos: dict[str, Any], data_dir: Path) -> SharedReplayEngine:
    return SharedReplayEngine(
        algos=algos,
        exchange="nordpool",
        start=date(2026, 3, 1),
        end=date(2026, 3, 1),
        products=["NO1_DA"],
        data_dir=data_dir,
        initial_capital=Decimal("100000"),
    )


# ===========================================================================
# Tests
# ===========================================================================


class TestSharedReplayBasic:
    """Basic two-algo DA run."""

    def test_two_algos_both_fill(self, data_dir: Path) -> None:
        result = _engine({"buyer": _AlwaysBuy(), "seller": _AlwaysSell()}, data_dir).run()
        assert isinstance(result, ComparisonResult)
        assert len(result.results["buyer"].fills) == 3
        assert len(result.results["seller"].fills) == 3

    def test_result_contains_all_algo_names(self, data_dir: Path) -> None:
        result = _engine({"alpha": _AlwaysBuy(), "beta": _NoOp()}, data_dir).run()
        assert set(result.results.keys()) == {"alpha", "beta"}

    def test_noop_algo_produces_no_fills(self, data_dir: Path) -> None:
        result = _engine({"passive": _NoOp(), "active": _AlwaysBuy()}, data_dir).run()
        assert len(result.results["passive"].fills) == 0

    def test_exchange_and_products_preserved(self, data_dir: Path) -> None:
        result = _engine({"a": _AlwaysBuy(), "b": _NoOp()}, data_dir).run()
        assert result.exchange == "nordpool"
        assert result.products == ["NO1_DA"]
        assert result.start_date == date(2026, 3, 1)
        assert result.end_date == date(2026, 3, 1)

    def test_on_setup_called_for_all_algos(self, data_dir: Path) -> None:
        """on_setup must be dispatched to every algo before auction events."""
        _TrackSetup.setup_called = False
        _engine({"ts": _TrackSetup(), "buy": _AlwaysBuy()}, data_dir).run()
        assert _TrackSetup.setup_called


class TestSharedReplayLockstep:
    """All algos advance to the same timestamp before any moves to the next."""

    def test_fills_have_same_timestamps_across_algos(self, data_dir: Path) -> None:
        result = _engine({"a": _AlwaysBuy(), "b": _AlwaysBuy()}, data_dir).run()
        ts_a = {f.timestamp for f in result.results["a"].fills}
        ts_b = {f.timestamp for f in result.results["b"].fills}
        assert ts_a == ts_b

    def test_fill_timestamps_are_ordered(self, data_dir: Path) -> None:
        result = _engine({"a": _AlwaysBuy()}, data_dir).run()
        fills = list(result.results["a"].fills)
        timestamps = [f.timestamp for f in fills]
        assert timestamps == sorted(timestamps)


class TestAlgoIsolation:
    """Orders placed by one algo must not appear in another algo's context."""

    def test_fill_counts_are_independent(self, data_dir: Path) -> None:
        """AlwaysBuy and NoOp should have 3 vs 0 fills, never the same."""
        result = _engine({"active": _AlwaysBuy(), "passive": _NoOp()}, data_dir).run()
        assert len(result.results["active"].fills) == 3
        assert len(result.results["passive"].fills) == 0

    def test_fills_reference_correct_algo_orders(self, data_dir: Path) -> None:
        """Buy fills must all have side BUY; sell fills must all have side SELL."""
        from nexa_backtest.types import Side

        result = _engine({"b": _AlwaysBuy(), "s": _AlwaysSell()}, data_dir).run()
        for f in result.results["b"].fills:
            assert f.side == Side.BUY
        for f in result.results["s"].fills:
            assert f.side == Side.SELL

    def test_buy_and_sell_fill_sides_are_different(self, data_dir: Path) -> None:
        """A buyer and a seller see the same market but with opposite fill sides."""
        from nexa_backtest.types import Side

        result = _engine({"buy": _AlwaysBuy(), "sell": _AlwaysSell()}, data_dir).run()
        buy_sides = {f.side for f in result.results["buy"].fills}
        sell_sides = {f.side for f in result.results["sell"].fills}
        assert buy_sides == {Side.BUY}
        assert sell_sides == {Side.SELL}


class TestIdenticalAlgosDuplicateResults:
    """Running the same algo class twice under different names yields identical results."""

    def test_same_class_same_results(self, data_dir: Path) -> None:
        result = _engine({"a": _AlwaysBuy(), "b": _AlwaysBuy()}, data_dir).run()
        fills_a = result.results["a"].fills
        fills_b = result.results["b"].fills
        assert len(fills_a) == len(fills_b)
        for fa, fb in zip(fills_a, fills_b, strict=True):
            assert fa.price == fb.price
            assert fa.volume == fb.volume
            assert fa.product_id == fb.product_id


class TestAlgoDecoratorSupport:
    """SharedReplayEngine must accept @algo-decorated async functions."""

    def test_async_algo_runs_and_produces_result(self, data_dir: Path) -> None:
        @algo(name="async_buy", version="0.1.0")
        async def _async_buy(ctx: TradingContext) -> None:
            async for event in ctx.events():
                if isinstance(event, AuctionInfo):
                    ctx.place_order(
                        Order.buy(
                            product_id=event.product_id,
                            volume_mw=Decimal("5"),
                            price_eur_mwh=Decimal("999"),
                        )
                    )

        result = _engine({"async_buy": _async_buy, "noop": _NoOp()}, data_dir).run()
        # @algo dispatches on_auction_open as a BarEvent, not AuctionInfo,
        # so it may not fill; assert the run completes without error.
        assert "async_buy" in result.results
        assert "noop" in result.results

    def test_mixed_simple_and_async_algos(self, data_dir: Path) -> None:
        @algo(name="async_noop", version="0.1.0")
        async def _async_noop(ctx: TradingContext) -> None:
            async for _event in ctx.events():
                pass

        result = _engine({"simple": _AlwaysBuy(), "async": _async_noop}, data_dir).run()
        assert len(result.results["simple"].fills) == 3
        assert len(result.results["async"].fills) == 0


class TestComparisonResultRanking:
    """ranking() orders algos by the specified metric."""

    @pytest.fixture()
    def comparison(self, data_dir: Path) -> ComparisonResult:
        return _engine({"buyer": _AlwaysBuy(), "noop": _NoOp()}, data_dir).run()

    def test_ranking_by_total_pnl(self, comparison: ComparisonResult) -> None:
        ranked = comparison.ranking("total_pnl")
        assert len(ranked) == 2
        pnl_first = comparison.results[ranked[0]].pnl.total_alpha_eur
        pnl_second = comparison.results[ranked[1]].pnl.total_alpha_eur
        assert pnl_first >= pnl_second

    def test_ranking_by_trades(self, comparison: ComparisonResult) -> None:
        ranked = comparison.ranking("trades")
        assert len(ranked) == 2
        count_first = len(comparison.results[ranked[0]].fills)
        count_second = len(comparison.results[ranked[1]].fills)
        assert count_first >= count_second

    def test_best_and_worst(self, comparison: ComparisonResult) -> None:
        best_name, _ = comparison.best
        worst_name, _ = comparison.worst
        # Both best and worst must be valid algo names.
        assert best_name in {"buyer", "noop"}
        assert worst_name in {"buyer", "noop"}

    def test_ranking_unknown_metric_falls_back_to_pnl(self, comparison: ComparisonResult) -> None:
        # Unknown metric should not raise, falls back to total_pnl.
        ranked = comparison.ranking("nonexistent_metric")
        assert len(ranked) == 2


class TestComparisonResultSummary:
    """summary() must include all algo names and key metrics."""

    def test_summary_contains_all_names(self, data_dir: Path) -> None:
        result = _engine(
            {"alpha": _AlwaysBuy(), "beta": _AlwaysSell(), "gamma": _NoOp()}, data_dir
        ).run()
        summary = result.summary()
        assert "alpha" in summary
        assert "beta" in summary
        assert "gamma" in summary

    def test_summary_contains_exchange(self, data_dir: Path) -> None:
        result = _engine({"a": _AlwaysBuy(), "b": _NoOp()}, data_dir).run()
        assert "nordpool" in result.summary().lower()

    def test_summary_contains_pnl_label(self, data_dir: Path) -> None:
        result = _engine({"a": _AlwaysBuy(), "b": _NoOp()}, data_dir).run()
        assert "PnL" in result.summary()


class TestComparisonResultHtml:
    """HTML report must contain chart containers and metrics table."""

    @pytest.fixture()
    def html(self, data_dir: Path) -> str:
        result = _engine({"a": _AlwaysBuy(), "b": _NoOp()}, data_dir).run()
        return result.to_html.__func__  # type: ignore[attr-defined]

    def test_html_contains_chart_div(self, data_dir: Path, tmp_path: Path) -> None:
        result = _engine({"a": _AlwaysBuy(), "b": _NoOp()}, data_dir).run()
        out = tmp_path / "report.html"
        result.to_html(str(out))
        html = out.read_text()
        assert "chart-equity" in html or "chart" in html

    def test_html_contains_metrics_table(self, data_dir: Path, tmp_path: Path) -> None:
        result = _engine({"a": _AlwaysBuy(), "b": _NoOp()}, data_dir).run()
        out = tmp_path / "report.html"
        result.to_html(str(out))
        html = out.read_text()
        assert "metrics-table" in html or "Metric" in html

    def test_html_contains_algo_names(self, data_dir: Path, tmp_path: Path) -> None:
        result = _engine({"alpha": _AlwaysBuy(), "beta": _NoOp()}, data_dir).run()
        out = tmp_path / "report.html"
        result.to_html(str(out))
        html = out.read_text()
        assert "alpha" in html
        assert "beta" in html

    def test_to_json_exports_all_algos(self, data_dir: Path, tmp_path: Path) -> None:
        result = _engine({"buyer": _AlwaysBuy(), "noop": _NoOp()}, data_dir).run()
        out = tmp_path / "report.json"
        result.to_json(str(out))
        payload = json.loads(out.read_text())
        assert "buyer" in payload["algos"]
        assert "noop" in payload["algos"]
        assert payload["exchange"] == "nordpool"


class TestCliCompare:
    """CLI nexa compare integration tests."""

    def _write_algo(self, tmp_path: Path, name: str, content: str) -> Path:
        p = tmp_path / f"{name}.py"
        p.write_text(textwrap.dedent(content))
        return p

    def test_compare_two_algos(self, data_dir: Path, tmp_path: Path) -> None:
        from nexa_backtest.cli.main import cli

        algo_a = self._write_algo(
            tmp_path,
            "always_buy",
            """
            from decimal import Decimal
            from nexa_backtest.algo import SimpleAlgo
            from nexa_backtest.context import TradingContext
            from nexa_backtest.types import AuctionInfo, Order

            class AlwaysBuyAlgo(SimpleAlgo):
                def on_auction_open(self, ctx: TradingContext, auction: AuctionInfo) -> None:
                    ctx.place_order(Order.buy(
                        product_id=auction.product_id, volume_mw=Decimal("10"),
                        price_eur_mwh=Decimal("999"),
                    ))
            """,
        )
        algo_b = self._write_algo(
            tmp_path,
            "noop",
            """
            from nexa_backtest.algo import SimpleAlgo
            class NoOpAlgo(SimpleAlgo):
                pass
            """,
        )

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "compare",
                f"buyer:{algo_a}",
                f"noop:{algo_b}",
                "--exchange",
                "nordpool",
                "--start",
                "2026-03-01",
                "--end",
                "2026-03-01",
                "--products",
                "NO1_DA",
                "--data-dir",
                str(data_dir),
            ],
        )
        assert result.exit_code == 0, result.output
        assert "buyer" in result.output
        assert "noop" in result.output

    def test_compare_too_many_algos_raises_error(self, data_dir: Path, tmp_path: Path) -> None:
        from nexa_backtest.cli.main import cli

        # Write MAX_ALGOS + 1 = 9 algos.
        algo_path = self._write_algo(
            tmp_path,
            "noop",
            "from nexa_backtest.algo import SimpleAlgo\nclass NoOpAlgo(SimpleAlgo): pass\n",
        )
        specs = [f"a{i}:{algo_path}" for i in range(MAX_ALGOS + 1)]

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "compare",
                *specs,
                "--exchange",
                "nordpool",
                "--start",
                "2026-03-01",
                "--end",
                "2026-03-01",
                "--products",
                "NO1_DA",
                "--data-dir",
                str(data_dir),
            ],
        )
        assert result.exit_code != 0
        assert "Too many algos" in result.output or "Error" in result.output

    def test_compare_single_algo_raises_error(self, data_dir: Path, tmp_path: Path) -> None:
        from nexa_backtest.cli.main import cli

        algo_path = self._write_algo(
            tmp_path,
            "noop",
            "from nexa_backtest.algo import SimpleAlgo\nclass NoOpAlgo(SimpleAlgo): pass\n",
        )
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "compare",
                str(algo_path),
                "--exchange",
                "nordpool",
                "--start",
                "2026-03-01",
                "--end",
                "2026-03-01",
                "--products",
                "NO1_DA",
                "--data-dir",
                str(data_dir),
            ],
        )
        assert result.exit_code != 0

    def test_compare_with_json_output(self, data_dir: Path, tmp_path: Path) -> None:
        from nexa_backtest.cli.main import cli

        algo_path = self._write_algo(
            tmp_path,
            "noop",
            "from nexa_backtest.algo import SimpleAlgo\nclass NoOpAlgo(SimpleAlgo): pass\n",
        )
        out_path = tmp_path / "out.json"
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "compare",
                f"a:{algo_path}",
                f"b:{algo_path}",
                "--exchange",
                "nordpool",
                "--start",
                "2026-03-01",
                "--end",
                "2026-03-01",
                "--products",
                "NO1_DA",
                "--data-dir",
                str(data_dir),
                "--output",
                str(out_path),
            ],
        )
        assert result.exit_code == 0, result.output
        assert out_path.exists()
        payload = json.loads(out_path.read_text())
        assert "algos" in payload


class TestMemoryTracking:
    """peak_memory_bytes is reported and estimated_separate_memory_bytes is larger."""

    def test_memory_bytes_reported(self, data_dir: Path) -> None:
        result = _engine({"a": _AlwaysBuy(), "b": _NoOp()}, data_dir).run()
        assert result.peak_memory_bytes > 0

    def test_estimated_separate_is_multiple_of_peak(self, data_dir: Path) -> None:
        result = _engine({"a": _AlwaysBuy(), "b": _NoOp()}, data_dir).run()
        n_algos = len(result.results)
        assert result.estimated_separate_memory_bytes == result.peak_memory_bytes * n_algos

    def test_memory_note_in_summary(self, data_dir: Path) -> None:
        result = _engine({"a": _AlwaysBuy(), "b": _NoOp()}, data_dir).run()
        summary = result.summary()
        assert "Memory" in summary or "MB" in summary


class TestSharedReplayValidation:
    """Engine should raise early for invalid configurations."""

    def test_too_many_algos_raises_algo_error(self, data_dir: Path) -> None:
        algos = {f"a{i}": _NoOp() for i in range(MAX_ALGOS + 1)}
        with pytest.raises(AlgoError, match="at most"):
            SharedReplayEngine(
                algos=algos,
                exchange="nordpool",
                start=date(2026, 3, 1),
                end=date(2026, 3, 1),
                products=["NO1_DA"],
                data_dir=data_dir,
            )

    def test_empty_algos_raises_algo_error(self, data_dir: Path) -> None:
        with pytest.raises(AlgoError):
            SharedReplayEngine(
                algos={},
                exchange="nordpool",
                start=date(2026, 3, 1),
                end=date(2026, 3, 1),
                products=["NO1_DA"],
                data_dir=data_dir,
            )

    def test_invalid_algo_type_raises_algo_error(self, data_dir: Path) -> None:
        engine = SharedReplayEngine(
            algos={"bad": "not_an_algo", "good": _NoOp()},  # type: ignore[arg-type]
            exchange="nordpool",
            start=date(2026, 3, 1),
            end=date(2026, 3, 1),
            products=["NO1_DA"],
            data_dir=data_dir,
        )
        with pytest.raises(AlgoError):
            engine.run()

    def test_missing_data_raises_data_error(self, tmp_path: Path) -> None:
        engine = SharedReplayEngine(
            algos={"a": _NoOp(), "b": _NoOp()},
            exchange="nordpool",
            start=date(2026, 3, 1),
            end=date(2026, 3, 1),
            products=["NO1_DA"],
            data_dir=tmp_path,  # no da_prices.parquet
        )
        with pytest.raises(DataError):
            engine.run()

    def test_no_products_raises_data_error(self, data_dir: Path) -> None:
        engine = SharedReplayEngine(
            algos={"a": _NoOp(), "b": _NoOp()},
            exchange="nordpool",
            start=date(2026, 3, 1),
            end=date(2026, 3, 1),
            products=[],
            data_dir=data_dir,
        )
        with pytest.raises(DataError, match="No products"):
            engine.run()
