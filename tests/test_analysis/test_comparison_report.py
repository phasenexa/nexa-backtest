"""Unit tests for comparison_report.py targeting uncovered lines.

Missing lines at time of writing:
    57-59  _json default handler (Decimal → float, TypeError for unknown types)
    105    drawdown peak update when equity rises above previous peak
    327    mem_note = "" branch when peak_memory_bytes == 0
"""

from __future__ import annotations

import dataclasses
import json
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest

from nexa_backtest.analysis.comparison_report import (
    _json,
    _overlaid_drawdown_traces,
    generate_comparison_html_report,
)

# ---------------------------------------------------------------------------
# Shared helpers (mirrors test_shared_replay.py)
# ---------------------------------------------------------------------------


def _write_da_prices(path: Path, rows: list[tuple[str, str, float, float]]) -> None:
    data = {
        "timestamp": pd.to_datetime([r[1] for r in rows], utc=True),
        "zone": [r[0] for r in rows],
        "price_eur_mwh": [r[2] for r in rows],
        "volume_mwh": [r[3] for r in rows],
    }
    pd.DataFrame(data).to_parquet(path / "da_prices.parquet", index=False)


@pytest.fixture()
def comparison(tmp_path: Path):
    """Run a minimal two-algo DA shared replay and return the ComparisonResult."""
    from decimal import Decimal

    from nexa_backtest.algo import SimpleAlgo
    from nexa_backtest.context import TradingContext
    from nexa_backtest.engines.shared import SharedReplayEngine
    from nexa_backtest.types import AuctionInfo, Order

    class _AlwaysBuy(SimpleAlgo):
        def on_auction_open(self, ctx: TradingContext, auction: AuctionInfo) -> None:
            ctx.place_order(
                Order.buy(
                    product_id=auction.product_id,
                    volume_mw=Decimal("10"),
                    price_eur_mwh=Decimal("999"),
                )
            )

    class _NoOp(SimpleAlgo):
        pass

    _write_da_prices(
        tmp_path,
        [
            ("NO1", "2026-03-01T00:00:00Z", 45.0, 1000.0),
            ("NO1", "2026-03-01T00:15:00Z", 50.0, 1000.0),
            ("NO1", "2026-03-01T00:30:00Z", 55.0, 1000.0),
        ],
    )

    return SharedReplayEngine(
        algos={"buyer": _AlwaysBuy(), "noop": _NoOp()},
        exchange="nordpool",
        start=date(2026, 3, 1),
        end=date(2026, 3, 1),
        products=["NO1_DA"],
        data_dir=tmp_path,
        initial_capital=Decimal("100000"),
    ).run()


def _make_zero_memory_comparison(comparison):
    """Return a copy of a ComparisonResult with peak_memory_bytes=0."""

    return dataclasses.replace(comparison, peak_memory_bytes=0, estimated_separate_memory_bytes=0)


# ---------------------------------------------------------------------------
# Tests for _json (lines 57-59)
# ---------------------------------------------------------------------------


class TestJsonHelper:
    """_json serialises Decimal values and raises TypeError for unknowns."""

    def test_decimal_is_serialised_as_float(self) -> None:
        payload = {"price": Decimal("50.75")}
        result = json.loads(_json(payload))
        assert result["price"] == pytest.approx(50.75)

    def test_non_serialisable_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="not JSON serialisable"):
            _json({"obj": object()})


# ---------------------------------------------------------------------------
# Tests for _overlaid_drawdown_traces (line 105)
# ---------------------------------------------------------------------------


class TestDrawdownPeakUpdate:
    """Equity rising above a previous peak must update the running peak."""

    def test_rising_equity_updates_peak(self, comparison) -> None:
        """Use the real comparison result (buyer has increasing equity) to exercise
        the `peak = eq` branch inside _overlaid_drawdown_traces."""
        from nexa_backtest.types import EquitySnapshot

        # Build a synthetic ComparisonResult with a rising-then-falling equity curve.
        snapshots = (
            EquitySnapshot(
                timestamp=datetime(2026, 3, 1, 0, 0, tzinfo=UTC),
                realised_pnl=Decimal("0"),
                unrealised_pnl=Decimal("0"),
                total_equity=Decimal("100000"),
                cash=Decimal("100000"),
                net_position_mw=Decimal("0"),
            ),
            # equity rises — triggers the `peak = eq` branch
            EquitySnapshot(
                timestamp=datetime(2026, 3, 1, 1, 0, tzinfo=UTC),
                realised_pnl=Decimal("500"),
                unrealised_pnl=Decimal("0"),
                total_equity=Decimal("100500"),
                cash=Decimal("100500"),
                net_position_mw=Decimal("0"),
            ),
            # equity then falls back — produces a negative drawdown
            EquitySnapshot(
                timestamp=datetime(2026, 3, 1, 2, 0, tzinfo=UTC),
                realised_pnl=Decimal("200"),
                unrealised_pnl=Decimal("0"),
                total_equity=Decimal("100200"),
                cash=Decimal("100200"),
                net_position_mw=Decimal("0"),
            ),
        )
        # Patch just the equity_curve of the first algo result.
        first_name = next(iter(comparison.results))
        patched_result = dataclasses.replace(comparison.results[first_name], equity_curve=snapshots)
        patched_comparison = dataclasses.replace(comparison, results={first_name: patched_result})

        traces_json = _overlaid_drawdown_traces(patched_comparison)
        traces = json.loads(traces_json)
        assert len(traces) == 1
        drawdowns = traces[0]["y"]
        # First snapshot is the starting peak — drawdown is 0.
        assert drawdowns[0] == pytest.approx(0.0)
        # Second snapshot is higher than the first — peak updates, drawdown still 0.
        assert drawdowns[1] == pytest.approx(0.0)
        # Third snapshot has fallen from the new peak.
        assert drawdowns[2] < 0


# ---------------------------------------------------------------------------
# Tests for generate_comparison_html_report (line 327)
# ---------------------------------------------------------------------------


class TestHtmlReportMemNote:
    """mem_note is empty when peak_memory_bytes == 0 (line 327)."""

    def test_no_memory_note_when_peak_zero(self, comparison) -> None:
        zero_mem = _make_zero_memory_comparison(comparison)
        html = generate_comparison_html_report(zero_mem)
        # Should not contain the memory efficiency note.
        assert "saved" not in html
        assert "MB" not in html

    def test_memory_note_present_when_peak_nonzero(self, comparison) -> None:
        nonzero = dataclasses.replace(
            comparison,
            peak_memory_bytes=10_000_000,
            estimated_separate_memory_bytes=20_000_000,
        )
        html = generate_comparison_html_report(nonzero)
        assert "MB" in html
        assert "saved" in html
