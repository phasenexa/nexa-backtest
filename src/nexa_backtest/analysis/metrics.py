"""BacktestResult, advanced metrics, and export methods.

:class:`BacktestResult` holds the complete output of a backtest run, exposes
metric computation helpers, and can produce text summaries, DataFrames, and
file exports (HTML, JSON, Parquet).

Metric functions (:func:`compute_sharpe`, :func:`compute_max_drawdown`,
:func:`compute_profit_factor`) operate on the equity curve and fill list —
not on raw market data — and are importable independently.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from nexa_backtest.analysis.pnl import PnlSummary
from nexa_backtest.types import EquitySnapshot, Fill, GateClosureSnapshot, Side

if TYPE_CHECKING:
    import pandas as pd


# ---------------------------------------------------------------------------
# Supporting types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DailyPnL:
    """Aggregated performance metrics for a single calendar day.

    Attributes:
        date: The calendar date.
        pnl: Total realised PnL for the day in EUR (VWAP-relative).
        volume_mw: Total traded volume in MW.
        trade_count: Number of fills on the day.
        vwap_edge: Average PnL per MW in EUR/MW (``pnl / volume_mw``).
            Zero if no volume.
    """

    date: date
    pnl: Decimal
    volume_mw: Decimal
    trade_count: int
    vwap_edge: Decimal


# ---------------------------------------------------------------------------
# Per-fill PnL helper (used by engine and BacktestResult)
# ---------------------------------------------------------------------------


def compute_fill_pnl(fill: Fill, market_vwap: Decimal) -> Decimal:
    """Compute the VWAP-relative PnL for a single fill.

    For buys: ``(market_vwap - fill.price) * fill.volume`` — positive when
    the algo bought below VWAP.

    For sells: ``(fill.price - market_vwap) * fill.volume`` — positive when
    the algo sold above VWAP.

    Args:
        fill: The fill to evaluate.
        market_vwap: Volume-weighted average clearing price across all products
            in the backtest period (the benchmark).

    Returns:
        PnL in EUR, positive meaning the fill benefited the algo vs VWAP.
    """
    if fill.side == Side.BUY:
        return (market_vwap - fill.price) * fill.volume
    return (fill.price - market_vwap) * fill.volume


# ---------------------------------------------------------------------------
# Metric computation functions
# ---------------------------------------------------------------------------


def compute_sharpe(equity_snapshots: list[EquitySnapshot]) -> Decimal | None:
    """Compute the annualised Sharpe ratio from an equity curve.

    Uses percentage returns between consecutive equity snapshots.
    Annualisation factor is derived from the actual average observation
    frequency using ``sqrt(252 * periods_per_day)``.

    Args:
        equity_snapshots: Chronological list of equity snapshots.

    Returns:
        Annualised Sharpe ratio, or ``None`` if fewer than two snapshots
        (impossible to compute a standard deviation).
    """
    if len(equity_snapshots) < 2:
        return None

    equities = np.array([float(s.total_equity) for s in equity_snapshots])
    returns = np.diff(equities) / equities[:-1]

    if len(returns) < 1:
        return None

    std_ret = float(np.std(returns, ddof=1))
    if std_ret == 0:
        return None

    mean_ret = float(np.mean(returns))

    timestamps = [s.timestamp for s in equity_snapshots]
    total_seconds = (timestamps[-1] - timestamps[0]).total_seconds()
    total_days = max(total_seconds / 86400.0, 1.0)
    n_obs = len(returns)
    periods_per_day = n_obs / total_days
    annualisation = math.sqrt(252.0 * periods_per_day)

    sharpe = (mean_ret / std_ret) * annualisation
    return Decimal(str(round(sharpe, 6)))


def compute_max_drawdown(
    equity_snapshots: list[EquitySnapshot],
) -> tuple[Decimal, Decimal]:
    """Compute maximum drawdown (absolute EUR and as fraction of peak equity).

    Args:
        equity_snapshots: Chronological list of equity snapshots.

    Returns:
        Tuple of ``(max_drawdown_eur, max_drawdown_fraction)``.
        The fraction is ``max_drawdown / peak_equity_at_that_point``.
        Both are zero for an empty or monotonically increasing equity curve.
    """
    if not equity_snapshots:
        return Decimal("0"), Decimal("0")

    peak = equity_snapshots[0].total_equity
    max_dd = Decimal("0")
    max_dd_pct = Decimal("0")

    for snap in equity_snapshots:
        if snap.total_equity > peak:
            peak = snap.total_equity
        dd = peak - snap.total_equity
        if dd > max_dd:
            max_dd = dd
            if peak > 0:
                max_dd_pct = dd / peak

    return max_dd, max_dd_pct


def compute_profit_factor(fills: list[Fill], market_vwap: Decimal) -> Decimal | None:
    """Compute profit factor: sum of gains / sum of losses.

    Args:
        fills: All fills from the backtest.
        market_vwap: Market VWAP used as the PnL benchmark.

    Returns:
        Profit factor as a Decimal, or ``None`` if there are no losing fills
        (undefined / infinite profit factor).
    """
    fill_pnls = [compute_fill_pnl(f, market_vwap) for f in fills]
    gains = sum((p for p in fill_pnls if p > 0), Decimal("0"))
    losses = sum((abs(p) for p in fill_pnls if p < 0), Decimal("0"))
    if losses == 0:
        return None
    return gains / losses


# ---------------------------------------------------------------------------
# BacktestResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BacktestResult:
    """Complete result of a backtest run.

    Holds all fills, PnL metrics, the equity curve, and advanced statistics.
    Provides export methods for HTML reports, JSON, and Parquet.

    Attributes:
        algo_name: Class name of the algo that was run.
        exchange: Exchange identifier, e.g. ``"nordpool"``.
        start: First delivery date included in the backtest.
        end: Last delivery date included in the backtest.
        fills: All fills produced during the run, in chronological order.
        pnl: Aggregated PnL and VWAP-relative performance metrics.
        equity_curve: Equity snapshots, one per auction day with activity.
        initial_capital: Starting capital in EUR.
        sharpe_ratio: Annualised Sharpe ratio, or ``None`` if < 2 snapshots.
        max_drawdown: Largest peak-to-trough drawdown in EUR.
        max_drawdown_pct: Max drawdown as a fraction of peak equity.
        profit_factor: Sum of gains / sum of losses, or ``None`` if no losses.
        avg_trade_pnl: Mean per-fill PnL in EUR.
        best_trade: Fill with the highest individual PnL, or ``None``.
        worst_trade: Fill with the lowest individual PnL, or ``None``.
        duration_days: Number of calendar days covered (end - start + 1).
    """

    # --- existing fields ---
    algo_name: str
    exchange: str
    start: date
    end: date
    fills: tuple[Fill, ...]
    pnl: PnlSummary

    # --- new in spec-03 ---
    equity_curve: tuple[EquitySnapshot, ...] = field(default_factory=tuple)
    initial_capital: Decimal = Decimal("0")
    sharpe_ratio: Decimal | None = None
    max_drawdown: Decimal = Decimal("0")
    max_drawdown_pct: Decimal = Decimal("0")
    profit_factor: Decimal | None = None
    avg_trade_pnl: Decimal = Decimal("0")
    best_trade: Fill | None = None
    worst_trade: Fill | None = None
    duration_days: int = 0

    # --- new in spec-05: gate closure NOP tracking ---
    gate_closure_positions: tuple[GateClosureSnapshot, ...] = field(default_factory=tuple)
    avg_gate_closure_nop_mw: Decimal = Decimal("0")
    max_gate_closure_nop_mw: Decimal = Decimal("0")

    # ------------------------------------------------------------------
    # Text summary
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Produce a human-readable text summary of the backtest result.

        Returns:
            Formatted multi-line string suitable for printing to stdout.
        """
        p = self.pnl
        sep = "=" * 62
        total_pnl = p.total_alpha_eur
        final_equity = self.initial_capital + total_pnl

        lines: list[str] = [
            sep,
            f"  Backtest Results: {self.start}  to  {self.end}  ({self.duration_days} days)",
            f"  Exchange: {self.exchange} | Algo: {self.algo_name}",
            sep,
            "",
            f"  Total PnL:        {total_pnl:>+14,.2f} EUR",
            f"  Market VWAP:      {p.market_vwap:>14.2f} EUR/MWh",
        ]

        if self.sharpe_ratio is not None:
            lines.append(f"  Sharpe Ratio:     {float(self.sharpe_ratio):>14.2f}")
        else:
            lines.append(f"  Sharpe Ratio:     {'N/A':>14}")

        dd_pct = float(self.max_drawdown_pct) * 100
        lines.append(
            f"  Max Drawdown:    {-float(self.max_drawdown):>+14,.2f} EUR  ({-dd_pct:.1f}%)"
        )

        if self.profit_factor is not None:
            lines.append(f"  Profit Factor:    {float(self.profit_factor):>14.2f}")
        else:
            lines.append(f"  Profit Factor:    {'N/A (no losses)':>14}")

        total_volume = sum((f.volume for f in self.fills), Decimal("0"))
        win_rate = (
            (p.buys.win_rate * p.buys.count + p.sells.win_rate * p.sells.count) / len(self.fills)
            if self.fills
            else 0.0
        )

        lines += [
            "",
            f"  Trades:           {len(self.fills):>14,d}",
            f"  Win Rate:         {win_rate:>14.1%}",
            f"  Avg Trade PnL:    {float(self.avg_trade_pnl):>+14,.2f} EUR",
        ]

        if self.best_trade is not None:
            bt_pnl = compute_fill_pnl(self.best_trade, p.market_vwap)
            ts = self.best_trade.timestamp.strftime("%Y-%m-%d %H:%M")
            lines.append(
                f"  Best Trade:      {float(bt_pnl):>+14,.2f} EUR  "
                f"({self.best_trade.product_id} {ts})"
            )

        if self.worst_trade is not None:
            wt_pnl = compute_fill_pnl(self.worst_trade, p.market_vwap)
            ts = self.worst_trade.timestamp.strftime("%Y-%m-%d %H:%M")
            lines.append(
                f"  Worst Trade:     {float(wt_pnl):>+14,.2f} EUR  "
                f"({self.worst_trade.product_id} {ts})"
            )

        if self.gate_closure_positions:
            lines += [
                "",
                "  Gate Closure NOP:",
                f"    Products closed:  {len(self.gate_closure_positions):>6d}",
                f"    Avg NOP at close: {float(self.avg_gate_closure_nop_mw):>10.2f} MW",
                f"    Max NOP at close: {float(self.max_gate_closure_nop_mw):>10.2f} MW",
            ]

        lines += [
            "",
            f"  Total Volume:     {float(total_volume):>14,.1f} MW",
            f"  Initial Capital:  {float(self.initial_capital):>14,.2f} EUR",
            f"  Final Equity:     {float(final_equity):>14,.2f} EUR",
            "",
        ]

        if p.buys.count > 0:
            sign = "-" if p.buys.vwap_alpha >= 0 else "+"
            alpha_str = f"{sign}{abs(p.buys.vwap_alpha):.2f}"
            note = "(bought above VWAP)" if p.buys.vwap_alpha >= 0 else "(bought below VWAP ✓)"
            lines += [
                "  Buys",
                f"    Fills:      {p.buys.count:>6d}",
                f"    Volume:     {p.buys.volume_mwh:>10.1f} MWh",
                f"    Avg Price:  {p.buys.avg_price:>10.2f} EUR/MWh",
                f"    vs VWAP:    {alpha_str:>10} EUR/MWh  {note}",
                f"    EUR Alpha:  {p.buys.total_alpha_eur:>+10.2f} EUR",
                f"    Win Rate:   {p.buys.win_rate:>9.1%}",
                "",
            ]

        if p.sells.count > 0:
            sign = "+" if p.sells.vwap_alpha >= 0 else "-"
            alpha_str = f"{sign}{abs(p.sells.vwap_alpha):.2f}"
            note = "(sold above VWAP ✓)" if p.sells.vwap_alpha <= 0 else "(sold below VWAP)"
            lines += [
                "  Sells",
                f"    Fills:      {p.sells.count:>6d}",
                f"    Volume:     {p.sells.volume_mwh:>10.1f} MWh",
                f"    Avg Price:  {p.sells.avg_price:>10.2f} EUR/MWh",
                f"    vs VWAP:    {alpha_str:>10} EUR/MWh  {note}",
                f"    EUR Alpha:  {p.sells.total_alpha_eur:>+10.2f} EUR",
                f"    Win Rate:   {p.sells.win_rate:>9.1%}",
                "",
            ]

        if p.buys.count == 0 and p.sells.count == 0:
            lines.append("  No fills recorded.")
            lines.append("")

        lines += [
            f"  Total Alpha:  {p.total_alpha_eur:>+10.2f} EUR",
            f"  Total Fills:  {len(self.fills):>6d}",
            sep,
        ]

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Daily PnL aggregation
    # ------------------------------------------------------------------

    def daily_pnl(self) -> list[DailyPnL]:
        """Aggregate fills into per-calendar-day performance buckets.

        Returns:
            List of :class:`DailyPnL` objects sorted by date, one per day
            that had at least one fill.
        """
        market_vwap = self.pnl.market_vwap
        daily_fills: dict[date, list[Fill]] = defaultdict(list)
        for fill in self.fills:
            daily_fills[fill.timestamp.date()].append(fill)

        result: list[DailyPnL] = []
        for day, day_fills in sorted(daily_fills.items()):
            pnl_eur = sum((compute_fill_pnl(f, market_vwap) for f in day_fills), Decimal("0"))
            volume = sum((f.volume for f in day_fills), Decimal("0"))
            vwap_edge = pnl_eur / volume if volume > 0 else Decimal("0")
            result.append(
                DailyPnL(
                    date=day,
                    pnl=pnl_eur,
                    volume_mw=volume,
                    trade_count=len(day_fills),
                    vwap_edge=vwap_edge,
                )
            )
        return result

    # ------------------------------------------------------------------
    # DataFrame methods
    # ------------------------------------------------------------------

    def equity_curve_df(self) -> pd.DataFrame:
        """Return the equity curve as a pandas DataFrame.

        Columns: ``timestamp``, ``realised_pnl``, ``unrealised_pnl``,
        ``total_equity``, ``cash``, ``net_position_mw``.

        Returns:
            DataFrame with one row per equity snapshot.

        Raises:
            ImportError: If pandas is not installed.
        """
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                "pandas is required for DataFrame output. "
                "Install it with: pip install nexa-backtest[pandas]"
            ) from exc

        rows = [
            {
                "timestamp": s.timestamp,
                "realised_pnl": float(s.realised_pnl),
                "unrealised_pnl": float(s.unrealised_pnl),
                "total_equity": float(s.total_equity),
                "cash": float(s.cash),
                "net_position_mw": float(s.net_position_mw),
            }
            for s in self.equity_curve
        ]
        return pd.DataFrame(rows)

    def trades_df(self) -> pd.DataFrame:
        """Return all fills as a pandas DataFrame.

        Columns: ``timestamp``, ``product_id``, ``side``, ``price``,
        ``volume``, ``order_id``.

        Returns:
            DataFrame with one row per fill.

        Raises:
            ImportError: If pandas is not installed.
        """
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                "pandas is required for DataFrame output. "
                "Install it with: pip install nexa-backtest[pandas]"
            ) from exc

        rows = [
            {
                "timestamp": f.timestamp,
                "product_id": f.product_id,
                "side": str(f.side),
                "price": float(f.price),
                "volume": float(f.volume),
                "order_id": f.order_id,
            }
            for f in self.fills
        ]
        return pd.DataFrame(rows)

    def daily_pnl_df(self) -> pd.DataFrame:
        """Return the daily PnL breakdown as a pandas DataFrame.

        Columns: ``date``, ``pnl``, ``volume_mw``, ``trade_count``,
        ``vwap_edge``.

        Returns:
            DataFrame with one row per trading day.

        Raises:
            ImportError: If pandas is not installed.
        """
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                "pandas is required for DataFrame output. "
                "Install it with: pip install nexa-backtest[pandas]"
            ) from exc

        rows = [
            {
                "date": d.date,
                "pnl": float(d.pnl),
                "volume_mw": float(d.volume_mw),
                "trade_count": d.trade_count,
                "vwap_edge": float(d.vwap_edge),
            }
            for d in self.daily_pnl()
        ]
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Export methods
    # ------------------------------------------------------------------

    def to_html(self, path: str) -> None:
        """Generate a self-contained HTML report and write it to ``path``.

        The report includes interactive Plotly charts (equity curve, drawdown,
        daily PnL, VWAP edge, trade distribution) and summary tables, styled
        with the Phase Nexa colour palette.

        Args:
            path: File path to write the HTML report to.
        """
        from nexa_backtest.analysis.report import generate_html_report

        html = generate_html_report(self)
        Path(path).write_text(html, encoding="utf-8")

    def to_json(self, path: str) -> None:
        """Serialise the backtest result to a JSON file.

        Decimal values are written as strings to preserve precision.
        The JSON contains top-level metrics, an ``equity_curve`` array,
        and a ``trades`` array.

        Args:
            path: File path to write the JSON to.
        """

        def _fill_to_dict(f: Fill) -> dict[str, str]:
            return {
                "order_id": f.order_id,
                "product_id": f.product_id,
                "side": str(f.side),
                "price": str(f.price),
                "volume": str(f.volume),
                "timestamp": f.timestamp.isoformat(),
            }

        def _snap_to_dict(s: EquitySnapshot) -> dict[str, str]:
            return {
                "timestamp": s.timestamp.isoformat(),
                "realised_pnl": str(s.realised_pnl),
                "unrealised_pnl": str(s.unrealised_pnl),
                "total_equity": str(s.total_equity),
                "cash": str(s.cash),
                "net_position_mw": str(s.net_position_mw),
            }

        data: dict[str, object] = {
            "algo_name": self.algo_name,
            "exchange": self.exchange,
            "start_date": self.start.isoformat(),
            "end_date": self.end.isoformat(),
            "duration_days": self.duration_days,
            "initial_capital": str(self.initial_capital),
            "total_pnl": str(self.pnl.total_alpha_eur),
            "market_vwap": str(self.pnl.market_vwap),
            "sharpe_ratio": str(self.sharpe_ratio) if self.sharpe_ratio is not None else None,
            "max_drawdown": str(self.max_drawdown),
            "max_drawdown_pct": str(self.max_drawdown_pct),
            "profit_factor": str(self.profit_factor) if self.profit_factor is not None else None,
            "avg_trade_pnl": str(self.avg_trade_pnl),
            "trade_count": len(self.fills),
            "equity_curve": [_snap_to_dict(s) for s in self.equity_curve],
            "trades": [_fill_to_dict(f) for f in self.fills],
        }
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")

    def to_parquet(self, path: str) -> None:
        """Write equity curve, trades, and daily PnL as Parquet files.

        Creates the following structure under ``path``::

            <path>/
              metadata.json
              equity_curve.parquet
              trades.parquet
              daily_pnl.parquet

        Args:
            path: Output directory path (created if it does not exist).
        """
        import pyarrow as pa
        import pyarrow.parquet as pq

        output_dir = Path(path)
        output_dir.mkdir(parents=True, exist_ok=True)

        metadata: dict[str, object] = {
            "algo_name": self.algo_name,
            "exchange": self.exchange,
            "start_date": self.start.isoformat(),
            "end_date": self.end.isoformat(),
            "duration_days": self.duration_days,
            "initial_capital": str(self.initial_capital),
            "total_pnl": str(self.pnl.total_alpha_eur),
            "market_vwap": str(self.pnl.market_vwap),
            "sharpe_ratio": str(self.sharpe_ratio) if self.sharpe_ratio is not None else None,
            "max_drawdown": str(self.max_drawdown),
            "max_drawdown_pct": str(self.max_drawdown_pct),
            "profit_factor": str(self.profit_factor) if self.profit_factor is not None else None,
            "avg_trade_pnl": str(self.avg_trade_pnl),
            "trade_count": len(self.fills),
        }
        (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        if self.equity_curve:
            eq_table = pa.table(
                {
                    "timestamp": pa.array(
                        [s.timestamp for s in self.equity_curve],
                        type=pa.timestamp("us", tz="UTC"),
                    ),
                    "realised_pnl": [float(s.realised_pnl) for s in self.equity_curve],
                    "unrealised_pnl": [float(s.unrealised_pnl) for s in self.equity_curve],
                    "total_equity": [float(s.total_equity) for s in self.equity_curve],
                    "cash": [float(s.cash) for s in self.equity_curve],
                    "net_position_mw": [float(s.net_position_mw) for s in self.equity_curve],
                }
            )
            pq.write_table(eq_table, output_dir / "equity_curve.parquet")

        if self.fills:
            trades_table = pa.table(
                {
                    "order_id": [f.order_id for f in self.fills],
                    "product_id": [f.product_id for f in self.fills],
                    "side": [str(f.side) for f in self.fills],
                    "price": [float(f.price) for f in self.fills],
                    "volume": [float(f.volume) for f in self.fills],
                    "timestamp": pa.array(
                        [f.timestamp for f in self.fills],
                        type=pa.timestamp("us", tz="UTC"),
                    ),
                }
            )
            pq.write_table(trades_table, output_dir / "trades.parquet")

        daily = self.daily_pnl()
        if daily:
            daily_table = pa.table(
                {
                    "date": pa.array([d.date for d in daily], type=pa.date32()),
                    "pnl": [float(d.pnl) for d in daily],
                    "volume_mw": [float(d.volume_mw) for d in daily],
                    "trade_count": [d.trade_count for d in daily],
                    "vwap_edge": [float(d.vwap_edge) for d in daily],
                }
            )
            pq.write_table(daily_table, output_dir / "daily_pnl.parquet")
