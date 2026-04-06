"""BacktestResult and summary formatting.

:class:`BacktestResult` holds the complete output of a backtest run including
all fills and PnL metrics, and can produce a human-readable text summary.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from nexa_backtest.analysis.pnl import PnlSummary
from nexa_backtest.types import Fill


@dataclass(frozen=True)
class BacktestResult:
    """Complete result of a backtest run.

    Attributes:
        algo_name: Class name of the algo that was run.
        exchange: Exchange identifier, e.g. ``"nordpool"``.
        start: First delivery date included in the backtest.
        end: Last delivery date included in the backtest.
        fills: All fills produced during the run, in chronological order.
        pnl: Aggregated PnL and VWAP-relative performance metrics.
    """

    algo_name: str
    exchange: str
    start: date
    end: date
    fills: tuple[Fill, ...]
    pnl: PnlSummary

    def summary(self) -> str:
        """Produce a human-readable text summary of the backtest result.

        Returns:
            Formatted multi-line string suitable for printing to stdout.
        """
        p = self.pnl
        sep = "=" * 62

        lines = [
            sep,
            "  nexa-backtest Result",
            sep,
            f"  Algo:     {self.algo_name}",
            f"  Exchange: {self.exchange}",
            f"  Period:   {self.start}  →  {self.end}",
            "",
            f"  Market VWAP:  {p.market_vwap:>10.2f} EUR/MWh",
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
