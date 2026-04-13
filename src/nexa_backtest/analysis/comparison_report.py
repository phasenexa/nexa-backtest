"""HTML report generation for multi-algo comparison results.

Generates a self-contained HTML file with embedded Plotly charts and
metrics comparison table, styled with the Phase Nexa colour palette.
The output file has no external dependencies and can be opened directly
in a browser.
"""

from __future__ import annotations

import json
from decimal import Decimal
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nexa_backtest.engines.shared import ComparisonResult

# ---------------------------------------------------------------------------
# Phase Nexa colour palette
# ---------------------------------------------------------------------------

_QUANTUM_BLUE = "#0E2A47"
_NEBULA = "#122F4D"
_TEXT = "#E8EDF2"
_TEXT_DIM = "#8A9BB0"
_VIOLET = "#8A6CFF"
_CORAL = "#FF6B6B"
_MINT = "#4DFFC3"

# Algo accent colours (same order as ALGO_COLOURS in shared.py)
_ALGO_COLOURS = [
    "#00E5FF",
    "#8A6CFF",
    "#FFD700",
    "#4DFFC3",
    "#FF6B6B",
    "#A8FF78",
    "#FF8C00",
    "#C0C0C0",
]


def _colour(index: int) -> str:
    """Return the accent colour for algo at position ``index`` (cycles)."""
    return _ALGO_COLOURS[index % len(_ALGO_COLOURS)]


# ---------------------------------------------------------------------------
# JSON serialisation helper
# ---------------------------------------------------------------------------


def _json(obj: Any) -> str:
    """Serialise Python objects to a JSON string safe to embed in HTML."""

    def _default(o: Any) -> Any:
        if isinstance(o, Decimal):
            return float(o)
        raise TypeError(f"Object of type {type(o).__name__} is not JSON serialisable")

    return json.dumps(obj, default=_default)


# ---------------------------------------------------------------------------
# Chart data builders
# ---------------------------------------------------------------------------


def _overlaid_equity_traces(comparison: ComparisonResult) -> str:
    """Return Plotly trace definitions for overlaid equity curves."""
    traces = []
    for i, (name, result) in enumerate(comparison.results.items()):
        snaps = result.equity_curve
        if not snaps:
            continue
        xs = [s.timestamp.isoformat() for s in snaps]
        ys = [float(s.total_equity) for s in snaps]
        colour = _colour(i)
        traces.append(
            {
                "type": "scatter",
                "x": xs,
                "y": ys,
                "mode": "lines",
                "name": name,
                "line": {"color": colour, "width": 2},
            }
        )
    return _json(traces)


def _overlaid_drawdown_traces(comparison: ComparisonResult) -> str:
    """Return Plotly trace definitions for the drawdown comparison chart."""
    traces = []
    for i, (name, result) in enumerate(comparison.results.items()):
        snaps = result.equity_curve
        if not snaps:
            continue
        xs = [s.timestamp.isoformat() for s in snaps]
        peak = float(snaps[0].total_equity)
        drawdowns = []
        for s in snaps:
            eq = float(s.total_equity)
            if eq > peak:
                peak = eq
            drawdowns.append(eq - peak)
        colour = _colour(i)
        traces.append(
            {
                "type": "scatter",
                "x": xs,
                "y": drawdowns,
                "mode": "lines",
                "name": name,
                "line": {"color": colour, "width": 1},
            }
        )
    return _json(traces)


def _daily_pnl_grouped_traces(comparison: ComparisonResult) -> str:
    """Return Plotly trace definitions for grouped daily PnL bar chart."""
    traces = []
    for i, (name, result) in enumerate(comparison.results.items()):
        daily = result.daily_pnl()
        if not daily:
            continue
        xs = [d.date.isoformat() for d in daily]
        ys = [float(d.pnl) for d in daily]
        colour = _colour(i)
        traces.append(
            {
                "type": "bar",
                "x": xs,
                "y": ys,
                "name": name,
                "marker": {"color": colour},
            }
        )
    return _json(traces)


def _cumulative_vwap_edge_traces(comparison: ComparisonResult) -> str:
    """Return Plotly trace definitions for cumulative VWAP edge."""
    traces = []
    for i, (name, result) in enumerate(comparison.results.items()):
        daily = result.daily_pnl()
        if not daily:
            continue
        xs: list[str] = []
        cumulative: list[float] = []
        running = 0.0
        for d in daily:
            xs.append(d.date.isoformat())
            running += float(d.vwap_edge)
            cumulative.append(running)
        colour = _colour(i)
        traces.append(
            {
                "type": "scatter",
                "x": xs,
                "y": cumulative,
                "mode": "lines",
                "name": name,
                "line": {"color": colour, "width": 2},
            }
        )
    return _json(traces)


# ---------------------------------------------------------------------------
# Table builders
# ---------------------------------------------------------------------------


def _metrics_table_html(comparison: ComparisonResult) -> str:
    """Return HTML for the metrics comparison table with best-value highlighting."""

    names = list(comparison.results.keys())

    # Collect per-metric values for highlighting.
    def _get(name: str) -> list[tuple[str, float]]:
        out = []
        for algo_name, result in comparison.results.items():
            val = float(comparison._metric_value(name, result))
            out.append((algo_name, val))
        return out

    metrics: list[tuple[str, str, str, bool]] = [
        # (metric_key, display_label, format_str, higher_is_better)
        ("total_pnl", "Total PnL (EUR)", "+,.2f", True),
        ("sharpe_ratio", "Sharpe Ratio", ".2f", True),
        ("max_drawdown", "Max Drawdown (EUR)", "+,.2f", True),
        ("profit_factor", "Profit Factor", ".2f", True),
        ("win_rate", "Win Rate", ".1%", True),
        ("trades", "Trade Count", "d", True),
    ]

    header_cells = "".join(f"<th>{n}</th>" for n in names)
    header = f"<tr><th>Metric</th>{header_cells}</tr>"

    rows_html = ""
    for metric_key, label, fmt, higher_better in metrics:
        vals = _get(metric_key)
        best_val = max(v for _, v in vals) if higher_better else min(v for _, v in vals)

        cells = ""
        for _, v in vals:
            is_best = abs(v - best_val) < 1e-9
            cls = " class='best'" if is_best else ""
            try:
                display = format(v, fmt)
            except (ValueError, TypeError):
                display = str(v)
            cells += f"<td{cls}>{display}</td>"
        rows_html += f"<tr><td class='label'>{label}</td>{cells}</tr>"

    return f"<table class='metrics-table'><thead>{header}</thead><tbody>{rows_html}</tbody></table>"


def _top_trades_html(comparison: ComparisonResult) -> str:
    """Return collapsible per-algo trade summary HTML sections."""
    from nexa_backtest.analysis.metrics import compute_fill_pnl

    sections = ""
    for i, (name, result) in enumerate(comparison.results.items()):
        colour = _colour(i)
        if not result.fills:
            body = "<p style='color:#8A9BB0;padding:10px'>No fills.</p>"
        else:
            market_vwap = result.pnl.market_vwap
            pvwaps = result.product_vwaps or None
            fill_pnls = [(f, compute_fill_pnl(f, market_vwap, pvwaps)) for f in result.fills]
            best10 = sorted(fill_pnls, key=lambda x: x[1], reverse=True)[:10]
            worst10 = sorted(fill_pnls, key=lambda x: x[1])[:10]

            def _table(title: str, rows: list[tuple[Any, Decimal]], _c: str = colour) -> str:
                header = (
                    "<tr><th>Time</th><th>Product</th><th>Side</th>"
                    "<th>Price</th><th>Volume</th><th>PnL</th></tr>"
                )
                body_rows = "".join(
                    f"<tr>"
                    f"<td>{f.timestamp.strftime('%Y-%m-%d %H:%M')}</td>"
                    f"<td>{f.product_id}</td>"
                    f"<td class='{'buy' if str(f.side) == 'BUY' else 'sell'}'>{f.side}</td>"
                    f"<td>{float(f.price):.2f}</td>"
                    f"<td>{float(f.volume):.1f}</td>"
                    f"<td class='{'pos' if pnl >= 0 else 'neg'}'>{float(pnl):+,.2f}</td>"
                    f"</tr>"
                    for f, pnl in rows
                )
                return (
                    f"<h4 style='color:{_c};margin:12px 0 6px'>{title}</h4>"
                    f"<table>{header}{body_rows}</table>"
                )

            body = _table("Top 10 Best Trades", best10) + _table("Top 10 Worst Trades", worst10)

        sections += f"""
        <details style='margin-bottom:12px;border:1px solid {colour}33;border-radius:6px;overflow:hidden'>
          <summary style='background:{_NEBULA};padding:12px 16px;cursor:pointer;
                          font-weight:600;color:{colour};list-style:none'>
            ▸ {name}
          </summary>
          <div style='padding:16px;background:{_QUANTUM_BLUE}'>
            {body}
          </div>
        </details>
        """
    return sections


# ---------------------------------------------------------------------------
# HTML assembly
# ---------------------------------------------------------------------------

_PLOTLY_LAYOUT = {
    "paper_bgcolor": _NEBULA,
    "plot_bgcolor": _NEBULA,
    "font": {"color": _TEXT, "family": "'JetBrains Mono', 'Courier New', monospace"},
    "xaxis": {"gridcolor": "#1E3A5F", "linecolor": "#1E3A5F"},
    "yaxis": {"gridcolor": "#1E3A5F", "linecolor": "#1E3A5F"},
    "margin": {"l": 50, "r": 20, "t": 40, "b": 40},
    "legend": {"bgcolor": "rgba(0,0,0,0)"},
    "barmode": "group",
}


def generate_comparison_html_report(comparison: ComparisonResult) -> str:
    """Generate a self-contained HTML comparison report.

    Contains overlaid equity curves, drawdown comparison, daily PnL grouped
    bar chart, cumulative VWAP edge chart, metrics comparison table (with
    best-value highlighting), and per-algo trade summaries.

    Args:
        comparison: The completed comparison result to report on.

    Returns:
        Complete HTML document as a string.
    """
    eq_traces = _overlaid_equity_traces(comparison)
    dd_traces = _overlaid_drawdown_traces(comparison)
    daily_traces = _daily_pnl_grouped_traces(comparison)
    vwap_traces = _cumulative_vwap_edge_traces(comparison)
    layout_json = _json(_PLOTLY_LAYOUT)

    metrics_table = _metrics_table_html(comparison)
    trade_sections = _top_trades_html(comparison)

    duration = (comparison.end_date - comparison.start_date).days + 1
    products_str = ", ".join(comparison.products)
    title = (
        f"Comparison Report — {comparison.exchange} | "
        f"{comparison.start_date} to {comparison.end_date}"
    )

    # Memory efficiency note
    if comparison.peak_memory_bytes > 0:
        saved = comparison.estimated_separate_memory_bytes - comparison.peak_memory_bytes
        mem_note = (
            f"Memory: {comparison.peak_memory_bytes / 1e6:.0f} MB "
            f"(saved ~{saved / 1e6:.0f} MB vs separate backtests)"
        )
    else:
        mem_note = ""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{title}</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin/>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet"/>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js" charset="utf-8"></script>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: {_QUANTUM_BLUE}; color: {_TEXT};
          font-family: 'Inter', sans-serif; font-size: 14px; line-height: 1.5; }}
  header {{ background: {_NEBULA}; border-bottom: 2px solid {_VIOLET}; padding: 20px 32px; }}
  header h1 {{ font-size: 20px; font-weight: 700; color: #00E5FF; letter-spacing: 0.03em; }}
  header p {{ color: {_TEXT_DIM}; font-size: 12px; margin-top: 4px;
              font-family: 'JetBrains Mono', monospace; }}
  .container {{ max-width: 1400px; margin: 0 auto; padding: 24px 32px; }}
  .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px; }}
  .card {{ background: {_NEBULA}; border: 1px solid #1E3A5F; border-radius: 8px;
           padding: 20px; overflow: hidden; }}
  .card h3 {{ font-size: 11px; font-weight: 600; text-transform: uppercase;
              letter-spacing: 0.1em; color: {_VIOLET}; margin-bottom: 16px; }}
  .chart-container {{ width: 100%; height: 300px; }}
  .chart-full {{ width: 100%; height: 360px; }}
  table {{ width: 100%; border-collapse: collapse; font-family: 'JetBrains Mono', monospace;
           font-size: 12px; }}
  th {{ color: {_TEXT_DIM}; font-weight: 600; text-align: left; padding: 6px 10px;
        border-bottom: 1px solid #1E3A5F; white-space: nowrap; }}
  td {{ padding: 5px 10px; border-bottom: 1px solid #0E2A47; }}
  td.label {{ color: {_TEXT_DIM}; }}
  td.best {{ color: #4DFFC3; font-weight: 700; }}
  td.pos {{ color: #4DFFC3; }}
  td.neg {{ color: {_CORAL}; }}
  td.buy {{ color: #00E5FF; }}
  td.sell {{ color: {_CORAL}; }}
  .metrics-table {{ margin-top: 8px; }}
  summary::-webkit-details-marker {{ display: none; }}
</style>
</head>
<body>
<header>
  <h1>Comparison Report</h1>
  <p>{comparison.exchange.upper()} | {products_str} | {comparison.start_date} to {comparison.end_date} ({duration} days)</p>
  {f'<p style="color:#8A9BB0;font-size:11px;margin-top:4px">{mem_note}</p>' if mem_note else ""}
</header>
<div class="container">

  <div class="card" style="margin-bottom:20px">
    <h3>Overlaid Equity Curves</h3>
    <div id="chart-equity" class="chart-full"></div>
  </div>

  <div class="grid-2">
    <div class="card">
      <h3>Drawdown Comparison</h3>
      <div id="chart-dd" class="chart-container"></div>
    </div>
    <div class="card">
      <h3>Daily PnL Comparison</h3>
      <div id="chart-daily" class="chart-container"></div>
    </div>
  </div>

  <div class="card" style="margin-bottom:20px">
    <h3>Cumulative VWAP Edge</h3>
    <div id="chart-vwap" class="chart-container"></div>
  </div>

  <div class="card" style="margin-bottom:20px">
    <h3>Metrics Comparison</h3>
    {metrics_table}
  </div>

  <div class="card">
    <h3>Per-Algo Trade Summaries</h3>
    {trade_sections}
  </div>

</div>
<script>
var layout = {layout_json};
Plotly.newPlot('chart-equity', {eq_traces}, Object.assign({{title:'Equity Curves'}}, layout));
Plotly.newPlot('chart-dd', {dd_traces}, Object.assign({{title:'Drawdown'}}, layout));
Plotly.newPlot('chart-daily', {daily_traces}, Object.assign({{title:'Daily PnL'}}, layout));
Plotly.newPlot('chart-vwap', {vwap_traces}, Object.assign({{title:'Cumulative VWAP Edge'}}, layout));
</script>
</body>
</html>"""

    return html
