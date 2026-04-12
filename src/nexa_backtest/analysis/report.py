"""HTML report generation for backtest results.

Generates a self-contained HTML file with embedded Plotly charts and
summary tables, styled with the Phase Nexa colour palette.  The output
file has no external dependencies and can be opened directly in a browser.
"""

from __future__ import annotations

import json
from decimal import Decimal
from typing import Any

from nexa_backtest.analysis.metrics import BacktestResult

# ---------------------------------------------------------------------------
# Phase Nexa colour palette
# ---------------------------------------------------------------------------

_QUANTUM_BLUE = "#0E2A47"
_NEBULA = "#122F4D"
_MINT = "#4DFFC3"
_CORAL = "#FF6B6B"
_CYAN = "#00E5FF"
_VIOLET = "#8A6CFF"
_GOLD = "#FFD700"
_TEXT = "#E8EDF2"
_TEXT_DIM = "#8A9BB0"

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


def _equity_curve_traces(result: BacktestResult) -> str:
    """Return Plotly trace definitions for the equity curve chart."""
    snaps = result.equity_curve
    if not snaps:
        return "[]"

    xs = [s.timestamp.isoformat() for s in snaps]
    ys = [float(s.total_equity) for s in snaps]
    initial = float(result.initial_capital)

    traces = [
        {
            "type": "scatter",
            "x": xs,
            "y": ys,
            "mode": "lines",
            "name": "Equity",
            "line": {"color": _CYAN, "width": 2},
        },
        {
            "type": "scatter",
            "x": [xs[0], xs[-1]],
            "y": [initial, initial],
            "mode": "lines",
            "name": "Initial Capital",
            "line": {"color": _VIOLET, "width": 1, "dash": "dash"},
        },
    ]
    return _json(traces)


def _drawdown_traces(result: BacktestResult) -> str:
    """Return Plotly trace definitions for the drawdown chart."""
    snaps = result.equity_curve
    if not snaps:
        return "[]"

    xs = [s.timestamp.isoformat() for s in snaps]
    peak = float(snaps[0].total_equity)
    drawdowns = []
    for s in snaps:
        eq = float(s.total_equity)
        if eq > peak:
            peak = eq
        drawdowns.append(eq - peak)  # negative or zero

    traces = [
        {
            "type": "scatter",
            "x": xs,
            "y": drawdowns,
            "mode": "lines",
            "fill": "tozeroy",
            "name": "Drawdown",
            "line": {"color": _CORAL, "width": 1},
            "fillcolor": "rgba(255,107,107,0.3)",
        }
    ]
    return _json(traces)


def _daily_pnl_traces(result: BacktestResult) -> str:
    """Return Plotly trace definitions for the daily PnL bar chart."""
    daily = result.daily_pnl()
    if not daily:
        return "[]"

    xs = [d.date.isoformat() for d in daily]
    ys = [float(d.pnl) for d in daily]
    colours = [_MINT if y >= 0 else _CORAL for y in ys]

    traces = [
        {
            "type": "bar",
            "x": xs,
            "y": ys,
            "name": "Daily PnL",
            "marker": {"color": colours},
        }
    ]
    return _json(traces)


def _vwap_edge_traces(result: BacktestResult) -> str:
    """Return Plotly trace definitions for the cumulative VWAP edge chart."""
    daily = result.daily_pnl()
    if not daily:
        return "[]"

    xs: list[str] = []
    cumulative: list[float] = []
    running = 0.0
    for d in daily:
        xs.append(d.date.isoformat())
        running += float(d.vwap_edge)
        cumulative.append(running)

    traces = [
        {
            "type": "scatter",
            "x": xs,
            "y": cumulative,
            "mode": "lines",
            "name": "Cumulative VWAP Edge",
            "line": {"color": _GOLD, "width": 2},
        }
    ]
    return _json(traces)


def _hourly_distribution_traces(result: BacktestResult) -> str:
    """Return Plotly trace definitions for the trade distribution by hour."""
    if not result.fills:
        return "[]"

    counts: dict[int, int] = dict.fromkeys(range(24), 0)
    for fill in result.fills:
        counts[fill.timestamp.hour] += 1

    hours = list(range(24))
    values = [counts[h] for h in hours]

    traces = [
        {
            "type": "bar",
            "x": [f"{h:02d}:00" for h in hours],
            "y": values,
            "name": "Trade Count",
            "marker": {"color": _VIOLET},
        }
    ]
    return _json(traces)


# ---------------------------------------------------------------------------
# Table builders
# ---------------------------------------------------------------------------


def _metric_rows(result: BacktestResult) -> list[tuple[str, str]]:
    """Return summary metric (label, value) pairs for the summary table."""
    p = result.pnl
    total_pnl = p.total_alpha_eur
    final_equity = result.initial_capital + total_pnl
    total_volume = sum((f.volume for f in result.fills), Decimal("0"))

    rows: list[tuple[str, str]] = [
        ("Period", f"{result.start} — {result.end} ({result.duration_days} days)"),
        ("Exchange", result.exchange),
        ("Algo", result.algo_name),
        ("Total PnL", f"{total_pnl:+,.2f} EUR"),
        ("Market VWAP", f"{p.market_vwap:.2f} EUR/MWh"),
        (
            "Sharpe Ratio",
            f"{float(result.sharpe_ratio):.2f}" if result.sharpe_ratio is not None else "N/A",
        ),
        (
            "Max Drawdown",
            (
                f"{-float(result.max_drawdown):+,.2f} EUR"
                f"  ({-float(result.max_drawdown_pct) * 100:.1f}%)"
            ),
        ),
        (
            "Profit Factor",
            f"{float(result.profit_factor):.2f}" if result.profit_factor is not None else "N/A",
        ),
        ("Trade Count", f"{len(result.fills):,d}"),
        ("Total Volume", f"{float(total_volume):,.1f} MW"),
        ("Avg Trade PnL", f"{float(result.avg_trade_pnl):+,.2f} EUR"),
        ("Initial Capital", f"{float(result.initial_capital):,.2f} EUR"),
        ("Final Equity", f"{float(final_equity):,.2f} EUR"),
    ]
    return rows


def _top_trades_rows(
    result: BacktestResult, *, best: bool
) -> list[tuple[str, str, str, str, str, str]]:
    """Return top-10 best or worst trade rows for the trades table."""
    from nexa_backtest.analysis.metrics import compute_fill_pnl

    market_vwap = result.pnl.market_vwap
    product_vwaps = result.product_vwaps or None
    fill_pnls = [(f, compute_fill_pnl(f, market_vwap, product_vwaps)) for f in result.fills]
    sorted_fills = sorted(fill_pnls, key=lambda x: x[1], reverse=best)[:10]

    rows = []
    for fill, pnl in sorted_fills:
        ts = fill.timestamp.strftime("%Y-%m-%d %H:%M")
        rows.append(
            (
                ts,
                fill.product_id,
                str(fill.side),
                f"{float(fill.price):.2f}",
                f"{float(fill.volume):.1f}",
                f"{float(pnl):+,.2f}",
            )
        )
    return rows


# ---------------------------------------------------------------------------
# HTML assembly
# ---------------------------------------------------------------------------


def _gate_closure_nop_traces(result: BacktestResult) -> str:
    """Return Plotly trace for NOP at gate closure over time."""
    snaps = result.gate_closure_positions
    if not snaps:
        return "[]"

    xs = [s.gate_closure_time.isoformat() for s in snaps]
    ys = [float(s.net_mw) for s in snaps]
    colours = [_MINT if y >= 0 else _CORAL for y in ys]

    traces = [
        {
            "type": "bar",
            "x": xs,
            "y": ys,
            "name": "NOP at Gate Closure",
            "marker": {"color": colours},
        }
    ]
    return _json(traces)


def generate_html_report(result: BacktestResult) -> str:
    """Generate a self-contained HTML backtest report.

    The report contains five Plotly charts and four data tables, styled
    with the Phase Nexa colour palette.  No external assets are required
    beyond the Plotly.js CDN script tag.

    Args:
        result: The completed backtest result to report on.

    Returns:
        Complete HTML document as a string.
    """
    eq_traces = _equity_curve_traces(result)
    dd_traces = _drawdown_traces(result)
    daily_traces = _daily_pnl_traces(result)
    vwap_traces = _vwap_edge_traces(result)
    hourly_traces = _hourly_distribution_traces(result)
    gate_nop_traces = _gate_closure_nop_traces(result)

    metric_rows_html = "".join(
        f"<tr><td>{label}</td><td>{value}</td></tr>" for label, value in _metric_rows(result)
    )

    def _trades_table(title: str, rows: list[tuple[str, str, str, str, str, str]]) -> str:
        header = "<tr><th>Time</th><th>Product</th><th>Side</th><th>Price</th><th>Volume</th><th>PnL</th></tr>"
        body = "".join(
            f"<tr>"
            f"<td>{r[0]}</td><td>{r[1]}</td>"
            f"<td class='{'buy' if r[2] == 'BUY' else 'sell'}'>{r[2]}</td>"
            f"<td>{r[3]}</td><td>{r[4]}</td>"
            f"<td class='{'pos' if r[5].startswith('+') else 'neg'}'>{r[5]}</td>"
            f"</tr>"
            for r in rows
        )
        return f"<h3>{title}</h3><table>{header}{body}</table>"

    best_rows = _top_trades_rows(result, best=True)
    worst_rows = _top_trades_rows(result, best=False)
    best_table_html = _trades_table("Top 10 Best Trades", best_rows)
    worst_table_html = _trades_table("Top 10 Worst Trades", worst_rows)

    daily_breakdown_rows = "".join(
        f"<tr>"
        f"<td>{d.date}</td>"
        f"<td class='{'pos' if d.pnl >= 0 else 'neg'}'>{float(d.pnl):+,.2f}</td>"
        f"<td>{float(d.volume_mw):,.1f}</td>"
        f"<td>{d.trade_count}</td>"
        f"<td>{float(d.vwap_edge):+,.2f}</td>"
        f"</tr>"
        for d in result.daily_pnl()
    )

    plotly_layout = _json(
        {
            "paper_bgcolor": _NEBULA,
            "plot_bgcolor": _NEBULA,
            "font": {"color": _TEXT, "family": "'JetBrains Mono', 'Courier New', monospace"},
            "xaxis": {"gridcolor": "#1E3A5F", "linecolor": "#1E3A5F"},
            "yaxis": {"gridcolor": "#1E3A5F", "linecolor": "#1E3A5F"},
            "margin": {"l": 50, "r": 20, "t": 40, "b": 40},
            "legend": {"bgcolor": "rgba(0,0,0,0)"},
        }
    )

    report_title = f"Backtest Report — {result.algo_name} | {result.start} to {result.end}"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{report_title}</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin/>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet"/>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js" charset="utf-8"></script>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

  body {{
    background: {_QUANTUM_BLUE};
    color: {_TEXT};
    font-family: 'Inter', sans-serif;
    font-size: 14px;
    line-height: 1.5;
  }}

  header {{
    background: {_NEBULA};
    border-bottom: 2px solid {_VIOLET};
    padding: 20px 32px;
  }}

  header h1 {{
    font-size: 20px;
    font-weight: 700;
    color: {_CYAN};
    letter-spacing: 0.03em;
  }}

  header p {{
    color: {_TEXT_DIM};
    font-size: 12px;
    margin-top: 4px;
    font-family: 'JetBrains Mono', 'Courier New', monospace;
  }}

  .container {{
    max-width: 1400px;
    margin: 0 auto;
    padding: 24px 32px;
  }}

  .grid-2 {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
    margin-bottom: 20px;
  }}

  .grid-3 {{
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 20px;
    margin-bottom: 20px;
  }}

  .card {{
    background: {_NEBULA};
    border: 1px solid #1E3A5F;
    border-radius: 8px;
    padding: 20px;
    overflow: hidden;
  }}

  .card h3 {{
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: {_VIOLET};
    margin-bottom: 16px;
  }}

  .chart-container {{
    width: 100%;
    height: 280px;
  }}

  .chart-full {{
    width: 100%;
    height: 320px;
  }}

  table {{
    width: 100%;
    border-collapse: collapse;
    font-family: 'JetBrains Mono', 'Courier New', monospace;
    font-size: 12px;
  }}

  th {{
    color: {_TEXT_DIM};
    font-weight: 600;
    text-align: left;
    padding: 6px 10px;
    border-bottom: 1px solid #1E3A5F;
    white-space: nowrap;
  }}

  td {{
    padding: 6px 10px;
    border-bottom: 1px solid rgba(30,58,95,0.5);
    white-space: nowrap;
  }}

  tr:last-child td {{ border-bottom: none; }}

  tr:hover td {{ background: rgba(0,229,255,0.04); }}

  .pos {{ color: {_MINT}; }}
  .neg {{ color: {_CORAL}; }}
  .buy {{ color: {_CYAN}; }}
  .sell {{ color: {_GOLD}; }}

  .metrics-table td:first-child {{
    color: {_TEXT_DIM};
    width: 40%;
  }}

  .metrics-table td:last-child {{
    color: {_GOLD};
    font-weight: 600;
    text-align: right;
  }}

  .section-title {{
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: {_VIOLET};
    margin: 28px 0 12px;
    padding-bottom: 6px;
    border-bottom: 1px solid #1E3A5F;
  }}

  footer {{
    text-align: center;
    color: {_TEXT_DIM};
    font-size: 11px;
    padding: 24px;
    border-top: 1px solid #1E3A5F;
    margin-top: 32px;
  }}
</style>
</head>
<body>
<header>
  <h1>nexa-backtest — {result.algo_name}</h1>
  <p>{result.exchange.upper()} &nbsp;|&nbsp; {result.start} to {result.end} &nbsp;|&nbsp; {result.duration_days} days</p>
</header>

<div class="container">

  <!-- Row 1: Equity + Drawdown -->
  <p class="section-title">Equity Curve</p>
  <div class="grid-2">
    <div class="card">
      <h3>Total Equity</h3>
      <div id="chart-equity" class="chart-container"></div>
    </div>
    <div class="card">
      <h3>Drawdown</h3>
      <div id="chart-drawdown" class="chart-container"></div>
    </div>
  </div>

  <!-- Row 2: Daily PnL + VWAP Edge + Hourly Distribution -->
  <p class="section-title">PnL Analysis</p>
  <div class="grid-3">
    <div class="card">
      <h3>Daily PnL</h3>
      <div id="chart-daily" class="chart-container"></div>
    </div>
    <div class="card">
      <h3>Cumulative VWAP Edge</h3>
      <div id="chart-vwap" class="chart-container"></div>
    </div>
    <div class="card">
      <h3>Trade Distribution by Hour</h3>
      <div id="chart-hourly" class="chart-container"></div>
    </div>
  </div>

  <!-- Summary metrics -->
  <p class="section-title">Summary Metrics</p>
  <div class="card">
    <table class="metrics-table">
      <tbody>
        {metric_rows_html}
      </tbody>
    </table>
  </div>

  <!-- Trades tables -->
  <p class="section-title">Trade Detail</p>
  <div class="grid-2">
    <div class="card">
      {best_table_html}
    </div>
    <div class="card">
      {worst_table_html}
    </div>
  </div>

  <!-- Gate Closure Exposure -->
  <p class="section-title">Gate Closure Exposure</p>
  <div class="grid-2">
    <div class="card">
      <h3>NOP at Gate Closure (MW)</h3>
      <div id="chart-gate-nop" class="chart-container"></div>
    </div>
    <div class="card">
      <table class="metrics-table">
        <tbody>
          <tr><td>Products Closed</td><td>{len(result.gate_closure_positions)}</td></tr>
          <tr><td>Avg NOP at Closure</td><td>{float(result.avg_gate_closure_nop_mw):.2f} MW</td></tr>
          <tr><td>Max NOP at Closure</td><td>{float(result.max_gate_closure_nop_mw):.2f} MW</td></tr>
        </tbody>
      </table>
    </div>
  </div>

  <!-- Daily breakdown -->
  <p class="section-title">Daily Breakdown</p>
  <div class="card">
    <table>
      <thead>
        <tr>
          <th>Date</th>
          <th>PnL (EUR)</th>
          <th>Volume (MW)</th>
          <th>Trades</th>
          <th>VWAP Edge (EUR/MW)</th>
        </tr>
      </thead>
      <tbody>
        {daily_breakdown_rows}
      </tbody>
    </table>
  </div>

</div>

<footer>
  Generated by nexa-backtest &nbsp;|&nbsp; Phase Nexa
</footer>

<script>
(function() {{
  var layout = {plotly_layout};
  var config = {{responsive: true, displayModeBar: false}};

  Plotly.newPlot('chart-equity', {eq_traces}, Object.assign({{}}, layout, {{title: false}}), config);
  Plotly.newPlot('chart-drawdown', {dd_traces}, Object.assign({{}}, layout, {{title: false}}), config);
  Plotly.newPlot('chart-daily', {daily_traces}, Object.assign({{}}, layout, {{title: false}}), config);
  Plotly.newPlot('chart-vwap', {vwap_traces}, Object.assign({{}}, layout, {{title: false}}), config);
  Plotly.newPlot('chart-hourly', {hourly_traces}, Object.assign({{}}, layout, {{title: false}}), config);
  Plotly.newPlot('chart-gate-nop', {gate_nop_traces}, Object.assign({{}}, layout, {{title: false}}), config);
}})();
</script>
</body>
</html>"""

    return html
