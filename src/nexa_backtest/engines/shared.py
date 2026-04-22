"""Shared data replay engine for multi-algo backtesting.

Runs multiple algos against the same historical data in a single pass.
Market data is loaded once; each algo gets its own independent
TradingContext with separate positions, orders, and equity tracking.
Algos advance in lockstep: all algos process timestamp T before T+1.

Typical usage::

    from nexa_backtest import SharedReplayEngine
    from my_algos import ConservativeAlgo, AggressiveAlgo

    comparison = SharedReplayEngine(
        algos={
            "conservative": ConservativeAlgo(),
            "aggressive": AggressiveAlgo(),
        },
        exchange="nordpool",
        start=date(2026, 3, 1),
        end=date(2026, 3, 31),
        products=["NO1_DA"],
        data_dir=Path("./data"),
        initial_capital=Decimal("100000"),
    ).run()

    print(comparison.summary())
    comparison.to_html("reports/comparison.html")
"""

from __future__ import annotations

import contextlib
import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

from nexa_backtest.algo import SimpleAlgo
from nexa_backtest.analysis.metrics import (
    BacktestResult,
    compute_fill_pnl,
    compute_max_drawdown,
    compute_profit_factor,
    compute_sharpe,
    compute_time_in_drawdown,
)
from nexa_backtest.analysis.pnl import compute_pnl
from nexa_backtest.analysis.vwap import compute_idc_vwaps, compute_market_vwap
from nexa_backtest.data.loader import ParquetLoader
from nexa_backtest.data.window import DataManifest, SlidingWindow
from nexa_backtest.engines.backtest import (
    _AUCTION_OFFSET,
    _GATE_CLOSURE_OFFSET,
    _IDC_GATE_CLOSURE_BEFORE_DELIVERY,
    _MTU_DURATION,
    AsyncAlgoDispatcher,
    SimpleAlgoDispatcher,
    _BacktestContext,
    _in_mtu,
    _is_da_product,
    _is_idc_product,
    _mtu_to_product_id,
    _parse_zone,
)
from nexa_backtest.engines.clock import SimulatedClock
from nexa_backtest.engines.matching import ContinuousMatchingEngine, DAAuctionMatcher
from nexa_backtest.exceptions import AlgoError, DataError, SignalError
from nexa_backtest.models.registry import ModelRegistry
from nexa_backtest.signals.base import SignalProvider
from nexa_backtest.signals.csv_loader import CsvSignalProvider
from nexa_backtest.signals.registry import SignalRegistry
from nexa_backtest.types import (
    AuctionInfo,
    EquitySnapshot,
    Fill,
    GateClosureSnapshot,
    HistoricalTrade,
)

logger = logging.getLogger(__name__)

#: Maximum number of algos supported in one SharedReplayEngine run.
MAX_ALGOS = 8

#: Algo accent colours (Phase Nexa palette).  Cycles if > 8 algos.
ALGO_COLOURS = [
    "#00E5FF",
    "#8A6CFF",
    "#FFD700",
    "#4DFFC3",
    "#FF6B6B",
    "#A8FF78",
    "#FF8C00",
    "#C0C0C0",
]


@dataclass
class _AlgoRunner:
    """Encapsulates all mutable state for one algo within the shared replay loop.

    Attributes:
        name: Display name for this algo.
        dispatcher: Routes engine lifecycle events to the algo.
        context: Trading context (positions, orders, fills) for this algo.
        idc_engine: Per-algo continuous matching engine (IDC only, else None).
        fills: All fills produced during the replay.
        equity_snapshots: Equity curve snapshots.
        gate_closure_snapshots: Gate closure NOP snapshots.
        cumulative_pnl: Running realised PnL in EUR.
        idc_vwap_accum: Accumulated IDC VWAP data per product.
        gate_warned: Products for which a gate closure warning has been sent.
    """

    name: str
    dispatcher: SimpleAlgoDispatcher | AsyncAlgoDispatcher
    context: _BacktestContext
    idc_engine: ContinuousMatchingEngine | None
    fills: list[Fill] = field(default_factory=list)
    equity_snapshots: list[EquitySnapshot] = field(default_factory=list)
    gate_closure_snapshots: list[GateClosureSnapshot] = field(default_factory=list)
    cumulative_pnl: Decimal = field(default_factory=lambda: Decimal("0"))
    idc_vwap_accum: dict[str, tuple[Decimal, Decimal]] = field(default_factory=dict)
    gate_warned: set[str] = field(default_factory=set)


@dataclass
class ComparisonResult:
    """Results from a multi-algo backtest run via SharedReplayEngine.

    Holds a BacktestResult for each named algo and provides comparison
    utilities: ranking by any metric, text summary, HTML and JSON exports,
    and memory efficiency tracking.

    Attributes:
        results: Per-algo results keyed by display name.
        start_date: First delivery date of the backtest period.
        end_date: Last delivery date of the backtest period (inclusive).
        exchange: Exchange identifier, e.g. ``"nordpool"``.
        products: Product specifications that were replayed.
        peak_memory_bytes: Peak memory used by the shared data layer (bytes).
        estimated_separate_memory_bytes: Estimated memory if each algo had run
            a separate backtest with its own data copy.
    """

    results: dict[str, BacktestResult]
    start_date: date
    end_date: date
    exchange: str
    products: list[str]
    peak_memory_bytes: int = 0
    estimated_separate_memory_bytes: int = 0

    def _metric_value(self, name: str, result: BacktestResult) -> Decimal:
        """Extract a scalar metric from a BacktestResult by key.

        Args:
            name: Metric identifier.  Supported: ``"total_pnl"``,
                ``"sharpe_ratio"``, ``"max_drawdown"``, ``"profit_factor"``,
                ``"win_rate"``, ``"trades"``.
            result: The backtest result to query.

        Returns:
            Metric value as Decimal.
        """
        if name == "total_pnl":
            return result.pnl.total_alpha_eur
        if name == "sharpe_ratio":
            return result.sharpe_ratio or Decimal("0")
        if name == "max_drawdown":
            # Less drawdown is better; negate for descending sort.
            return -result.max_drawdown
        if name == "profit_factor":
            return result.profit_factor or Decimal("0")
        if name == "win_rate":
            p = result.pnl
            if not result.fills:
                return Decimal("0")
            total = p.buys.count + p.sells.count
            if total == 0:
                return Decimal("0")
            weighted = p.buys.win_rate * p.buys.count + p.sells.win_rate * p.sells.count
            return Decimal(str(weighted / total))
        if name == "trades":
            return Decimal(len(result.fills))
        return result.pnl.total_alpha_eur

    def ranking(self, metric: str = "total_pnl") -> list[str]:
        """Rank algos by a metric, best to worst.

        Args:
            metric: Metric to rank by.  Supported values: ``"total_pnl"``,
                ``"sharpe_ratio"``, ``"max_drawdown"`` (less negative is
                better), ``"profit_factor"``, ``"win_rate"``, ``"trades"``.
                Defaults to ``"total_pnl"``.

        Returns:
            List of algo display names ordered from best to worst.
        """
        scored = [(n, self._metric_value(metric, r)) for n, r in self.results.items()]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [n for n, _ in scored]

    @property
    def best(self) -> tuple[str, BacktestResult]:
        """The algo with the highest total PnL.

        Returns:
            ``(name, result)`` tuple for the best-performing algo.
        """
        return max(
            self.results.items(),
            key=lambda kv: kv[1].pnl.total_alpha_eur,
        )

    @property
    def worst(self) -> tuple[str, BacktestResult]:
        """The algo with the lowest total PnL.

        Returns:
            ``(name, result)`` tuple for the worst-performing algo.
        """
        return min(
            self.results.items(),
            key=lambda kv: kv[1].pnl.total_alpha_eur,
        )

    def summary(self) -> str:
        """Side-by-side text summary of all algos.

        Returns:
            Formatted multi-line string suitable for printing to stdout.
        """
        names = list(self.results.keys())
        col_w = 18
        label_w = 18
        n_algos = len(names)
        duration = (self.end_date - self.start_date).days + 1
        products_str = ", ".join(self.products)
        sep = "=" * (label_w + col_w * n_algos + 2)

        lines: list[str] = [
            sep,
            f"  Comparison Results: {self.start_date} to {self.end_date} ({duration} days)",
            f"  Exchange: {self.exchange} | Products: {products_str}",
            sep,
            "",
            "  " + " " * label_w + "".join(f"{nm:>{col_w}}" for nm in names),
        ]

        def _row(label: str, values: list[str]) -> str:
            return "  " + f"{label:<{label_w}}" + "".join(f"{v:>{col_w}}" for v in values)

        pnl_vals = [f"{float(r.pnl.total_alpha_eur):+,.2f} EUR" for r in self.results.values()]
        vwap_vals = [f"{float(r.pnl.market_vwap):.2f} EUR/MWh" for r in self.results.values()]
        sharpe_vals = [
            f"{float(r.sharpe_ratio):.2f}" if r.sharpe_ratio is not None else "N/A"
            for r in self.results.values()
        ]
        dd_vals = [f"{-float(r.max_drawdown):+,.2f} EUR" for r in self.results.values()]
        pf_vals = [
            f"{float(r.profit_factor):.2f}" if r.profit_factor is not None else "N/A"
            for r in self.results.values()
        ]
        wr_vals: list[str] = []
        for r in self.results.values():
            p = r.pnl
            total = p.buys.count + p.sells.count
            if total > 0:
                wr = (p.buys.win_rate * p.buys.count + p.sells.win_rate * p.sells.count) / total
                wr_vals.append(f"{wr:.1%}")
            else:
                wr_vals.append("N/A")
        trade_vals = [str(len(r.fills)) for r in self.results.values()]

        lines += [
            _row("Total PnL", pnl_vals),
            _row("vs VWAP", vwap_vals),
            _row("Sharpe", sharpe_vals),
            _row("Max Drawdown", dd_vals),
            _row("Profit Factor", pf_vals),
            _row("Win Rate", wr_vals),
            _row("Trades", trade_vals),
            "",
        ]

        best_name, best_result = self.best
        lines.append(
            f"  Best by PnL: {best_name} ({float(best_result.pnl.total_alpha_eur):+,.2f} EUR)"
        )

        sharpe_ranking = self.ranking("sharpe_ratio")
        best_sharpe = sharpe_ranking[0]
        best_sharpe_val = self.results[best_sharpe].sharpe_ratio
        if best_sharpe_val is not None:
            lines.append(
                f"  Best risk-adjusted: {best_sharpe} (Sharpe {float(best_sharpe_val):.2f})"
            )

        if self.peak_memory_bytes > 0:
            saved = self.estimated_separate_memory_bytes - self.peak_memory_bytes
            peak_mb = self.peak_memory_bytes / (1024 * 1024)
            saved_mb = saved / (1024 * 1024)
            lines += [
                "",
                f"  Memory: {peak_mb:.0f} MB (saved ~{saved_mb:.0f} MB vs separate backtests)",
            ]

        lines.append(sep)
        return "\n".join(lines)

    def to_html(self, path: str) -> None:
        """Write a self-contained HTML comparison report.

        Args:
            path: Destination file path (will be created or overwritten).
        """
        from nexa_backtest.analysis.comparison_report import generate_comparison_html_report

        html = generate_comparison_html_report(self)
        Path(path).write_text(html, encoding="utf-8")

    def to_json(self, path: str) -> None:
        """Export comparison results as a JSON file.

        Args:
            path: Destination file path (will be created or overwritten).
        """

        def _dec(v: object) -> object:
            if isinstance(v, Decimal):
                return float(v)
            return v

        payload: dict[str, Any] = {
            "start_date": str(self.start_date),
            "end_date": str(self.end_date),
            "exchange": self.exchange,
            "products": self.products,
            "peak_memory_bytes": self.peak_memory_bytes,
            "estimated_separate_memory_bytes": self.estimated_separate_memory_bytes,
            "algos": {},
        }

        for name, result in self.results.items():
            p = result.pnl
            payload["algos"][name] = {
                "algo_name": result.algo_name,
                "total_pnl_eur": _dec(p.total_alpha_eur),
                "market_vwap_eur_mwh": _dec(p.market_vwap),
                "sharpe_ratio": (
                    _dec(result.sharpe_ratio) if result.sharpe_ratio is not None else None
                ),
                "max_drawdown_eur": _dec(result.max_drawdown),
                "max_drawdown_pct": _dec(result.max_drawdown_pct),
                "profit_factor": (
                    _dec(result.profit_factor) if result.profit_factor is not None else None
                ),
                "trade_count": len(result.fills),
                "initial_capital": _dec(result.initial_capital),
                "final_equity": _dec(result.initial_capital + p.total_alpha_eur),
            }

        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


class SharedReplayEngine:
    """Run multiple algos against the same historical data.

    Market data is loaded once.  Each algo gets its own TradingContext with
    independent positions, orders, and equity tracking.  Algos advance in
    lockstep: all algos process timestamp T before moving to T+1.

    Algos do not affect each other.  Algo A's orders do not appear in Algo B's
    view of the market.  Each algo sees the same historical order book and
    clearing prices (price-taker assumption applied per algo).

    Args:
        algos: Named algos to run.  Keys are display names (e.g.
            ``"conservative"``), values are SimpleAlgo instances or
            ``@algo``-decorated async functions.  Insertion order is preserved
            in results.  Maximum MAX_ALGOS algos.
        exchange: Exchange identifier, e.g. ``"nordpool"``.
        start: First delivery date to include in the backtest.
        end: Last delivery date to include (inclusive).
        products: Product specifications, e.g. ``["NO1_DA"]``.
        signals: Explicitly constructed signal providers (shared across algos).
        models: Registry of ML models (shared across algos).
        data_dir: Directory containing market data files.
        initial_capital: Starting capital in EUR, applied equally to every algo.

    Raises:
        AlgoError: If more than MAX_ALGOS algos are provided, or if any
            algo is not a SimpleAlgo instance or ``@algo`` function.
        DataError: If required data files are missing or malformed.
    """

    def __init__(
        self,
        algos: dict[str, SimpleAlgo | Any],
        exchange: str,
        start: date,
        end: date,
        products: list[str],
        signals: list[SignalProvider] | None = None,
        models: ModelRegistry | None = None,
        data_dir: str | Path = "./data",
        initial_capital: Decimal = Decimal("100000"),
    ) -> None:
        if len(algos) > MAX_ALGOS:
            raise AlgoError(
                f"SharedReplayEngine supports at most {MAX_ALGOS} algos; "
                f"got {len(algos)}.  More than {MAX_ALGOS} makes the comparison "
                "report unreadable and memory usage unreasonable."
            )
        if not algos:
            raise AlgoError("algos must not be empty.")

        self._algos = algos
        self._exchange = exchange
        self._start = start
        self._end = end
        self._products = products
        self._signals: list[SignalProvider] = signals or []
        self._models = models
        self._data_dir = Path(data_dir)
        self._initial_capital = initial_capital

    def run(self) -> ComparisonResult:
        """Execute the shared replay and return a comparison of all algos.

        Returns:
            ComparisonResult with per-algo BacktestResult objects.

        Raises:
            DataError: If required data files are missing or malformed.
        """
        if not self._products:
            raise DataError("No products specified.")

        da_products = [p for p in self._products if _is_da_product(p)]
        idc_products = [p for p in self._products if _is_idc_product(p)]

        if not da_products and not idc_products:
            raise DataError(
                f"Cannot determine market type for products {self._products}. "
                "Use 'NO1_DA' for day-ahead or 'NO1-QH' for intraday continuous."
            )

        zone = _parse_zone(self._products[0])
        registry = SignalRegistry()
        for provider in self._signals:
            registry.register(provider)

        runners = self._build_runners(zone, registry)
        peak_memory: int = 0

        if da_products:
            da_memory = self._run_da_shared(zone, registry, runners)
            peak_memory = max(peak_memory, da_memory)

        if idc_products:
            idc_peak = self._run_idc_shared(zone, registry, runners)
            peak_memory = max(peak_memory, idc_peak)

        n_algos = len(runners)
        estimated_separate = peak_memory * n_algos

        results: dict[str, BacktestResult] = {}
        for runner in runners:
            results[runner.name] = self._build_result(runner, zone)

        return ComparisonResult(
            results=results,
            start_date=self._start,
            end_date=self._end,
            exchange=self._exchange,
            products=self._products,
            peak_memory_bytes=peak_memory,
            estimated_separate_memory_bytes=estimated_separate,
        )

    def _build_runners(self, zone: str, registry: SignalRegistry) -> list[_AlgoRunner]:
        """Construct one _AlgoRunner per algo.

        Args:
            zone: Bidding zone derived from the first product spec.
            registry: Shared signal registry.

        Returns:
            List of runners in insertion order.
        """
        has_idc = any(_is_idc_product(p) for p in self._products)
        placeholder_time = datetime.combine(self._start, time(0, 0), tzinfo=UTC)
        clock = SimulatedClock(initial_time=placeholder_time)

        runners: list[_AlgoRunner] = []
        for name, algo_or_fn in self._algos.items():
            if isinstance(algo_or_fn, SimpleAlgo):
                dispatcher: SimpleAlgoDispatcher | AsyncAlgoDispatcher = SimpleAlgoDispatcher(
                    algo_or_fn
                )
            elif callable(algo_or_fn) and getattr(algo_or_fn, "_is_algo", False):
                dispatcher = AsyncAlgoDispatcher(algo_or_fn)
            else:
                raise AlgoError(
                    f"Algo '{name}' must be a SimpleAlgo instance or a function "
                    "decorated with @algo."
                )

            idc_engine: ContinuousMatchingEngine | None = (
                ContinuousMatchingEngine() if has_idc else None
            )
            context = _BacktestContext(
                clock=clock,
                signal_registry=registry,
                idc_engine=idc_engine,
                model_registry=self._models,
            )
            runners.append(
                _AlgoRunner(
                    name=name,
                    dispatcher=dispatcher,
                    context=context,
                    idc_engine=idc_engine,
                )
            )

        return runners

    def _run_da_shared(
        self,
        zone: str,
        registry: SignalRegistry,
        runners: list[_AlgoRunner],
    ) -> int:
        """Run the DA auction loop dispatching to all runners in lockstep.

        Args:
            zone: Bidding zone.
            registry: Shared signal registry.
            runners: All algo runners (mutated in place).

        Returns:
            Estimated memory usage of the shared DA data in bytes.
        """
        loader = ParquetLoader(self._data_dir)
        market_data = loader.load_da_prices(zone, self._start, self._end)
        da_memory = int(market_data.memory_usage(deep=True).sum())

        first_auction = _auction_time(self._start)
        clock = SimulatedClock(initial_time=first_auction - timedelta(minutes=1))
        for runner in runners:
            runner.context._clock = clock

        market_vwap = compute_market_vwap(market_data)
        market_data["delivery_date"] = market_data["timestamp"].dt.date

        for runner in runners:
            runner.dispatcher.on_setup(runner.context)
        self._discover_signals(registry, runners)

        for delivery_date, day_data in market_data.groupby("delivery_date"):
            auction_t = _auction_time(delivery_date)  # type: ignore[arg-type]
            clock.advance_to(auction_t)
            gate_closure = auction_t + _GATE_CLOSURE_OFFSET

            for runner in runners:
                runner.context._reset_day()
                for _, row in day_data.iterrows():
                    pid: str = str(row["product_id"])
                    runner.context._clearing_prices[pid] = Decimal(str(row["price_eur_mwh"]))
                    runner.context._gate_closures[pid] = gate_closure
                    ts = row["timestamp"]
                    runner.context._product_delivery_starts[pid] = (
                        ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts
                    )

            for runner in runners:
                self._dispatch_signals(runner, registry)

            for _, product_row in day_data.sort_values("timestamp").iterrows():
                auction_info = AuctionInfo(
                    product_id=str(product_row["product_id"]),
                    auction_type="DA",
                    gate_closure_time=gate_closure,
                    zone=zone,
                )
                for runner in runners:
                    runner.dispatcher.on_auction_open(runner.context, auction_info)

            fill_time = gate_closure
            for runner in runners:
                orders = list(runner.context._pending_orders.values())
                runner.context._pending_orders.clear()
                day_fills: list[Fill] = []
                for order in orders:
                    clearing = runner.context._clearing_prices.get(order.product_id)
                    if clearing is None:
                        logger.warning(
                            "Order for unknown product '%s' rejected (runner '%s').",
                            order.product_id,
                            runner.name,
                        )
                        continue
                    matcher = DAAuctionMatcher(clearing_price=clearing, fill_timestamp=fill_time)
                    match_result = matcher.match(order)
                    if match_result.fill is not None:
                        runner.context._record_fill(match_result.fill)
                        runner.fills.append(match_result.fill)
                        day_fills.append(match_result.fill)
                        runner.dispatcher.on_fill(runner.context, match_result.fill)

                if day_fills:
                    day_pnl = sum(
                        (compute_fill_pnl(f, market_vwap) for f in day_fills),
                        Decimal("0"),
                    )
                    runner.cumulative_pnl += day_pnl
                    runner.equity_snapshots.append(
                        EquitySnapshot(
                            timestamp=fill_time,
                            realised_pnl=runner.cumulative_pnl,
                            unrealised_pnl=Decimal("0"),
                            total_equity=self._initial_capital + runner.cumulative_pnl,
                            cash=self._initial_capital + runner.cumulative_pnl,
                            net_position_mw=Decimal("0"),
                        )
                    )

        for runner in runners:
            runner.dispatcher.on_teardown(runner.context)

        return da_memory

    def _run_idc_shared(
        self,
        zone: str,
        registry: SignalRegistry,
        runners: list[_AlgoRunner],
    ) -> int:
        """Run the IDC continuous loop dispatching to all runners in lockstep.

        The SlidingWindow is loaded once and shared.  Each runner has its own
        ContinuousMatchingEngine.

        Args:
            zone: Bidding zone.
            registry: Shared signal registry.
            runners: All algo runners (mutated in place).

        Returns:
            Peak memory usage of the shared sliding window in bytes.
        """
        try:
            manifest = DataManifest(data_dir=self._data_dir, zone=zone)
        except FileNotFoundError as exc:
            raise DataError(str(exc)) from exc

        window = SlidingWindow(manifest=manifest)
        start_dt = datetime.combine(self._start, time(0, 0), tzinfo=UTC)
        end_dt = datetime.combine(self._end + timedelta(days=1), time(0, 0), tzinfo=UTC)
        clock = SimulatedClock(initial_time=start_dt - timedelta(minutes=1))

        for runner in runners:
            runner.context._clock = clock
            if runner.idc_engine is not None:
                _register_idc_gate_closures(runner.idc_engine, start_dt, end_dt, zone)
            current = start_dt
            while current < end_dt:
                pid = _mtu_to_product_id(current, zone)
                runner.context._product_delivery_starts[pid] = current
                current += _MTU_DURATION

        for runner in runners:
            runner.dispatcher.on_setup(runner.context)
        self._discover_signals(registry, runners)

        peak_memory = 0
        current_time = start_dt

        while current_time < end_dt:
            next_time = current_time + _MTU_DURATION
            clock.advance_to(current_time)

            # Advance the shared window — loaded ONCE for all algos.
            window.advance_to(current_time)
            peak_memory = max(peak_memory, window.memory_usage_bytes)

            # Read historical events once per MTU.
            mtu_events = list(window.events_between(current_time, next_time))

            # Replay events through each runner's own matching engine.
            for runner in runners:
                if runner.idc_engine is None:
                    continue
                for event in mtu_events:
                    try:
                        fills = runner.idc_engine.process_historical_event(event)
                    except Exception as exc:
                        with contextlib.suppress(Exception):
                            runner.dispatcher.on_error(runner.context, exc)
                        continue

                    if isinstance(event, HistoricalTrade):
                        prev = runner.idc_vwap_accum.get(
                            event.product_id, (Decimal("0"), Decimal("0"))
                        )
                        runner.idc_vwap_accum[event.product_id] = (
                            prev[0] + event.price_eur_mwh * event.volume_mw,
                            prev[1] + event.volume_mw,
                        )

                    runner.dispatcher.on_market_event(runner.context, event)
                    for fill in fills:
                        runner.context._record_fill(fill)
                        runner.fills.append(fill)
                        runner.dispatcher.on_fill(runner.context, fill)

                for product_id, closure_time in list(runner.idc_engine._gate_closures.items()):
                    if (
                        product_id not in runner.gate_warned
                        and current_time < closure_time <= next_time
                    ):
                        runner.dispatcher.on_gate_closure_warning(
                            runner.context, product_id, closure_time - current_time
                        )
                        runner.gate_warned.add(product_id)

            for runner in runners:
                self._dispatch_signals(runner, registry)
                runner.dispatcher.on_bar(runner.context)

            for runner in runners:
                if runner.idc_engine is None:
                    continue
                pending = list(runner.context._idc_pending_orders.values())
                runner.context._idc_pending_orders.clear()
                for order in pending:
                    fills = runner.idc_engine.place_algo_order(order, current_time)
                    for fill in fills:
                        runner.context._record_fill(fill)
                        runner.fills.append(fill)
                        runner.dispatcher.on_fill(runner.context, fill)

                closed_products = runner.idc_engine.check_gate_closures(current_time)
                for product_id in closed_products:
                    pos = runner.context.get_position(product_id)
                    delivery_start = runner.context._product_delivery_starts.get(product_id)
                    if delivery_start is not None:
                        runner.gate_closure_snapshots.append(
                            GateClosureSnapshot(
                                product_id=product_id,
                                gate_closure_time=current_time,
                                delivery_start=delivery_start,
                                net_mw=pos.net_mw,
                            )
                        )
                    runner.dispatcher.on_gate_closure(runner.context, product_id)

                bar_all_fills = [
                    f for f in runner.fills if _in_mtu(f.timestamp, current_time, next_time)
                ]
                if bar_all_fills:
                    running_vwaps, _ = compute_idc_vwaps(runner.idc_vwap_accum)
                    bar_pnl = sum(
                        (
                            compute_fill_pnl(f, running_vwaps.get(f.product_id, Decimal("0")))
                            for f in bar_all_fills
                        ),
                        Decimal("0"),
                    )
                    runner.cumulative_pnl += bar_pnl
                    unrealised = runner.context.get_unrealised_pnl()
                    net_mw = sum(
                        (p.net_mw for p in runner.context.get_all_positions().values()),
                        Decimal("0"),
                    )
                    runner.equity_snapshots.append(
                        EquitySnapshot(
                            timestamp=current_time,
                            realised_pnl=runner.cumulative_pnl,
                            unrealised_pnl=unrealised,
                            total_equity=(
                                self._initial_capital + runner.cumulative_pnl + unrealised
                            ),
                            cash=self._initial_capital + runner.cumulative_pnl,
                            net_position_mw=net_mw,
                        )
                    )

            current_time = next_time

        for runner in runners:
            runner.dispatcher.on_teardown(runner.context)

        return peak_memory

    def _build_result(self, runner: _AlgoRunner, zone: str) -> BacktestResult:
        """Build a BacktestResult from a completed _AlgoRunner.

        Args:
            runner: The completed runner.
            zone: Bidding zone (used to load DA data for VWAP computation).

        Returns:
            Fully computed BacktestResult.
        """
        import pandas as pd

        all_fills = runner.fills
        equity_snapshots = sorted(runner.equity_snapshots, key=lambda s: s.timestamp)

        per_product_vwap, idc_portfolio_vwap = compute_idc_vwaps(runner.idc_vwap_accum)

        try:
            loader = ParquetLoader(self._data_dir)
            market_data = loader.load_da_prices(zone, self._start, self._end)
        except DataError:
            market_data = pd.DataFrame(
                columns=["product_id", "timestamp", "zone", "price_eur_mwh", "volume_mwh"]
            )

        if len(market_data) > 0:
            market_vwap = compute_market_vwap(market_data)
        else:
            market_vwap = idc_portfolio_vwap if idc_portfolio_vwap > 0 else Decimal("0")

        pnl = compute_pnl(
            all_fills,
            market_data,
            product_vwaps=per_product_vwap or None,
            idc_portfolio_vwap=idc_portfolio_vwap if idc_portfolio_vwap > 0 else None,
        )

        sharpe = compute_sharpe(equity_snapshots)
        max_dd, max_dd_pct = compute_max_drawdown(equity_snapshots)
        dd_time = compute_time_in_drawdown(equity_snapshots)
        product_vwaps_for_metrics = per_product_vwap or {}
        profit_fac = compute_profit_factor(
            all_fills, market_vwap, product_vwaps_for_metrics or None
        )
        avg_trade = pnl.total_alpha_eur / Decimal(len(all_fills)) if all_fills else Decimal("0")

        best_fill: Fill | None = None
        worst_fill: Fill | None = None
        if all_fills:
            fill_pnls = [
                (f, compute_fill_pnl(f, market_vwap, product_vwaps_for_metrics or None))
                for f in all_fills
            ]
            best_fill = max(fill_pnls, key=lambda x: x[1])[0]
            worst_fill = min(fill_pnls, key=lambda x: x[1])[0]

        duration = (self._end - self._start).days + 1

        nop_values = [abs(s.net_mw) for s in runner.gate_closure_snapshots if s.net_mw != 0]
        avg_gate_nop = (
            sum(nop_values, Decimal("0")) / Decimal(len(nop_values)) if nop_values else Decimal("0")
        )
        max_gate_nop = max(nop_values, default=Decimal("0"))

        algo_or_fn = self._algos[runner.name]
        if isinstance(algo_or_fn, SimpleAlgo):
            algo_name = type(algo_or_fn).__name__
        else:
            algo_name = getattr(
                algo_or_fn, "_algo_name", getattr(algo_or_fn, "__name__", runner.name)
            )

        return BacktestResult(
            algo_name=algo_name,
            exchange=self._exchange,
            start=self._start,
            end=self._end,
            fills=tuple(all_fills),
            pnl=pnl,
            equity_curve=tuple(equity_snapshots),
            initial_capital=self._initial_capital,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            max_drawdown_pct=max_dd_pct,
            time_in_drawdown=dd_time,
            profit_factor=profit_fac,
            avg_trade_pnl=avg_trade,
            best_trade=best_fill,
            worst_trade=worst_fill,
            duration_days=duration,
            gate_closure_positions=tuple(runner.gate_closure_snapshots),
            avg_gate_closure_nop_mw=avg_gate_nop,
            max_gate_closure_nop_mw=max_gate_nop,
            product_vwaps=product_vwaps_for_metrics,
        )

    def _dispatch_signals(self, runner: _AlgoRunner, registry: SignalRegistry) -> None:
        """Dispatch current signal values to a single runner."""
        for signal_name in runner.dispatcher.subscribed_signals:
            if registry.has(signal_name):
                try:
                    value = runner.context.get_signal(signal_name)
                    runner.dispatcher.on_signal(runner.context, signal_name, value)
                except SignalError:
                    logger.debug(
                        "No signal '%s' at %s (runner '%s') — skipping.",
                        signal_name,
                        runner.context.now().isoformat(),
                        runner.name,
                    )

    def _discover_signals(self, registry: SignalRegistry, runners: list[_AlgoRunner]) -> None:
        """Auto-register CSV providers for subscribed but unregistered signals."""
        signals_dir = self._data_dir / "signals"
        subscribed: set[str] = set()
        for runner in runners:
            subscribed.update(runner.dispatcher.subscribed_signals)

        for signal_name in subscribed:
            if registry.has(signal_name):
                continue
            csv_path = signals_dir / f"{signal_name}.csv"
            if not csv_path.exists():
                raise DataError(
                    f"Signal '{signal_name}' is subscribed by an algo but no CSV "
                    f"was found at '{csv_path}'."
                )
            provider = CsvSignalProvider(
                name=signal_name,
                path=csv_path,
                unit="",
                description=f"Auto-loaded from {csv_path.name}",
            )
            registry.register(provider)
            logger.info("Auto-loaded signal '%s' from %s", signal_name, csv_path)


def _auction_time(delivery_date: date) -> datetime:
    """Return D-1 12:00 UTC as the simulated clock time for the DA auction."""
    delivery_dt = datetime.combine(delivery_date, time(0, 0), tzinfo=UTC)
    return delivery_dt + _AUCTION_OFFSET


def _register_idc_gate_closures(
    engine: ContinuousMatchingEngine,
    start_dt: datetime,
    end_dt: datetime,
    zone: str,
) -> None:
    """Pre-register gate closure times for all IDC products in the replay range."""
    current = start_dt
    while current < end_dt:
        product_id = _mtu_to_product_id(current, zone)
        gate_closure = current - _IDC_GATE_CLOSURE_BEFORE_DELIVERY
        engine.set_gate_closure(product_id, gate_closure)
        current += _MTU_DURATION
