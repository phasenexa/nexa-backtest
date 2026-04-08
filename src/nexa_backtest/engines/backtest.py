"""BacktestEngine: simulated clock, DA auction replay, IDC continuous replay.

The engine loads historical market data, advances a simulated clock through
each delivery period, calls the algo's lifecycle hooks, matches orders
against historical prices/order book data, and returns a
:class:`BacktestResult`.

Two market modes are supported:

- **DA (Day-Ahead)**: Products ending in ``_DA``, e.g. ``NO1_DA``.
  Orders are batch-matched at auction gate closure against the historical
  clearing price using the price-taker assumption.

- **IDC (Intraday Continuous)**: Products matching the pattern
  ``{zone}-QH`` or ``{zone}-QH-{time}``, e.g. ``NO1-QH``.
  Orders are continuously matched against the historical order book
  using price-time priority.  Data is loaded via a sliding window of
  Parquet row groups.

Mixed DA + IDC products in a single backtest are supported: both matching
engines run in parallel sharing the same simulated clock.
"""

from __future__ import annotations

import contextlib
import logging
import uuid
from collections import defaultdict
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
)
from nexa_backtest.analysis.pnl import compute_pnl
from nexa_backtest.analysis.vwap import compute_market_vwap
from nexa_backtest.context import SignalValue, TradingContext
from nexa_backtest.data.loader import ParquetLoader
from nexa_backtest.data.window import DataManifest, SlidingWindow
from nexa_backtest.engines.clock import SimulatedClock
from nexa_backtest.engines.matching import ContinuousMatchingEngine, DAAuctionMatcher
from nexa_backtest.exceptions import DataError, SignalError
from nexa_backtest.signals.base import SignalProvider
from nexa_backtest.signals.csv_loader import CsvSignalProvider
from nexa_backtest.signals.registry import SignalRegistry
from nexa_backtest.types import (
    MTU,
    AuctionInfo,
    CancelResult,
    EquitySnapshot,
    Fill,
    Order,
    OrderBook,
    OrderResult,
    OrderStatus,
    Position,
    PriceLevel,
    Side,
)

logger = logging.getLogger(__name__)

# Simulated time of DA auction relative to delivery day midnight UTC.
_AUCTION_OFFSET = timedelta(days=-1, hours=12)  # D-1 12:00 UTC
_GATE_CLOSURE_OFFSET = timedelta(hours=1)  # gate closes 1h after auction opens
_MTU_DURATION = timedelta(minutes=15)

# IDC gate closure: 30 minutes before delivery start (Nord Pool Nordics default)
_IDC_GATE_CLOSURE_BEFORE_DELIVERY = timedelta(minutes=30)

# Gate closure warning threshold
_IDC_GATE_WARNING_BEFORE = timedelta(minutes=5)


def _zero_position(product_id: str) -> Position:
    return Position(
        product_id=product_id,
        net_mw=Decimal("0"),
        avg_entry_price=Decimal("0"),
        unrealised_pnl=Decimal("0"),
    )


def _is_idc_product(spec: str) -> bool:
    """Return True if ``spec`` looks like an IDC product specifier."""
    return "-QH" in spec or "-QH-" in spec


def _is_da_product(spec: str) -> bool:
    """Return True if ``spec`` looks like a DA product specifier."""
    return spec.endswith("_DA")


def _parse_zone(spec: str) -> str:
    """Extract the bidding zone from a product spec.

    Examples::

        "NO1_DA" → "NO1"
        "NO1-QH" → "NO1"
        "NO1-QH-0900" → "NO1"
    """
    return spec.split("_")[0].split("-")[0]


class _BacktestContext:
    """Internal :class:`~nexa_backtest.context.TradingContext` implementation.

    Created by :class:`BacktestEngine` for each run.  The algo receives this
    object via every lifecycle hook but never knows its concrete type.

    When ``_idc_engine`` is set, the context operates in IDC mode: order book
    queries are delegated to the matching engine and ``place_order()`` routes
    orders to a pending queue for the engine loop to process after each
    ``on_bar`` call.
    """

    def __init__(
        self,
        clock: SimulatedClock,
        signal_registry: SignalRegistry,
        idc_engine: ContinuousMatchingEngine | None = None,
    ) -> None:
        self._clock = clock
        self._signal_registry = signal_registry
        self._idc_engine = idc_engine

        # Shared fill history
        self._fills: list[Fill] = []
        self._position_fills: dict[str, list[Fill]] = defaultdict(list)

        # DA-mode state
        self._pending_orders: dict[str, Order] = {}
        self._clearing_prices: dict[str, Decimal] = {}
        self._gate_closures: dict[str, datetime] = {}

        # IDC-mode state: orders placed during on_bar() await engine processing
        self._idc_pending_orders: dict[str, Order] = {}

    # ------------------------------------------------------------------
    # Time
    # ------------------------------------------------------------------

    def now(self) -> datetime:
        """Return current simulated time."""
        return self._clock.now()

    def time_to_gate_closure(self, product_id: str) -> timedelta:
        """Return time remaining until gate closure for ``product_id``."""
        if product_id in self._gate_closures:
            remaining = self._gate_closures[product_id] - self._clock.now()
            return max(remaining, timedelta(0))
        if self._idc_engine is not None:
            closure = self._idc_engine._gate_closures.get(product_id)
            if closure is not None:
                remaining = closure - self._clock.now()
                return max(remaining, timedelta(0))
        return timedelta(0)

    def current_mtu(self) -> MTU:
        """Return the MTU whose delivery period contains ``now()``."""
        now = self._clock.now()
        minute_slot = (now.minute // 15) * 15
        start = now.replace(minute=minute_slot, second=0, microsecond=0)
        end = start + _MTU_DURATION
        return MTU(start=start, end=end, zone="")

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------

    def get_orderbook(self, product_id: str) -> OrderBook:
        """Return the current order book snapshot for a product.

        In IDC mode returns the historical order book state at the current
        simulated time.  In DA mode returns a synthetic book based on the
        clearing price.
        """
        if self._idc_engine is not None:
            return self._idc_engine.get_orderbook(product_id, self._clock.now())
        # DA fallback — synthetic book from clearing price
        price = self._clearing_prices.get(product_id)
        ts = self._clock.now()
        if price is None:
            return OrderBook(product_id=product_id, bids=[], asks=[], timestamp=ts)
        level = PriceLevel(price=price, volume=Decimal("1000"))
        return OrderBook(product_id=product_id, bids=[level], asks=[level], timestamp=ts)

    def get_best_bid(self, product_id: str) -> PriceLevel | None:
        """Return the best bid for ``product_id``."""
        return self.get_orderbook(product_id).best_bid

    def get_best_ask(self, product_id: str) -> PriceLevel | None:
        """Return the best ask for ``product_id``."""
        return self.get_orderbook(product_id).best_ask

    def get_last_price(self, product_id: str) -> Decimal | None:
        """Return the last traded price for ``product_id``."""
        if self._idc_engine is not None:
            return self._idc_engine.get_last_trade_price(product_id)
        return self._clearing_prices.get(product_id)

    def get_vwap(self, product_id: str) -> Decimal | None:
        """Return the session VWAP for ``product_id``."""
        if self._idc_engine is not None:
            return self._idc_engine.get_last_trade_price(product_id)
        return self._clearing_prices.get(product_id)

    # ------------------------------------------------------------------
    # Order management
    # ------------------------------------------------------------------

    def place_order(self, order: Order) -> OrderResult:
        """Submit an order for matching.

        In DA mode the order is queued for deferred matching at gate closure.
        In IDC mode the order is queued for processing by the engine after the
        current ``on_bar()`` call returns.
        """
        if self._idc_engine is not None:
            self._idc_pending_orders[order.order_id] = order
        else:
            self._pending_orders[order.order_id] = order
        return OrderResult(order_id=order.order_id, status=OrderStatus.ACCEPTED)

    def cancel_order(self, order_id: str) -> CancelResult:
        """Cancel a pending or resting order."""
        # IDC pending queue
        if order_id in self._idc_pending_orders:
            del self._idc_pending_orders[order_id]
            return CancelResult(order_id=order_id, status="cancelled")
        # IDC resting in matching engine
        if self._idc_engine is not None:
            return self._idc_engine.cancel_algo_order(order_id)
        # DA pending queue
        if order_id in self._pending_orders:
            del self._pending_orders[order_id]
            return CancelResult(order_id=order_id, status="cancelled")
        return CancelResult(order_id=order_id, status="not_found")

    def modify_order(self, order_id: str, **changes: Any) -> OrderResult:
        """Cancel and resubmit an order with updated fields."""
        # Try IDC pending queue first
        source = None
        if order_id in self._idc_pending_orders:
            source = self._idc_pending_orders
        elif order_id in self._pending_orders:
            source = self._pending_orders

        if source is None:
            return OrderResult(
                order_id=order_id,
                status=OrderStatus.REJECTED,
                rejection_reason=f"Order '{order_id}' not found.",
            )

        old_order = source.pop(order_id)
        new_data = old_order.model_dump()
        new_data.update(changes)
        new_data["order_id"] = str(uuid.uuid4())
        try:
            new_order = Order.model_validate(new_data)
        except Exception as exc:
            return OrderResult(
                order_id=order_id,
                status=OrderStatus.REJECTED,
                rejection_reason=str(exc),
            )
        if self._idc_engine is not None:
            self._idc_pending_orders[new_order.order_id] = new_order
        else:
            self._pending_orders[new_order.order_id] = new_order
        return OrderResult(order_id=new_order.order_id, status=OrderStatus.ACCEPTED)

    # ------------------------------------------------------------------
    # Positions
    # ------------------------------------------------------------------

    def get_position(self, product_id: str) -> Position:
        """Return the current net position for ``product_id``."""
        fills = self._position_fills.get(product_id, [])
        if not fills:
            return _zero_position(product_id)

        net_mw = Decimal("0")
        total_cost = Decimal("0")
        for f in fills:
            if f.side == Side.BUY:
                net_mw += f.volume
                total_cost += f.price * f.volume
            else:
                net_mw -= f.volume
                total_cost -= f.price * f.volume

        if net_mw == 0:
            return _zero_position(product_id)

        avg_price = abs(total_cost / net_mw)
        if self._idc_engine is not None:
            mark = self._idc_engine.get_last_trade_price(product_id) or avg_price
        else:
            mark = self._clearing_prices.get(product_id, avg_price)
        unrealised = (mark - avg_price) * net_mw

        return Position(
            product_id=product_id,
            net_mw=net_mw,
            avg_entry_price=avg_price,
            unrealised_pnl=unrealised,
        )

    def get_all_positions(self) -> dict[str, Position]:
        """Return all non-zero positions."""
        result: dict[str, Position] = {}
        for product_id in self._position_fills:
            pos = self.get_position(product_id)
            if pos.net_mw != 0:
                result[product_id] = pos
        return result

    def get_unrealised_pnl(self) -> Decimal:
        """Return total unrealised PnL across all positions."""
        return sum(
            (p.unrealised_pnl for p in self.get_all_positions().values()),
            Decimal("0"),
        )

    # ------------------------------------------------------------------
    # Signals
    # ------------------------------------------------------------------

    def get_signal(self, name: str) -> SignalValue:
        """Return the latest signal value visible at the current simulated time."""
        provider = self._signal_registry.get(name)
        return provider.get_value(self._clock.now())

    def get_signal_history(self, name: str, lookback: int) -> list[SignalValue]:
        """Return the most recent ``lookback`` values visible at the current time."""
        provider = self._signal_registry.get(name)
        return provider.get_history_at(self._clock.now(), lookback)

    # ------------------------------------------------------------------
    # ML models (stub — implemented in Stage 3)
    # ------------------------------------------------------------------

    def predict(self, model_name: str, features: dict[str, Any]) -> Any:
        """Run inference on a registered ML model.

        Raises:
            NotImplementedError: ML model registry is not yet implemented.
        """
        raise NotImplementedError(
            "ML model inference is not yet supported. It will be added in a later stage."
        )

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def log(self, message: str, level: str = "info") -> None:
        """Emit a log message tagged with the current simulated time."""
        algo_logger = logging.getLogger("nexa_backtest.algo")
        log_fn = getattr(algo_logger, level, algo_logger.info)
        log_fn("[%s] %s", self._clock.now().isoformat(), message)

    # ------------------------------------------------------------------
    # Engine-internal helpers (not part of TradingContext)
    # ------------------------------------------------------------------

    def _reset_day(self) -> None:
        """Clear pending orders and clearing prices for the next auction day."""
        self._pending_orders.clear()
        self._clearing_prices.clear()
        self._gate_closures.clear()

    def _record_fill(self, fill: Fill) -> None:
        """Record a fill in the position tracker."""
        self._fills.append(fill)
        self._position_fills[fill.product_id].append(fill)


def _check_context_protocol(ctx: TradingContext) -> None:  # pragma: no cover
    """Compile-time check that _BacktestContext satisfies TradingContext."""


_check_context_protocol(_BacktestContext.__new__(_BacktestContext))


class BacktestEngine:
    """Runs a backtest against historical market data (DA and/or IDC).

    The engine detects market mode from the product specs:

    - Products ending in ``_DA`` use the DA auction matching engine.
    - Products containing ``-QH`` use the IDC continuous matching engine.
    - Mixed: both engines run in parallel sharing the same simulated clock.

    For DA products the engine:

    1. Loads clearing prices from ``{data_dir}/da_prices.parquet``.
    2. Advances the simulated clock to D-1 12:00 UTC for each delivery day.
    3. Dispatches signal updates.
    4. Calls ``on_auction_open`` once per product.
    5. Matches all orders against the clearing price (price-taker).
    6. Calls ``on_fill`` for each fill.

    For IDC products the engine:

    1. Loads event data from ``{data_dir}/idc_events/{zone}_*.parquet``
       via a :class:`~nexa_backtest.data.window.SlidingWindow`.
    2. Advances the simulated clock to each 15-minute MTU boundary.
    3. Replays historical events through the
       :class:`~nexa_backtest.engines.matching.ContinuousMatchingEngine`.
    4. Calls ``on_fill`` for any fills triggered by historical events.
    5. Calls ``on_bar`` at each MTU boundary.
    6. Processes orders placed during ``on_bar`` against the current book.
    7. Checks gate closures and fires ``on_gate_closure``/``on_cancel``.

    Args:
        algo: The trading algorithm to run.
        exchange: Exchange identifier, e.g. ``"nordpool"``.
        start: First delivery date to include.
        end: Last delivery date to include (inclusive).
        products: Product specifications, e.g. ``["NO1_DA"]`` or
            ``["NO1-QH"]``.
        data_dir: Directory containing market data files.
        capital: Starting capital in EUR (informational).
        signals: Explicitly constructed signal providers.

    Example::

        engine = BacktestEngine(
            algo=MyAlgo(),
            exchange="nordpool",
            start=date(2026, 3, 1),
            end=date(2026, 3, 31),
            products=["NO1-QH"],
            data_dir=Path("data/nordpool/"),
            capital=Decimal("100000"),
        )
        result = engine.run()
        print(result.summary())
    """

    def __init__(
        self,
        algo: SimpleAlgo,
        exchange: str,
        start: date,
        end: date,
        products: list[str],
        data_dir: Path,
        capital: Decimal,
        signals: list[SignalProvider] | None = None,
    ) -> None:
        self._algo = algo
        self._exchange = exchange
        self._start = start
        self._end = end
        self._products = products
        self._data_dir = data_dir
        self._capital = capital
        self._signals: list[SignalProvider] = signals or []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> BacktestResult:
        """Execute the backtest and return results.

        Returns:
            :class:`~nexa_backtest.analysis.metrics.BacktestResult`.

        Raises:
            :class:`~nexa_backtest.exceptions.DataError`: If required data
                files are missing or malformed.
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

        # Build signal registry
        registry = SignalRegistry()
        for provider in self._signals:
            registry.register(provider)

        # Determine the zone (first product wins)
        zone = _parse_zone(self._products[0])

        all_fills: list[Fill] = []
        equity_snapshots: list[EquitySnapshot] = []
        cumulative_pnl = Decimal("0")

        # --- DA run ---
        if da_products:
            da_fills, da_snapshots, da_pnl, context, _, market_vwap = self._run_da(zone, registry)
            all_fills.extend(da_fills)
            equity_snapshots.extend(da_snapshots)
            cumulative_pnl += da_pnl
        else:
            context = None
            market_vwap = None

        # --- IDC run ---
        if idc_products:
            idc_fills, idc_snapshots, idc_pnl, idc_context, _ = self._run_idc(zone, registry)
            all_fills.extend(idc_fills)
            equity_snapshots.extend(idc_snapshots)
            cumulative_pnl += idc_pnl
            if context is None:
                context = idc_context

        # Compute analysis
        market_data = self._load_da_or_empty(zone, market_vwap)
        if market_vwap is None:
            market_vwap = compute_market_vwap(market_data) if len(market_data) > 0 else Decimal("0")
        pnl = compute_pnl(all_fills, market_data)

        equity_snapshots.sort(key=lambda s: s.timestamp)

        sharpe = compute_sharpe(equity_snapshots)
        max_dd, max_dd_pct = compute_max_drawdown(equity_snapshots)
        profit_fac = compute_profit_factor(all_fills, market_vwap or Decimal("0"))

        avg_trade = pnl.total_alpha_eur / Decimal(len(all_fills)) if all_fills else Decimal("0")

        best_fill: Fill | None = None
        worst_fill: Fill | None = None
        if all_fills:
            mv = market_vwap or Decimal("0")
            fill_pnls = [(f, compute_fill_pnl(f, mv)) for f in all_fills]
            best_fill = max(fill_pnls, key=lambda x: x[1])[0]
            worst_fill = min(fill_pnls, key=lambda x: x[1])[0]

        duration = (self._end - self._start).days + 1

        return BacktestResult(
            algo_name=type(self._algo).__name__,
            exchange=self._exchange,
            start=self._start,
            end=self._end,
            fills=tuple(all_fills),
            pnl=pnl,
            equity_curve=tuple(equity_snapshots),
            initial_capital=self._capital,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            max_drawdown_pct=max_dd_pct,
            profit_factor=profit_fac,
            avg_trade_pnl=avg_trade,
            best_trade=best_fill,
            worst_trade=worst_fill,
            duration_days=duration,
        )

    # ------------------------------------------------------------------
    # DA backtest loop
    # ------------------------------------------------------------------

    def _run_da(
        self,
        zone: str,
        registry: SignalRegistry,
    ) -> tuple[
        list[Fill],
        list[EquitySnapshot],
        Decimal,
        _BacktestContext,
        SimulatedClock,
        Decimal,
    ]:
        """Run the DA auction backtest loop."""
        loader = ParquetLoader(self._data_dir)
        market_data = loader.load_da_prices(zone, self._start, self._end)

        first_auction = self._auction_time(self._start)
        clock = SimulatedClock(initial_time=first_auction - timedelta(minutes=1))
        context = _BacktestContext(clock=clock, signal_registry=registry)

        self._algo.on_setup(context)
        self._discover_signals(registry)

        market_vwap = compute_market_vwap(market_data)
        market_data["delivery_date"] = market_data["timestamp"].dt.date

        all_fills: list[Fill] = []
        equity_snapshots: list[EquitySnapshot] = []
        cumulative_pnl = Decimal("0")

        for delivery_date, day_data in market_data.groupby("delivery_date"):
            auction_time = self._auction_time(delivery_date)  # type: ignore[arg-type]
            clock.advance_to(auction_time)

            context._reset_day()
            gate_closure = auction_time + _GATE_CLOSURE_OFFSET
            for _, row in day_data.iterrows():
                pid: str = str(row["product_id"])
                context._clearing_prices[pid] = Decimal(str(row["price_eur_mwh"]))
                context._gate_closures[pid] = gate_closure

            for signal_name in self._algo._subscribed_signals:
                if registry.has(signal_name):
                    try:
                        value = context.get_signal(signal_name)
                        self._algo.on_signal(context, signal_name, value)
                    except SignalError:
                        logger.debug(
                            "No value yet for signal '%s' at %s — skipping.",
                            signal_name,
                            auction_time.isoformat(),
                        )

            for _, product_row in day_data.sort_values("timestamp").iterrows():
                auction_info = AuctionInfo(
                    product_id=str(product_row["product_id"]),
                    auction_type="DA",
                    gate_closure_time=gate_closure,
                    zone=zone,
                )
                self._algo.on_auction_open(context, auction_info)

            fill_time = gate_closure
            orders = list(context._pending_orders.values())
            context._pending_orders.clear()

            day_fills: list[Fill] = []
            for order in orders:
                clearing = context._clearing_prices.get(order.product_id)
                if clearing is None:
                    logger.warning("Order for unknown product '%s' rejected.", order.product_id)
                    continue
                matcher = DAAuctionMatcher(clearing_price=clearing, fill_timestamp=fill_time)
                match_result = matcher.match(order)
                if match_result.fill is not None:
                    context._record_fill(match_result.fill)
                    all_fills.append(match_result.fill)
                    day_fills.append(match_result.fill)
                    self._algo.on_fill(context, match_result.fill)

            if day_fills:
                day_pnl = sum(
                    (compute_fill_pnl(f, market_vwap) for f in day_fills),
                    Decimal("0"),
                )
                cumulative_pnl += day_pnl
                equity_snapshots.append(
                    EquitySnapshot(
                        timestamp=fill_time,
                        realised_pnl=cumulative_pnl,
                        unrealised_pnl=Decimal("0"),
                        total_equity=self._capital + cumulative_pnl,
                        cash=self._capital + cumulative_pnl,
                        net_position_mw=Decimal("0"),
                    )
                )

        self._algo.on_teardown(context)
        return all_fills, equity_snapshots, cumulative_pnl, context, clock, market_vwap

    # ------------------------------------------------------------------
    # IDC backtest loop
    # ------------------------------------------------------------------

    def _run_idc(
        self,
        zone: str,
        registry: SignalRegistry,
    ) -> tuple[
        list[Fill],
        list[EquitySnapshot],
        Decimal,
        _BacktestContext,
        SimulatedClock,
    ]:
        """Run the IDC continuous backtest loop."""
        # Build sliding window
        try:
            manifest = DataManifest(data_dir=self._data_dir, zone=zone)
        except FileNotFoundError as exc:
            raise DataError(str(exc)) from exc

        window = SlidingWindow(manifest=manifest)
        matching_engine = ContinuousMatchingEngine()

        start_dt = datetime.combine(self._start, time(0, 0), tzinfo=UTC)
        clock = SimulatedClock(initial_time=start_dt - timedelta(minutes=1))
        context = _BacktestContext(
            clock=clock, signal_registry=registry, idc_engine=matching_engine
        )

        self._algo.on_setup(context)
        self._discover_signals(registry)

        # Register gate closures for all products that will trade
        end_dt = datetime.combine(self._end + timedelta(days=1), time(0, 0), tzinfo=UTC)
        self._register_idc_gate_closures(matching_engine, start_dt, end_dt, zone)

        all_fills: list[Fill] = []
        equity_snapshots: list[EquitySnapshot] = []
        cumulative_pnl = Decimal("0")

        current_time = start_dt
        while current_time < end_dt:
            next_time = current_time + _MTU_DURATION
            clock.advance_to(current_time)

            # Advance window and process historical events for this MTU
            window.advance_to(current_time)
            for event in window.events_between(current_time, next_time):
                try:
                    fills = matching_engine.process_historical_event(event)
                except Exception as exc:
                    with contextlib.suppress(Exception):
                        self._algo.on_error(context, exc)
                    continue
                for fill in fills:
                    context._record_fill(fill)
                    all_fills.append(fill)
                    self._algo.on_fill(context, fill)

            # Signal updates at each bar
            for signal_name in self._algo._subscribed_signals:
                if registry.has(signal_name):
                    try:
                        value = context.get_signal(signal_name)
                        self._algo.on_signal(context, signal_name, value)
                    except SignalError:
                        pass

            # on_bar: algo evaluates market and places orders
            self._algo.on_bar(context)

            # Process orders placed during on_bar
            bar_fills: list[Fill] = []
            pending = list(context._idc_pending_orders.values())
            context._idc_pending_orders.clear()
            for order in pending:
                fills = matching_engine.place_algo_order(order, current_time)
                for fill in fills:
                    context._record_fill(fill)
                    all_fills.append(fill)
                    bar_fills.append(fill)
                    self._algo.on_fill(context, fill)

            # Gate closure checks
            closed_products = matching_engine.check_gate_closures(current_time)
            for product_id in closed_products:
                self._algo.on_gate_closure(context, product_id)
                # on_cancel for any orders that were cancelled by gate closure
                # (they were already removed from matching_engine in check_gate_closures)

            # Equity snapshot at each bar where fills occurred
            bar_all_fills = [f for f in all_fills if _in_mtu(f.timestamp, current_time, next_time)]
            if bar_all_fills:
                # Use zero VWAP as fallback for IDC — we don't have market-wide VWAP
                bar_pnl = sum(
                    (compute_fill_pnl(f, Decimal("0")) for f in bar_all_fills),
                    Decimal("0"),
                )
                cumulative_pnl += bar_pnl
                unrealised = context.get_unrealised_pnl()
                net_mw = sum(
                    (p.net_mw for p in context.get_all_positions().values()),
                    Decimal("0"),
                )
                equity_snapshots.append(
                    EquitySnapshot(
                        timestamp=current_time,
                        realised_pnl=cumulative_pnl,
                        unrealised_pnl=unrealised,
                        total_equity=self._capital + cumulative_pnl + unrealised,
                        cash=self._capital + cumulative_pnl,
                        net_position_mw=net_mw,
                    )
                )

            current_time = next_time

        self._algo.on_teardown(context)
        return all_fills, equity_snapshots, cumulative_pnl, context, clock

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_da_or_empty(self, zone: str, market_vwap: Decimal | None) -> Any:
        """Load DA prices if available, else return an empty DataFrame."""
        import pandas as pd

        try:
            loader = ParquetLoader(self._data_dir)
            return loader.load_da_prices(zone, self._start, self._end)
        except DataError:
            return pd.DataFrame(
                columns=["product_id", "timestamp", "zone", "price_eur_mwh", "volume_mwh"]
            )

    @staticmethod
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

    def _discover_signals(self, registry: SignalRegistry) -> None:
        """Auto-register CSV providers for subscribed but unregistered signals."""
        signals_dir = self._data_dir / "signals"
        for signal_name in self._algo._subscribed_signals:
            if registry.has(signal_name):
                continue
            csv_path = signals_dir / f"{signal_name}.csv"
            if not csv_path.exists():
                raise DataError(
                    f"Signal '{signal_name}' is subscribed by the algo but no CSV "
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

    @staticmethod
    def _auction_time(delivery_date: date) -> datetime:
        """Return D-1 12:00 UTC as the simulated clock time for the DA auction."""
        delivery_dt = datetime.combine(delivery_date, time(0, 0), tzinfo=UTC)
        return delivery_dt + _AUCTION_OFFSET

    def _parse_zone(self) -> str:
        """Extract zone from the first product spec."""
        return _parse_zone(self._products[0])


def _mtu_to_product_id(mtu_start: datetime, zone: str) -> str:
    """Convert an MTU start time to a Nord Pool IDC product ID.

    Example::

        datetime(2026, 3, 1, 9, 0, tzinfo=UTC), "NO1" → "NO1-QH-0900"
    """
    return f"{zone}-QH-{mtu_start.hour:02d}{mtu_start.minute:02d}"


def _in_mtu(ts: datetime, mtu_start: datetime, mtu_end: datetime) -> bool:
    """Return True if ``ts`` falls within ``[mtu_start, mtu_end)``."""
    return mtu_start <= ts < mtu_end
