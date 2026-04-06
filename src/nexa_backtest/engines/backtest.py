"""BacktestEngine: simulated clock, DA auction replay, and signal dispatch.

The engine loads historical clearing prices, advances a simulated clock
through each delivery day, calls the algo's lifecycle hooks, matches orders
against historical clearing prices, and returns a :class:`BacktestResult`.

The engine also implements :class:`~nexa_backtest.context.TradingContext` via
an internal :class:`_BacktestContext` object that is passed to the algo. Algo
code never sees the engine directly.
"""

from __future__ import annotations

import logging
import uuid
from collections import defaultdict
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

from nexa_backtest.algo import SimpleAlgo
from nexa_backtest.analysis.metrics import BacktestResult
from nexa_backtest.analysis.pnl import compute_pnl
from nexa_backtest.context import SignalValue, TradingContext
from nexa_backtest.data.loader import ParquetLoader
from nexa_backtest.engines.clock import SimulatedClock
from nexa_backtest.engines.matching import DAAuctionMatcher
from nexa_backtest.exceptions import DataError, SignalError
from nexa_backtest.signals.base import SignalProvider
from nexa_backtest.signals.csv_loader import CsvSignalProvider
from nexa_backtest.signals.registry import SignalRegistry
from nexa_backtest.types import (
    MTU,
    AuctionInfo,
    CancelResult,
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
# We set the clock to D-1 12:00 UTC so publication_offset-based filters work
# correctly for day-ahead forecasts.
_AUCTION_OFFSET = timedelta(days=-1, hours=12)  # D-1 12:00 UTC
_GATE_CLOSURE_OFFSET = timedelta(hours=1)  # gate closes 1h after auction opens


def _zero_position(product_id: str) -> Position:
    return Position(
        product_id=product_id,
        net_mw=Decimal("0"),
        avg_entry_price=Decimal("0"),
        unrealised_pnl=Decimal("0"),
    )


class _BacktestContext:
    """Internal :class:`~nexa_backtest.context.TradingContext` implementation.

    Created by :class:`BacktestEngine` for each run. The algo receives this
    object via every lifecycle hook but never knows its concrete type.
    """

    def __init__(
        self,
        clock: SimulatedClock,
        signal_registry: SignalRegistry,
    ) -> None:
        self._clock = clock
        self._signal_registry = signal_registry

        # Mutable state updated by the engine between calls
        self._pending_orders: dict[str, Order] = {}
        self._fills: list[Fill] = []
        self._position_fills: dict[str, list[Fill]] = defaultdict(list)
        self._clearing_prices: dict[str, Decimal] = {}
        self._gate_closures: dict[str, datetime] = {}

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
        return timedelta(0)

    def current_mtu(self) -> MTU:
        """Return the MTU whose delivery period contains ``now()``."""
        now = self._clock.now()
        minute_slot = (now.minute // 15) * 15
        start = now.replace(minute=minute_slot, second=0, microsecond=0)
        end = start + timedelta(minutes=15)
        return MTU(start=start, end=end, zone="")

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------

    def get_orderbook(self, product_id: str) -> OrderBook:
        """Return a synthetic order book based on the DA clearing price."""
        price = self._clearing_prices.get(product_id)
        ts = self._clock.now()
        if price is None:
            return OrderBook(
                product_id=product_id,
                best_bid=None,
                best_ask=None,
                timestamp=ts,
            )
        level = PriceLevel(price=price, volume=Decimal("1000"))
        return OrderBook(
            product_id=product_id,
            best_bid=level,
            best_ask=level,
            timestamp=ts,
        )

    def get_best_bid(self, product_id: str) -> PriceLevel | None:
        """Return the best bid for ``product_id``."""
        return self.get_orderbook(product_id).best_bid

    def get_best_ask(self, product_id: str) -> PriceLevel | None:
        """Return the best ask for ``product_id``."""
        return self.get_orderbook(product_id).best_ask

    def get_last_price(self, product_id: str) -> Decimal | None:
        """Return the last DA clearing price for ``product_id``."""
        return self._clearing_prices.get(product_id)

    def get_vwap(self, product_id: str) -> Decimal | None:
        """Return the DA clearing price as session VWAP for ``product_id``."""
        return self._clearing_prices.get(product_id)

    # ------------------------------------------------------------------
    # Order management
    # ------------------------------------------------------------------

    def place_order(self, order: Order) -> OrderResult:
        """Accept an order for deferred matching against the clearing price."""
        self._pending_orders[order.order_id] = order
        return OrderResult(order_id=order.order_id, status=OrderStatus.ACCEPTED)

    def cancel_order(self, order_id: str) -> CancelResult:
        """Cancel a pending order before it is matched."""
        if order_id in self._pending_orders:
            del self._pending_orders[order_id]
            return CancelResult(order_id=order_id, status="cancelled")
        return CancelResult(order_id=order_id, status="not_found")

    def modify_order(self, order_id: str, **changes: Any) -> OrderResult:
        """Cancel and resubmit an order with updated fields."""
        if order_id not in self._pending_orders:
            return OrderResult(
                order_id=order_id,
                status=OrderStatus.REJECTED,
                rejection_reason=f"Order '{order_id}' not found in pending orders.",
            )
        old_order = self._pending_orders.pop(order_id)
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
    """Runs a DA backtest against historical clearing price data.

    The engine:

    1. Loads DA clearing prices from ``{data_dir}/da_prices.parquet``.
    2. Advances the simulated clock to D-1 12:00 UTC for each delivery day.
    3. Dispatches signal updates to the algo for subscribed signals.
    4. Calls :meth:`~nexa_backtest.algo.SimpleAlgo.on_auction_open` once per
       delivery product.
    5. Matches all orders placed during those calls against the clearing price
       using the price-taker assumption.
    6. Calls :meth:`~nexa_backtest.algo.SimpleAlgo.on_fill` for each fill.
    7. Returns a :class:`~nexa_backtest.analysis.metrics.BacktestResult`.

    **Signal auto-discovery**

    If the algo calls :meth:`~nexa_backtest.algo.SimpleAlgo.subscribe_signal`
    during :meth:`~nexa_backtest.algo.SimpleAlgo.on_setup` and no explicit
    provider is registered for that name, the engine looks for::

        {data_dir}/signals/{signal_name}.csv

    A :class:`~nexa_backtest.exceptions.DataError` is raised if the file is
    not found.

    Args:
        algo: The trading algorithm to run.
        exchange: Exchange identifier, e.g. ``"nordpool"``.
        start: First delivery date to include.
        end: Last delivery date to include (inclusive).
        products: Product specifications, e.g. ``["NO1_DA"]``.
            Format is ``{zone}_{type}`` where type is ``DA``.
        data_dir: Directory containing ``da_prices.parquet`` and optionally a
            ``signals/`` subdirectory.
        capital: Starting capital in EUR (informational only for now).
        signals: Explicitly constructed signal providers. Auto-discovered CSV
            providers are added for any subscribed signals not covered here.

    Example::

        engine = BacktestEngine(
            algo=MyAlgo(),
            exchange="nordpool",
            start=date(2026, 3, 1),
            end=date(2026, 3, 31),
            products=["NO1_DA"],
            data_dir=Path("data/"),
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
            :class:`~nexa_backtest.analysis.metrics.BacktestResult` containing
            all fills, PnL metrics, and VWAP benchmarking.

        Raises:
            :class:`~nexa_backtest.exceptions.DataError`: If required data
                files are missing or malformed.
        """
        zone = self._parse_zone()

        # Load market data
        loader = ParquetLoader(self._data_dir)
        market_data = loader.load_da_prices(zone, self._start, self._end)

        # Build signal registry from explicitly passed providers
        registry = SignalRegistry()
        for provider in self._signals:
            registry.register(provider)

        # Initialise clock to just before the first auction (D-1 12:00 UTC of start)
        first_auction = self._auction_time(self._start)
        clock = SimulatedClock(initial_time=first_auction - timedelta(minutes=1))
        context = _BacktestContext(clock=clock, signal_registry=registry)

        # on_setup: algo registers signal subscriptions and sets state
        self._algo.on_setup(context)

        # Auto-discover CSVs for any subscribed signals not yet registered
        self._discover_signals(registry)

        # Group products by delivery day
        market_data["delivery_date"] = market_data["timestamp"].dt.date

        all_fills: list[Fill] = []

        for delivery_date, day_data in market_data.groupby("delivery_date"):
            auction_time = self._auction_time(delivery_date)  # type: ignore[arg-type]
            clock.advance_to(auction_time)

            # Populate clearing prices and gate closures for this day
            context._reset_day()
            gate_closure = auction_time + _GATE_CLOSURE_OFFSET
            for _, row in day_data.iterrows():
                pid: str = str(row["product_id"])
                context._clearing_prices[pid] = Decimal(str(row["price_eur_mwh"]))
                context._gate_closures[pid] = gate_closure

            # Emit on_signal for each subscribed signal
            for signal_name in self._algo._subscribed_signals:
                if registry.has(signal_name):
                    try:
                        value = context.get_signal(signal_name)
                        self._algo.on_signal(context, signal_name, value)
                    except SignalError:
                        logger.debug(
                            "No value yet for signal '%s' at %s — skipping on_signal.",
                            signal_name,
                            auction_time.isoformat(),
                        )

            # Call on_auction_open for every product in this day
            for _, product_row in day_data.sort_values("timestamp").iterrows():
                auction_info = AuctionInfo(
                    product_id=str(product_row["product_id"]),
                    auction_type="DA",
                    gate_closure_time=gate_closure,
                    zone=zone,
                )
                self._algo.on_auction_open(context, auction_info)

            # Match all pending orders against clearing prices
            fill_time = gate_closure
            orders = list(context._pending_orders.values())
            context._pending_orders.clear()

            for order in orders:
                clearing = context._clearing_prices.get(order.product_id)
                if clearing is None:
                    logger.warning(
                        "Order for unknown product '%s' rejected — no clearing price.",
                        order.product_id,
                    )
                    continue

                matcher = DAAuctionMatcher(clearing_price=clearing, fill_timestamp=fill_time)
                result = matcher.match(order)

                if result.fill is not None:
                    context._record_fill(result.fill)
                    all_fills.append(result.fill)
                    self._algo.on_fill(context, result.fill)

        self._algo.on_teardown(context)

        pnl = compute_pnl(all_fills, market_data)

        return BacktestResult(
            algo_name=type(self._algo).__name__,
            exchange=self._exchange,
            start=self._start,
            end=self._end,
            fills=tuple(all_fills),
            pnl=pnl,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _parse_zone(self) -> str:
        """Extract zone from the first product spec, e.g. ``"NO1_DA"`` -> ``"NO1"``."""
        if not self._products:
            raise DataError("No products specified. Pass e.g. products=['NO1_DA'].")
        spec = self._products[0]
        parts = spec.split("_")
        if len(parts) < 2:
            raise DataError(
                f"Cannot parse product spec '{spec}'. "
                "Expected format: {{zone}}_{{type}}, e.g. 'NO1_DA'."
            )
        return parts[0]

    @staticmethod
    def _auction_time(delivery_date: date) -> datetime:
        """Return the simulated clock time for the DA auction of ``delivery_date``.

        Set to D-1 12:00 UTC, which matches Nord Pool DA gate closure and
        ensures forecasts with 36h+ publication_offset are visible for all
        delivery products on day D.
        """
        delivery_dt = datetime.combine(delivery_date, time(0, 0), tzinfo=UTC)
        return delivery_dt + _AUCTION_OFFSET

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
                    f"was found at '{csv_path}'. Either create the file or pass a "
                    f"SignalProvider explicitly to BacktestEngine."
                )
            provider = CsvSignalProvider(
                name=signal_name,
                path=csv_path,
                unit="",
                description=f"Auto-loaded from {csv_path.name}",
            )
            registry.register(provider)
            logger.info("Auto-loaded signal '%s' from %s", signal_name, csv_path)
