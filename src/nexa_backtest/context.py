"""TradingContext protocol definition.

This module defines the interface that algo code uses to interact with the
market. The same protocol is implemented by the backtest, paper, and live
engines, so algo code never needs to know which engine is running underneath.

The protocol is defined using ``typing.Protocol`` for structural subtyping,
enabling mypy to verify compliance without inheritance.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict

from nexa_backtest.types import (
    MTU,
    CancelResult,
    Order,
    OrderBook,
    OrderResult,
    Position,
    PriceLevel,
)


class SignalValue(BaseModel):
    """A single timestamped signal observation.

    Attributes:
        name: Signal identifier, e.g. ``"wind_generation_forecast"``.
        timestamp: Timezone-aware time this value is valid for.
        value: Numeric signal value. Units depend on the signal definition.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    timestamp: datetime
    value: float


class TradingContext(Protocol):
    """The interface exposed to algo code.

    All methods are documented with their semantics under backtesting,
    paper trading, and live trading to make the invariants clear.

    Algo code must only interact with the market through this protocol.
    Importing engine-specific classes directly breaks the engine-agnostic
    contract and will be caught by the validation pipeline.
    """

    # ------------------------------------------------------------------
    # Time
    # ------------------------------------------------------------------

    def now(self) -> datetime:
        """Return the current simulated or wall-clock time.

        Returns:
            Timezone-aware current time. Under the backtest engine this is
            the simulated clock time. Under paper/live engines this is the
            real wall-clock time.
        """
        ...

    def time_to_gate_closure(self, product_id: str) -> timedelta:
        """Return time remaining until gate closure for a product.

        Args:
            product_id: Exchange product identifier.

        Returns:
            Remaining time as a timedelta. Returns ``timedelta(0)`` if gate
            has already closed.
        """
        ...

    def current_mtu(self) -> MTU:
        """Return the current market time unit.

        Returns:
            The MTU whose delivery period contains ``now()``.
        """
        ...

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------

    def get_orderbook(self, product_id: str) -> OrderBook:
        """Return the current order book snapshot for a product.

        Args:
            product_id: Exchange product identifier.

        Returns:
            Order book snapshot with best bid and ask.
        """
        ...

    def get_best_bid(self, product_id: str) -> PriceLevel | None:
        """Return the best bid for a product.

        Args:
            product_id: Exchange product identifier.

        Returns:
            Best bid price level, or ``None`` if no bids exist.
        """
        ...

    def get_best_ask(self, product_id: str) -> PriceLevel | None:
        """Return the best ask for a product.

        Args:
            product_id: Exchange product identifier.

        Returns:
            Best ask price level, or ``None`` if no offers exist.
        """
        ...

    def get_last_price(self, product_id: str) -> Decimal | None:
        """Return the last traded price for a product.

        Args:
            product_id: Exchange product identifier.

        Returns:
            Last traded price in EUR/MWh, or ``None`` if no trades yet.
        """
        ...

    def get_vwap(self, product_id: str) -> Decimal | None:
        """Return the volume-weighted average price for a product.

        Computed over all trades in the current session.

        Args:
            product_id: Exchange product identifier.

        Returns:
            VWAP in EUR/MWh, or ``None`` if no trades yet.
        """
        ...

    # ------------------------------------------------------------------
    # Order management
    # ------------------------------------------------------------------

    def place_order(self, order: Order) -> OrderResult:
        """Submit an order to the exchange.

        Args:
            order: The order to submit. Use ``Order.buy()``, ``Order.sell()``,
                ``Order.market()``, or ``Order.block_bid()`` to construct.

        Returns:
            Result containing the order status and fill details if immediately matched.
        """
        ...

    def cancel_order(self, order_id: str) -> CancelResult:
        """Cancel a previously submitted order.

        Args:
            order_id: ID of the order to cancel.

        Returns:
            Result with status ``"cancelled"`` or ``"not_found"``.
        """
        ...

    def modify_order(self, order_id: str, **changes: Any) -> OrderResult:
        """Modify a previously submitted order.

        Equivalent to cancel + resubmit. The original order ID is retired and
        a new order is created with the updated fields.

        Args:
            order_id: ID of the order to modify.
            **changes: Fields to update, e.g. ``price_eur_mwh=42.5``.

        Returns:
            Result for the new replacement order.
        """
        ...

    # ------------------------------------------------------------------
    # Positions
    # ------------------------------------------------------------------

    def get_position(self, product_id: str) -> Position:
        """Return the current position for a product.

        Args:
            product_id: Exchange product identifier.

        Returns:
            Position with net volume, average entry price, and unrealised PnL.
            Returns a zero position if no trades have been made.
        """
        ...

    def get_all_positions(self) -> dict[str, Position]:
        """Return all current positions keyed by product ID.

        Returns:
            Dictionary mapping product IDs to positions. Only includes products
            with non-zero positions.
        """
        ...

    def get_unrealised_pnl(self) -> Decimal:
        """Return total unrealised PnL across all positions.

        Returns:
            Unrealised PnL in EUR. Positive means profit.
        """
        ...

    # ------------------------------------------------------------------
    # Signals
    # ------------------------------------------------------------------

    def get_signal(self, name: str) -> SignalValue:
        """Return the most recent value for a named signal.

        Respects the signal's ``publication_offset`` to prevent look-ahead
        bias: returns the value that was *known* at ``now()``, not a future
        value.

        Args:
            name: Signal name, e.g. ``"wind_generation_forecast"``.

        Returns:
            Most recent signal value valid at the current simulated time.
        """
        ...

    def get_signal_history(self, name: str, lookback: int) -> list[SignalValue]:
        """Return the most recent N values for a named signal.

        Args:
            name: Signal name.
            lookback: Number of historical values to return, most recent last.

        Returns:
            List of signal values in chronological order.
        """
        ...

    # ------------------------------------------------------------------
    # ML models
    # ------------------------------------------------------------------

    def predict(self, model_name: str, features: dict[str, Any]) -> Any:
        """Run inference on a registered ML model.

        Args:
            model_name: Name the model was registered under in ``ModelRegistry``.
            features: Input feature dictionary matching the model's input schema.

        Returns:
            Model output. Type depends on the model's output schema.
        """
        ...

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def log(self, message: str, level: str = "info") -> None:
        """Emit a log message tagged with the current simulated time.

        Args:
            message: Human-readable message.
            level: Log level string: ``"debug"``, ``"info"``, ``"warning"``,
                or ``"error"``. Defaults to ``"info"``.
        """
        ...
