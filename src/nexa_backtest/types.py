"""Core domain types for nexa-backtest.

All types are immutable data containers using Pydantic v2 with frozen=True.
Prices and volumes use decimal.Decimal to avoid floating-point precision issues.
All datetimes are timezone-aware.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, computed_field


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


class Side(StrEnum):
    """Direction of a trade or order."""

    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(StrEnum):
    """Lifecycle state of an order."""

    PENDING = "PENDING"
    ACCEPTED = "ACCEPTED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


class MTU(BaseModel):
    """A market time unit representing a 15-minute delivery period.

    EU power markets use 15-minute MTUs as of 30 Sept 2025. The backtester
    also supports hourly resolution for pre-2025 data.

    Attributes:
        start: Timezone-aware start of the delivery period.
        end: Timezone-aware end of the delivery period.
        zone: Bidding zone identifier, e.g. ``"NO1"``.
    """

    model_config = ConfigDict(frozen=True)

    start: datetime
    end: datetime
    zone: str


class PriceLevel(BaseModel):
    """A single price level in an order book (bid or ask side).

    Attributes:
        price: Price in EUR/MWh.
        volume: Available volume in MW.
    """

    model_config = ConfigDict(frozen=True)

    price: Decimal
    volume: Decimal


class OrderBook(BaseModel):
    """Snapshot of the order book for a product.

    Attributes:
        product_id: Exchange product identifier, e.g. ``"NO1-QH-0900"``.
        bids: All bid price levels, sorted best (highest price) first.
        asks: All ask price levels, sorted best (lowest price) first.
        timestamp: Timezone-aware time at which this snapshot was taken.
        best_bid: Best (highest) bid level, or ``None`` if no bids.
        best_ask: Best (lowest) ask level, or ``None`` if no asks.
        spread: ``best_ask.price - best_bid.price``, or ``None`` if either
            side is empty.
        mid_price: ``(best_bid.price + best_ask.price) / 2``, or ``None``
            if either side is empty.
    """

    model_config = ConfigDict(frozen=True)

    product_id: str
    bids: list[PriceLevel]
    asks: list[PriceLevel]
    timestamp: datetime

    @computed_field  # type: ignore[prop-decorator]
    @property
    def best_bid(self) -> PriceLevel | None:
        """Highest bid price level, or ``None`` if the bid side is empty."""
        return self.bids[0] if self.bids else None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def best_ask(self) -> PriceLevel | None:
        """Lowest ask price level, or ``None`` if the ask side is empty."""
        return self.asks[0] if self.asks else None

    @property
    def spread(self) -> Decimal | None:
        """Best ask minus best bid. ``None`` if either side is empty."""
        if self.best_bid and self.best_ask:
            return self.best_ask.price - self.best_bid.price
        return None

    @property
    def mid_price(self) -> Decimal | None:
        """Mid-point price. ``None`` if either side is empty."""
        if self.best_bid and self.best_ask:
            return (self.best_bid.price + self.best_ask.price) / Decimal("2")
        return None


class Order(BaseModel):
    """A trading order submitted to an exchange.

    Use the class methods ``Order.buy()``, ``Order.sell()``,
    ``Order.market()``, and ``Order.block_bid()`` to construct orders rather
    than instantiating directly.

    Attributes:
        order_id: Unique identifier assigned at creation.
        product_id: Exchange product identifier.
        side: ``Side.BUY`` or ``Side.SELL``.
        volume_mw: Order volume in MW. Must be positive.
        price_eur_mwh: Limit price in EUR/MWh, or ``None`` for market orders.
        is_block_bid: Whether this is a block bid spanning multiple MTUs.
        block_start: Start of block delivery period (block bids only).
        block_end: End of block delivery period (block bids only).
    """

    model_config = ConfigDict(frozen=True)

    order_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    product_id: str
    side: Side
    volume_mw: Decimal
    price_eur_mwh: Decimal | None = None
    is_block_bid: bool = False
    block_start: datetime | None = None
    block_end: datetime | None = None

    @classmethod
    def buy(
        cls,
        product_id: str,
        volume_mw: Decimal | float | int,
        price_eur_mwh: Decimal | float | int,
    ) -> Order:
        """Create a limit buy order.

        Args:
            product_id: Exchange product identifier.
            volume_mw: Order volume in MW.
            price_eur_mwh: Maximum price willing to pay in EUR/MWh.

        Returns:
            A new buy limit order.
        """
        return cls(
            product_id=product_id,
            side=Side.BUY,
            volume_mw=Decimal(str(volume_mw)),
            price_eur_mwh=Decimal(str(price_eur_mwh)),
        )

    @classmethod
    def sell(
        cls,
        product_id: str,
        volume_mw: Decimal | float | int,
        price_eur_mwh: Decimal | float | int,
    ) -> Order:
        """Create a limit sell order.

        Args:
            product_id: Exchange product identifier.
            volume_mw: Order volume in MW.
            price_eur_mwh: Minimum price willing to accept in EUR/MWh.

        Returns:
            A new sell limit order.
        """
        return cls(
            product_id=product_id,
            side=Side.SELL,
            volume_mw=Decimal(str(volume_mw)),
            price_eur_mwh=Decimal(str(price_eur_mwh)),
        )

    @classmethod
    def market(
        cls,
        product_id: str,
        side: Side,
        volume_mw: Decimal | float | int,
    ) -> Order:
        """Create a market order with no price limit.

        Args:
            product_id: Exchange product identifier.
            side: ``Side.BUY`` or ``Side.SELL``.
            volume_mw: Order volume in MW.

        Returns:
            A new market order.
        """
        return cls(
            product_id=product_id,
            side=side,
            volume_mw=Decimal(str(volume_mw)),
            price_eur_mwh=None,
        )

    @classmethod
    def block_bid(
        cls,
        product_id: str,
        side: Side,
        volume_mw: Decimal | float | int,
        price_eur_mwh: Decimal | float | int,
        block_start: datetime,
        block_end: datetime,
    ) -> Order:
        """Create a block bid spanning multiple MTUs.

        Block bids are executed only if the average clearing price across all
        MTUs in the block meets the limit price.

        Args:
            product_id: Exchange product identifier for the block.
            side: ``Side.BUY`` or ``Side.SELL``.
            volume_mw: Flat volume in MW across all MTUs.
            price_eur_mwh: Limit price in EUR/MWh.
            block_start: Timezone-aware start of the block delivery period.
            block_end: Timezone-aware end of the block delivery period.

        Returns:
            A new block bid order.
        """
        return cls(
            product_id=product_id,
            side=side,
            volume_mw=Decimal(str(volume_mw)),
            price_eur_mwh=Decimal(str(price_eur_mwh)),
            is_block_bid=True,
            block_start=block_start,
            block_end=block_end,
        )


class Fill(BaseModel):
    """Record of a trade execution.

    Attributes:
        order_id: ID of the order that was filled.
        product_id: Exchange product identifier.
        price: Execution price in EUR/MWh.
        volume: Filled volume in MW.
        timestamp: Timezone-aware time at which the fill occurred.
        side: ``Side.BUY`` or ``Side.SELL`` from the algo's perspective.
    """

    model_config = ConfigDict(frozen=True)

    order_id: str
    product_id: str
    price: Decimal
    volume: Decimal
    timestamp: datetime
    side: Side


class OrderResult(BaseModel):
    """Result of submitting an order to the exchange.

    Attributes:
        order_id: ID of the submitted order.
        status: Current status of the order.
        fill: Fill details if the order was immediately matched, else ``None``.
        rejection_reason: Human-readable reason if status is ``REJECTED``.
    """

    model_config = ConfigDict(frozen=True)

    order_id: str
    status: OrderStatus
    fill: Fill | None = None
    rejection_reason: str | None = None


class CancelResult(BaseModel):
    """Result of a cancel request.

    Attributes:
        order_id: ID of the order cancellation was attempted for.
        status: ``"cancelled"`` on success, ``"not_found"`` if unknown order.
    """

    model_config = ConfigDict(frozen=True)

    order_id: str
    status: str  # "cancelled" | "not_found"


class Position(BaseModel):
    """Net position for a single product.

    Attributes:
        product_id: Exchange product identifier.
        net_mw: Net volume in MW (positive = long, negative = short).
        avg_entry_price: Volume-weighted average entry price in EUR/MWh.
        unrealised_pnl: Unrealised profit/loss in EUR based on current market price.
    """

    model_config = ConfigDict(frozen=True)

    product_id: str
    net_mw: Decimal
    avg_entry_price: Decimal
    unrealised_pnl: Decimal


class AuctionInfo(BaseModel):
    """Metadata about an upcoming auction.

    Attributes:
        product_id: Exchange product identifier.
        auction_type: ``"DA"`` for day-ahead or ``"IDA"`` for intraday auction.
        gate_closure_time: Timezone-aware deadline for order submission.
        zone: Bidding zone identifier, e.g. ``"NO1"``.
    """

    model_config = ConfigDict(frozen=True)

    product_id: str
    auction_type: str  # "DA" | "IDA"
    gate_closure_time: datetime
    zone: str


@dataclass(frozen=True)
class MarketEvent:
    """Base class for all IDC market events replayed by the backtest engine.

    Attributes:
        timestamp: Timezone-aware time at which the event occurred.
        product_id: Exchange product identifier, e.g. ``"NO1-QH-0900"``.
    """

    timestamp: datetime
    product_id: str


@dataclass(frozen=True)
class MarketDataUpdate(MarketEvent):
    """A change to the historical order book.

    Attributes:
        event_type: One of ``"new"``, ``"modify"``, ``"cancel"``, ``"trade"``.
        order_id: Exchange order identifier.
        side: ``"buy"`` or ``"sell"``.
        price_eur_mwh: Order price in EUR/MWh.
        volume_mw: Original order volume in MW.
        remaining_mw: Remaining volume after the event.
    """

    event_type: str
    order_id: str
    side: str
    price_eur_mwh: Decimal
    volume_mw: Decimal
    remaining_mw: Decimal


@dataclass(frozen=True)
class HistoricalTrade(MarketEvent):
    """A trade that occurred in the historical data.

    Used by :class:`~nexa_backtest.engines.matching.ContinuousMatchingEngine`
    to determine whether the algo's resting orders would have been hit.

    Attributes:
        trade_id: Exchange trade identifier.
        price_eur_mwh: Trade execution price in EUR/MWh.
        volume_mw: Traded volume in MW.
        aggressor_side: ``"buy"`` if a buyer placed the aggressive order,
            ``"sell"`` if a seller did, or ``None`` when not available in
            the exchange data export.
    """

    trade_id: str
    price_eur_mwh: Decimal
    volume_mw: Decimal
    aggressor_side: str | None


@dataclass(frozen=True)
class GateClosureWarning(MarketEvent):
    """Warning that gate closure for a product is approaching.

    Attributes:
        remaining: Time until gate closes.
    """

    remaining: timedelta


@dataclass(frozen=True)
class GateClosureEvent(MarketEvent):
    """Gate has closed for a product; no further orders may be submitted."""


@dataclass(frozen=True)
class SignalUpdate(MarketEvent):
    """A subscribed signal has a new value.

    Delivered to ``@algo`` functions via :meth:`TradingContext.events`.

    Attributes:
        name: Signal name matching the subscription.
        value: The latest signal value visible at the current simulated time.
    """

    name: str
    value: SignalValue


@dataclass(frozen=True)
class BarEvent(MarketEvent):
    """MTU boundary tick delivered to ``@algo`` functions.

    Equivalent to :meth:`~nexa_backtest.algo.SimpleAlgo.on_bar` in the
    hook-based API.

    Attributes:
        mtu: The MTU that just started.
    """

    mtu: MTU


@dataclass(frozen=True)
class FillEvent(MarketEvent):
    """One of the algo's orders was filled.

    Delivered to ``@algo`` functions via :meth:`TradingContext.events`.
    Equivalent to :meth:`~nexa_backtest.algo.SimpleAlgo.on_fill`.

    Attributes:
        fill: Details of the fill, including price, volume, and side.
    """

    fill: Fill


@dataclass(frozen=True)
class CancelEvent(MarketEvent):
    """One of the algo's orders was cancelled.

    Delivered to ``@algo`` functions via :meth:`TradingContext.events`.
    Equivalent to :meth:`~nexa_backtest.algo.SimpleAlgo.on_cancel`.

    Attributes:
        order_id: ID of the cancelled order.
        reason: One of ``"gate_closure"``, ``"user_cancel"``,
            ``"exchange_cancel"``.
    """

    order_id: str
    reason: str


@dataclass(frozen=True)
class DeliveryPosition:
    """Net open position for a specific delivery period, aggregated across
    all products covering that period.

    For most IDC backtests one product maps to one delivery period, so
    ``net_mw`` equals the single ``Position.net_mw``.  This type becomes
    meaningful when multiple products (e.g. hourly + quarter-hourly) map to
    the same 15-minute window.

    Attributes:
        delivery_start: Timezone-aware start of the delivery period.
        delivery_end: Timezone-aware end of the delivery period.
        net_mw: Net volume in MW (positive = long, negative = short),
            summed across all contributing positions.
        positions: Per-product positions that contribute to this window.
    """

    delivery_start: datetime
    delivery_end: datetime
    net_mw: Decimal
    positions: tuple[Position, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class GateClosureSnapshot:
    """Snapshot of the algo's net position when gate closed for a product.

    Recorded by the engine each time a product's gate closes.  A non-zero
    ``net_mw`` indicates the algo is carrying open IDC exposure into
    physical delivery.

    Attributes:
        product_id: Exchange product identifier.
        gate_closure_time: Timezone-aware time gate closed.
        delivery_start: Timezone-aware start of the delivery period.
        net_mw: Net position in MW at gate closure.
    """

    product_id: str
    gate_closure_time: datetime
    delivery_start: datetime
    net_mw: Decimal


@dataclass(frozen=True)
class EquitySnapshot:
    """Capital state at a point in time.

    Recorded after each auction period with trading activity, these snapshots
    form the equity curve used to compute time-series metrics such as Sharpe
    ratio and max drawdown.

    For DA-only backtesting, ``unrealised_pnl`` is always zero because DA
    auctions settle immediately at the clearing price. The field exists for
    IDC compatibility in later stages.

    Attributes:
        timestamp: Timezone-aware time of this snapshot (typically fill time).
        realised_pnl: Cumulative realised PnL to this point in EUR.
        unrealised_pnl: Mark-to-market of open positions in EUR (zero for DA).
        total_equity: ``initial_capital + realised_pnl + unrealised_pnl``.
        cash: Cash remaining after settled trades.
        net_position_mw: Total net MW across all products (zero after DA settlement).
    """

    timestamp: datetime
    realised_pnl: Decimal
    unrealised_pnl: Decimal
    total_equity: Decimal
    cash: Decimal
    net_position_mw: Decimal
