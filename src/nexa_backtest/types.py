"""Core domain types for nexa-backtest.

All types are immutable data containers using Pydantic v2 with frozen=True.
Prices and volumes use decimal.Decimal to avoid floating-point precision issues.
All datetimes are timezone-aware.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


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
    """Snapshot of the best bid and ask for a product.

    Attributes:
        product_id: Exchange product identifier, e.g. ``"NO1-QH-0900"``.
        best_bid: Best bid price level, or ``None`` if no bids.
        best_ask: Best ask price level, or ``None`` if no offers.
        timestamp: Timezone-aware time at which this snapshot was taken.
    """

    model_config = ConfigDict(frozen=True)

    product_id: str
    best_bid: PriceLevel | None
    best_ask: PriceLevel | None
    timestamp: datetime


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
