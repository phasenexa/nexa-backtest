"""Shared pytest fixtures for nexa-backtest tests."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from nexa_backtest.types import (
    MTU,
    AuctionInfo,
    Order,
    OrderBook,
    PriceLevel,
)


@pytest.fixture
def utc_now() -> datetime:
    """Fixed UTC timestamp for use in tests."""
    return datetime(2026, 1, 15, 12, 0, 0, tzinfo=UTC)


@pytest.fixture
def sample_mtu() -> MTU:
    """A 15-minute MTU for NO1 at 09:00 UTC on 2026-01-15."""
    return MTU(
        start=datetime(2026, 1, 15, 9, 0, tzinfo=UTC),
        end=datetime(2026, 1, 15, 9, 15, tzinfo=UTC),
        zone="NO1",
    )


@pytest.fixture
def sample_mtu_hourly() -> MTU:
    """An hourly MTU for NO1 at 09:00 UTC on 2026-01-15."""
    return MTU(
        start=datetime(2026, 1, 15, 9, 0, tzinfo=UTC),
        end=datetime(2026, 1, 15, 10, 0, tzinfo=UTC),
        zone="NO1",
    )


@pytest.fixture
def sample_product_id() -> str:
    """Standard product ID used across tests."""
    return "NO1-QH-0900"


@pytest.fixture
def clearing_price() -> Decimal:
    """Sample DA clearing price in EUR/MWh."""
    return Decimal("50.00")


@pytest.fixture
def sample_buy_order(sample_product_id: str) -> Order:
    """A limit buy order priced above the clearing price (should fill)."""
    return Order.buy(product_id=sample_product_id, volume_mw=10, price_eur_mwh=55)


@pytest.fixture
def sample_sell_order(sample_product_id: str) -> Order:
    """A limit sell order priced below the clearing price (should fill)."""
    return Order.sell(product_id=sample_product_id, volume_mw=10, price_eur_mwh=45)


@pytest.fixture
def sample_orderbook(sample_product_id: str, utc_now: datetime) -> OrderBook:
    """A simple order book with one bid and one ask."""
    return OrderBook(
        product_id=sample_product_id,
        bids=[PriceLevel(price=Decimal("49.50"), volume=Decimal("20"))],
        asks=[PriceLevel(price=Decimal("50.50"), volume=Decimal("15"))],
        timestamp=utc_now,
    )


@pytest.fixture
def sample_auction_info(sample_product_id: str) -> AuctionInfo:
    """Sample DA auction info for NO1."""
    return AuctionInfo(
        product_id=sample_product_id,
        auction_type="DA",
        gate_closure_time=datetime(2026, 1, 14, 11, 0, tzinfo=UTC),
        zone="NO1",
    )
