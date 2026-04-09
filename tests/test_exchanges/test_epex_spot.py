"""Tests for the EPEX SPOT exchange adapter (task 05)."""

from __future__ import annotations

from datetime import timedelta
from decimal import Decimal

import pytest

from nexa_backtest.exchanges.epex_spot import (
    EpexSpotAdapter,
)
from nexa_backtest.exchanges.nordpool import NordPoolAdapter
from nexa_backtest.types import Order


@pytest.fixture
def epex() -> EpexSpotAdapter:
    """EPEX SPOT adapter for DE-LU area."""
    return EpexSpotAdapter("DE-LU")


# ---------------------------------------------------------------------------
# Capabilities
# ---------------------------------------------------------------------------


def test_epex_exchange_id(epex: EpexSpotAdapter) -> None:
    assert epex.capabilities.exchange_id == "epex_spot"


def test_epex_supports_block_bids(epex: EpexSpotAdapter) -> None:
    assert epex.capabilities.supports_block_bids is True


def test_epex_does_not_support_linked_orders(epex: EpexSpotAdapter) -> None:
    assert epex.capabilities.supports_linked_orders is False


def test_epex_does_not_support_market_orders(epex: EpexSpotAdapter) -> None:
    assert epex.capabilities.supports_market_orders is False


def test_epex_supports_partial_fills(epex: EpexSpotAdapter) -> None:
    assert epex.capabilities.supports_partial_fills is True


def test_epex_min_price_is_minus_500(epex: EpexSpotAdapter) -> None:
    assert epex.capabilities.min_price_eur_mwh == Decimal("-500")


def test_epex_max_price_is_4000(epex: EpexSpotAdapter) -> None:
    """EPEX SPOT max price (4,000 EUR/MWh) differs from Nord Pool (3,000)."""
    assert epex.capabilities.max_price_eur_mwh == Decimal("4000")


def test_epex_mtu_duration_is_15_minutes(epex: EpexSpotAdapter) -> None:
    assert epex.capabilities.mtu_duration_minutes == 15


def test_epex_min_volume(epex: EpexSpotAdapter) -> None:
    assert epex.capabilities.min_volume_mw == Decimal("0.1")


def test_epex_gate_closure_30_min(epex: EpexSpotAdapter) -> None:
    assert epex.capabilities.gate_closure_minutes_before_delivery == 30


def test_epex_supports_continuous_trading(epex: EpexSpotAdapter) -> None:
    assert epex.supports_continuous_trading is True


def test_epex_supports_auction_trading(epex: EpexSpotAdapter) -> None:
    assert epex.supports_auction_trading is True


# ---------------------------------------------------------------------------
# EPEX vs Nord Pool capability differences
# ---------------------------------------------------------------------------


def test_epex_max_price_differs_from_nordpool() -> None:
    """EPEX max price (4,000) must be higher than Nord Pool (3,000)."""
    nordpool = NordPoolAdapter("NO1")
    epex = EpexSpotAdapter("DE-LU")
    assert epex.capabilities.max_price_eur_mwh > nordpool.capabilities.max_price_eur_mwh


def test_epex_exchange_id_differs_from_nordpool() -> None:
    nordpool = NordPoolAdapter("NO1")
    epex = EpexSpotAdapter("DE-LU")
    assert epex.capabilities.exchange_id != nordpool.capabilities.exchange_id


# ---------------------------------------------------------------------------
# Products
# ---------------------------------------------------------------------------


def test_epex_products_returns_96_qh(epex: EpexSpotAdapter) -> None:
    products = epex.get_products()
    assert len(products) == 96


def test_epex_products_use_area_prefix(epex: EpexSpotAdapter) -> None:
    products = epex.get_products()
    assert all(p.startswith("DE-LU-QH-") for p in products)


def test_epex_products_include_midnight(epex: EpexSpotAdapter) -> None:
    products = epex.get_products()
    assert "DE-LU-QH-0000" in products


def test_epex_products_include_last_quarter(epex: EpexSpotAdapter) -> None:
    products = epex.get_products()
    assert "DE-LU-QH-2345" in products


def test_epex_different_area() -> None:
    fr = EpexSpotAdapter("FR")
    products = fr.get_products()
    assert all(p.startswith("FR-QH-") for p in products)
    assert len(products) == 96


# ---------------------------------------------------------------------------
# Order book
# ---------------------------------------------------------------------------


def test_epex_orderbook_is_empty(epex: EpexSpotAdapter) -> None:
    ob = epex.get_orderbook("DE-LU-QH-0900")
    assert ob.product_id == "DE-LU-QH-0900"
    assert ob.bids == []
    assert ob.asks == []


# ---------------------------------------------------------------------------
# Order validation
# ---------------------------------------------------------------------------


def test_epex_validate_order_valid(epex: EpexSpotAdapter) -> None:
    order = Order.buy("DE-LU-QH-0900", volume_mw=1, price_eur_mwh=50)
    assert epex.validate_order(order) is None


def test_epex_validate_order_below_min_volume(epex: EpexSpotAdapter) -> None:
    order = Order.buy("DE-LU-QH-0900", volume_mw=Decimal("0.05"), price_eur_mwh=50)
    error = epex.validate_order(order)
    assert error is not None
    assert "minimum" in error.lower()


def test_epex_validate_order_price_too_low(epex: EpexSpotAdapter) -> None:
    order = Order.buy("DE-LU-QH-0900", volume_mw=1, price_eur_mwh=-600)
    error = epex.validate_order(order)
    assert error is not None
    assert "minimum" in error.lower()


def test_epex_validate_order_price_too_high(epex: EpexSpotAdapter) -> None:
    order = Order.buy("DE-LU-QH-0900", volume_mw=1, price_eur_mwh=4500)
    error = epex.validate_order(order)
    assert error is not None
    assert "maximum" in error.lower()


def test_epex_validate_order_at_max_price(epex: EpexSpotAdapter) -> None:
    """Order at exactly 4,000 EUR/MWh must be valid (EPEX allows it)."""
    order = Order.buy("DE-LU-QH-0900", volume_mw=1, price_eur_mwh=4000)
    assert epex.validate_order(order) is None


def test_epex_validate_order_at_max_price_fails_for_nordpool() -> None:
    """An order at 4,000 EUR/MWh is valid for EPEX but invalid for Nord Pool."""
    nordpool = NordPoolAdapter("NO1")
    order = Order.buy("NO1-QH-0900", volume_mw=1, price_eur_mwh=4000)
    # Nord Pool max is 3,000
    error = nordpool.validate_order(order)
    assert error is not None
    assert "maximum" in error.lower()


def test_epex_submit_order_raises(epex: EpexSpotAdapter) -> None:
    order = Order.buy("DE-LU-QH-0900", volume_mw=1, price_eur_mwh=50)
    with pytest.raises(NotImplementedError):
        epex.submit_order(order)


def test_epex_cancel_order_raises(epex: EpexSpotAdapter) -> None:
    with pytest.raises(NotImplementedError):
        epex.cancel_order("some-order-id")


# ---------------------------------------------------------------------------
# Gate closure offsets
# ---------------------------------------------------------------------------


def test_epex_idc_gate_closure_offset(epex: EpexSpotAdapter) -> None:
    assert epex.gate_closure_offset("IDC") == timedelta(minutes=30)


def test_epex_da_gate_closure_offset(epex: EpexSpotAdapter) -> None:
    assert epex.gate_closure_offset("DA") == timedelta(hours=1)


def test_epex_ida_gate_closure_offset(epex: EpexSpotAdapter) -> None:
    assert epex.gate_closure_offset("IDA") == timedelta(hours=1)
