"""Tests for exchanges/nordpool.py: NordPoolAdapter and helpers."""

from __future__ import annotations

from datetime import timedelta
from decimal import Decimal

import pytest

from nexa_backtest.exchanges.nordpool import (
    NORDPOOL_DA_GATE_CLOSURE,
    NORDPOOL_IDC_GATE_CLOSURE,
    NORDPOOL_MAX_PRICE,
    NORDPOOL_MIN_PRICE,
    NORDPOOL_MIN_VOLUME,
    NordPoolAdapter,
    is_idc_product,
)
from nexa_backtest.types import Order, Side  # Side used for Order.market()


# ---------------------------------------------------------------------------
# is_idc_product
# ---------------------------------------------------------------------------


class TestIsIdcProduct:
    def test_valid_qh_products(self) -> None:
        assert is_idc_product("NO1-QH-0900") is True
        assert is_idc_product("SE3-QH-1345") is True
        assert is_idc_product("DK1-QH-0000") is True
        assert is_idc_product("FI-QH-2345") is True

    def test_invalid_products(self) -> None:
        assert is_idc_product("NO1_DA") is False
        assert is_idc_product("NO1-QH") is False
        assert is_idc_product("NO1-QH-900") is False  # only 3 digits
        assert is_idc_product("") is False
        assert is_idc_product("NO1-HOURLY-0900") is False


# ---------------------------------------------------------------------------
# NordPoolAdapter construction and capabilities
# ---------------------------------------------------------------------------


class TestNordPoolAdapterCapabilities:
    def test_zone_stored(self) -> None:
        adapter = NordPoolAdapter("NO1")
        assert adapter._zone == "NO1"

    def test_capabilities_exchange_id(self) -> None:
        adapter = NordPoolAdapter("SE2")
        assert adapter.capabilities.exchange_id == "nordpool"

    def test_capabilities_block_bids_supported(self) -> None:
        adapter = NordPoolAdapter("NO1")
        assert adapter.capabilities.supports_block_bids is True

    def test_capabilities_market_orders_not_supported(self) -> None:
        adapter = NordPoolAdapter("NO1")
        assert adapter.capabilities.supports_market_orders is False

    def test_capabilities_partial_fills_supported(self) -> None:
        adapter = NordPoolAdapter("NO1")
        assert adapter.capabilities.supports_partial_fills is True

    def test_capabilities_price_limits(self) -> None:
        adapter = NordPoolAdapter("NO1")
        assert adapter.capabilities.min_price_eur_mwh == NORDPOOL_MIN_PRICE
        assert adapter.capabilities.max_price_eur_mwh == NORDPOOL_MAX_PRICE

    def test_capabilities_min_volume(self) -> None:
        adapter = NordPoolAdapter("NO1")
        assert adapter.capabilities.min_volume_mw == NORDPOOL_MIN_VOLUME

    def test_capabilities_mtu_duration(self) -> None:
        adapter = NordPoolAdapter("NO1")
        assert adapter.capabilities.mtu_duration_minutes == 15

    def test_supports_continuous_trading(self) -> None:
        adapter = NordPoolAdapter("NO1")
        assert adapter.supports_continuous_trading is True


# ---------------------------------------------------------------------------
# get_products
# ---------------------------------------------------------------------------


class TestGetProducts:
    def test_returns_96_products(self) -> None:
        adapter = NordPoolAdapter("NO1")
        products = adapter.get_products()
        assert len(products) == 96

    def test_product_format(self) -> None:
        adapter = NordPoolAdapter("NO1")
        products = adapter.get_products()
        assert "NO1-QH-0000" in products
        assert "NO1-QH-0900" in products
        assert "NO1-QH-2345" in products

    def test_zone_prefix_respected(self) -> None:
        adapter = NordPoolAdapter("SE3")
        products = adapter.get_products()
        assert all(p.startswith("SE3-QH-") for p in products)

    def test_all_quarter_hours(self) -> None:
        adapter = NordPoolAdapter("DK1")
        products = adapter.get_products()
        # Check a sample of quarter-hour products
        for hh in ["0000", "0015", "0030", "0045", "0100", "2330", "2345"]:
            assert f"DK1-QH-{hh}" in products


# ---------------------------------------------------------------------------
# get_orderbook
# ---------------------------------------------------------------------------


class TestGetOrderBook:
    def test_returns_empty_orderbook(self) -> None:
        adapter = NordPoolAdapter("NO1")
        book = adapter.get_orderbook("NO1-QH-0900")
        assert book.product_id == "NO1-QH-0900"
        assert book.bids == []
        assert book.asks == []

    def test_orderbook_has_timestamp(self) -> None:
        adapter = NordPoolAdapter("NO1")
        book = adapter.get_orderbook("NO1-QH-1200")
        assert book.timestamp is not None


# ---------------------------------------------------------------------------
# submit_order / cancel_order raise NotImplementedError
# ---------------------------------------------------------------------------


class TestNotImplementedMethods:
    def test_submit_order_raises(self) -> None:
        adapter = NordPoolAdapter("NO1")
        order = Order.buy("NO1-QH-0900", volume_mw=1, price_eur_mwh=Decimal("50"))
        with pytest.raises(NotImplementedError, match="submit_order"):
            adapter.submit_order(order)

    def test_cancel_order_raises(self) -> None:
        adapter = NordPoolAdapter("NO1")
        with pytest.raises(NotImplementedError, match="cancel_order"):
            adapter.cancel_order("some-order-id")


# ---------------------------------------------------------------------------
# gate_closure_offset
# ---------------------------------------------------------------------------


class TestGateClosureOffset:
    def test_idc_offset(self) -> None:
        adapter = NordPoolAdapter("NO1")
        assert adapter.gate_closure_offset("IDC") == NORDPOOL_IDC_GATE_CLOSURE
        assert adapter.gate_closure_offset("IDC") == timedelta(minutes=30)

    def test_da_offset(self) -> None:
        adapter = NordPoolAdapter("NO1")
        assert adapter.gate_closure_offset("DA") == NORDPOOL_DA_GATE_CLOSURE
        assert adapter.gate_closure_offset("DA") == timedelta(hours=1)

    def test_unknown_type_defaults_to_idc(self) -> None:
        adapter = NordPoolAdapter("NO1")
        # Default fallback returns IDC offset
        assert adapter.gate_closure_offset("other") == NORDPOOL_IDC_GATE_CLOSURE


# ---------------------------------------------------------------------------
# validate_order
# ---------------------------------------------------------------------------


class TestValidateOrder:
    def test_valid_buy_order(self) -> None:
        adapter = NordPoolAdapter("NO1")
        order = Order.buy("NO1-QH-0900", volume_mw=Decimal("1"), price_eur_mwh=Decimal("50"))
        assert adapter.validate_order(order) is None

    def test_valid_sell_order(self) -> None:
        adapter = NordPoolAdapter("NO1")
        order = Order.sell("NO1-QH-0900", volume_mw=Decimal("5"), price_eur_mwh=Decimal("55"))
        assert adapter.validate_order(order) is None

    def test_volume_below_minimum(self) -> None:
        adapter = NordPoolAdapter("NO1")
        order = Order.buy("NO1-QH-0900", volume_mw=Decimal("0.05"), price_eur_mwh=Decimal("50"))
        error = adapter.validate_order(order)
        assert error is not None
        assert "minimum" in error.lower()

    def test_price_below_minimum(self) -> None:
        adapter = NordPoolAdapter("NO1")
        order = Order.sell(
            "NO1-QH-0900",
            volume_mw=Decimal("1"),
            price_eur_mwh=NORDPOOL_MIN_PRICE - Decimal("1"),
        )
        error = adapter.validate_order(order)
        assert error is not None
        assert "below" in error.lower()

    def test_price_above_maximum(self) -> None:
        adapter = NordPoolAdapter("NO1")
        order = Order.buy(
            "NO1-QH-0900",
            volume_mw=Decimal("1"),
            price_eur_mwh=NORDPOOL_MAX_PRICE + Decimal("1"),
        )
        error = adapter.validate_order(order)
        assert error is not None
        assert "above" in error.lower()

    def test_market_order_rejected(self) -> None:
        adapter = NordPoolAdapter("NO1")
        order = Order.market("NO1-QH-0900", Side.BUY, volume_mw=Decimal("1"))
        error = adapter.validate_order(order)
        assert error is not None
        assert "market order" in error.lower()

    def test_price_at_minimum_is_valid(self) -> None:
        adapter = NordPoolAdapter("NO1")
        order = Order.sell("NO1-QH-0900", volume_mw=Decimal("1"), price_eur_mwh=NORDPOOL_MIN_PRICE)
        assert adapter.validate_order(order) is None

    def test_price_at_maximum_is_valid(self) -> None:
        adapter = NordPoolAdapter("NO1")
        order = Order.buy("NO1-QH-0900", volume_mw=Decimal("1"), price_eur_mwh=NORDPOOL_MAX_PRICE)
        assert adapter.validate_order(order) is None
