"""Tests for exchanges/base.py: ExchangeCapabilities."""

from __future__ import annotations

from decimal import Decimal

import pytest
from pydantic import ValidationError

from nexa_backtest.exchanges.base import ExchangeCapabilities


class TestExchangeCapabilities:
    def test_defaults(self) -> None:
        caps = ExchangeCapabilities(exchange_id="test_exchange")
        assert caps.exchange_id == "test_exchange"
        assert caps.supports_block_bids is False
        assert caps.supports_linked_orders is False
        assert caps.supports_market_orders is False
        assert caps.supports_partial_fills is False
        assert caps.min_volume_mw == Decimal("0.1")
        assert caps.max_volume_mw is None
        assert caps.min_price_eur_mwh == Decimal("-500")
        assert caps.max_price_eur_mwh == Decimal("3000")
        assert caps.mtu_duration_minutes == 15
        assert caps.gate_closure_minutes_before_delivery == 60

    def test_custom_values(self) -> None:
        caps = ExchangeCapabilities(
            exchange_id="nordpool",
            supports_block_bids=True,
            supports_partial_fills=True,
            min_volume_mw=Decimal("1.0"),
            max_volume_mw=Decimal("500.0"),
            min_price_eur_mwh=Decimal("-9999"),
            max_price_eur_mwh=Decimal("9999"),
            mtu_duration_minutes=60,
            gate_closure_minutes_before_delivery=30,
        )
        assert caps.supports_block_bids is True
        assert caps.supports_partial_fills is True
        assert caps.max_volume_mw == Decimal("500.0")
        assert caps.mtu_duration_minutes == 60

    def test_frozen_model_prevents_mutation(self) -> None:
        caps = ExchangeCapabilities(exchange_id="test")
        with pytest.raises(ValidationError):
            caps.exchange_id = "other"  # type: ignore[misc]

    def test_equality(self) -> None:
        a = ExchangeCapabilities(exchange_id="test")
        b = ExchangeCapabilities(exchange_id="test")
        assert a == b
