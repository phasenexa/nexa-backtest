"""Tests for Step 4: Exchange Feature Compatibility."""

from __future__ import annotations

import sys
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

from nexa_backtest.exchanges.base import ExchangeCapabilities
from nexa_backtest.validation import feature_check
from nexa_backtest.validation.feature_check import FeatureCheck
from tests.test_validation.conftest import fixture_path


def test_block_bid_passes_nordpool() -> None:
    """Nord Pool supports block bids — step should not fail."""
    result = FeatureCheck().run(fixture_path("unsupported_feature.py"), "nordpool")
    # Decimal("10") is treated as dynamic by the AST check — so "warn" is expected.
    assert result.status in ("pass", "warn")
    # There should be no "block_bid" error message.
    assert not any("block_bid" in m.lower() or "block bid" in m.lower() for m in result.messages)


def test_market_order_fails_epex_spot(tmp_path) -> None:
    """EPEX SPOT does not support market orders (no limit price)."""
    algo = tmp_path / "algo.py"
    algo.write_text(
        "from nexa_backtest.algo import SimpleAlgo\n"
        "from nexa_backtest.context import TradingContext\n"
        "from nexa_backtest.types import AuctionInfo, Order\n"
        "from decimal import Decimal\n\n"
        "class MarketAlgo(SimpleAlgo):\n"
        "    def on_auction_open(self, ctx: TradingContext, auction: AuctionInfo) -> None:\n"
        "        ctx.place_order(Order.market(product_id=auction.product_id,\n"
        "            volume_mw=Decimal('10')))\n"
    )
    result = FeatureCheck().run(str(algo), "epex_spot")
    assert result.status == "fail"
    assert any("market" in m.lower() for m in result.messages)


def test_valid_algo_passes_nordpool() -> None:
    # valid_simple_algo uses Decimal("10") which is AST-dynamic — warn expected.
    result = FeatureCheck().run(fixture_path("valid_simple_algo.py"), "nordpool")
    assert result.status in ("pass", "warn")


def test_unknown_exchange_skips() -> None:
    result = FeatureCheck().run(fixture_path("valid_simple_algo.py"), "unknown_exchange")
    assert result.status == "skip"
    assert any("unknown" in m.lower() for m in result.messages)


def test_volume_below_minimum_is_error(tmp_path: Path) -> None:
    algo = tmp_path / "algo.py"
    algo.write_text(
        "from nexa_backtest.algo import SimpleAlgo\n"
        "from nexa_backtest.context import TradingContext\n"
        "from nexa_backtest.types import AuctionInfo, Order\n"
        "from decimal import Decimal\n\n"
        "class TinyAlgo(SimpleAlgo):\n"
        "    def on_auction_open(self, ctx: TradingContext, auction: AuctionInfo) -> None:\n"
        "        ctx.place_order(Order.buy(product_id=auction.product_id,\n"
        "            volume_mw=0.001, price_eur_mwh=50.0))\n"
    )
    result = FeatureCheck().run(str(algo), "nordpool")
    assert result.status == "fail"
    assert any("minimum" in m.lower() for m in result.messages)


def test_dynamic_volume_warns(tmp_path: Path) -> None:
    algo = tmp_path / "algo.py"
    algo.write_text(
        "from nexa_backtest.algo import SimpleAlgo\n"
        "from nexa_backtest.context import TradingContext\n"
        "from nexa_backtest.types import AuctionInfo, Order\n\n"
        "class DynAlgo(SimpleAlgo):\n"
        "    def on_auction_open(self, ctx: TradingContext, auction: AuctionInfo) -> None:\n"
        "        vol = float(ctx.get_signal('vol').value)\n"
        "        ctx.place_order(Order.buy(product_id=auction.product_id,\n"
        "            volume_mw=vol, price_eur_mwh=50.0))\n"
    )
    result = FeatureCheck().run(str(algo), "nordpool")
    assert result.status == "warn"
    assert any("dynamic" in m.lower() or "unable to validate" in m.lower() for m in result.messages)


def test_file_not_found_is_error() -> None:
    result = FeatureCheck().run("/nonexistent/path/algo.py", "nordpool")
    assert result.status == "fail"
    assert any("cannot read" in m.lower() for m in result.messages)


def test_syntax_error_file_is_skip(tmp_path: Path) -> None:
    algo = tmp_path / "algo.py"
    algo.write_text("def broken(\n")
    result = FeatureCheck().run(str(algo), "nordpool")
    assert result.status == "skip"
    assert any("syntax" in m.lower() for m in result.messages)


def test_exclusive_group_fails_when_linked_orders_unsupported(tmp_path: Path) -> None:
    # Both nordpool and epex_spot have supports_linked_orders=False.
    algo = tmp_path / "algo.py"
    algo.write_text(
        "from nexa_backtest.algo import SimpleAlgo\n"
        "from nexa_backtest.context import TradingContext\n"
        "from nexa_backtest.types import AuctionInfo, Order\n\n"
        "class ExcAlgo(SimpleAlgo):\n"
        "    def on_auction_open(self, ctx: TradingContext, auction: AuctionInfo) -> None:\n"
        "        ctx.place_order(Order.buy(product_id=auction.product_id,\n"
        "            volume_mw=1.0, price_eur_mwh=50.0, exclusive_group='g1'))\n"
    )
    result = FeatureCheck().run(str(algo), "nordpool")
    assert result.status == "fail"
    assert any("exclusive" in m.lower() or "linked" in m.lower() for m in result.messages)


def test_price_below_minimum_is_error(tmp_path: Path) -> None:
    # Nord Pool minimum price is -500 EUR/MWh.
    algo = tmp_path / "algo.py"
    algo.write_text(
        "from nexa_backtest.algo import SimpleAlgo\n"
        "from nexa_backtest.context import TradingContext\n"
        "from nexa_backtest.types import AuctionInfo, Order\n\n"
        "class CheapAlgo(SimpleAlgo):\n"
        "    def on_auction_open(self, ctx: TradingContext, auction: AuctionInfo) -> None:\n"
        "        ctx.place_order(Order.sell(product_id=auction.product_id,\n"
        "            volume_mw=1.0, price_eur_mwh=-600.0))\n"
    )
    result = FeatureCheck().run(str(algo), "nordpool")
    assert result.status == "fail"
    assert any("minimum" in m.lower() or "below" in m.lower() for m in result.messages)


def test_price_above_maximum_is_error(tmp_path: Path) -> None:
    # Nord Pool maximum price is 3000 EUR/MWh.
    algo = tmp_path / "algo.py"
    algo.write_text(
        "from nexa_backtest.algo import SimpleAlgo\n"
        "from nexa_backtest.context import TradingContext\n"
        "from nexa_backtest.types import AuctionInfo, Order\n\n"
        "class ExpensiveAlgo(SimpleAlgo):\n"
        "    def on_auction_open(self, ctx: TradingContext, auction: AuctionInfo) -> None:\n"
        "        ctx.place_order(Order.buy(product_id=auction.product_id,\n"
        "            volume_mw=1.0, price_eur_mwh=3500.0))\n"
    )
    result = FeatureCheck().run(str(algo), "nordpool")
    assert result.status == "fail"
    assert any("maximum" in m.lower() or "exceeds" in m.lower() for m in result.messages)


def test_negative_price_via_unary_op_is_checked(tmp_path: Path) -> None:
    # -600.0 is ast.UnaryOp(USub, Constant(600.0)) — exercises _extract_numeric UnaryOp path.
    algo = tmp_path / "algo.py"
    algo.write_text(
        "from nexa_backtest.algo import SimpleAlgo\n"
        "from nexa_backtest.context import TradingContext\n"
        "from nexa_backtest.types import AuctionInfo, Order\n\n"
        "class NegPriceAlgo(SimpleAlgo):\n"
        "    def on_auction_open(self, ctx: TradingContext, auction: AuctionInfo) -> None:\n"
        "        ctx.place_order(Order.sell(product_id=auction.product_id,\n"
        "            volume_mw=1.0, price_eur_mwh=-600.0))\n"
    )
    result = FeatureCheck().run(str(algo), "nordpool")
    assert result.status == "fail"
    assert any("-600" in m or "minimum" in m.lower() for m in result.messages)


def test_block_bid_fails_on_unsupported_exchange(tmp_path: Path) -> None:
    # Inject a test exchange that does not support block bids.
    _orig = dict(feature_check._EXCHANGE_CAPABILITIES)
    feature_check._EXCHANGE_CAPABILITIES["no_block"] = ExchangeCapabilities(
        exchange_id="no_block",
        supports_block_bids=False,
        supports_linked_orders=False,
        supports_market_orders=False,
        supports_partial_fills=False,
        min_volume_mw=Decimal("0.1"),
        max_volume_mw=None,
        min_price_eur_mwh=Decimal("-500"),
        max_price_eur_mwh=Decimal("3000"),
        mtu_duration_minutes=15,
        gate_closure_minutes_before_delivery=30,
    )
    try:
        algo = tmp_path / "algo.py"
        algo.write_text(
            "from nexa_backtest.types import Order\n\n"
            "Order.block_bid(product_id='x', volume_mw=1.0, price_eur_mwh=50.0)\n"
        )
        result = FeatureCheck().run(str(algo), "no_block")
        assert result.status == "fail"
        assert any("block_bid" in m.lower() or "block bid" in m.lower() for m in result.messages)
    finally:
        feature_check._EXCHANGE_CAPABILITIES.clear()
        feature_check._EXCHANGE_CAPABILITIES.update(_orig)


def test_volume_above_maximum_is_error(tmp_path: Path) -> None:
    # Inject a test exchange with a max_volume_mw limit.
    _orig = dict(feature_check._EXCHANGE_CAPABILITIES)
    feature_check._EXCHANGE_CAPABILITIES["small_exchange"] = ExchangeCapabilities(
        exchange_id="small_exchange",
        supports_block_bids=False,
        supports_linked_orders=False,
        supports_market_orders=False,
        supports_partial_fills=False,
        min_volume_mw=Decimal("0.1"),
        max_volume_mw=Decimal("10"),
        min_price_eur_mwh=Decimal("-500"),
        max_price_eur_mwh=Decimal("3000"),
        mtu_duration_minutes=15,
        gate_closure_minutes_before_delivery=30,
    )
    try:
        algo = tmp_path / "algo.py"
        algo.write_text(
            "from nexa_backtest.types import Order\n\n"
            "Order.buy(product_id='x', volume_mw=500.0, price_eur_mwh=50.0)\n"
        )
        result = FeatureCheck().run(str(algo), "small_exchange")
        assert result.status == "fail"
        assert any("maximum" in m.lower() or "exceeds" in m.lower() for m in result.messages)
    finally:
        feature_check._EXCHANGE_CAPABILITIES.clear()
        feature_check._EXCHANGE_CAPABILITIES.update(_orig)


def test_populate_capabilities_handles_import_errors() -> None:
    # Verify that import failures inside _populate_capabilities are swallowed.
    _orig = dict(feature_check._EXCHANGE_CAPABILITIES)
    feature_check._EXCHANGE_CAPABILITIES.clear()
    try:
        with patch.dict(
            sys.modules,
            {
                "nexa_backtest.exchanges.nordpool": None,
                "nexa_backtest.exchanges.epex_spot": None,
            },
        ):
            feature_check._populate_capabilities()
        # Both imports failed → nothing added.
        assert len(feature_check._EXCHANGE_CAPABILITIES) == 0
    finally:
        feature_check._EXCHANGE_CAPABILITIES.clear()
        feature_check._EXCHANGE_CAPABILITIES.update(_orig)
