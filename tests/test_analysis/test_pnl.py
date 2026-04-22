"""Tests for analysis/pnl.py: compute_pnl and _side_summary."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pandas as pd
import pytest

from nexa_backtest.analysis.pnl import compute_pnl
from nexa_backtest.types import Fill, Side


def _fill(side: Side, price: float, volume: float = 10.0, product_id: str = "P1") -> Fill:
    return Fill(
        order_id="o1",
        product_id=product_id,
        side=side,
        price=Decimal(str(price)),
        volume=Decimal(str(volume)),
        timestamp=datetime(2026, 3, 1, 12, 0, tzinfo=UTC),
    )


def _market_data(prices: list[float], volumes: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"price_eur_mwh": prices, "volume_mwh": volumes})


class TestComputePnlEmptyMarketData:
    def test_empty_market_data_gives_zero_vwap(self) -> None:
        df = pd.DataFrame(columns=["price_eur_mwh", "volume_mwh"])
        result = compute_pnl([], df)
        assert result.market_vwap == Decimal("0")

    def test_market_data_without_volume_col_gives_zero_vwap(self) -> None:
        df = pd.DataFrame({"price_eur_mwh": [50.0]})
        result = compute_pnl([], df)
        assert result.market_vwap == Decimal("0")

    def test_no_fills_all_counts_zero(self) -> None:
        df = _market_data([50.0], [100.0])
        result = compute_pnl([], df)
        assert result.buys.count == 0
        assert result.sells.count == 0
        assert result.total_alpha_eur == Decimal("0")


class TestComputePnlBuys:
    def test_buy_below_vwap_is_positive_alpha(self) -> None:
        # VWAP = 50, buy at 40 → bought below VWAP → positive alpha
        df = _market_data([50.0], [100.0])
        fills = [_fill(Side.BUY, 40.0, 10.0)]
        result = compute_pnl(fills, df)
        assert result.buys.vwap_alpha < Decimal("0")  # alpha = avg - vwap = 40 - 50 = -10
        assert result.buys.total_alpha_eur > Decimal("0")  # -alpha * volume = 100 EUR

    def test_buy_above_vwap_is_negative_alpha(self) -> None:
        df = _market_data([50.0], [100.0])
        fills = [_fill(Side.BUY, 60.0, 10.0)]
        result = compute_pnl(fills, df)
        assert result.buys.vwap_alpha > Decimal("0")  # 60 - 50 = +10 (bad)

    def test_buy_win_rate_counted_correctly(self) -> None:
        # 2 fills: 40 (below vwap=50, win) and 60 (above, loss)
        df = _market_data([50.0], [100.0])
        fills = [_fill(Side.BUY, 40.0), _fill(Side.BUY, 60.0)]
        result = compute_pnl(fills, df)
        assert result.buys.win_rate == pytest.approx(0.5)


class TestComputePnlSells:
    def test_sell_above_vwap_is_positive_alpha(self) -> None:
        # VWAP = 50, sell at 60 → sold above VWAP → positive alpha
        df = _market_data([50.0], [100.0])
        fills = [_fill(Side.SELL, 60.0, 10.0)]
        result = compute_pnl(fills, df)
        # vwap_alpha for sells = market_vwap - avg_price = 50 - 60 = -10
        assert result.sells.vwap_alpha < Decimal("0")
        assert result.sells.total_alpha_eur > Decimal("0")  # -(-10) * 10 = 100

    def test_sell_below_vwap_is_negative_alpha(self) -> None:
        df = _market_data([50.0], [100.0])
        fills = [_fill(Side.SELL, 40.0, 10.0)]
        result = compute_pnl(fills, df)
        # vwap_alpha = 50 - 40 = +10 (bad — sold below VWAP)
        assert result.sells.vwap_alpha > Decimal("0")

    def test_sell_win_rate_counted_correctly(self) -> None:
        # 2 sells: 60 (above vwap=50, win) and 40 (below, loss)
        df = _market_data([50.0], [100.0])
        fills = [_fill(Side.SELL, 60.0), _fill(Side.SELL, 40.0)]
        result = compute_pnl(fills, df)
        assert result.sells.win_rate == pytest.approx(0.5)

    def test_sell_count_and_volume(self) -> None:
        df = _market_data([50.0], [100.0])
        fills = [_fill(Side.SELL, 55.0, 20.0), _fill(Side.SELL, 45.0, 10.0)]
        result = compute_pnl(fills, df)
        assert result.sells.count == 2
        assert result.sells.volume_mwh == Decimal("30")

    def test_total_alpha_combines_buys_and_sells(self) -> None:
        df = _market_data([50.0], [100.0])
        fills = [
            _fill(Side.BUY, 40.0, 10.0),
            _fill(Side.SELL, 60.0, 10.0),
        ]
        result = compute_pnl(fills, df)
        assert result.total_alpha_eur == result.buys.total_alpha_eur + result.sells.total_alpha_eur


class TestPnlSummaryDerivedMetrics:
    """Tests for the derived metrics on PnlSummary."""

    def test_pnl_per_mwh(self) -> None:
        df = _market_data([50.0], [100.0])
        fills = [_fill(Side.BUY, 40.0, 10.0)]  # alpha = (50-40)*10 = 100
        result = compute_pnl(fills, df)
        # pnl_per_mwh = 100 / 10 = 10
        assert float(result.pnl_per_mwh) == pytest.approx(10.0)

    def test_pnl_per_mwh_no_fills(self) -> None:
        df = _market_data([50.0], [100.0])
        result = compute_pnl([], df)
        assert result.pnl_per_mwh == Decimal("0")

    def test_win_rate_combined(self) -> None:
        df = _market_data([50.0], [100.0])
        fills = [
            _fill(Side.BUY, 40.0),  # win (below VWAP)
            _fill(Side.BUY, 60.0),  # loss (above VWAP)
            _fill(Side.SELL, 60.0),  # win (above VWAP)
        ]
        result = compute_pnl(fills, df)
        # 2 wins out of 3 fills
        assert result.win_rate == pytest.approx(2 / 3)
        assert result.loss_rate == pytest.approx(1 / 3)

    def test_long_short_pct(self) -> None:
        df = _market_data([50.0], [100.0])
        fills = [
            _fill(Side.BUY, 40.0),
            _fill(Side.BUY, 45.0),
            _fill(Side.SELL, 55.0),
        ]
        result = compute_pnl(fills, df)
        assert result.long_pct == pytest.approx(2 / 3)
        assert result.short_pct == pytest.approx(1 / 3)

    def test_all_buys_gives_100_pct_longs(self) -> None:
        df = _market_data([50.0], [100.0])
        fills = [_fill(Side.BUY, 40.0)]
        result = compute_pnl(fills, df)
        assert result.long_pct == pytest.approx(1.0)
        assert result.short_pct == pytest.approx(0.0)

    def test_no_fills_gives_zero_pcts(self) -> None:
        df = _market_data([50.0], [100.0])
        result = compute_pnl([], df)
        assert result.long_pct == 0.0
        assert result.short_pct == 0.0
        assert result.win_rate == 0.0
        assert result.loss_rate == 1.0
