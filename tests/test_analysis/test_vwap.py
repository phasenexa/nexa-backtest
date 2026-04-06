"""Tests for analysis/vwap.py: compute_market_vwap."""

from __future__ import annotations

from decimal import Decimal

import pandas as pd

from nexa_backtest.analysis.vwap import compute_market_vwap


class TestComputeMarketVwap:
    def test_empty_dataframe_returns_zero(self) -> None:
        df = pd.DataFrame(columns=["price_eur_mwh", "volume_mwh"])
        assert compute_market_vwap(df) == Decimal("0")

    def test_missing_volume_column_returns_zero(self) -> None:
        df = pd.DataFrame({"price_eur_mwh": [50.0, 60.0]})
        assert compute_market_vwap(df) == Decimal("0")

    def test_zero_total_volume_returns_zero(self) -> None:
        df = pd.DataFrame({"price_eur_mwh": [50.0, 60.0], "volume_mwh": [0.0, 0.0]})
        assert compute_market_vwap(df) == Decimal("0")

    def test_single_row_returns_that_price(self) -> None:
        df = pd.DataFrame({"price_eur_mwh": [50.0], "volume_mwh": [100.0]})
        result = compute_market_vwap(df)
        assert result == Decimal("50")

    def test_equal_volumes_returns_arithmetic_mean(self) -> None:
        df = pd.DataFrame({"price_eur_mwh": [40.0, 60.0], "volume_mwh": [100.0, 100.0]})
        result = compute_market_vwap(df)
        assert result == Decimal("50")

    def test_unequal_volumes_weighted_correctly(self) -> None:
        # 30 * 200 + 60 * 100 = 6000 + 6000 = 12000 / 300 = 40.0
        df = pd.DataFrame({"price_eur_mwh": [30.0, 60.0], "volume_mwh": [200.0, 100.0]})
        result = compute_market_vwap(df)
        assert result == Decimal("40")

    def test_result_is_decimal(self) -> None:
        df = pd.DataFrame({"price_eur_mwh": [45.5], "volume_mwh": [100.0]})
        result = compute_market_vwap(df)
        assert isinstance(result, Decimal)
