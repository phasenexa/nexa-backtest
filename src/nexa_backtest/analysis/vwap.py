"""VWAP benchmark calculation.

VWAP (Volume-Weighted Average Price) is the primary benchmark for evaluating
execution quality. If an algo cannot beat VWAP, a simple time-weighted passive
execution strategy would achieve better results.
"""

from __future__ import annotations

from decimal import Decimal

import pandas as pd


def compute_market_vwap(market_data: pd.DataFrame) -> Decimal:
    """Compute the volume-weighted average clearing price across all products.

    This is the benchmark price: the average price a passive participant would
    have paid (for buys) or received (for sells) if they executed at each
    clearing price proportional to market volume.

    Args:
        market_data: DataFrame with ``price_eur_mwh`` (float) and
            ``volume_mwh`` (float) columns for all products in the period.

    Returns:
        Market VWAP in EUR/MWh, or ``Decimal("0")`` if the data is empty
        or has no volume.
    """
    if market_data.empty or "volume_mwh" not in market_data.columns:
        return Decimal("0")

    total_vol = market_data["volume_mwh"].sum()
    if total_vol <= 0:
        return Decimal("0")

    price_x_vol = (market_data["price_eur_mwh"] * market_data["volume_mwh"]).sum()
    return Decimal(str(round(price_x_vol / total_vol, 6)))
