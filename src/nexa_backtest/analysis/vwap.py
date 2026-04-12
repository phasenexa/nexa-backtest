"""VWAP benchmark calculation.

VWAP (Volume-Weighted Average Price) is the primary benchmark for evaluating
execution quality. If an algo cannot beat VWAP, a simple time-weighted passive
execution strategy would achieve better results.
"""

from __future__ import annotations

from decimal import Decimal

import pandas as pd


def compute_idc_vwaps(
    accum: dict[str, tuple[Decimal, Decimal]],
) -> tuple[dict[str, Decimal], Decimal]:
    """Compute per-product and portfolio IDC market VWAPs from trade accumulators.

    Converts raw ``(sum_notional, sum_volume)`` accumulators — built
    incrementally as ``HistoricalTrade`` events flow through the IDC replay
    loop — into per-product VWAPs and a single portfolio-level VWAP.

    This is O(P) in memory (P = number of products), compatible with
    the sliding-window replay design.

    Args:
        accum: Per-product accumulators mapping ``product_id`` to
            ``(sum_notional_eur, sum_volume_mw)``.

    Returns:
        ``(per_product_vwap, portfolio_vwap)`` where ``per_product_vwap``
        maps each product to its market VWAP in EUR/MWh and
        ``portfolio_vwap`` is the volume-weighted average across all
        products. Both are zero when no trades were observed.
    """
    per_product: dict[str, Decimal] = {}
    total_notional = Decimal("0")
    total_volume = Decimal("0")
    for pid, (notional, volume) in accum.items():
        if volume > 0:
            per_product[pid] = Decimal(str(round(float(notional / volume), 6)))
            total_notional += notional
            total_volume += volume
    portfolio = (
        Decimal(str(round(float(total_notional / total_volume), 6)))
        if total_volume > 0
        else Decimal("0")
    )
    return per_product, portfolio


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
