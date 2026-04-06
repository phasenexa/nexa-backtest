"""PnL calculation engine.

Computes realised profit/loss and VWAP-relative performance metrics from a
list of fills and market clearing price data.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

import pandas as pd

from nexa_backtest.analysis.vwap import compute_market_vwap
from nexa_backtest.types import Fill, Side


@dataclass(frozen=True)
class SideSummary:
    """Aggregated statistics for one side (buys or sells) of the strategy.

    Attributes:
        count: Number of fills.
        volume_mwh: Total filled volume in MWh.
        avg_price: Volume-weighted average fill price in EUR/MWh.
            Zero if no fills.
        vwap_alpha: Difference between the algo's average price and the
            market VWAP in EUR/MWh.  For buys: ``avg_price - market_vwap``
            (negative = bought below VWAP, good). For sells:
            ``market_vwap - avg_price`` (negative = sold above VWAP, good).
        total_alpha_eur: ``vwap_alpha * volume_mwh`` in EUR.
        win_rate: Fraction of fills where the price beat market VWAP (buys:
            price < VWAP; sells: price > VWAP).
    """

    count: int
    volume_mwh: Decimal
    avg_price: Decimal
    vwap_alpha: Decimal
    total_alpha_eur: Decimal
    win_rate: float


@dataclass(frozen=True)
class PnlSummary:
    """Aggregated PnL metrics for a completed backtest run.

    Attributes:
        market_vwap: Volume-weighted average clearing price across **all**
            products in the backtest period (the benchmark).
        buys: Statistics for the algo's buy fills.
        sells: Statistics for the algo's sell fills.
        total_alpha_eur: Combined VWAP alpha across buys and sells in EUR.
    """

    market_vwap: Decimal
    buys: SideSummary
    sells: SideSummary
    total_alpha_eur: Decimal


def _side_summary(fills: list[Fill], market_vwap: Decimal, side: Side) -> SideSummary:
    """Compute aggregated statistics for one side of fills."""
    side_fills = [f for f in fills if f.side == side]
    if not side_fills:
        return SideSummary(
            count=0,
            volume_mwh=Decimal("0"),
            avg_price=Decimal("0"),
            vwap_alpha=Decimal("0"),
            total_alpha_eur=Decimal("0"),
            win_rate=0.0,
        )

    total_volume: Decimal = sum((f.volume for f in side_fills), Decimal("0"))
    total_cost: Decimal = sum((f.price * f.volume for f in side_fills), Decimal("0"))
    avg_price = total_cost / total_volume if total_volume else Decimal("0")

    if side == Side.BUY:
        vwap_alpha = avg_price - market_vwap  # negative = good (bought below VWAP)
        win_count = sum(1 for f in side_fills if f.price < market_vwap)
    else:
        vwap_alpha = market_vwap - avg_price  # negative = sold above VWAP (good)
        win_count = sum(1 for f in side_fills if f.price > market_vwap)

    total_alpha = -vwap_alpha * total_volume  # EUR benefit vs passive VWAP execution
    win_rate = win_count / len(side_fills)

    return SideSummary(
        count=len(side_fills),
        volume_mwh=total_volume,
        avg_price=avg_price,
        vwap_alpha=vwap_alpha,
        total_alpha_eur=total_alpha,
        win_rate=win_rate,
    )


def compute_pnl(fills: list[Fill], market_data: pd.DataFrame) -> PnlSummary:
    """Compute PnL metrics from fills and market clearing price data.

    The benchmark is the market VWAP: the volume-weighted average clearing
    price across *all* products in ``market_data``, not just those traded.
    A positive total alpha means the algo executed better than passive
    VWAP participation would have.

    Args:
        fills: All fills produced during the backtest run.
        market_data: DataFrame returned by
            :meth:`~nexa_backtest.data.loader.ParquetLoader.load_da_prices`,
            containing ``price_eur_mwh`` and ``volume_mwh`` for all products
            in the period.

    Returns:
        :class:`PnlSummary` with market VWAP and per-side metrics.
    """
    market_vwap = compute_market_vwap(market_data)

    buys = _side_summary(fills, market_vwap, Side.BUY)
    sells = _side_summary(fills, market_vwap, Side.SELL)

    total_alpha = buys.total_alpha_eur + sells.total_alpha_eur

    return PnlSummary(
        market_vwap=market_vwap,
        buys=buys,
        sells=sells,
        total_alpha_eur=total_alpha,
    )
