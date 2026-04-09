"""Valid @algo decorated async function: passes all validation steps."""

from __future__ import annotations

from decimal import Decimal

from nexa_backtest.algo import algo
from nexa_backtest.context import TradingContext
from nexa_backtest.types import BarEvent, Order


@algo(name="valid_async", version="1.0.0")
async def run(ctx: TradingContext) -> None:
    async for event in ctx.events():
        if isinstance(event, BarEvent):
            ctx.place_order(
                Order.buy(
                    product_id=event.product_id,
                    volume_mw=Decimal("5"),
                    price_eur_mwh=Decimal("45"),
                )
            )
