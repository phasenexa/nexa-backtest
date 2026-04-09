"""@algo function that never calls ctx.events() — should warn in step 3."""

from __future__ import annotations

from nexa_backtest.algo import algo
from nexa_backtest.context import TradingContext


@algo(name="no_events", version="1.0.0")
async def run(ctx: TradingContext) -> None:
    # This algo never consumes any events — it will do nothing.
    ctx.log("Starting but never iterating events")
