"""Example async algo: spread scalper using the @algo decorator.

Demonstrates the @algo decorator API for complex strategies that need
fine-grained control over the event stream. This algo watches the intraday
continuous order book and places buy orders when the spread exceeds a
threshold, aiming to capture mean-reversion.

The @algo decorator is the recommended API for quants who need:

- Access to all event types (market data, fills, gate closures, bars)
- Fine-grained control over when to act (not just at auction open)
- Clean sequential logic without class state management

Contrast with ``simple_idc_algo.py`` which uses the hook-based SimpleAlgo
approach for the same IDC market.

Run via::

    nexa run examples/async_spread_algo.py \\
        --exchange nordpool \\
        --start 2026-03-01 \\
        --end 2026-03-31 \\
        --products NO1-QH \\
        --data-dir tests/fixtures/nordpool \\
        --capital 100000
"""

from __future__ import annotations

from decimal import Decimal

from nexa_backtest.algo import algo
from nexa_backtest.context import TradingContext
from nexa_backtest.types import (
    BarEvent,
    FillEvent,
    GateClosureWarning,
    MarketDataUpdate,
    Order,
)

# Strategy parameters
MIN_SPREAD_EUR = Decimal("2.0")   # Only trade when spread > 2 EUR/MWh
MAX_POSITION_MW = Decimal("50")   # Never exceed 50 MW net long
ORDER_VOLUME_MW = Decimal("5")    # 5 MW per order
GATE_CLOSURE_BUFFER_MINUTES = 10  # Stop trading 10 minutes before gate


@algo(name="spread_scalper", version="1.0.0")
async def run(ctx: TradingContext) -> None:
    """Spread scalper that buys when the bid-ask spread is wide.

    Listens to the event stream and:
    - On MarketDataUpdate: evaluates whether the spread justifies a trade.
    - On BarEvent: logs current position at each 15-minute MTU boundary.
    - On FillEvent: logs fill details.
    - On GateClosureWarning: stops placing orders for products near gate.

    Args:
        ctx: The trading context (engine-agnostic — same code in backtest
            and live).
    """
    ctx.log("Spread scalper starting.")

    # Track which products are near gate closure.
    near_gate: set[str] = set()

    async for event in ctx.events():
        match event:
            case MarketDataUpdate(product_id=product_id):
                _handle_market_data(ctx, product_id, near_gate)

            case BarEvent(product_id=product_id):
                _handle_bar(ctx, product_id)

            case FillEvent(fill=fill):
                ctx.log(
                    f"FILL  {fill.side:<4} {fill.volume:>6} MW "
                    f"@ {fill.price:>8.2f} EUR/MWh  [{fill.product_id}]"
                )

            case GateClosureWarning(product_id=product_id):
                near_gate.add(product_id)
                ctx.log(f"Gate closure warning for {product_id} — pausing orders.")

    ctx.log("Spread scalper complete.")
    _log_summary(ctx)


# ---------------------------------------------------------------------------
# Event handlers
# ---------------------------------------------------------------------------


def _handle_market_data(
    ctx: TradingContext,
    product_id: str,
    near_gate: set[str],
) -> None:
    """Evaluate the order book and place a buy if the spread is wide enough."""
    if product_id in near_gate:
        return

    time_to_gate = ctx.time_to_gate_closure(product_id)
    if time_to_gate is not None and time_to_gate.total_seconds() < GATE_CLOSURE_BUFFER_MINUTES * 60:
        near_gate.add(product_id)
        return

    best_bid = ctx.get_best_bid(product_id)
    best_ask = ctx.get_best_ask(product_id)

    if best_bid is None or best_ask is None:
        return

    spread = best_ask.price - best_bid.price
    if spread < MIN_SPREAD_EUR:
        return

    position = ctx.get_position(product_id)
    if position is not None and position.net_mw >= MAX_POSITION_MW:
        return

    # Place a limit buy just above the best bid, inside the spread.
    bid_price = best_bid.price + Decimal("0.50")
    ctx.place_order(
        Order.buy(
            product_id=product_id,
            volume_mw=ORDER_VOLUME_MW,
            price_eur_mwh=bid_price,
        )
    )
    ctx.log(
        f"ORDER BUY  {ORDER_VOLUME_MW} MW @ {bid_price:.2f} EUR/MWh "
        f"(spread={spread:.2f})  [{product_id}]"
    )


def _handle_bar(ctx: TradingContext, product_id: str) -> None:
    """Log the net position at each MTU boundary."""
    position = ctx.get_position(product_id)
    if position and position.net_mw != Decimal("0"):
        ctx.log(
            f"BAR   position={position.net_mw:+.1f} MW  [{product_id}]  "
            f"@ {ctx.now().strftime('%H:%M')}"
        )


def _log_summary(ctx: TradingContext) -> None:
    """Log a summary of unrealised PnL at the end of the backtest."""
    unrealised = ctx.get_unrealised_pnl()
    ctx.log(f"Unrealised PnL: {unrealised:+.2f} EUR")
