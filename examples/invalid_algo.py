"""Deliberately broken algo — illustrates every validation failure category.

This file intentionally violates all six validation rules so that the
validation pipeline has something concrete to catch. Do NOT use this as a
template. See ``notebooks/05-validation_walkthrough.ipynb`` for a full
explanation of each finding.

Violations introduced:
    Step 1 - Syntax & Style (ruff):
        - ``badAlgo`` class name violates PascalCase convention (N801, warn).
        - ``calculate_alpha()`` is never defined anywhere (F821, hard error).
    Step 2 - Type Safety (mypy --strict):
        - ``on_auction_open`` is missing type annotations on ``ctx`` and
          ``auction``, triggering ``no-untyped-def``.
    Step 3 - Interface Compliance:
        - ``on_setup(self)`` is missing the required ``ctx`` parameter.
        - ``ctx.get_signal("wind_forecast")`` is called without a matching
          ``self.subscribe_signal("wind_forecast")`` in ``on_setup``.
    Step 4 - Exchange Features:
        - ``Order.buy(volume_mw=0.001, ...)`` — below the 0.1 MW minimum.
        - ``Order.buy(price_eur_mwh=9999, ...)`` — above the 3000 EUR/MWh
          Nord Pool maximum (and 4000 EUR/MWh EPEX maximum).
        - ``Order.market(...)`` — market orders are not supported on Nord
          Pool or EPEX SPOT.
    Step 5 - Look-Ahead Bias:
        - ``prices["price"].shift(-2)`` — negative shift reads future rows.
        - ``prices.sort_values("price").iloc[0]`` — sort then iloc may
          expose chronologically future data.
        - ``ctx.get_signal_history("wind_forecast", lookback=200)`` — 200
          MTUs (50 hours) is an unusually large look-back window.
    Step 6 - Resource Safety:
        - ``import requests`` — network call not available in backtest mode.
        - ``import threading`` — threading is not safe with a simulated clock.
        - ``time.sleep(0.1)`` — pauses the real clock, not the simulated one.
        - ``datetime.now()`` and ``datetime.utcnow()`` — wall-clock time;
          use ``ctx.now()`` instead.
        - ``open("prices.csv")`` inside ``on_auction_open`` — file I/O in
          the hot path; move to ``on_setup``.
        - Module-level ``PRICE_CACHE`` and ``ORDER_LOG`` are mutable; they
          leak state between backtest runs.
"""

from __future__ import annotations

import time  # Step 6: enables time.sleep() below
import requests  # Step 6: network call — not available in backtest mode  # noqa: F401
import threading  # Step 6: threading warning  # noqa: F401
from datetime import datetime  # Step 6: wall-clock time
from decimal import Decimal

import pandas as pd

from nexa_backtest.algo import SimpleAlgo
from nexa_backtest.context import TradingContext
from nexa_backtest.types import AuctionInfo, Fill, Order  # noqa: F401

# Step 6: Mutable module-level variables leak state between backtest runs.
PRICE_CACHE: list = []
ORDER_LOG: dict = {}


class badAlgo(SimpleAlgo):  # Step 1: N801 — class name should be PascalCase (BadAlgo)
    """An algo that deliberately violates every validation rule."""

    # Step 3: on_setup is missing the required 'ctx' parameter.
    def on_setup(self) -> None:  # type: ignore[override]
        self.threshold = 5  # int, not Decimal — type error when used in arithmetic

    def on_auction_open(self, ctx, auction) -> None:  # Step 2: missing type annotations
        # Step 6: datetime.now() returns wall-clock time — use ctx.now() instead.
        now = datetime.now()
        _ = now  # suppress "unused variable" noise

        # Step 6: time.sleep() pauses the real clock, not the simulated clock.
        time.sleep(0.1)

        # Step 5: negative shift — accesses 2 future rows of price data.
        prices = pd.DataFrame({"price": [100.0, 200.0, 300.0]})
        _ = prices["price"].shift(-2)

        # Step 5: sort_values().iloc may access chronologically future rows.
        _ = prices.sort_values("price").iloc[0]

        # Step 6: file I/O inside a hot-path hook (runs on every auction tick).
        with open("prices.csv") as f:  # noqa: PTH123
            _ = f.read()

        # Step 3: get_signal without a matching subscribe_signal in on_setup.
        _ = ctx.get_signal("wind_forecast")

        # Step 4: literal floats so the feature check can evaluate them statically.
        # volume_mw=0.001 < 0.1 MW minimum; price_eur_mwh=9999 > 3000 (Nord Pool) / 4000 (EPEX).
        ctx.place_order(
            Order.buy(
                product_id=auction.product_id,
                volume_mw=0.001,   # type: ignore[arg-type]
                price_eur_mwh=9999,  # type: ignore[arg-type]
            )
        )

        # Step 4: Order.market() is not supported on Nord Pool or EPEX SPOT.
        ctx.place_order(
            Order.market(
                product_id=auction.product_id,
                volume_mw=10,   # type: ignore[arg-type]
            )
        )

        # Step 5: lookback=200 MTUs (50 h) is an unusually large window.
        _ = ctx.get_signal_history("wind_forecast", lookback=200)

        # Step 1: F821 hard error — calculate_alpha is never defined anywhere in this file.
        result = calculate_alpha()  # type: ignore[name-defined]
        _ = result

    def on_fill(self, ctx: TradingContext, fill: Fill) -> None:
        # Step 6: datetime.utcnow() — use ctx.now() instead.
        t = datetime.utcnow()
        ctx.log(f"Fill at {t}")

    def on_teardown(self, ctx: TradingContext) -> None:
        pass
