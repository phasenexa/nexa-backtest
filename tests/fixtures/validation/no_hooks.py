"""SimpleAlgo with no hooks implemented — should error in step 3."""

from __future__ import annotations

from nexa_backtest.algo import SimpleAlgo
from nexa_backtest.context import TradingContext  # noqa: F401


class NoHooksAlgo(SimpleAlgo):
    """An algo that does absolutely nothing."""

    pass
