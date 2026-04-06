"""nexa-backtest: Backtesting framework for European power markets.

Key public exports for Task 01. Additional exports will be added as
subsequent tasks are implemented.
"""

from nexa_backtest._version import __version__
from nexa_backtest.context import SignalValue, TradingContext
from nexa_backtest.exceptions import (
    AlgoError,
    DataError,
    ExchangeError,
    MatchingError,
    NexaBacktestError,
    SignalError,
    UnsupportedFeatureError,
    ValidationError,
)
from nexa_backtest.types import (
    MTU,
    AuctionInfo,
    CancelResult,
    Fill,
    Order,
    OrderBook,
    OrderResult,
    OrderStatus,
    Position,
    PriceLevel,
    Side,
)

__all__ = [
    "MTU",
    "AlgoError",
    "AuctionInfo",
    "CancelResult",
    "DataError",
    "ExchangeError",
    "Fill",
    "MatchingError",
    "NexaBacktestError",
    "Order",
    "OrderBook",
    "OrderResult",
    "OrderStatus",
    "Position",
    "PriceLevel",
    "Side",
    "SignalError",
    "SignalValue",
    "TradingContext",
    "UnsupportedFeatureError",
    "ValidationError",
    "__version__",
]
