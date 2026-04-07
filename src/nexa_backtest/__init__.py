"""nexa-backtest: Backtesting framework for European power markets."""

from nexa_backtest._version import __version__
from nexa_backtest.algo import SimpleAlgo
from nexa_backtest.analysis.metrics import BacktestResult, DailyPnL
from nexa_backtest.context import SignalValue, TradingContext
from nexa_backtest.engines.backtest import BacktestEngine
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
from nexa_backtest.signals.base import SignalProvider, SignalSchema
from nexa_backtest.signals.csv_loader import CsvSignalProvider
from nexa_backtest.signals.registry import SignalRegistry
from nexa_backtest.types import (
    MTU,
    AuctionInfo,
    CancelResult,
    EquitySnapshot,
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
    "BacktestEngine",
    "BacktestResult",
    "CancelResult",
    "CsvSignalProvider",
    "DailyPnL",
    "DataError",
    "EquitySnapshot",
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
    "SignalProvider",
    "SignalRegistry",
    "SignalSchema",
    "SignalValue",
    "SimpleAlgo",
    "TradingContext",
    "UnsupportedFeatureError",
    "ValidationError",
    "__version__",
]
