"""nexa-backtest: Backtesting framework for European power markets."""

from nexa_backtest._version import __version__
from nexa_backtest.algo import SimpleAlgo, algo
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
from nexa_backtest.exchanges.epex_spot import EpexSpotAdapter
from nexa_backtest.exchanges.nordpool import NordPoolAdapter
from nexa_backtest.signals.base import SignalProvider, SignalSchema
from nexa_backtest.signals.csv_loader import CsvSignalProvider
from nexa_backtest.signals.registry import SignalRegistry
from nexa_backtest.types import (
    MTU,
    AuctionInfo,
    BarEvent,
    CancelEvent,
    CancelResult,
    DeliveryPosition,
    EquitySnapshot,
    Fill,
    FillEvent,
    GateClosureEvent,
    GateClosureSnapshot,
    GateClosureWarning,
    HistoricalTrade,
    MarketDataUpdate,
    MarketEvent,
    Order,
    OrderBook,
    OrderResult,
    OrderStatus,
    Position,
    PriceLevel,
    Side,
    SignalUpdate,
)

__all__ = [
    "MTU",
    "AlgoError",
    "AuctionInfo",
    "BacktestEngine",
    "BacktestResult",
    "BarEvent",
    "CancelEvent",
    "CancelResult",
    "CsvSignalProvider",
    "DailyPnL",
    "DataError",
    "DeliveryPosition",
    "EpexSpotAdapter",
    "EquitySnapshot",
    "ExchangeError",
    "Fill",
    "FillEvent",
    "GateClosureEvent",
    "GateClosureSnapshot",
    "GateClosureWarning",
    "HistoricalTrade",
    "MarketDataUpdate",
    "MarketEvent",
    "MatchingError",
    "NexaBacktestError",
    "NordPoolAdapter",
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
    "SignalUpdate",
    "SignalValue",
    "SimpleAlgo",
    "TradingContext",
    "UnsupportedFeatureError",
    "ValidationError",
    "__version__",
    "algo",
]
