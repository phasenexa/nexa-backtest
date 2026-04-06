"""Exception hierarchy for nexa-backtest."""


class NexaBacktestError(Exception):
    """Base exception for all nexa-backtest errors."""


class AlgoError(NexaBacktestError):
    """Raised when user algo code raises an unexpected error."""


class DataError(NexaBacktestError):
    """Raised when data loading or schema validation fails."""


class ExchangeError(NexaBacktestError):
    """Raised when an exchange adapter encounters an error."""


class ValidationError(NexaBacktestError):
    """Raised when pre-run validation fails."""


class UnsupportedFeatureError(ValidationError):
    """Raised when an algo uses a feature not supported by the target exchange."""


class MatchingError(NexaBacktestError):
    """Raised when the matching engine encounters an error."""


class SignalError(NexaBacktestError):
    """Raised when a signal provider encounters an error."""
