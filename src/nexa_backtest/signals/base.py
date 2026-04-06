"""SignalProvider protocol and supporting types.

This module defines the interface that signal data sources must implement.
All signal providers must respect ``publication_offset`` to prevent look-ahead
bias: at simulated time T, only values published before T are visible.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Protocol

import pandas as pd
from pydantic import BaseModel, ConfigDict

from nexa_backtest.context import SignalValue


class SignalSchema(BaseModel):
    """Describes the structure and semantics of a signal.

    Attributes:
        name: Signal identifier, e.g. ``"wind_generation_forecast"``.
        dtype: Python type of the value field, e.g. ``float``.
        frequency: How often the signal updates, e.g. ``timedelta(minutes=15)``.
        description: Human-readable description of what the signal represents.
        unit: Physical unit of the value, e.g. ``"EUR/MWh"``, ``"MW"``, ``"m/s"``.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    dtype: type[Any]
    frequency: timedelta
    description: str
    unit: str


class SignalProvider(Protocol):
    """Protocol for signal data sources.

    Implementations must respect ``publication_offset`` to prevent look-ahead
    bias: ``get_value(t)`` must only return data that was known at time ``t``.

    Custom signal providers only need to implement this protocol. The simplest
    path is :class:`~nexa_backtest.signals.csv_loader.CsvSignalProvider`.
    """

    @property
    def name(self) -> str:
        """Signal name used to register and look up this provider."""
        ...

    @property
    def schema(self) -> SignalSchema:
        """Metadata describing the signal's content and structure."""
        ...

    def get_value(self, timestamp: datetime) -> SignalValue:
        """Return the latest value visible at ``timestamp``.

        Respects ``publication_offset`` so that future values are never
        returned: only values whose publication time is <= ``timestamp``.

        Args:
            timestamp: The current simulated time.

        Returns:
            The most recent :class:`~nexa_backtest.context.SignalValue`
            visible at ``timestamp``.

        Raises:
            :class:`~nexa_backtest.exceptions.SignalError`: If no value is
                available at ``timestamp``.
        """
        ...

    def get_range(self, start: datetime, end: datetime) -> pd.Series[Any]:
        """Return all values in the half-open interval ``[start, end)``.

        Returns raw values by their delivery timestamp without applying
        ``publication_offset`` filtering.

        Args:
            start: Start of the range (inclusive).
            end: End of the range (exclusive).

        Returns:
            A :class:`pandas.Series` of float values indexed by their delivery
            timestamps in ascending order.
        """
        ...

    def get_history_at(self, timestamp: datetime, lookback: int) -> list[SignalValue]:
        """Return the most recent ``lookback`` values visible at ``timestamp``.

        Like :meth:`get_value`, this method respects ``publication_offset``
        so no future values are returned.

        Args:
            timestamp: The current simulated time.
            lookback: Maximum number of historical values to return.

        Returns:
            Signal values in chronological order (oldest first), capped at
            ``lookback`` entries.
        """
        ...
