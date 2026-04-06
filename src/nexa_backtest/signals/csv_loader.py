"""CsvSignalProvider: load a CSV file as a signal.

This is the simplest way for a customer to bring their own forecast data
without implementing :class:`~nexa_backtest.signals.base.SignalProvider`
from scratch.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from nexa_backtest.context import SignalValue
from nexa_backtest.data.schema import SIGNAL_CSV_TIMESTAMP_COL, SIGNAL_CSV_VALUE_COL
from nexa_backtest.exceptions import DataError, SignalError
from nexa_backtest.signals.base import SignalSchema

logger = logging.getLogger(__name__)


class CsvSignalProvider:
    """Loads a CSV file and serves it as a named signal.

    Expected CSV format::

        timestamp,value
        2026-03-15T00:00:00+01:00,42.31
        2026-03-15T00:15:00+01:00,41.87

    The ``timestamp`` column must contain timezone-aware datetimes in ISO 8601
    format. The ``value`` column must be numeric (parsed as float). Additional
    columns are silently ignored.

    **Look-ahead bias and publication_offset**

    Set ``publication_offset`` for forecast data. A value with timestamp T
    (the delivery period it describes) was published at T - publication_offset,
    so it becomes visible when the simulated clock reaches T - publication_offset.

    In code: ``get_value(current_time)`` returns the latest row where::

        row.timestamp <= current_time + publication_offset

    Example: a wind forecast published 6 hours ahead (``publication_offset=
    timedelta(hours=6)``). At 00:00, only forecasts up to 06:00 are visible.

    If ``publication_offset`` is not set a warning is logged, because values
    will be available at their exact timestamp — correct for actuals but
    look-ahead bias for forecasts.

    Args:
        name: Signal name used for registration and lookup.
        path: Path to the CSV file. Resolved at construction time.
        unit: Physical unit string, e.g. ``"EUR/MWh"`` or ``"MW"``.
        description: Human-readable description of the signal.
        frequency: Expected update cadence. Defaults to 1-hour intervals.
        publication_offset: How far ahead of delivery the forecast was
            published. See above for semantics. Pass ``None`` for actuals.

    Raises:
        :class:`~nexa_backtest.exceptions.DataError`: If the file is missing,
            lacks required columns, or contains unparseable values.
    """

    def __init__(
        self,
        name: str,
        path: str | Path,
        unit: str,
        description: str,
        frequency: timedelta = timedelta(hours=1),
        publication_offset: timedelta | None = None,
    ) -> None:
        self._name = name
        self._path = Path(path)
        self._unit = unit
        self._description = description
        self._frequency = frequency
        self._publication_offset = publication_offset

        if publication_offset is None:
            logger.warning(
                "Signal '%s' has no publication_offset. Values are available at their "
                "timestamp. This is correct for actuals but may introduce look-ahead "
                "bias for forecasts — consider whether your data is a forecast.",
                name,
            )

        self._data = self._load()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self) -> pd.DataFrame:
        """Load, validate, and sort the CSV file."""
        if not self._path.exists():
            raise DataError(
                f"Signal CSV not found: {self._path}. Check the file path or the data directory."
            )

        try:
            df = pd.read_csv(self._path)
        except Exception as exc:
            raise DataError(f"Failed to read signal CSV '{self._path}': {exc}") from exc

        for col in (SIGNAL_CSV_TIMESTAMP_COL, SIGNAL_CSV_VALUE_COL):
            if col not in df.columns:
                raise DataError(
                    f"Signal CSV '{self._path}' is missing required column '{col}'. "
                    f"Found columns: {list(df.columns)}"
                )

        try:
            df[SIGNAL_CSV_TIMESTAMP_COL] = pd.to_datetime(df[SIGNAL_CSV_TIMESTAMP_COL], utc=True)
        except Exception as exc:
            raise DataError(
                f"Signal CSV '{self._path}' contains invalid timestamps: {exc}"
            ) from exc

        try:
            df[SIGNAL_CSV_VALUE_COL] = df[SIGNAL_CSV_VALUE_COL].astype(float)
        except Exception as exc:
            raise DataError(
                f"Signal CSV '{self._path}' contains non-numeric values "
                f"in column '{SIGNAL_CSV_VALUE_COL}': {exc}"
            ) from exc

        result = (
            df[[SIGNAL_CSV_TIMESTAMP_COL, SIGNAL_CSV_VALUE_COL]]
            .sort_values(SIGNAL_CSV_TIMESTAMP_COL)
            .reset_index(drop=True)
        )
        return result

    def _effective_cutoff(self, current_time: datetime) -> pd.Timestamp:
        """Return the timestamp cutoff for values visible at ``current_time``."""
        offset = self._publication_offset if self._publication_offset is not None else timedelta(0)
        return pd.Timestamp(current_time + offset)

    # ------------------------------------------------------------------
    # SignalProvider protocol
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Signal name."""
        return self._name

    @property
    def publication_offset(self) -> timedelta | None:
        """Publication offset, or ``None`` if unset (actuals mode)."""
        return self._publication_offset

    @property
    def schema(self) -> SignalSchema:
        """Schema metadata describing this signal."""
        return SignalSchema(
            name=self._name,
            dtype=float,
            frequency=self._frequency,
            description=self._description,
            unit=self._unit,
        )

    def get_value(self, timestamp: datetime) -> SignalValue:
        """Return the latest value visible at ``timestamp``.

        Respects ``publication_offset``: only rows whose delivery timestamp
        is <= ``timestamp + publication_offset`` are considered.

        Args:
            timestamp: Current simulated time.

        Returns:
            Most recent :class:`~nexa_backtest.context.SignalValue` visible
            at ``timestamp``.

        Raises:
            :class:`~nexa_backtest.exceptions.SignalError`: If no value is
                visible at ``timestamp`` (e.g. all values are in the future).
        """
        cutoff = self._effective_cutoff(timestamp)
        mask = self._data[SIGNAL_CSV_TIMESTAMP_COL] <= cutoff
        visible = self._data[mask]

        if visible.empty:
            first_ts = self._data[SIGNAL_CSV_TIMESTAMP_COL].iloc[0]
            raise SignalError(
                f"No value available for signal '{self._name}' at time {timestamp}. "
                f"Earliest data timestamp: {first_ts}. "
                f"publication_offset: {self._publication_offset}."
            )

        row = visible.iloc[-1]
        ts: datetime = row[SIGNAL_CSV_TIMESTAMP_COL].to_pydatetime()
        val: float = float(row[SIGNAL_CSV_VALUE_COL])
        return SignalValue(name=self._name, timestamp=ts, value=val)

    def get_range(self, start: datetime, end: datetime) -> pd.Series[Any]:
        """Return values in the half-open interval ``[start, end)``.

        Returns raw delivery-timestamp values without applying
        ``publication_offset``.

        Args:
            start: Start of the range (inclusive).
            end: End of the range (exclusive).

        Returns:
            :class:`pandas.Series` with a UTC datetime index and float values,
            sorted ascending.
        """
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)

        mask = (self._data[SIGNAL_CSV_TIMESTAMP_COL] >= start_ts) & (
            self._data[SIGNAL_CSV_TIMESTAMP_COL] < end_ts
        )
        subset = self._data[mask].copy()
        return subset.set_index(SIGNAL_CSV_TIMESTAMP_COL)[SIGNAL_CSV_VALUE_COL]

    def get_history_at(self, timestamp: datetime, lookback: int) -> list[SignalValue]:
        """Return the most recent ``lookback`` values visible at ``timestamp``.

        Respects ``publication_offset`` so no future values are included.

        Args:
            timestamp: Current simulated time.
            lookback: Maximum number of historical values to return.

        Returns:
            Signal values in chronological order (oldest first), with at most
            ``lookback`` entries.
        """
        cutoff = self._effective_cutoff(timestamp)
        mask = self._data[SIGNAL_CSV_TIMESTAMP_COL] <= cutoff
        visible = self._data[mask].tail(lookback)

        result: list[SignalValue] = []
        for _, row in visible.iterrows():
            ts: datetime = row[SIGNAL_CSV_TIMESTAMP_COL].to_pydatetime()
            val: float = float(row[SIGNAL_CSV_VALUE_COL])
            result.append(SignalValue(name=self._name, timestamp=ts, value=val))
        return result
