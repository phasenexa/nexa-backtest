"""Clock implementations for backtest and real-time engines.

Two implementations are provided:

- ``SimulatedClock``: advances instantly between timestamps. Used by the
  backtest engine to replay historical data.
- ``RealtimeClock``: returns the actual wall-clock time. Used by paper and
  live engines.

Both implement the ``Clock`` protocol so engine code can be written against
the protocol rather than a concrete class.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Protocol


class Clock(Protocol):
    """Minimal clock interface required by all engines."""

    def now(self) -> datetime:
        """Return the current time as a timezone-aware datetime.

        Returns:
            Current time. Always timezone-aware.
        """
        ...


class SimulatedClock:
    """A clock that advances instantly to explicit timestamps.

    Used by the backtest engine to replay historical data without waiting for
    real time to pass. The clock starts at the first timestamp supplied to
    ``advance_to`` or at the provided initial time.

    Args:
        initial_time: Optional starting time. If omitted the clock starts at
            the Unix epoch in UTC.

    Example:
        >>> from datetime import datetime, timezone
        >>> clock = SimulatedClock()
        >>> t1 = datetime(2026, 1, 1, 8, 0, tzinfo=timezone.utc)
        >>> t2 = datetime(2026, 1, 1, 8, 15, tzinfo=timezone.utc)
        >>> clock.advance_to(t1)
        >>> clock.advance_to(t2)
        >>> clock.now() == t2
        True
    """

    def __init__(self, initial_time: datetime | None = None) -> None:
        if initial_time is not None:
            if initial_time.tzinfo is None:
                raise ValueError("SimulatedClock requires a timezone-aware initial_time")
            self._current: datetime = initial_time
        else:
            self._current = datetime(1970, 1, 1, tzinfo=UTC)
        self._timestamps: list[datetime] = []
        self._index: int = -1

    def now(self) -> datetime:
        """Return the current simulated time.

        Returns:
            Current simulated time as a timezone-aware datetime.
        """
        return self._current

    def advance_to(self, timestamp: datetime) -> None:
        """Advance the clock to a specific timestamp.

        The timestamp must be at or after the current simulated time.

        Args:
            timestamp: Target time. Must be timezone-aware and >= ``now()``.

        Raises:
            ValueError: If ``timestamp`` is naive or earlier than ``now()``.
        """
        if timestamp.tzinfo is None:
            raise ValueError("advance_to requires a timezone-aware timestamp")
        if timestamp < self._current:
            raise ValueError(f"Cannot advance clock backwards: {timestamp} < {self._current}")
        self._current = timestamp
        self._timestamps.append(timestamp)
        self._index = len(self._timestamps) - 1

    def peek_next(self) -> datetime | None:
        """Return the next scheduled timestamp without advancing the clock.

        Returns:
            The next timestamp in the schedule, or ``None`` if no further
            timestamps have been added after the current position.
        """
        next_index = self._index + 1
        if next_index < len(self._timestamps):
            return self._timestamps[next_index]
        return None


class RealtimeClock:
    """A clock backed by the real system wall clock.

    Used by paper and live engines where simulated time is not needed.
    Always returns a UTC-aware datetime.

    Example:
        >>> clock = RealtimeClock()
        >>> clock.now().tzinfo is not None
        True
    """

    def now(self) -> datetime:
        """Return the current wall-clock time in UTC.

        Returns:
            Current UTC time as a timezone-aware datetime.
        """
        return datetime.now(tz=UTC)
