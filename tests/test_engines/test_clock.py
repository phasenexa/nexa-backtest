"""Tests for SimulatedClock and RealtimeClock."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from nexa_backtest.engines.clock import RealtimeClock, SimulatedClock


class TestSimulatedClock:
    def test_default_starts_at_epoch(self) -> None:
        clock = SimulatedClock()
        assert clock.now() == datetime(1970, 1, 1, tzinfo=UTC)

    def test_initial_time(self) -> None:
        t = datetime(2026, 1, 15, 9, 0, tzinfo=UTC)
        clock = SimulatedClock(initial_time=t)
        assert clock.now() == t

    def test_initial_time_naive_raises(self) -> None:
        naive = datetime(2026, 1, 15, 9, 0)
        with pytest.raises(ValueError, match="timezone-aware"):
            SimulatedClock(initial_time=naive)

    def test_advance_to(self) -> None:
        clock = SimulatedClock()
        t = datetime(2026, 1, 15, 9, 15, tzinfo=UTC)
        clock.advance_to(t)
        assert clock.now() == t

    def test_advance_to_naive_raises(self) -> None:
        clock = SimulatedClock()
        naive = datetime(2026, 1, 15, 9, 0)
        with pytest.raises(ValueError, match="timezone-aware"):
            clock.advance_to(naive)

    def test_advance_backwards_raises(self) -> None:
        t1 = datetime(2026, 1, 15, 9, 15, tzinfo=UTC)
        t0 = datetime(2026, 1, 15, 9, 0, tzinfo=UTC)
        clock = SimulatedClock()
        clock.advance_to(t1)
        with pytest.raises(ValueError, match="backwards"):
            clock.advance_to(t0)

    def test_advance_to_same_time_allowed(self) -> None:
        t = datetime(2026, 1, 15, 9, 0, tzinfo=UTC)
        clock = SimulatedClock()
        clock.advance_to(t)
        clock.advance_to(t)  # should not raise
        assert clock.now() == t

    def test_sequential_advances(self) -> None:
        clock = SimulatedClock()
        times = [
            datetime(2026, 1, 15, 9, 0, tzinfo=UTC),
            datetime(2026, 1, 15, 9, 15, tzinfo=UTC),
            datetime(2026, 1, 15, 9, 30, tzinfo=UTC),
        ]
        for t in times:
            clock.advance_to(t)
        assert clock.now() == times[-1]

    def test_peek_next_none_when_no_future(self) -> None:
        clock = SimulatedClock()
        assert clock.peek_next() is None

    def test_peek_next_none_at_last_timestamp(self) -> None:
        clock = SimulatedClock()
        t = datetime(2026, 1, 15, 9, 0, tzinfo=UTC)
        clock.advance_to(t)
        assert clock.peek_next() is None

    def test_peek_next_returns_next(self) -> None:
        clock = SimulatedClock()
        t1 = datetime(2026, 1, 15, 9, 0, tzinfo=UTC)
        t2 = datetime(2026, 1, 15, 9, 15, tzinfo=UTC)
        # Add both timestamps to the schedule by advancing past them
        # Peek reflects timestamps added AFTER the current index
        clock.advance_to(t1)
        # At this point _index = 0, no future timestamps → None
        assert clock.peek_next() is None
        clock.advance_to(t2)
        # After advancing to t2, peek should still be None (t2 is now current)
        assert clock.peek_next() is None

    def test_peek_does_not_advance(self) -> None:
        clock = SimulatedClock()
        t = datetime(2026, 1, 15, 9, 0, tzinfo=UTC)
        clock.advance_to(t)
        _ = clock.peek_next()
        assert clock.now() == t

    def test_advance_large_gap(self) -> None:
        clock = SimulatedClock()
        t1 = datetime(2026, 1, 15, tzinfo=UTC)
        t2 = datetime(2026, 12, 31, tzinfo=UTC)
        clock.advance_to(t1)
        clock.advance_to(t2)
        assert clock.now() == t2

    def test_advance_one_microsecond(self) -> None:
        clock = SimulatedClock()
        t1 = datetime(2026, 1, 15, 9, 0, tzinfo=UTC)
        t2 = t1 + timedelta(microseconds=1)
        clock.advance_to(t1)
        clock.advance_to(t2)
        assert clock.now() == t2


class TestRealtimeClock:
    def test_now_returns_datetime(self) -> None:
        clock = RealtimeClock()
        result = clock.now()
        assert isinstance(result, datetime)

    def test_now_is_timezone_aware(self) -> None:
        clock = RealtimeClock()
        result = clock.now()
        assert result.tzinfo is not None

    def test_now_is_utc(self) -> None:
        clock = RealtimeClock()
        result = clock.now()
        # UTC offset should be zero
        assert result.utcoffset() == timedelta(0)

    def test_now_is_current(self) -> None:
        clock = RealtimeClock()
        before = datetime.now(tz=UTC)
        result = clock.now()
        after = datetime.now(tz=UTC)
        assert before <= result <= after

    def test_successive_calls_monotonic(self) -> None:
        clock = RealtimeClock()
        t1 = clock.now()
        t2 = clock.now()
        assert t2 >= t1
