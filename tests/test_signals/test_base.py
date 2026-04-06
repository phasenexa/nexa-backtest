"""Tests for signals/base.py: SignalSchema and SignalProvider protocol."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError as PydanticValidationError

from nexa_backtest.context import SignalValue
from nexa_backtest.signals.base import SignalSchema


class TestSignalSchema:
    def test_construction(self):
        schema = SignalSchema(
            name="wind_forecast",
            dtype=float,
            frequency=timedelta(minutes=15),
            description="Wind generation forecast for NO1",
            unit="MW",
        )
        assert schema.name == "wind_forecast"
        assert schema.dtype is float
        assert schema.frequency == timedelta(minutes=15)
        assert schema.description == "Wind generation forecast for NO1"
        assert schema.unit == "MW"

    def test_immutable(self):
        schema = SignalSchema(
            name="test",
            dtype=float,
            frequency=timedelta(hours=1),
            description="test",
            unit="MW",
        )
        with pytest.raises(PydanticValidationError):  # frozen model
            schema.name = "other"  # type: ignore[misc]

    def test_int_dtype(self):
        schema = SignalSchema(
            name="count",
            dtype=int,
            frequency=timedelta(hours=1),
            description="A count",
            unit="",
        )
        assert schema.dtype is int


class TestSignalValue:
    """Tests for the SignalValue used by signals (defined in context.py)."""

    def test_construction(self):
        ts = datetime(2026, 3, 1, 12, 0, tzinfo=UTC)
        sv = SignalValue(name="test_signal", timestamp=ts, value=42.5)
        assert sv.name == "test_signal"
        assert sv.timestamp == ts
        assert sv.value == pytest.approx(42.5)

    def test_immutable(self):
        ts = datetime(2026, 3, 1, tzinfo=UTC)
        sv = SignalValue(name="s", timestamp=ts, value=1.0)
        with pytest.raises(PydanticValidationError):  # frozen model
            sv.value = 2.0  # type: ignore[misc]

    def test_value_is_float(self):
        ts = datetime(2026, 3, 1, tzinfo=UTC)
        sv = SignalValue(name="s", timestamp=ts, value=100)
        assert isinstance(sv.value, float)
