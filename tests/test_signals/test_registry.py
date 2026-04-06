"""Tests for signals/registry.py."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import pytest

from nexa_backtest.context import SignalValue
from nexa_backtest.exceptions import SignalError
from nexa_backtest.signals.base import SignalSchema
from nexa_backtest.signals.registry import SignalRegistry


class _FakeProvider:
    """Minimal SignalProvider implementation for registry tests."""

    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def schema(self) -> SignalSchema:
        return SignalSchema(
            name=self._name,
            dtype=float,
            frequency=timedelta(hours=1),
            description="",
            unit="",
        )

    def get_value(self, timestamp: datetime) -> SignalValue:
        return SignalValue(name=self._name, timestamp=timestamp, value=0.0)

    def get_range(self, start: datetime, end: datetime) -> pd.Series[Any]:
        return pd.Series(dtype=float)

    def get_history_at(self, timestamp: datetime, lookback: int) -> list[SignalValue]:
        return []


class TestSignalRegistry:
    def test_register_and_get(self):
        registry = SignalRegistry()
        provider = _FakeProvider("wind")
        registry.register(provider)
        assert registry.get("wind") is provider

    def test_has(self):
        registry = SignalRegistry()
        assert not registry.has("wind")
        registry.register(_FakeProvider("wind"))
        assert registry.has("wind")

    def test_get_missing_raises(self):
        registry = SignalRegistry()
        with pytest.raises(SignalError, match="not found"):
            registry.get("missing")

    def test_get_missing_lists_available(self):
        registry = SignalRegistry()
        registry.register(_FakeProvider("alpha"))
        registry.register(_FakeProvider("beta"))
        with pytest.raises(SignalError, match="alpha"):
            registry.get("gamma")

    def test_list_signals_sorted(self):
        registry = SignalRegistry()
        registry.register(_FakeProvider("zebra"))
        registry.register(_FakeProvider("alpha"))
        assert registry.list_signals() == ["alpha", "zebra"]

    def test_list_signals_empty(self):
        registry = SignalRegistry()
        assert registry.list_signals() == []

    def test_register_replaces_existing(self):
        registry = SignalRegistry()
        p1 = _FakeProvider("sig")
        p2 = _FakeProvider("sig")
        registry.register(p1)
        registry.register(p2)
        assert registry.get("sig") is p2
