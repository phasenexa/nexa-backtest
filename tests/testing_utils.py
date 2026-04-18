"""Internal test utilities for nexa-backtest.

These helpers intentionally access private engine implementation details to
support unit tests that need to construct engine state without running a full
backtest.  They must not be imported from algo code.
"""

from __future__ import annotations

from datetime import UTC, datetime

from nexa_backtest.engines.backtest import _BacktestContext
from nexa_backtest.engines.clock import SimulatedClock
from nexa_backtest.signals.registry import SignalRegistry


def make_minimal_backtest_context(
    products: dict[str, datetime] | None = None,
    initial_time: datetime | None = None,
) -> _BacktestContext:
    """Create a minimal ``_BacktestContext`` for engine unit tests.

    Args:
        products: Optional mapping of ``product_id → delivery_start`` to
            pre-populate ``_product_delivery_starts``.
        initial_time: Simulated clock start time.  Defaults to
            ``2026-03-01T09:00:00Z``.

    Returns:
        A ``_BacktestContext`` with no fills, no IDC engine, and no model
        registry.  Callers may call ``ctx._record_fill(fill)`` and access
        ``ctx._product_delivery_starts`` directly for state setup.
    """
    clock = SimulatedClock(
        initial_time=initial_time or datetime(2026, 3, 1, 9, 0, tzinfo=UTC),
    )
    ctx = _BacktestContext(clock=clock, signal_registry=SignalRegistry())
    if products:
        ctx._product_delivery_starts = products
    return ctx
