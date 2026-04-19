"""Shared utilities for validation check modules."""

from __future__ import annotations

import time


def _elapsed_ms(start: float) -> int:
    """Return milliseconds elapsed since ``start`` (from :func:`time.monotonic`)."""
    return int((time.monotonic() - start) * 1000)
