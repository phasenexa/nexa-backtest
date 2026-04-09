"""Shared fixtures for validation tests."""

from __future__ import annotations

from pathlib import Path

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "validation"


def fixture_path(name: str) -> str:
    """Return the absolute path to a validation fixture file."""
    return str(FIXTURES_DIR / name)
