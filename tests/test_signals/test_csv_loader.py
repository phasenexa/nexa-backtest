"""Tests for signals/csv_loader.py: CsvSignalProvider."""

from __future__ import annotations

import textwrap
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

from nexa_backtest.exceptions import DataError, SignalError
from nexa_backtest.signals.base import SignalSchema
from nexa_backtest.signals.csv_loader import CsvSignalProvider

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VALID_CSV = textwrap.dedent("""\
    timestamp,value
    2026-03-01T00:00:00+00:00,40.00
    2026-03-01T00:15:00+00:00,41.00
    2026-03-01T00:30:00+00:00,42.00
    2026-03-01T00:45:00+00:00,43.00
    2026-03-01T01:00:00+00:00,44.00
""")


def _make_csv(tmp_path: Path, content: str, name: str = "signal.csv") -> Path:
    p = tmp_path / name
    p.write_text(content)
    return p


def _make_provider(
    tmp_path: Path,
    content: str = _VALID_CSV,
    publication_offset: timedelta | None = None,
) -> CsvSignalProvider:
    path = _make_csv(tmp_path, content)
    return CsvSignalProvider(
        name="test_signal",
        path=path,
        unit="EUR/MWh",
        description="Test",
        publication_offset=publication_offset,
    )


# ---------------------------------------------------------------------------
# Construction and loading
# ---------------------------------------------------------------------------


class TestCsvSignalProviderLoad:
    def test_loads_valid_csv(self, tmp_path):
        provider = _make_provider(tmp_path)
        assert provider.name == "test_signal"

    def test_missing_file_raises_data_error(self, tmp_path):
        with pytest.raises(DataError, match="not found"):
            CsvSignalProvider(name="x", path=tmp_path / "missing.csv", unit="MW", description="")

    def test_missing_timestamp_column_raises(self, tmp_path):
        csv = "val\n1.0\n2.0\n"
        with pytest.raises(DataError, match="timestamp"):
            _make_provider(tmp_path, content=csv)

    def test_missing_value_column_raises(self, tmp_path):
        csv = "timestamp\n2026-03-01T00:00:00+00:00\n"
        with pytest.raises(DataError, match="value"):
            _make_provider(tmp_path, content=csv)

    def test_bad_timestamps_raise_data_error(self, tmp_path):
        csv = "timestamp,value\nnot-a-date,1.0\n"
        with pytest.raises(DataError, match="timestamp"):
            _make_provider(tmp_path, content=csv)

    def test_bad_values_raise_data_error(self, tmp_path):
        csv = "timestamp,value\n2026-03-01T00:00:00+00:00,abc\n"
        with pytest.raises(DataError, match="non-numeric"):
            _make_provider(tmp_path, content=csv)

    def test_extra_columns_ignored(self, tmp_path):
        csv = "timestamp,value,zone\n2026-03-01T00:00:00+00:00,40.0,NO1\n"
        provider = _make_provider(tmp_path, content=csv)
        v = provider.get_value(datetime(2026, 3, 1, 1, tzinfo=UTC))
        assert v.value == pytest.approx(40.0)

    def test_warns_when_no_publication_offset(self, tmp_path, caplog):
        import logging

        with caplog.at_level(logging.WARNING, logger="nexa_backtest.signals.csv_loader"):
            _make_provider(tmp_path, publication_offset=None)
        assert "publication_offset" in caplog.text

    def test_read_csv_failure_raises_data_error(self, tmp_path):
        path = _make_csv(tmp_path, _VALID_CSV)
        with (
            patch("pandas.read_csv", side_effect=OSError("disk error")),
            pytest.raises(DataError, match="Failed to read signal CSV"),
        ):
            CsvSignalProvider(name="x", path=path, unit="MW", description="")


class TestCsvSignalProviderProperties:
    def test_publication_offset_property(self, tmp_path):
        offset = timedelta(hours=6)
        provider = _make_provider(tmp_path, publication_offset=offset)
        assert provider.publication_offset == offset

    def test_publication_offset_none_when_not_set(self, tmp_path):
        provider = _make_provider(tmp_path, publication_offset=None)
        assert provider.publication_offset is None

    def test_schema_property(self, tmp_path):
        provider = CsvSignalProvider(
            name="wind_forecast",
            path=_make_csv(tmp_path, _VALID_CSV),
            unit="MW",
            description="Wind power forecast",
            frequency=timedelta(minutes=15),
            publication_offset=timedelta(hours=6),
        )
        schema = provider.schema
        assert isinstance(schema, SignalSchema)
        assert schema.name == "wind_forecast"
        assert schema.unit == "MW"
        assert schema.description == "Wind power forecast"


# ---------------------------------------------------------------------------
# get_value
# ---------------------------------------------------------------------------


class TestGetValue:
    def test_returns_latest_visible_value(self, tmp_path):
        provider = _make_provider(tmp_path, publication_offset=timedelta(0))
        # At 00:30 we should see the 00:30 row (3rd)
        t = datetime(2026, 3, 1, 0, 30, tzinfo=UTC)
        v = provider.get_value(t)
        assert v.value == pytest.approx(42.0)

    def test_returns_value_at_exact_timestamp(self, tmp_path):
        provider = _make_provider(tmp_path, publication_offset=timedelta(0))
        t = datetime(2026, 3, 1, 0, 15, tzinfo=UTC)
        v = provider.get_value(t)
        assert v.value == pytest.approx(41.0)

    def test_no_value_before_first_row_raises(self, tmp_path):
        provider = _make_provider(tmp_path, publication_offset=timedelta(0))
        t = datetime(2026, 2, 28, 23, 59, tzinfo=UTC)
        with pytest.raises(SignalError, match="No value"):
            provider.get_value(t)

    def test_signal_value_has_correct_name(self, tmp_path):
        provider = _make_provider(tmp_path, publication_offset=timedelta(0))
        t = datetime(2026, 3, 1, 1, tzinfo=UTC)
        v = provider.get_value(t)
        assert v.name == "test_signal"

    def test_signal_value_timestamp_is_utc_aware(self, tmp_path):
        provider = _make_provider(tmp_path, publication_offset=timedelta(0))
        t = datetime(2026, 3, 1, 1, tzinfo=UTC)
        v = provider.get_value(t)
        assert v.timestamp.tzinfo is not None


# ---------------------------------------------------------------------------
# Look-ahead bias (publication_offset)
# ---------------------------------------------------------------------------


class TestPublicationOffset:
    """Verify that publication_offset correctly gates visibility."""

    def test_future_value_not_visible_before_publication(self, tmp_path):
        # publication_offset=15min: a value with timestamp T is visible at T-15min
        # Equivalently: get_value(t) returns rows where timestamp <= t + 15min
        provider = _make_provider(tmp_path, publication_offset=timedelta(minutes=15))

        # At t=00:00, we can see rows with timestamp <= 00:15
        t = datetime(2026, 3, 1, 0, 0, tzinfo=UTC)
        v = provider.get_value(t)
        assert v.timestamp <= datetime(2026, 3, 1, 0, 15, tzinfo=UTC)
        # Latest visible is the 00:15 row (value=41.0)
        assert v.value == pytest.approx(41.0)

    def test_value_visible_after_publication_time(self, tmp_path):
        # At t=00:30, with offset=15min, we can see up to 00:45
        provider = _make_provider(tmp_path, publication_offset=timedelta(minutes=15))
        t = datetime(2026, 3, 1, 0, 30, tzinfo=UTC)
        v = provider.get_value(t)
        assert v.value == pytest.approx(43.0)  # the 00:45 row

    def test_no_look_ahead_without_offset(self, tmp_path):
        """At time T, only rows with timestamp <= T are returned."""
        provider = _make_provider(tmp_path, publication_offset=timedelta(0))
        # At 00:00, only the 00:00 row is visible (not 00:15 or later)
        t = datetime(2026, 3, 1, 0, 0, tzinfo=UTC)
        v = provider.get_value(t)
        assert v.value == pytest.approx(40.0)

    def test_no_value_before_publication_time_raises(self, tmp_path):
        """With offset=0 and only future rows, SignalError is raised."""
        csv = "timestamp,value\n2026-03-02T00:00:00+00:00,50.0\n"
        provider = _make_provider(tmp_path, content=csv, publication_offset=timedelta(0))
        t = datetime(2026, 3, 1, 0, 0, tzinfo=UTC)
        with pytest.raises(SignalError):
            provider.get_value(t)

    def test_large_offset_makes_future_values_visible(self, tmp_path):
        """With offset=24h, we can see values 24 hours into the future."""
        provider = _make_provider(tmp_path, publication_offset=timedelta(hours=24))
        # At 2026-02-28T00:00, offset=24h -> visible up to 2026-03-01T00:00
        t = datetime(2026, 2, 28, 0, 0, tzinfo=UTC)
        v = provider.get_value(t)
        assert v.value == pytest.approx(40.0)  # the first row at 2026-03-01T00:00


# ---------------------------------------------------------------------------
# get_range
# ---------------------------------------------------------------------------


class TestGetRange:
    def test_returns_values_in_range(self, tmp_path):
        provider = _make_provider(tmp_path, publication_offset=timedelta(0))
        start = datetime(2026, 3, 1, 0, 0, tzinfo=UTC)
        end = datetime(2026, 3, 1, 0, 30, tzinfo=UTC)
        series = provider.get_range(start, end)
        assert len(series) == 2  # 00:00 and 00:15 (00:30 is exclusive)

    def test_empty_range(self, tmp_path):
        provider = _make_provider(tmp_path, publication_offset=timedelta(0))
        start = datetime(2026, 3, 2, 0, 0, tzinfo=UTC)
        end = datetime(2026, 3, 2, 1, 0, tzinfo=UTC)
        series = provider.get_range(start, end)
        assert series.empty


# ---------------------------------------------------------------------------
# get_history_at
# ---------------------------------------------------------------------------


class TestGetHistoryAt:
    def test_returns_last_n_visible_values(self, tmp_path):
        provider = _make_provider(tmp_path, publication_offset=timedelta(0))
        t = datetime(2026, 3, 1, 1, 0, tzinfo=UTC)  # all 5 rows visible
        history = provider.get_history_at(t, lookback=3)
        assert len(history) == 3
        # Should be chronological: 00:30, 00:45, 01:00
        assert history[0].value == pytest.approx(42.0)
        assert history[2].value == pytest.approx(44.0)

    def test_capped_at_available_values(self, tmp_path):
        provider = _make_provider(tmp_path, publication_offset=timedelta(0))
        t = datetime(2026, 3, 1, 0, 15, tzinfo=UTC)  # only 2 rows visible
        history = provider.get_history_at(t, lookback=10)
        assert len(history) == 2

    def test_respects_publication_offset(self, tmp_path):
        # offset=15min at t=00:00: visible up to 00:15 → 2 rows
        provider = _make_provider(tmp_path, publication_offset=timedelta(minutes=15))
        t = datetime(2026, 3, 1, 0, 0, tzinfo=UTC)
        history = provider.get_history_at(t, lookback=10)
        assert len(history) == 2
        assert all(sv.timestamp <= datetime(2026, 3, 1, 0, 15, tzinfo=UTC) for sv in history)
