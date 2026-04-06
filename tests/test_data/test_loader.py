"""Tests for data/loader.py: ParquetLoader error paths."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from nexa_backtest.data.loader import ParquetLoader
from nexa_backtest.exceptions import DataError


def _write_parquet(path: Path, df: pd.DataFrame) -> None:
    df.to_parquet(path / "da_prices.parquet", index=False)


class TestParquetLoaderErrors:
    def test_missing_file_raises_data_error(self, tmp_path: Path) -> None:
        loader = ParquetLoader(tmp_path)
        with pytest.raises(DataError, match="da_prices"):
            loader.load_da_prices("NO1", date(2026, 3, 1), date(2026, 3, 31))

    def test_missing_required_columns_raises_data_error(self, tmp_path: Path) -> None:
        # Write a parquet with wrong columns
        df = pd.DataFrame({"col_a": [1, 2], "col_b": [3, 4]})
        _write_parquet(tmp_path, df)
        loader = ParquetLoader(tmp_path)
        with pytest.raises(DataError, match="missing required columns"):
            loader.load_da_prices("NO1", date(2026, 3, 1), date(2026, 3, 31))

    def test_no_data_for_zone_raises_data_error(self, tmp_path: Path) -> None:
        df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2026-03-01T00:00:00Z"], utc=True),
                "zone": ["DE"],  # wrong zone
                "price_eur_mwh": [50.0],
                "volume_mwh": [1000.0],
            }
        )
        _write_parquet(tmp_path, df)
        loader = ParquetLoader(tmp_path)
        with pytest.raises(DataError, match="No DA prices found"):
            loader.load_da_prices("NO1", date(2026, 3, 1), date(2026, 3, 31))

    def test_tz_naive_timestamps_are_localised_to_utc(self, tmp_path: Path) -> None:
        # Write parquet with tz-naive timestamps
        df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2026-03-01T00:00:00"]),  # no tz
                "zone": ["NO1"],
                "price_eur_mwh": [45.0],
                "volume_mwh": [1000.0],
            }
        )
        _write_parquet(tmp_path, df)
        loader = ParquetLoader(tmp_path)
        result = loader.load_da_prices("NO1", date(2026, 3, 1), date(2026, 3, 1))
        assert result["timestamp"].dt.tz is not None
        assert len(result) == 1

    def test_corrupt_parquet_raises_data_error(self, tmp_path: Path) -> None:
        # Write invalid bytes to da_prices.parquet
        (tmp_path / "da_prices.parquet").write_bytes(b"not a parquet file")
        loader = ParquetLoader(tmp_path)
        with pytest.raises(DataError, match="Failed to read DA prices"):
            loader.load_da_prices("NO1", date(2026, 3, 1), date(2026, 3, 31))
