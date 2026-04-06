"""Parquet-based data loader for historical market data.

DA clearing prices are small enough to load entirely into memory (~1.7 MB per
zone per year). IDC order book data uses windowed replay (Stage 2).
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from nexa_backtest.data.schema import (
    DA_PRICE_FILENAME,
    DA_PRICE_PRICE_COL,
    DA_PRICE_REQUIRED_COLUMNS,
    DA_PRICE_TIMESTAMP_COL,
    DA_PRICE_VOLUME_COL,
    DA_PRICE_ZONE_COL,
)
from nexa_backtest.exceptions import DataError


class ParquetLoader:
    """Loads historical market data from Parquet files.

    DA clearing prices are loaded entirely into memory. The loader validates
    the file schema and filters to the requested zone and date range.

    Args:
        data_dir: Directory containing market data files. The loader expects
            ``da_prices.parquet`` to be present within this directory.

    Example::

        loader = ParquetLoader(Path("data/"))
        df = loader.load_da_prices("NO1", date(2026, 3, 1), date(2026, 3, 31))
    """

    def __init__(self, data_dir: Path) -> None:
        self._data_dir = data_dir

    def load_da_prices(self, zone: str, start: date, end: date) -> pd.DataFrame:
        """Load DA clearing prices for a zone and date range.

        Reads ``da_prices.parquet`` from the data directory and filters to the
        requested zone and date range (inclusive on both ends).

        Args:
            zone: Bidding zone identifier, e.g. ``"NO1"``.
            start: First delivery date to include (inclusive).
            end: Last delivery date to include (inclusive).

        Returns:
            DataFrame with columns: ``product_id`` (str), ``timestamp``
            (tz-aware UTC datetime), ``zone`` (str), ``price_eur_mwh``
            (float64), ``volume_mwh`` (float64). Sorted ascending by
            timestamp.

        Raises:
            :class:`~nexa_backtest.exceptions.DataError`: If the file does
                not exist, is missing required columns, or contains no data
                for the requested zone and period.
        """
        path = self._data_dir / DA_PRICE_FILENAME
        if not path.exists():
            raise DataError(
                f"DA prices file not found: {path}. "
                f"Expected file: {DA_PRICE_FILENAME} in {self._data_dir}"
            )

        try:
            df = pd.read_parquet(path)
        except Exception as exc:
            raise DataError(f"Failed to read DA prices from {path}: {exc}") from exc

        missing = DA_PRICE_REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise DataError(
                f"DA prices file {path} is missing required columns: {sorted(missing)}. "
                f"Found: {sorted(df.columns)}"
            )

        # Ensure timestamp is timezone-aware UTC
        if df[DA_PRICE_TIMESTAMP_COL].dt.tz is None:
            df[DA_PRICE_TIMESTAMP_COL] = df[DA_PRICE_TIMESTAMP_COL].dt.tz_localize("UTC")
        else:
            df[DA_PRICE_TIMESTAMP_COL] = df[DA_PRICE_TIMESTAMP_COL].dt.tz_convert("UTC")

        mask = (
            (df[DA_PRICE_ZONE_COL] == zone)
            & (df[DA_PRICE_TIMESTAMP_COL].dt.date >= start)
            & (df[DA_PRICE_TIMESTAMP_COL].dt.date <= end)
        )
        result = df[mask].copy()

        if result.empty:
            raise DataError(
                f"No DA prices found for zone '{zone}' between {start} and {end}. "
                f"Verify that {path} contains data for this zone and period."
            )

        # Compute product IDs from zone + delivery timestamp
        result["product_id"] = (
            result[DA_PRICE_ZONE_COL]
            + "_"
            + result[DA_PRICE_TIMESTAMP_COL].dt.strftime("%Y%m%dT%H%MZ")
        )

        cols = [
            "product_id",
            DA_PRICE_TIMESTAMP_COL,
            DA_PRICE_ZONE_COL,
            DA_PRICE_PRICE_COL,
            DA_PRICE_VOLUME_COL,
        ]
        out: pd.DataFrame = result[cols].sort_values(DA_PRICE_TIMESTAMP_COL).reset_index(drop=True)
        return out
