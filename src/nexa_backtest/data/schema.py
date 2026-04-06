"""Standard column schemas for historical data files.

These constants define the expected column names and types for Parquet files
used by :class:`~nexa_backtest.data.loader.ParquetLoader`.
"""

from __future__ import annotations

# DA clearing price schema
# Parquet columns: timestamp (datetime64[ns, UTC]), zone (str),
#                  price_eur_mwh (float64), volume_mwh (float64)
DA_PRICE_TIMESTAMP_COL = "timestamp"
DA_PRICE_ZONE_COL = "zone"
DA_PRICE_PRICE_COL = "price_eur_mwh"
DA_PRICE_VOLUME_COL = "volume_mwh"

DA_PRICE_REQUIRED_COLUMNS: frozenset[str] = frozenset(
    {
        DA_PRICE_TIMESTAMP_COL,
        DA_PRICE_ZONE_COL,
        DA_PRICE_PRICE_COL,
        DA_PRICE_VOLUME_COL,
    }
)

# Default filename for DA clearing prices within a data directory
DA_PRICE_FILENAME = "da_prices.parquet"

# Signal CSV schema
SIGNAL_CSV_TIMESTAMP_COL = "timestamp"
SIGNAL_CSV_VALUE_COL = "value"
