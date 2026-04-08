"""Standard column schemas for historical data files.

These constants define the expected column names and types for Parquet files
used by :class:`~nexa_backtest.data.loader.ParquetLoader`.
"""

from __future__ import annotations

import pyarrow as pa

# ---------------------------------------------------------------------------
# DA clearing price schema
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# IDC order book events schema
# ---------------------------------------------------------------------------
# Parquet columns for intraday continuous order book event stream.
# Windowed replay via SlidingWindow — do NOT load an entire year at once.
#
# event_type values: "new", "modify", "cancel", "trade"
# side values:       "buy", "sell"
#
# For "trade" events the row represents the state of the RESTING order after
# the trade.  An additional nullable ``aggressor_side`` column records which
# side placed the aggressive (crossing) order so the matching engine can
# determine whether a resting algo order would have been hit.
IDC_EVENTS_TIMESTAMP_COL = "timestamp"
IDC_EVENTS_EVENT_TYPE_COL = "event_type"
IDC_EVENTS_ORDER_ID_COL = "order_id"
IDC_EVENTS_ZONE_COL = "zone"
IDC_EVENTS_PRODUCT_ID_COL = "product_id"
IDC_EVENTS_SIDE_COL = "side"
IDC_EVENTS_PRICE_COL = "price_eur_mwh"
IDC_EVENTS_VOLUME_COL = "volume_mw"
IDC_EVENTS_REMAINING_COL = "remaining_mw"
IDC_EVENTS_AGGRESSOR_SIDE_COL = "aggressor_side"
IDC_EVENTS_TRADE_ID_COL = "trade_id"

IDC_EVENTS_SCHEMA: pa.Schema = pa.schema(
    [
        (IDC_EVENTS_TIMESTAMP_COL, pa.timestamp("ns", tz="UTC")),
        (IDC_EVENTS_EVENT_TYPE_COL, pa.string()),
        (IDC_EVENTS_ORDER_ID_COL, pa.string()),
        (IDC_EVENTS_ZONE_COL, pa.string()),
        (IDC_EVENTS_PRODUCT_ID_COL, pa.string()),
        (IDC_EVENTS_SIDE_COL, pa.string()),
        (IDC_EVENTS_PRICE_COL, pa.float64()),
        (IDC_EVENTS_VOLUME_COL, pa.float64()),
        (IDC_EVENTS_REMAINING_COL, pa.float64()),
        # nullable — null for non-trade events
        (IDC_EVENTS_AGGRESSOR_SIDE_COL, pa.string()),
        # nullable — null for non-trade events
        (IDC_EVENTS_TRADE_ID_COL, pa.string()),
    ]
)

# Default subdirectory for IDC event Parquet files within a data directory
IDC_EVENTS_SUBDIR = "idc_events"

# ---------------------------------------------------------------------------
# IDC trades schema (separate from order book events)
# ---------------------------------------------------------------------------
# Parquet columns for a normalised trade tape.
# aggressor_side may be null when the exchange data does not include it.
IDC_TRADES_SCHEMA: pa.Schema = pa.schema(
    [
        ("timestamp", pa.timestamp("ns", tz="UTC")),
        ("trade_id", pa.string()),
        ("zone", pa.string()),
        ("product_id", pa.string()),
        ("price_eur_mwh", pa.float64()),
        ("volume_mw", pa.float64()),
        ("aggressor_side", pa.string()),  # nullable
    ]
)

# ---------------------------------------------------------------------------
# Signal CSV schema
# ---------------------------------------------------------------------------
SIGNAL_CSV_TIMESTAMP_COL = "timestamp"
SIGNAL_CSV_VALUE_COL = "value"
