"""Sliding window manager for IDC order book data.

IDC data volumes are ~10 GB per zone per year and cannot be loaded entirely
into memory.  This module implements a windowed replay strategy: Parquet row
groups are loaded on demand as the simulated clock advances and evicted once
they fall outside the lookback window.

Typical memory usage during replay is 200-500 MB regardless of replay period
length.  Peak memory is bounded by the number of row groups inside
``[now - lookback, now + lookahead]``.

Usage::

    manifest = DataManifest(data_dir=Path("data/"), zone="NO1")
    window = SlidingWindow(manifest=manifest)

    for mtu_start in mtu_boundaries:
        window.advance_to(mtu_start)
        for event in window.events_between(mtu_start, mtu_start + timedelta(minutes=15)):
            ...
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from nexa_backtest.data.schema import (
    IDC_EVENTS_AGGRESSOR_SIDE_COL,
    IDC_EVENTS_EVENT_TYPE_COL,
    IDC_EVENTS_ORDER_ID_COL,
    IDC_EVENTS_PRICE_COL,
    IDC_EVENTS_PRODUCT_ID_COL,
    IDC_EVENTS_REMAINING_COL,
    IDC_EVENTS_SIDE_COL,
    IDC_EVENTS_SUBDIR,
    IDC_EVENTS_TIMESTAMP_COL,
    IDC_EVENTS_TRADE_ID_COL,
    IDC_EVENTS_VOLUME_COL,
)
from nexa_backtest.types import (
    HistoricalTrade,
    MarketDataUpdate,
    MarketEvent,
)


@dataclass(frozen=True)
class RowGroupRef:
    """Pointer to a specific row group inside a Parquet file.

    Row group metadata (min/max timestamps) is read from the Parquet file
    footer without loading any data.

    Attributes:
        file_path: Absolute path to the Parquet file.
        row_group_index: Zero-based index of the row group within the file.
        min_timestamp: Earliest event timestamp in this row group.
        max_timestamp: Latest event timestamp in this row group.
        num_rows: Number of rows in this row group.
    """

    file_path: Path
    row_group_index: int
    min_timestamp: datetime
    max_timestamp: datetime
    num_rows: int


class DataManifest:
    """Index of all Parquet files and their row group timestamp ranges.

    Built once at startup, kept entirely in memory (~KB).  Knows where every
    chunk of IDC data lives on disk without loading any event data.

    Args:
        data_dir: Root data directory.  IDC event files are expected at
            ``{data_dir}/idc_events/{ZONE}_*.parquet``.
        zone: Bidding zone identifier, e.g. ``"NO1"``.

    Raises:
        FileNotFoundError: If ``{data_dir}/idc_events`` does not exist.
    """

    def __init__(self, data_dir: Path, zone: str) -> None:
        self._zone = zone
        events_dir = data_dir / IDC_EVENTS_SUBDIR
        if not events_dir.exists():
            raise FileNotFoundError(
                f"IDC events directory not found: {events_dir}. "
                f"Expected Parquet files at {events_dir}/{zone}_*.parquet"
            )
        self._refs: list[RowGroupRef] = self._build(events_dir, zone)

    @property
    def zone(self) -> str:
        """Bidding zone this manifest covers."""
        return self._zone

    @property
    def row_groups(self) -> list[RowGroupRef]:
        """All row group references, sorted by min_timestamp."""
        return list(self._refs)

    def row_groups_for_range(self, start: datetime, end: datetime) -> list[RowGroupRef]:
        """Return all row groups that overlap with ``[start, end)``.

        A row group overlaps when its min_timestamp < end AND its
        max_timestamp >= start.

        Args:
            start: Start of the time range (inclusive).
            end: End of the time range (exclusive).

        Returns:
            List of :class:`RowGroupRef` whose data may contain events in the
            requested range, sorted by min_timestamp.
        """
        return [ref for ref in self._refs if ref.min_timestamp < end and ref.max_timestamp >= start]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build(events_dir: Path, zone: str) -> list[RowGroupRef]:
        """Scan the events directory and build row group references."""
        pattern = f"{zone}_*.parquet"
        files = sorted(events_dir.glob(pattern))

        refs: list[RowGroupRef] = []
        for file_path in files:
            pf = pq.ParquetFile(file_path)
            metadata = pf.metadata
            for rg_idx in range(metadata.num_row_groups):
                rg = metadata.row_group(rg_idx)
                min_ts, max_ts = DataManifest._extract_ts_range(rg)
                if min_ts is None or max_ts is None:
                    # Row group has no timestamp statistics — skip
                    continue
                refs.append(
                    RowGroupRef(
                        file_path=file_path,
                        row_group_index=rg_idx,
                        min_timestamp=min_ts,
                        max_timestamp=max_ts,
                        num_rows=rg.num_rows,
                    )
                )

        refs.sort(key=lambda r: r.min_timestamp)
        return refs

    @staticmethod
    def _extract_ts_range(
        rg: pq.RowGroupMetaData,
    ) -> tuple[datetime | None, datetime | None]:
        """Extract min/max timestamps from row group column statistics."""
        for col_idx in range(rg.num_columns):
            col = rg.column(col_idx)
            if col.path_in_schema == IDC_EVENTS_TIMESTAMP_COL:
                stats = col.statistics
                if stats is None or not stats.has_min_max:
                    return None, None
                # PyArrow returns timestamps as Python datetime for ns[UTC]
                min_val = stats.min
                max_val = stats.max
                if isinstance(min_val, datetime) and isinstance(max_val, datetime):
                    return min_val, max_val
        return None, None


class SlidingWindow:
    """Maintains an in-memory window of IDC event data.

    Loads Parquet row groups from disk as the simulated clock advances and
    evicts row groups that have fallen outside the lookback window.  Window
    transitions happen between MTU boundaries — never mid-event-processing.

    **Price-taker / memory guarantee**: peak memory usage is bounded by the
    number of row groups covering ``[now - lookback, now + lookahead]``.
    With default parameters (4h lookback + 1h lookahead) and ~64 MB row
    groups this is typically 200-500 MB regardless of replay period length.

    Args:
        manifest: Pre-built :class:`DataManifest` for the target zone.
        lookback: How far behind ``now`` to keep loaded data.  Default 4h.
        lookahead: How far ahead of ``now`` to pre-load data.  Default 1h.
    """

    def __init__(
        self,
        manifest: DataManifest,
        lookback: timedelta = timedelta(hours=4),
        lookahead: timedelta = timedelta(hours=1),
    ) -> None:
        self._manifest = manifest
        self._lookback = lookback
        self._lookahead = lookahead
        # Loaded tables keyed by (file_path, row_group_index)
        self._loaded: dict[tuple[Path, int], pa.Table] = {}
        self._current_time: datetime | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def advance_to(self, timestamp: datetime) -> None:
        """Move the window forward to ``timestamp``.

        Loads row groups that overlap ``[timestamp - lookback,
        timestamp + lookahead]`` and evicts row groups that fall entirely
        before ``timestamp - lookback``.

        This method must only be called between MTU boundaries, never
        mid-event-processing.

        Args:
            timestamp: New simulated clock position (timezone-aware).
        """
        self._current_time = timestamp
        window_start = timestamp - self._lookback
        window_end = timestamp + self._lookahead

        needed = {
            (ref.file_path, ref.row_group_index): ref
            for ref in self._manifest.row_groups_for_range(window_start, window_end)
        }

        # Evict row groups no longer needed
        to_evict = [key for key in self._loaded if key not in needed]
        for key in to_evict:
            del self._loaded[key]

        # Load row groups not yet in memory
        for key, ref in needed.items():
            if key not in self._loaded:
                self._loaded[key] = self._read_row_group(ref)

    def events_between(self, start: datetime, end: datetime) -> Iterator[MarketEvent]:
        """Yield market events in the time range ``[start, end)``.

        Events are yielded in timestamp order.  Converts each row to a
        :class:`~nexa_backtest.types.MarketDataUpdate` or
        :class:`~nexa_backtest.types.HistoricalTrade` as appropriate.

        Args:
            start: Start of the time range (inclusive, timezone-aware).
            end: End of the time range (exclusive, timezone-aware).

        Yields:
            :class:`~nexa_backtest.types.MarketEvent` subclasses in
            chronological order.
        """
        if not self._loaded:
            return

        # Concatenate all loaded tables that may contain events in [start, end)
        tables: list[pa.Table] = []
        for (fp, rg_idx), table in self._loaded.items():
            ref = self._find_ref(fp, rg_idx)
            if ref is None:
                continue
            if ref.min_timestamp < end and ref.max_timestamp >= start:
                tables.append(table)

        if not tables:
            return

        combined = pa.concat_tables(tables)

        # Filter to the requested time range using PyArrow compute
        import pyarrow.compute as pc

        ts_col = combined.column(IDC_EVENTS_TIMESTAMP_COL)
        mask = pc.and_(pc.greater_equal(ts_col, pa.scalar(start)), pc.less(ts_col, pa.scalar(end)))
        filtered = combined.filter(mask)

        if filtered.num_rows == 0:
            return

        # Sort by timestamp
        sort_keys = [(IDC_EVENTS_TIMESTAMP_COL, "ascending")]
        sort_indices = pc.sort_indices(filtered, sort_keys=sort_keys)
        filtered = filtered.take(sort_indices)

        # Convert rows to MarketEvent objects
        yield from self._rows_to_events(filtered)

    @property
    def memory_usage_bytes(self) -> int:
        """Current memory footprint of all loaded PyArrow tables in bytes.

        Use this in tests and monitoring to verify that peak memory stays
        bounded during long replay periods.

        Returns:
            Total bytes used by loaded row groups.
        """
        return sum(table.nbytes for table in self._loaded.values())

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _find_ref(self, file_path: Path, rg_idx: int) -> RowGroupRef | None:
        """Look up a RowGroupRef from the manifest."""
        for ref in self._manifest.row_groups:
            if ref.file_path == file_path and ref.row_group_index == rg_idx:
                return ref
        return None

    @staticmethod
    def _read_row_group(ref: RowGroupRef) -> pa.Table:
        """Read a single row group from a Parquet file."""
        pf = pq.ParquetFile(ref.file_path)
        return pf.read_row_group(ref.row_group_index)

    @staticmethod
    def _rows_to_events(table: pa.Table) -> Iterator[MarketEvent]:
        """Convert a PyArrow Table to MarketEvent objects row by row."""
        # Extract columns as Python lists (zero-copy to numpy first for speed)
        ts_list = table.column(IDC_EVENTS_TIMESTAMP_COL).to_pylist()
        event_type_list = table.column(IDC_EVENTS_EVENT_TYPE_COL).to_pylist()
        order_id_list = table.column(IDC_EVENTS_ORDER_ID_COL).to_pylist()
        product_id_list = table.column(IDC_EVENTS_PRODUCT_ID_COL).to_pylist()
        side_list = table.column(IDC_EVENTS_SIDE_COL).to_pylist()
        price_list = table.column(IDC_EVENTS_PRICE_COL).to_pylist()
        volume_list = table.column(IDC_EVENTS_VOLUME_COL).to_pylist()
        remaining_list = table.column(IDC_EVENTS_REMAINING_COL).to_pylist()

        # Optional columns (may not be present in all files)
        has_aggressor = IDC_EVENTS_AGGRESSOR_SIDE_COL in table.schema.names
        has_trade_id = IDC_EVENTS_TRADE_ID_COL in table.schema.names
        aggressor_list = (
            table.column(IDC_EVENTS_AGGRESSOR_SIDE_COL).to_pylist()
            if has_aggressor
            else [None] * table.num_rows
        )
        trade_id_list = (
            table.column(IDC_EVENTS_TRADE_ID_COL).to_pylist()
            if has_trade_id
            else [None] * table.num_rows
        )

        for i in range(table.num_rows):
            ts: datetime = ts_list[i]
            event_type: str = event_type_list[i]
            product_id: str = product_id_list[i]
            price = Decimal(str(price_list[i]))
            volume = Decimal(str(volume_list[i]))
            remaining = Decimal(str(remaining_list[i]))

            if event_type == "trade":
                trade_id: str = trade_id_list[i] or f"trade-{i}"
                yield HistoricalTrade(
                    timestamp=ts,
                    product_id=product_id,
                    trade_id=trade_id,
                    price_eur_mwh=price,
                    volume_mw=volume,
                    aggressor_side=aggressor_list[i],
                )
            else:
                yield MarketDataUpdate(
                    timestamp=ts,
                    product_id=product_id,
                    event_type=event_type,
                    order_id=order_id_list[i],
                    side=side_list[i],
                    price_eur_mwh=price,
                    volume_mw=volume,
                    remaining_mw=remaining,
                )


__all__ = [
    "DataManifest",
    "RowGroupRef",
    "SlidingWindow",
]
