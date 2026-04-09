"""Tests for the EPEX SPOT data normaliser (task 05)."""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import pytest

from nexa_backtest.data.parsers.epex import parse_epex_csv, parse_epex_df
from nexa_backtest.exceptions import DataError

# ---------------------------------------------------------------------------
# Minimal EPEX fixture
# ---------------------------------------------------------------------------

_EPEX_CSV_CONTENT = """\
OrderId,InitialId,ParentId,Side,Product,DeliveryStart,DeliveryEnd,CreationTime,DeliveryArea,ExecutionRestriction,UserDefinedBlock,LinkedBasketId,RevisionNo,ActionCode,TransactionTime,ValidityTime,Price,Currency,Quantity,QuantityUnit,Volume,VolumeUnit
ORD-001,ORD-001,,BUY,QH,2026-03-01T09:00:00Z,2026-03-01T09:15:00Z,2026-03-01T08:00:00Z,DE-LU,,,,0,A01,,2026-03-01T09:00:00Z,55.50,EUR,10,MW,,MWh
ORD-002,ORD-002,,SELL,QH,2026-03-01T09:00:00Z,2026-03-01T09:15:00Z,2026-03-01T08:05:00Z,DE-LU,,,,0,A01,,2026-03-01T09:00:00Z,48.00,EUR,5,MW,,MWh
ORD-001,ORD-001,,BUY,QH,2026-03-01T09:00:00Z,2026-03-01T09:15:00Z,2026-03-01T08:10:00Z,DE-LU,,,,1,A02,,2026-03-01T09:00:00Z,54.00,EUR,10,MW,,MWh
ORD-003,ORD-003,,BUY,QH,2026-03-01T09:00:00Z,2026-03-01T09:15:00Z,2026-03-01T08:20:00Z,DE-LU,,,,0,A03,,2026-03-01T09:00:00Z,54.00,EUR,10,MW,,MWh
"""


@pytest.fixture
def raw_epex_df() -> pd.DataFrame:
    """Minimal raw EPEX CSV loaded into a DataFrame."""
    return pd.read_csv(io.StringIO(_EPEX_CSV_CONTENT))


@pytest.fixture
def normalised_epex_df(raw_epex_df: pd.DataFrame) -> pd.DataFrame:
    """EPEX DataFrame normalised to standard IDC schema."""
    return parse_epex_df(raw_epex_df)


# ---------------------------------------------------------------------------
# Schema compliance
# ---------------------------------------------------------------------------


def test_output_has_required_columns(normalised_epex_df: pd.DataFrame) -> None:
    required = {
        "timestamp",
        "event_type",
        "order_id",
        "zone",
        "product_id",
        "side",
        "price_eur_mwh",
        "volume_mw",
        "remaining_mw",
        "aggressor_side",
        "trade_id",
    }
    assert required.issubset(set(normalised_epex_df.columns))


def test_output_row_count_matches_input(
    raw_epex_df: pd.DataFrame, normalised_epex_df: pd.DataFrame
) -> None:
    assert len(normalised_epex_df) == len(raw_epex_df)


def test_timestamps_are_utc(normalised_epex_df: pd.DataFrame) -> None:
    ts = normalised_epex_df["timestamp"]
    assert str(ts.dt.tz) == "UTC"


def test_sorted_by_timestamp(normalised_epex_df: pd.DataFrame) -> None:
    timestamps = normalised_epex_df["timestamp"].tolist()
    assert timestamps == sorted(timestamps)


# ---------------------------------------------------------------------------
# ActionCode mapping
# ---------------------------------------------------------------------------


def test_action_code_a01_maps_to_new(normalised_epex_df: pd.DataFrame) -> None:
    rows = normalised_epex_df[normalised_epex_df["order_id"] == "ORD-002"]
    assert rows.iloc[0]["event_type"] == "new"


def test_action_code_a02_maps_to_modify(normalised_epex_df: pd.DataFrame) -> None:
    # ORD-001 has RevisionNo=1 and ActionCode=A02
    modify_rows = normalised_epex_df[normalised_epex_df["event_type"] == "modify"]
    assert len(modify_rows) >= 1


def test_action_code_a03_maps_to_cancel(normalised_epex_df: pd.DataFrame) -> None:
    cancel_rows = normalised_epex_df[normalised_epex_df["event_type"] == "cancel"]
    assert len(cancel_rows) == 1


# ---------------------------------------------------------------------------
# Field mappings
# ---------------------------------------------------------------------------


def test_side_buy_maps_to_lowercase(normalised_epex_df: pd.DataFrame) -> None:
    buy_rows = normalised_epex_df[normalised_epex_df["order_id"] == "ORD-001"]
    assert buy_rows.iloc[0]["side"] == "buy"


def test_side_sell_maps_to_lowercase(normalised_epex_df: pd.DataFrame) -> None:
    sell_rows = normalised_epex_df[normalised_epex_df["order_id"] == "ORD-002"]
    assert sell_rows.iloc[0]["side"] == "sell"


def test_product_id_uses_area_and_delivery_time(normalised_epex_df: pd.DataFrame) -> None:
    # DeliveryStart=09:00 → "DE-LU-QH-0900"
    assert (normalised_epex_df["product_id"] == "DE-LU-QH-0900").all()


def test_zone_is_delivery_area(normalised_epex_df: pd.DataFrame) -> None:
    assert (normalised_epex_df["zone"] == "DE-LU").all()


def test_price_eur_mwh_is_numeric(normalised_epex_df: pd.DataFrame) -> None:
    assert normalised_epex_df["price_eur_mwh"].dtype == "float64"
    first_row = normalised_epex_df[normalised_epex_df["order_id"] == "ORD-001"].iloc[0]
    assert first_row["price_eur_mwh"] in {55.5, 54.0}


def test_volume_mw_is_numeric(normalised_epex_df: pd.DataFrame) -> None:
    assert normalised_epex_df["volume_mw"].dtype == "float64"


# ---------------------------------------------------------------------------
# aggressor_side is None
# ---------------------------------------------------------------------------


def test_aggressor_side_is_none(normalised_epex_df: pd.DataFrame) -> None:
    """aggressor_side must be None for all rows (EPEX does not export it)."""
    assert normalised_epex_df["aggressor_side"].isna().all()


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_parse_epex_df_raises_on_missing_columns() -> None:
    df = pd.DataFrame({"OrderId": ["1"], "Side": ["BUY"]})
    with pytest.raises(DataError, match="missing required columns"):
        parse_epex_df(df)


def test_parse_epex_csv_raises_on_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "nonexistent.csv"
    with pytest.raises(DataError, match="Failed to read"):
        parse_epex_csv(missing)


def test_parse_epex_csv_round_trip(tmp_path: Path) -> None:
    """Write a minimal CSV and parse it back to verify the pipeline."""
    csv_file = tmp_path / "epex_test.csv"
    csv_file.write_text(_EPEX_CSV_CONTENT)

    result = parse_epex_csv(csv_file)
    assert len(result) == 4
    assert set(result["zone"]) == {"DE-LU"}
    assert result["aggressor_side"].isna().all()
