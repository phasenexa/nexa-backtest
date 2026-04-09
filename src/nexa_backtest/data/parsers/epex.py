"""EPEX SPOT data normaliser.

Reads EPEX SPOT native CSV or Parquet export format and converts it to the
standard IDC events schema used throughout nexa-backtest.

EPEX SPOT export columns
------------------------
``OrderId``, ``InitialId``, ``ParentId``, ``Side``, ``Product``,
``DeliveryStart``, ``DeliveryEnd``, ``CreationTime``, ``DeliveryArea``,
``ExecutionRestriction``, ``UserDefinedBlock``, ``LinkedBasketId``,
``RevisionNo``, ``ActionCode``, ``TransactionTime``, ``ValidityTime``,
``Price``, ``Currency``, ``Quantity``, ``QuantityUnit``, ``Volume``,
``VolumeUnit``

Standard IDC events schema (output)
------------------------------------
``timestamp`` (datetime64[ns, UTC]), ``event_type`` (str), ``order_id`` (str),
``zone`` (str), ``product_id`` (str), ``side`` (str), ``price_eur_mwh``
(float64), ``volume_mw`` (float64), ``remaining_mw`` (float64),
``aggressor_side`` (str, nullable), ``trade_id`` (str, nullable)

ActionCode mapping:
    A01 → ``"new"``
    A02 → ``"modify"``
    A03 → ``"cancel"``
    (Any trade records) → ``"trade"``

Notes
-----
- ``aggressor_side`` is not directly available in EPEX data. It defaults to
  ``None`` and the matching engine falls back to its heuristic logic.
- ``remaining_mw`` is approximated from ``Quantity`` as EPEX does not always
  provide remaining volume explicitly.  RevisionNo can be used to detect
  order modifications, but the actual remaining volume is not exported.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

# EPEX ActionCode → standard event_type mapping
_ACTION_CODE_MAP: dict[str, str] = {
    "A01": "new",
    "A02": "modify",
    "A03": "cancel",
}

_REQUIRED_COLUMNS = frozenset(
    {
        "CreationTime",
        "ActionCode",
        "OrderId",
        "Side",
        "DeliveryArea",
        "DeliveryStart",
        "Price",
        "Quantity",
    }
)


def _build_product_id(area: str, delivery_start: pd.Timestamp) -> str:
    """Construct a product ID from delivery area and start time.

    Args:
        area: EPEX delivery area code, e.g. ``"DE-LU"``.
        delivery_start: Delivery period start timestamp.

    Returns:
        Product ID in ``{area}-QH-{HHMM}`` format,
        e.g. ``"DE-LU-QH-0900"``.
    """
    hhmm = delivery_start.strftime("%H%M")
    return f"{area}-QH-{hhmm}"


def parse_epex_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise an EPEX SPOT DataFrame to the standard IDC events schema.

    Args:
        df: Raw EPEX export DataFrame.  Must contain the columns:
            ``CreationTime``, ``ActionCode``, ``OrderId``, ``Side``,
            ``DeliveryArea``, ``DeliveryStart``, ``Price``, ``Quantity``.

    Returns:
        Normalised DataFrame with columns matching the standard IDC events
        schema.  ``aggressor_side`` is always ``None``.

    Raises:
        :class:`~nexa_backtest.exceptions.DataError`: If required columns
            are missing from ``df``.
    """
    import pandas as pd

    from nexa_backtest.exceptions import DataError

    missing = _REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise DataError(
            f"EPEX data is missing required columns: {sorted(missing)}. "
            f"Available columns: {sorted(df.columns.tolist())}"
        )

    result = pd.DataFrame()

    # timestamp: parse CreationTime as UTC
    result["timestamp"] = pd.to_datetime(df["CreationTime"], utc=True)

    # event_type: map ActionCode
    result["event_type"] = df["ActionCode"].map(_ACTION_CODE_MAP).fillna("new")

    # order_id
    result["order_id"] = df["OrderId"].astype(str)

    # zone / delivery area
    result["zone"] = df["DeliveryArea"].astype(str)

    # product_id: area + delivery start time
    delivery_starts = pd.to_datetime(df["DeliveryStart"], utc=True)
    result["product_id"] = [
        _build_product_id(str(area), ds)
        for area, ds in zip(df["DeliveryArea"], delivery_starts, strict=False)
    ]

    # side: BUY → "buy", SELL → "sell"
    result["side"] = df["Side"].str.lower()

    # price_eur_mwh
    result["price_eur_mwh"] = pd.to_numeric(df["Price"], errors="coerce").astype("float64")

    # volume_mw: from Quantity column
    result["volume_mw"] = pd.to_numeric(df["Quantity"], errors="coerce").astype("float64")

    # remaining_mw: EPEX does not export remaining volume directly.
    # Use Quantity as an approximation (correct for new orders; may over-state
    # for partially-filled orders after modification).
    result["remaining_mw"] = result["volume_mw"]

    # aggressor_side: not available in EPEX exports
    result["aggressor_side"] = None

    # trade_id: not applicable for order-book events (only for trade records)
    result["trade_id"] = None

    # Sort by timestamp ascending
    result = result.sort_values("timestamp").reset_index(drop=True)

    return result


def parse_epex_csv(path: Path | str) -> pd.DataFrame:
    """Load and normalise an EPEX SPOT CSV export file.

    Args:
        path: Path to the EPEX SPOT CSV file.

    Returns:
        Normalised DataFrame with the standard IDC events schema.

    Raises:
        :class:`~nexa_backtest.exceptions.DataError`: If the file cannot be
            read or required columns are missing.
    """
    import pandas as pd

    from nexa_backtest.exceptions import DataError

    try:
        raw = pd.read_csv(path)
    except Exception as exc:
        raise DataError(f"Failed to read EPEX CSV file '{path}': {exc}") from exc

    return parse_epex_df(raw)
