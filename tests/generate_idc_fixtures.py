"""Generate synthetic IDC test fixtures for nexa-backtest.

Produces a small deterministic Parquet file with realistic IDC order book
events for use in unit and integration tests.

Run this script to (re)generate the fixture files::

    python tests/generate_idc_fixtures.py

Outputs:
    tests/fixtures/nordpool/idc_events/NO1_2026_03.parquet

The fixture covers one day (2026-03-01) for three Nord Pool NO1 quarter-
hourly products:

    NO1-QH-0800, NO1-QH-0815, NO1-QH-0830

Each product has approximately 50 events per hour: a mix of new orders,
cancellations, and trades.  Prices follow a random walk around 50 EUR/MWh
with a ±5 EUR range.  Some trade events include ``aggressor_side`` and some
do not, to exercise both matching code paths.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from nexa_backtest.data.schema import IDC_EVENTS_SCHEMA

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
SEED = 42
ZONE = "NO1"
DATE = datetime(2026, 3, 1, tzinfo=UTC)
PRODUCTS = ["NO1-QH-0800", "NO1-QH-0815", "NO1-QH-0830"]

BASE_PRICE = 50.0  # EUR/MWh
PRICE_RANGE = 5.0  # ±EUR/MWh
PRICE_STEP = 0.10  # EUR/MWh tick size

EVENTS_PER_PRODUCT_PER_HOUR = 50
HOURS_IN_DAY = 24
FRACTION_TRADES = 0.30  # 30% of events are trades
FRACTION_CANCEL = 0.20  # 20% of new orders are later cancelled
AGGRESSOR_MISSING_RATE = 0.25  # 25% of trades have aggressor_side=None

# Row group target: one row group per 4 hours of data
ROWS_PER_ROW_GROUP = EVENTS_PER_PRODUCT_PER_HOUR * 4 * len(PRODUCTS)

OUTPUT_DIR = Path(__file__).parent / "fixtures" / "nordpool" / "idc_events"
OUTPUT_FILE = OUTPUT_DIR / f"{ZONE}_2026_03.parquet"


def _round_price(price: float) -> float:
    return round(price / PRICE_STEP) * PRICE_STEP


def generate_idc_events(
    seed: int = SEED,
) -> pd.DataFrame:
    """Generate synthetic IDC order book events for all products.

    The generator simulates a realistic intraday order book:

    1. Active orders accumulate through the day as random-walk prices.
    2. At each event slot, one of three things happens with weighted
       probability: new order, cancel of an existing order, or trade.
    3. Trades hit resting orders on either side; aggressor_side is
       occasionally omitted to test the fallback path.

    Args:
        seed: Random seed for reproducibility.

    Returns:
        DataFrame conforming to the IDC events schema.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []

    total_events = EVENTS_PER_PRODUCT_PER_HOUR * HOURS_IN_DAY
    interval_seconds = 3600 / EVENTS_PER_PRODUCT_PER_HOUR

    for product_id in PRODUCTS:
        # Active orders tracked during generation: {order_id: {side, price, remaining}}
        active_orders: dict[str, dict[str, float | str]] = {}
        mid_price = BASE_PRICE + rng.uniform(-2.0, 2.0)

        for i in range(total_events):
            # Simulated timestamp: spread events evenly across the day
            ts = DATE + timedelta(seconds=i * interval_seconds)

            # Advance mid price with small random walk
            mid_price += rng.normal(0, 0.05)
            lo, hi = BASE_PRICE - PRICE_RANGE, BASE_PRICE + PRICE_RANGE
            mid_price = float(np.clip(mid_price, lo, hi))

            # Decide event type
            r = rng.random()
            if r < FRACTION_TRADES and active_orders:
                # --- Trade event ---
                # Pick a random active order as the resting side
                resting_id = rng.choice(list(active_orders.keys()))
                resting = active_orders[resting_id]
                trade_volume = float(
                    np.clip(rng.exponential(3.0), 0.5, float(resting["remaining"]))
                )
                trade_price = _round_price(float(resting["price"]))
                new_remaining = float(resting["remaining"]) - trade_volume
                aggressor_side: str | None = None
                if rng.random() > AGGRESSOR_MISSING_RATE:
                    aggressor_side = "sell" if resting["side"] == "buy" else "buy"

                trade_id = str(uuid.uuid4())
                rows.append(
                    {
                        "timestamp": ts,
                        "event_type": "trade",
                        "order_id": resting_id,
                        "zone": ZONE,
                        "product_id": product_id,
                        "side": resting["side"],
                        "price_eur_mwh": trade_price,
                        "volume_mw": trade_volume,
                        "remaining_mw": max(new_remaining, 0.0),
                        "aggressor_side": aggressor_side,
                        "trade_id": trade_id,
                    }
                )
                if new_remaining <= 0.0:
                    del active_orders[resting_id]
                else:
                    active_orders[resting_id]["remaining"] = new_remaining  # type: ignore[assignment]

            elif r < FRACTION_TRADES + FRACTION_CANCEL and active_orders:
                # --- Cancel event ---
                cancel_id = rng.choice(list(active_orders.keys()))
                cancelled = active_orders.pop(cancel_id)
                rows.append(
                    {
                        "timestamp": ts,
                        "event_type": "cancel",
                        "order_id": cancel_id,
                        "zone": ZONE,
                        "product_id": product_id,
                        "side": cancelled["side"],
                        "price_eur_mwh": float(cancelled["price"]),
                        "volume_mw": float(cancelled["remaining"]),
                        "remaining_mw": 0.0,
                        "aggressor_side": None,
                        "trade_id": None,
                    }
                )

            else:
                # --- New order event ---
                side = "buy" if rng.random() < 0.5 else "sell"
                spread = _round_price(rng.uniform(1.0, 3.0))
                if side == "buy":
                    price = _round_price(mid_price - spread)
                else:
                    price = _round_price(mid_price + spread)

                volume = float(np.clip(rng.exponential(5.0), 0.5, 50.0))
                volume = round(volume, 1)
                order_id = str(uuid.uuid4())
                active_orders[order_id] = {"side": side, "price": price, "remaining": volume}
                rows.append(
                    {
                        "timestamp": ts,
                        "event_type": "new",
                        "order_id": order_id,
                        "zone": ZONE,
                        "product_id": product_id,
                        "side": side,
                        "price_eur_mwh": price,
                        "volume_mw": volume,
                        "remaining_mw": volume,
                        "aggressor_side": None,
                        "trade_id": None,
                    }
                )

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def write_fixture(df: pd.DataFrame, output_path: Path, rows_per_rg: int) -> None:
    """Write the DataFrame to Parquet with multiple row groups.

    Multiple row groups are required for the SlidingWindow tests to verify
    that eviction and selective loading work correctly.

    Args:
        df: Events DataFrame.
        output_path: Destination Parquet file path.
        rows_per_rg: Maximum rows per row group.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    table = pa.Table.from_pandas(df, schema=IDC_EVENTS_SCHEMA, preserve_index=False)

    writer = pq.ParquetWriter(output_path, schema=IDC_EVENTS_SCHEMA)
    total_rows = len(table)
    for start in range(0, total_rows, rows_per_rg):
        chunk = table.slice(start, min(rows_per_rg, total_rows - start))
        writer.write_table(chunk)
    writer.close()


def main() -> None:
    """Generate and write all IDC fixture files."""
    print(f"Generating IDC events for {ZONE} on {DATE.date()} ...")
    df = generate_idc_events()
    print(f"  Generated {len(df)} events across {len(PRODUCTS)} products")
    write_fixture(df, OUTPUT_FILE, ROWS_PER_ROW_GROUP)
    print(f"  Written: {OUTPUT_FILE}")

    # Report row group distribution
    pf = pq.ParquetFile(OUTPUT_FILE)
    print(f"  Row groups: {pf.metadata.num_row_groups}")
    for i in range(pf.metadata.num_row_groups):
        rg = pf.metadata.row_group(i)
        print(f"    RG {i}: {rg.num_rows} rows")

    print("Done.")


if __name__ == "__main__":
    main()
