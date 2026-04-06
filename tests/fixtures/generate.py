"""Generate synthetic test fixture data for nexa-backtest.

Run this script to (re)generate the fixture files used by tests and examples::

    python tests/fixtures/generate.py

Outputs:
    - tests/fixtures/da_prices.parquet
    - tests/fixtures/signals/price_forecast.csv
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

FIXTURES_DIR = Path(__file__).parent
SIGNALS_DIR = FIXTURES_DIR / "signals"

# Synthetic data parameters
ZONE = "NO1"
START = datetime(2026, 3, 1, tzinfo=UTC)
END = datetime(2026, 4, 1, tzinfo=UTC)  # exclusive
MTU_MINUTES = 15
BASE_PRICE = 45.0  # EUR/MWh
PRICE_STD = 8.0
DAILY_AMPLITUDE = 10.0
SEED = 42

# Publication offset for the price_forecast signal: 36h ahead of delivery.
# At auction time D-1 12:00 UTC, get_value returns T <= D-1 12:00 + 36h = D+1 00:00,
# so all day-D forecasts are visible.
FORECAST_PUBLICATION_OFFSET_HOURS = 36
FORECAST_NOISE_STD = 2.0


def _generate_timestamps(start: datetime, end: datetime, step_minutes: int) -> list[datetime]:
    """Generate equally-spaced timezone-aware timestamps."""
    stamps = []
    t = start
    while t < end:
        stamps.append(t)
        t += timedelta(minutes=step_minutes)
    return stamps


def generate_da_prices(
    zone: str = ZONE,
    start: datetime = START,
    end: datetime = END,
    mtu_minutes: int = MTU_MINUTES,
    seed: int = SEED,
) -> pd.DataFrame:
    """Generate a synthetic DA clearing price DataFrame.

    Prices follow a sinusoidal daily pattern with Gaussian noise.

    Args:
        zone: Bidding zone identifier.
        start: First delivery timestamp (inclusive).
        end: Last delivery timestamp (exclusive).
        mtu_minutes: MTU duration in minutes (15 or 60).
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with columns: timestamp, zone, price_eur_mwh, volume_mwh.
    """
    rng = np.random.default_rng(seed)
    timestamps = _generate_timestamps(start, end, mtu_minutes)
    n = len(timestamps)

    hour_of_day = np.array([ts.hour + ts.minute / 60 for ts in timestamps])
    prices = (
        BASE_PRICE
        + DAILY_AMPLITUDE * np.sin(hour_of_day * np.pi / 12 - np.pi / 2)
        + rng.normal(0, PRICE_STD, n)
    )
    prices = np.maximum(prices, 5.0)

    volumes = rng.uniform(500, 2000, n)

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "zone": zone,
            "price_eur_mwh": prices,
            "volume_mwh": volumes,
        }
    )


def generate_price_forecast(
    da_prices: pd.DataFrame,
    noise_std: float = FORECAST_NOISE_STD,
    seed: int = SEED + 1,
) -> pd.DataFrame:
    """Generate a synthetic price forecast CSV with slight noise.

    The forecast timestamps match the DA delivery timestamps. The engine will
    use publication_offset=36h so the forecast for day D is visible at D-1
    12:00 UTC during the DA auction.

    Args:
        da_prices: DataFrame from :func:`generate_da_prices`.
        noise_std: Standard deviation of forecast error in EUR/MWh.
        seed: Random seed.

    Returns:
        DataFrame with columns: timestamp, value.
    """
    rng = np.random.default_rng(seed)
    n = len(da_prices)
    noise = rng.normal(0, noise_std, n)
    values = da_prices["price_eur_mwh"].to_numpy() + noise

    return pd.DataFrame(
        {
            "timestamp": da_prices["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
            "value": values,
        }
    )


def main() -> None:
    """Generate and write all fixture files."""
    SIGNALS_DIR.mkdir(exist_ok=True)

    print(f"Generating DA prices for {ZONE} {START.date()} - {END.date()} ...")
    da_prices = generate_da_prices()
    parquet_path = FIXTURES_DIR / "da_prices.parquet"
    da_prices.to_parquet(parquet_path, index=False)
    print(f"  Written: {parquet_path}  ({len(da_prices)} rows)")

    print("Generating price_forecast signal CSV ...")
    forecast = generate_price_forecast(da_prices)
    csv_path = SIGNALS_DIR / "price_forecast.csv"
    forecast.to_csv(csv_path, index=False)
    print(f"  Written: {csv_path}  ({len(forecast)} rows)")

    print("Done.")


if __name__ == "__main__":
    main()
