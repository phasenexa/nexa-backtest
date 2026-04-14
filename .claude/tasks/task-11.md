# Task 11: Imbalance Settlement Backtesting

## Goal

Let BRPs backtest strategies that deliberately hold positions through
gate closure and take the TSO's imbalance settlement price instead of
trading out. The backtester already tracks what you bought and sold.
This task adds what happens after delivery: the TSO settles your
imbalance at a price that might be better or worse than the market.

After this task, a customer can answer: "what if I'd held this
position instead of closing it in IDC? Would the imbalance settlement
price have been more favourable?"

This is an opt-in feature. If no imbalance configuration is provided,
the backtester behaves exactly as before. Customers who don't care
about imbalance settlement never see any of this.

---

## Context: How Imbalance Settlement Works

A Balance Responsible Party (BRP) nominates a schedule with the TSO:
"I will deliver 50 MW during the 08:00-08:15 MTU." After delivery,
the TSO measures what actually happened. If the BRP delivered 53 MW,
there's a 3 MW imbalance. The TSO settles this at the imbalance
settlement price.

The settlement price depends on the system's regulation state:

- **System needed upward regulation** (short on power): the
  regulation price is typically higher than spot. If your imbalance
  helped (you delivered more than nominated), you get the favourable
  regulation price. If it hurt (you delivered less), you get the
  unfavourable imbalance price.
- **System needed downward regulation** (long on power): the
  regulation price is typically lower than spot. Same logic,
  reversed direction.
- **No regulation**: settlement at spot price. No gain, no penalty.

Nordic markets use different pricing models:

- **Dual pricing**: two prices depending on whether you helped or
  hurt. Favourable imbalances get the regulation price, unfavourable
  get a worse price.
- **Single pricing**: one price regardless of direction. Norway has
  moved towards this for some settlement periods.

The pricing model varies by TSO, zone, and time period. The
settlement engine must handle all cases.

---

## What to build

### 1. `settlement/types.py` - Imbalance Types

```python
@dataclass(frozen=True)
class ImbalancePrice:
    """Settlement prices for a single MTU as published by the TSO."""
    timestamp: datetime
    zone: str
    regulation_direction: str        # "up", "down", "none"
    regulation_price_eur_mwh: Decimal
    imbalance_price_up_eur_mwh: Decimal    # Price for positive imbalance
    imbalance_price_down_eur_mwh: Decimal  # Price for negative imbalance
    system_imbalance_mw: Decimal     # Net system imbalance
    pricing_model: str               # "single" or "dual"

@dataclass(frozen=True)
class Nomination:
    """Nominated volume for a delivery period."""
    delivery_start: datetime
    delivery_end: datetime
    zone: str
    volume_mw: Decimal

@dataclass(frozen=True)
class PhysicalPosition:
    """Actual generation or consumption for a delivery period."""
    delivery_start: datetime
    delivery_end: datetime
    zone: str
    volume_mw: Decimal

@dataclass(frozen=True)
class ImbalanceSettlement:
    """Settlement result for a single MTU."""
    delivery_start: datetime
    delivery_end: datetime
    zone: str
    nominated_mw: Decimal
    actual_mw: Decimal
    imbalance_mw: Decimal            # actual - nominated
    regulation_direction: str
    settlement_price_eur_mwh: Decimal  # The price applied
    settlement_eur: Decimal            # imbalance_mw * settlement_price
    direction_alignment: str           # "favourable" or "unfavourable"
    pricing_model: str
```

### 2. `settlement/config.py` - Imbalance Configuration

The opt-in boundary. If this config is not provided to the engine,
no settlement logic runs.

```python
@dataclass
class ImbalanceConfig:
    """Configuration for imbalance settlement backtesting.

    All fields are data sources. The settlement engine reads from
    these during the backtest.

    Args:
        zone: Bidding zone for settlement.
        tso: TSO identifier for pricing rules ("statnett",
            "fingrid", "energinet", "svk").
        settlement_prices: Loader for historical imbalance
            settlement prices.
        nominations: Loader for historical nominations, or a
            static schedule, or None if the algo manages
            nominations via ctx.set_nomination().
        physical_schedule: Signal or loader providing actual
            physical generation/consumption. Can be historical
            actuals or a deterministic schedule for "what if"
            analysis.
    """
    zone: str
    tso: str
    settlement_prices: ParquetLoader
    nominations: ParquetLoader | list[Nomination] | None = None
    physical_schedule: SignalProvider | list[PhysicalPosition] | None = None
```

When `nominations` is None, the algo is expected to call
`ctx.set_nomination()` during the backtest. This supports algos
that dynamically decide their nomination strategy.

When `physical_schedule` is None, actual physical position equals
the nomination (no imbalance). This is the "perfect delivery"
assumption, useful for testing the settlement engine in isolation.

### 3. `settlement/engine.py` - Settlement Engine

Calculates imbalance settlement after each delivery period.

```python
class SettlementEngine:
    """Post-delivery imbalance settlement calculation.

    Called by the BacktestEngine after each MTU's delivery period
    has passed. Looks up the imbalance price, determines direction
    alignment, and calculates the settlement amount.
    """

    def __init__(
        self,
        config: ImbalanceConfig,
        pricing_rules: PricingRules,
    ) -> None: ...

    def settle(
        self,
        delivery_start: datetime,
        delivery_end: datetime,
        nominated_mw: Decimal,
        actual_mw: Decimal,
    ) -> ImbalanceSettlement:
        """Calculate settlement for a single MTU.

        1. imbalance_mw = actual_mw - nominated_mw
        2. Look up regulation direction and prices for this MTU
        3. Determine if the imbalance is favourable or unfavourable
        4. Apply the correct settlement price
        5. settlement_eur = imbalance_mw * settlement_price
        """
```

**Settlement logic in detail:**

```python
imbalance_mw = actual_mw - nominated_mw

if imbalance_mw == 0:
    # No imbalance, no settlement
    settlement_price = Decimal("0")
    alignment = "none"

elif pricing_model == "single":
    # Single pricing: one price regardless of direction
    settlement_price = regulation_price
    alignment = "neutral"

elif pricing_model == "dual":
    if regulation_direction == "up":
        if imbalance_mw > 0:
            # Over-delivered when system needed more. Helped.
            settlement_price = regulation_price  # Favourable
            alignment = "favourable"
        else:
            # Under-delivered when system needed more. Hurt.
            settlement_price = imbalance_price_down  # Unfavourable
            alignment = "unfavourable"

    elif regulation_direction == "down":
        if imbalance_mw < 0:
            # Under-delivered when system had too much. Helped.
            settlement_price = regulation_price  # Favourable
            alignment = "favourable"
        else:
            # Over-delivered when system had too much. Hurt.
            settlement_price = imbalance_price_up  # Unfavourable
            alignment = "unfavourable"

    else:
        # No regulation: settle at spot/regulation price
        settlement_price = regulation_price
        alignment = "neutral"

settlement_eur = imbalance_mw * settlement_price
```

### 4. `settlement/pricing.py` - TSO-Specific Pricing Rules

Each TSO has slightly different rules. Abstract behind a protocol:

```python
class PricingRules(Protocol):
    """TSO-specific settlement pricing rules."""

    @property
    def tso_name(self) -> str: ...

    def get_settlement_price(
        self,
        imbalance_mw: Decimal,
        imbalance_price: ImbalancePrice,
    ) -> tuple[Decimal, str]:
        """Return (settlement_price, alignment).

        Each TSO may have different logic for determining which
        price applies. This method encapsulates those rules.
        """
        ...

def get_pricing_rules(tso: str) -> PricingRules:
    """Factory for TSO-specific pricing rules."""
    # "statnett" -> StatnettPricingRules
    # "fingrid" -> FingridPricingRules
    # etc.
```

Start with Statnett (Norwegian market) since Photon asked for it.
Other TSOs can be added as customer demand requires.

```python
class StatnettPricingRules:
    """Norwegian imbalance settlement rules.

    Norway uses a combination of single and dual pricing depending
    on the settlement period. As of 2025, the trend is towards
    single pricing for most periods, with dual pricing for periods
    with significant regulation.

    References:
        - Statnett balance settlement documentation
        - Nordic Balancing Model regulations
    """
    tso_name = "statnett"

    def get_settlement_price(
        self,
        imbalance_mw: Decimal,
        imbalance_price: ImbalancePrice,
    ) -> tuple[Decimal, str]:
        ...
```

### 5. Data Schema and Loader

**Imbalance price Parquet schema:**

```python
IMBALANCE_PRICE_SCHEMA = pa.schema([
    ("timestamp", pa.timestamp("ns", tz="UTC")),
    ("zone", pa.string()),
    ("regulation_direction", pa.string()),     # "up", "down", "none"
    ("regulation_price_eur_mwh", pa.float64()),
    ("imbalance_price_up_eur_mwh", pa.float64()),
    ("imbalance_price_down_eur_mwh", pa.float64()),
    ("system_imbalance_mw", pa.float64()),
    ("pricing_model", pa.string()),            # "single", "dual"
])
```

**Nomination Parquet schema:**

```python
NOMINATION_SCHEMA = pa.schema([
    ("delivery_start", pa.timestamp("ns", tz="UTC")),
    ("delivery_end", pa.timestamp("ns", tz="UTC")),
    ("zone", pa.string()),
    ("volume_mw", pa.float64()),
])
```

**Physical position Parquet schema:**

```python
PHYSICAL_POSITION_SCHEMA = pa.schema([
    ("delivery_start", pa.timestamp("ns", tz="UTC")),
    ("delivery_end", pa.timestamp("ns", tz="UTC")),
    ("zone", pa.string()),
    ("volume_mw", pa.float64()),
])
```

Data volumes are small (96 rows per day per zone for each, all
loaded entirely into memory like DA data).

**File layout:**

```text
data/
  imbalance/
    statnett/
      settlement_prices/
        NO1_2026.parquet
      nominations/
        NO1_2026_03.parquet    # Optional, may come from algo
      physical/
        NO1_2026_03.parquet    # Optional, defaults to nomination
```

Add loaders to `data/loader.py` for these schemas.

### 6. Update `context.py` - New TradingContext Methods

```python
class TradingContext(Protocol):
    # Existing methods unchanged...

    # New for imbalance settlement:
    def get_nomination(self, product_id: str) -> Decimal | None:
        """Current nominated volume for a delivery period.
        Returns None if no nomination has been set and no
        nomination data was provided in ImbalanceConfig."""

    def set_nomination(self, product_id: str, volume_mw: Decimal) -> None:
        """Set or update nomination for a delivery period.
        Must be called before nomination gate closure.
        Raises AlgoError if called after gate closure or if
        imbalance settlement is not configured."""

    def get_imbalance_price(
        self, delivery_start: datetime
    ) -> ImbalancePrice | None:
        """Settlement prices for a delivery period.
        Only available after the delivery period has passed
        (in backtest mode, this means the clock has advanced
        past the delivery period). Returns None if imbalance
        settlement is not configured or delivery hasn't occurred."""

    def get_expected_imbalance(self, product_id: str) -> Decimal | None:
        """Estimated imbalance based on current nomination vs
        the physical schedule. Useful for deciding whether to
        trade out or hold a position.
        Returns None if imbalance settlement is not configured."""
```

These methods must return `None` (not raise errors) when imbalance
settlement is not configured. This ensures algos that don't use
imbalance features work on engines without imbalance config.

### 7. Update `engines/backtest.py`

Wire the settlement engine into the backtest loop:

```python
# Extended backtest loop (additions marked with ###):

# For each MTU:
#   1. Advance clock
#   2. Dispatch market events to algo
#   3. Run matching engine
#   4. Update positions
#   5. Record equity snapshot
#   ### 6. If past a delivery period AND imbalance config exists:
#   ###    a. Get nomination for this delivery period
#   ###    b. Get actual physical position
#   ###    c. Run settlement engine
#   ###    d. Add settlement PnL to equity
#   ###    e. Record ImbalanceSettlement
```

Settlement happens after the delivery period has passed, not during
trading. In the backtest loop, this means: when the clock advances
past delivery_end for a product, run settlement for that MTU.

Accept `ImbalanceConfig` as an optional parameter:

```python
result = BacktestEngine(
    algo=algo,
    exchange="nordpool",
    start=date(2026, 3, 1),
    end=date(2026, 3, 31),
    products=["NO1_DA", "NO1-QH"],
    imbalance=ImbalanceConfig(
        zone="NO1",
        tso="statnett",
        settlement_prices=ParquetLoader("data/imbalance/statnett/settlement_prices"),
        nominations=ParquetLoader("data/imbalance/statnett/nominations"),
        physical_schedule=CsvSignalProvider(
            name="physical_schedule",
            path="data/physical_NO1.csv",
            unit="MW",
        ),
    ),
    initial_capital=100_000,
).run()
```

### 8. Update `BacktestResult`

```python
class BacktestResult:
    # Existing fields unchanged...

    # New (only populated when imbalance config is provided):
    settlements: list[ImbalanceSettlement] | None
    settlement_pnl: Decimal | None       # Sum of all settlement amounts
    trading_pnl: Decimal | None          # PnL from fills only (existing total_pnl)
    total_pnl: Decimal                   # trading_pnl + settlement_pnl (updated)
    avg_imbalance_mw: Decimal | None     # Average absolute imbalance
    favourable_settlement_pct: Decimal | None  # % of settlements that were favourable
    settlement_count: int | None
```

When imbalance is not configured, all settlement fields are None
and `total_pnl` equals the existing trading PnL (no behaviour
change).

When imbalance IS configured, `total_pnl` becomes
`trading_pnl + settlement_pnl`. The existing `total_pnl` field
changes meaning slightly: it now includes settlement. Add
`trading_pnl` as the explicit field for the trading-only component.

### 9. Update `summary()` Output

```
Backtest Results: 2026-03-01 to 2026-03-31 (31 days)
Exchange: Nord Pool | Products: NO1_DA, NO1-QH
Imbalance Settlement: Statnett (NO1)

  Trading PnL:     +12,340.50 EUR
  Settlement PnL:   +3,480.20 EUR
  Total PnL:       +15,820.70 EUR

  vs VWAP:           +0.65 EUR/MWh (+3.2%)
  Sharpe Ratio:      1.62
  Max Drawdown:     -4,200.00 EUR (-3.8%)

  Trades:            186
  Settlements:       2,976 (96 MTUs x 31 days)
  Favourable:        58.2%
  Avg Imbalance:     2.4 MW
```

### 10. Update HTML Report

Add an "Imbalance Settlement" section to the HTML report when
imbalance data is present:

**Charts:**

1. **Settlement PnL over time** - cumulative line chart showing
   settlement PnL accruing over the backtest period.
2. **Trading vs Settlement PnL** - stacked area chart showing the
   two components of total PnL.
3. **Imbalance volume distribution** - histogram of imbalance_mw
   values. Shows how often and how much the BRP was out of position.
4. **Direction alignment** - pie chart or bar showing % favourable
   vs unfavourable vs neutral settlements.

**Tables:**

1. **Settlement summary** - key metrics (total settlement PnL,
   avg imbalance, favourable %).
2. **Top 10 best settlements** - MTUs where the imbalance strategy
   paid off most.
3. **Top 10 worst settlements** - MTUs where holding the position
   cost the most.
4. **Daily breakdown** - extended to include settlement PnL per day.

Use the existing Phase Nexa styling. Settlement-specific colours:
favourable settlements in Mint (#4DFFC3), unfavourable in Electric
Coral (#FF6B6B).

### 11. "What If" Comparison

Add a helper that answers: "what if I had traded out instead of
taking the imbalance settlement?"

```python
class BacktestResult:
    def settlement_vs_trading_analysis(self) -> SettlementAnalysis | None:
        """Compare settlement outcomes against what the algo would
        have received by trading out at the last available IDC price.

        For each settled MTU, compares:
        - Settlement amount (what the BRP actually received/paid)
        - Trading counterfactual (what they'd have got selling/buying
          the imbalance volume at the last IDC VWAP before gate closure)

        Returns None if imbalance is not configured.
        """

@dataclass(frozen=True)
class SettlementAnalysis:
    total_settlement_pnl: Decimal
    total_trading_counterfactual: Decimal
    settlement_advantage: Decimal          # settlement - counterfactual
    mtus_settlement_better: int
    mtus_trading_better: int
    summary: str
```

This requires the last IDC price before gate closure for each MTU,
which is already available from the matching engine's trade history.

---

## Example Algo

Create `examples/imbalance_strategy.py`:

```python
"""
Example: deliberate imbalance strategy.

This algo trades in the DA auction but intentionally holds a long
position through gate closure when it forecasts upward regulation,
expecting the imbalance settlement price to be higher than the
intraday price.

Uses a regulation direction forecast signal to decide when to hold
vs when to trade out.
"""

class ImbalanceAlgo(SimpleAlgo):
    def on_setup(self, ctx: TradingContext) -> None:
        self.subscribe_signal("regulation_forecast")
        self.subscribe_signal("price_forecast")

    def on_auction_open(self, ctx: TradingContext, auction: AuctionInfo) -> None:
        forecast = ctx.get_signal("price_forecast").value
        # Always buy in DA
        ctx.place_order(Order.buy(
            product=auction.product_id,
            volume_mw=10,
            price_eur=Decimal(str(forecast)),
        ))

    def on_gate_closure(self, ctx: TradingContext, product_id: str) -> None:
        regulation = ctx.get_signal("regulation_forecast").value
        pos = ctx.get_position(product_id)

        if regulation == "up" and pos.net_mw > 0:
            # Expect upward regulation, hold the long position
            ctx.log(f"Holding {pos.net_mw} MW, expecting up regulation")
            # Set nomination below our actual expected delivery
            expected_delivery = pos.net_mw
            ctx.set_nomination(product_id, expected_delivery - pos.net_mw)
        else:
            # Trade out the position in IDC
            if pos.net_mw != 0:
                ctx.place_order(Order.market(
                    product=product_id,
                    volume_mw=-pos.net_mw,
                ))
```

Create supporting fixture data:

- `tests/fixtures/imbalance/statnett/settlement_prices/NO1_2026.parquet`
- `tests/fixtures/imbalance/regulation_forecast.csv`
- `tests/fixtures/imbalance/physical_NO1.csv`

Generate with `tests/generate_imbalance_fixtures.py` using a
deterministic seed. The fixture should cover one month with realistic
patterns: regulation up ~40% of MTUs, down ~35%, none ~25%.

---

## Tests

1. **Settlement calculation - dual pricing, favourable**: imbalance
   in same direction as regulation. Verify favourable price applied.

2. **Settlement calculation - dual pricing, unfavourable**: imbalance
   opposing regulation. Verify unfavourable price applied.

3. **Settlement calculation - single pricing**: verify same price
   applied regardless of direction.

4. **Settlement calculation - no regulation**: verify regulation
   price applied, alignment is "neutral".

5. **Settlement calculation - zero imbalance**: verify zero
   settlement amount.

6. **Statnett pricing rules**: test with realistic Norwegian data.
   Verify correct price selection for each scenario.

7. **ImbalanceConfig opt-in**: run a backtest without ImbalanceConfig.
   Verify all settlement fields on BacktestResult are None. Verify
   total_pnl equals trading PnL. Verify no settlement section in
   HTML report.

8. **ImbalanceConfig with data**: run a backtest with ImbalanceConfig.
   Verify settlements are calculated, settlement_pnl is populated,
   total_pnl = trading_pnl + settlement_pnl.

9. **ctx.set_nomination()**: algo sets nominations dynamically.
   Verify the settlement engine uses the algo's nominations, not
   the data file.

10. **ctx.get_expected_imbalance()**: verify it returns the correct
    estimate based on current nomination vs physical schedule.

11. **ctx methods without config**: verify get_nomination(),
    get_imbalance_price(), get_expected_imbalance() all return None
    when imbalance is not configured.

12. **ctx.set_nomination() without config**: verify it raises
    AlgoError with a clear message.

13. **Settlement vs trading analysis**: run a backtest with
    imbalance, call settlement_vs_trading_analysis(). Verify the
    counterfactual uses the last IDC price before gate closure.

14. **HTML report with settlement section**: verify the report
    includes the settlement charts and tables when imbalance data
    is present. Verify it does not include them when absent.

15. **Summary output**: verify the text summary shows trading PnL,
    settlement PnL, and total PnL separately.

16. **End-to-end with example algo**: run the imbalance_strategy.py
    example against fixture data. Verify it produces a complete
    result with both trading and settlement PnL.

---

## Package Structure

New files:

```text
src/nexa_backtest/
    settlement/
        __init__.py
        types.py         # ImbalancePrice, Nomination, PhysicalPosition,
                         # ImbalanceSettlement
        config.py        # ImbalanceConfig
        engine.py        # SettlementEngine
        pricing.py       # PricingRules protocol, StatnettPricingRules
```

Modified files:

```text
    context.py           # New methods (get_nomination, set_nomination, etc.)
    types.py             # (no changes, new types are in settlement/types.py)
    engines/backtest.py  # Wire settlement into the loop
    analysis/pnl.py      # Add settlement PnL component
    analysis/metrics.py  # Add settlement metrics
    analysis/report.py   # Add settlement section to HTML report
    data/loader.py       # Add imbalance price and nomination loaders
    data/schema.py       # Add imbalance Parquet schemas
```

---

## What NOT to build

- Pricing rules for Fingrid, Energinet, or SVK. Start with Statnett
  only. Other TSOs follow the same pattern and can be added when
  customers ask for them.
- Nomination optimisation ("what nomination minimises my expected
  settlement cost"). That's a separate tool, not a backtester
  feature.
- Real-time imbalance price feeds (that's nexa-marketdata scope).
- Imbalance forecasting models (that's nexa-forecast scope). The
  backtester consumes forecasts as signals, it doesn't produce them.
- Reserve market participation (FCR, aFRR, mFRR activation). These
  are separate revenue streams with their own settlement rules.
  Out of scope.
- Multi-zone imbalance netting. Some BRPs operate across zones and
  can net imbalances. This is a v2 concern.

---

## Acceptance criteria

1. `make ci` passes
2. Imbalance settlement is fully opt-in. Backtests without
   ImbalanceConfig behave identically to before.
3. When configured, the settlement engine correctly calculates
   settlement amounts using TSO-specific pricing rules.
4. Both dual and single pricing models are supported.
5. `total_pnl` includes settlement PnL when imbalance is configured.
6. Algos can manage nominations dynamically via `ctx.set_nomination()`
7. The HTML report includes a settlement section when imbalance
   data is present.
8. The "settlement vs trading" analysis shows the counterfactual.
9. The example algo demonstrates a deliberate imbalance strategy.
10. Statnett pricing rules are implemented and tested.
11. All new types have type hints and frozen Pydantic models where
    appropriate.
12. All new public API has Google-style docstrings.
