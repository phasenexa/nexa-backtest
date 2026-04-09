"""Step 4: Exchange feature compatibility check.

Cross-references the algo's Order constructions against the target exchange's
ExchangeCapabilities using AST analysis. Only statically visible values can
be checked; dynamic computations are flagged as "unable to validate".
"""

from __future__ import annotations

import ast
import time
from decimal import Decimal
from pathlib import Path

from nexa_backtest.exchanges.base import ExchangeCapabilities
from nexa_backtest.validation.runner import StepResult

# Map exchange name → capabilities factory.
# Each factory returns a capabilities object without needing a live adapter.
_EXCHANGE_CAPABILITIES: dict[str, ExchangeCapabilities] = {}


def _get_capabilities(exchange: str) -> ExchangeCapabilities | None:
    """Return capabilities for ``exchange``, or None if unknown."""
    if not _EXCHANGE_CAPABILITIES:
        _populate_capabilities()
    return _EXCHANGE_CAPABILITIES.get(exchange.lower())


def _populate_capabilities() -> None:
    try:
        from nexa_backtest.exchanges.nordpool import NordPoolAdapter

        _EXCHANGE_CAPABILITIES["nordpool"] = NordPoolAdapter("NO1").capabilities
    except Exception:
        pass

    try:
        from nexa_backtest.exchanges.epex_spot import EpexSpotAdapter

        _EXCHANGE_CAPABILITIES["epex_spot"] = EpexSpotAdapter("DE-LU").capabilities
    except Exception:
        pass


class FeatureCheck:
    """Step 4: Exchange feature compatibility check.

    Parses the algo AST and cross-references all Order factory calls against
    the target exchange's ExchangeCapabilities.

    Limitations (reported as warnings, not errors):

    - Only checks statically visible literal values. Dynamic volumes/prices
      (``volume_mw=calculate_volume()``) cannot be validated.
    - Only checks keyword arguments. Positional-only calls are harder to
      validate reliably.
    """

    def run(self, algo_path: str, exchange: str) -> StepResult:
        """Check feature compatibility of ``algo_path`` against ``exchange``.

        Args:
            algo_path: Path to the algo Python file.
            exchange: Exchange identifier, e.g. ``"nordpool"``.

        Returns:
            :class:`~nexa_backtest.validation.runner.StepResult`.
        """
        start = time.monotonic()
        path = Path(algo_path)

        caps = _get_capabilities(exchange)
        if caps is None:
            return StepResult(
                name="Exchange Features",
                status="skip",
                messages=[
                    f"Unknown exchange '{exchange}'. Cannot perform feature compatibility check."
                ],
                duration_ms=_elapsed_ms(start),
            )

        try:
            source = path.read_text(encoding="utf-8")
        except OSError as exc:
            return StepResult(
                name="Exchange Features",
                status="fail",
                messages=[f"Cannot read file: {exc}"],
                duration_ms=_elapsed_ms(start),
            )

        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError:
            return StepResult(
                name="Exchange Features",
                status="skip",
                messages=["Skipped: syntax error prevents parsing."],
                duration_ms=_elapsed_ms(start),
            )

        errors: list[str] = []
        warnings: list[str] = []

        _check_order_calls(tree, path.name, caps, errors, warnings)

        messages = errors + warnings
        if errors:
            status = "fail"
        elif warnings:
            status = "warn"
        else:
            status = "pass"

        return StepResult(
            name="Exchange Features",
            status=status,
            messages=messages,
            duration_ms=_elapsed_ms(start),
        )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

# Order factory methods and whether they need block bid support.
_ORDER_FACTORIES = {
    "buy": False,
    "sell": False,
    "market": True,  # needs supports_market_orders
    "block_bid": True,  # needs supports_block_bids
}


def _check_order_calls(
    tree: ast.Module,
    filename: str,
    caps: ExchangeCapabilities,
    errors: list[str],
    warnings: list[str],
) -> None:
    """Walk the AST and check every Order factory call."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        # Match Order.buy(...), Order.sell(...), Order.market(...), Order.block_bid(...)
        if not (isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name)):
            continue
        if func.value.id != "Order":
            continue

        method = func.attr
        line = node.lineno

        if method == "block_bid" and not caps.supports_block_bids:
            errors.append(
                f"{filename}:{line}: Order.block_bid() used, but "
                f"{caps.exchange_id} does not support block bids."
            )

        if method == "market" and not caps.supports_market_orders:
            errors.append(
                f"{filename}:{line}: Order.market() (no limit price) used, but "
                f"{caps.exchange_id} does not support market orders."
            )

        # Extract keyword arguments for further checks.
        kwargs = {kw.arg: kw.value for kw in node.keywords if kw.arg is not None}

        # Check exclusive_group (linked orders).
        if "exclusive_group" in kwargs and not caps.supports_linked_orders:
            errors.append(
                f"{filename}:{line}: 'exclusive_group' parameter used, but "
                f"{caps.exchange_id} does not support linked/exclusive orders."
            )

        # Check volume.
        if "volume_mw" in kwargs:
            vol_node = kwargs["volume_mw"]
            vol = _extract_numeric(vol_node)
            if vol is not None:
                if Decimal(str(vol)) < caps.min_volume_mw:
                    errors.append(
                        f"{filename}:{line}: volume_mw={vol} is below the "
                        f"{caps.exchange_id} minimum of {caps.min_volume_mw} MW."
                    )
                if caps.max_volume_mw is not None and Decimal(str(vol)) > caps.max_volume_mw:
                    errors.append(
                        f"{filename}:{line}: volume_mw={vol} exceeds the "
                        f"{caps.exchange_id} maximum of {caps.max_volume_mw} MW."
                    )
            else:
                warnings.append(
                    f"{filename}:{line}: volume_mw is computed dynamically — "
                    "unable to validate against exchange limits."
                )

        # Check price.
        if "price_eur_mwh" in kwargs:
            price_node = kwargs["price_eur_mwh"]
            price = _extract_numeric(price_node)
            if price is not None:
                if Decimal(str(price)) < caps.min_price_eur_mwh:
                    errors.append(
                        f"{filename}:{line}: price_eur_mwh={price} is below the "
                        f"{caps.exchange_id} minimum of {caps.min_price_eur_mwh} EUR/MWh."
                    )
                if Decimal(str(price)) > caps.max_price_eur_mwh:
                    errors.append(
                        f"{filename}:{line}: price_eur_mwh={price} exceeds the "
                        f"{caps.exchange_id} maximum of {caps.max_price_eur_mwh} EUR/MWh."
                    )


def _extract_numeric(node: ast.expr) -> float | int | None:
    """Extract a literal numeric value from an AST node, or None if dynamic."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    # Handle unary minus: -10.0 is ast.UnaryOp(USub, Constant(10.0))
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.USub)
        and isinstance(node.operand, ast.Constant)
        and isinstance(node.operand.value, (int, float))
    ):
        return -node.operand.value
    return None


def _elapsed_ms(start: float) -> int:
    return int((time.monotonic() - start) * 1000)
