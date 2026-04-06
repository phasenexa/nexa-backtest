"""Tests for cli/main.py: nexa run command."""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
from click.testing import CliRunner

from nexa_backtest.algo import SimpleAlgo
from nexa_backtest.cli.main import cli, find_algo_class

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_algo(path: Path, code: str) -> Path:
    path.write_text(textwrap.dedent(code))
    return path


def _write_da_prices(path: Path) -> None:
    """Write a minimal single-day DA prices parquet."""
    data = {
        "timestamp": pd.to_datetime(["2026-03-01T00:00:00Z", "2026-03-01T00:15:00Z"], utc=True),
        "zone": ["NO1", "NO1"],
        "price_eur_mwh": [45.0, 50.0],
        "volume_mwh": [1000.0, 1000.0],
    }
    pd.DataFrame(data).to_parquet(path / "da_prices.parquet", index=False)


# ---------------------------------------------------------------------------
# Tests: algo discovery
# ---------------------------------------------------------------------------


class TestFindAlgoClass:
    def test_finds_single_subclass(self, tmp_path: Path) -> None:
        code = """
            from nexa_backtest.algo import SimpleAlgo
            class MyAlgo(SimpleAlgo):
                pass
        """
        algo_file = _write_algo(tmp_path / "algo.py", code)
        cls = find_algo_class(str(algo_file))
        assert cls.__name__ == "MyAlgo"
        assert issubclass(cls, SimpleAlgo)

    def test_no_subclass_raises(self, tmp_path: Path) -> None:
        code = "x = 1"
        algo_file = _write_algo(tmp_path / "algo.py", code)
        import click

        with pytest.raises(click.ClickException, match="No SimpleAlgo"):
            find_algo_class(str(algo_file))

    def test_invalid_spec_raises_click_exception(self, tmp_path: Path) -> None:
        code = "x = 1"
        algo_file = _write_algo(tmp_path / "algo.py", code)
        import click

        with patch("importlib.util.spec_from_file_location", return_value=None):
            with pytest.raises(click.ClickException, match="Cannot load module"):
                find_algo_class(str(algo_file))

    def test_multiple_subclasses_raises(self, tmp_path: Path) -> None:
        code = """
            from nexa_backtest.algo import SimpleAlgo
            class AlgoA(SimpleAlgo):
                pass
            class AlgoB(SimpleAlgo):
                pass
        """
        algo_file = _write_algo(tmp_path / "algo.py", code)
        import click

        with pytest.raises(click.ClickException, match="Multiple"):
            find_algo_class(str(algo_file))


# ---------------------------------------------------------------------------
# Tests: nexa run CLI command
# ---------------------------------------------------------------------------


class TestRunCommand:
    @pytest.fixture
    def data_dir(self, tmp_path: Path) -> Path:
        _write_da_prices(tmp_path)
        return tmp_path

    @pytest.fixture
    def algo_file(self, tmp_path: Path) -> Path:
        code = """
            from nexa_backtest.algo import SimpleAlgo
            class NoOpAlgo(SimpleAlgo):
                pass
        """
        return _write_algo(tmp_path / "noop_algo.py", code)

    def test_run_succeeds_and_prints_summary(self, algo_file: Path, data_dir: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "run",
                str(algo_file),
                "--exchange",
                "nordpool",
                "--start",
                "2026-03-01",
                "--end",
                "2026-03-01",
                "--products",
                "NO1_DA",
                "--data-dir",
                str(data_dir),
                "--capital",
                "100000",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "nordpool" in result.output

    def test_run_with_missing_parquet_exits_nonzero(self, algo_file: Path, tmp_path: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "run",
                str(algo_file),
                "--exchange",
                "nordpool",
                "--start",
                "2026-03-01",
                "--end",
                "2026-03-01",
                "--products",
                "NO1_DA",
                "--data-dir",
                str(tmp_path),  # exists but no parquet
                "--capital",
                "100000",
            ],
        )
        assert result.exit_code != 0

    def test_run_loads_algo_finds_subclass(self, data_dir: Path, tmp_path: Path) -> None:
        code = """
            from nexa_backtest.algo import SimpleAlgo
            from nexa_backtest.context import TradingContext
            from nexa_backtest.types import AuctionInfo, Order
            from decimal import Decimal

            class BuyAlgo(SimpleAlgo):
                def on_auction_open(self, ctx: TradingContext, auction: AuctionInfo) -> None:
                    ctx.place_order(Order.buy(
                        product_id=auction.product_id,
                        volume_mw=Decimal('1'),
                        price_eur_mwh=Decimal('999'),
                    ))
        """
        algo_file = _write_algo(tmp_path / "buy_algo.py", code)
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "run",
                str(algo_file),
                "--exchange",
                "nordpool",
                "--start",
                "2026-03-01",
                "--end",
                "2026-03-01",
                "--products",
                "NO1_DA",
                "--data-dir",
                str(data_dir),
                "--capital",
                "100000",
            ],
        )
        assert result.exit_code == 0, result.output
        # Summary should show fills
        assert "Fills" in result.output

    def test_run_with_no_subclass_in_algo_file_exits_nonzero(
        self, data_dir: Path, tmp_path: Path
    ) -> None:
        algo_file = _write_algo(tmp_path / "empty.py", "x = 1\n")
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "run",
                str(algo_file),
                "--exchange",
                "nordpool",
                "--start",
                "2026-03-01",
                "--end",
                "2026-03-01",
                "--products",
                "NO1_DA",
                "--data-dir",
                str(data_dir),
                "--capital",
                "100000",
            ],
        )
        assert result.exit_code != 0
        assert "SimpleAlgo" in result.output
