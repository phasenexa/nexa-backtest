"""Tests for Step 5: Look-Ahead Bias Detection."""

from __future__ import annotations

from pathlib import Path

from nexa_backtest.validation.lookahead_check import LookaheadCheck
from tests.test_validation.conftest import fixture_path


def test_lookahead_bias_fixture_warns() -> None:
    result = LookaheadCheck().run(fixture_path("lookahead_bias.py"))
    assert result.status == "warn"
    # Both shift(-1) and large lookback should be flagged.
    combined = " ".join(result.messages).lower()
    assert "shift" in combined or "lookback" in combined


def test_negative_shift_warns(tmp_path: Path) -> None:
    algo = tmp_path / "algo.py"
    algo.write_text(
        "from nexa_backtest.algo import SimpleAlgo\n"
        "from nexa_backtest.context import TradingContext\n"
        "import pandas as pd\n\n"
        "class ShiftAlgo(SimpleAlgo):\n"
        "    def on_bar(self, ctx: TradingContext) -> None:\n"
        "        df = pd.DataFrame({'x': [1, 2, 3]})\n"
        "        future = df['x'].shift(-1)\n"
    )
    result = LookaheadCheck().run(str(algo))
    assert result.status == "warn"
    assert any("shift" in m.lower() for m in result.messages)


def test_positive_shift_passes(tmp_path: Path) -> None:
    algo = tmp_path / "algo.py"
    algo.write_text(
        "from nexa_backtest.algo import SimpleAlgo\n"
        "from nexa_backtest.context import TradingContext\n"
        "import pandas as pd\n\n"
        "class ShiftAlgo(SimpleAlgo):\n"
        "    def on_bar(self, ctx: TradingContext) -> None:\n"
        "        df = pd.DataFrame({'x': [1, 2, 3]})\n"
        "        past = df['x'].shift(1)\n"
    )
    result = LookaheadCheck().run(str(algo))
    assert result.status == "pass"


def test_large_lookback_warns(tmp_path: Path) -> None:
    algo = tmp_path / "algo.py"
    algo.write_text(
        "from nexa_backtest.algo import SimpleAlgo\n"
        "from nexa_backtest.context import TradingContext\n\n"
        "class LookbackAlgo(SimpleAlgo):\n"
        "    def on_bar(self, ctx: TradingContext) -> None:\n"
        "        history = ctx.get_signal_history('forecast', lookback=96)\n"
    )
    result = LookaheadCheck().run(str(algo))
    assert result.status == "warn"
    assert any("lookback" in m.lower() for m in result.messages)


def test_small_lookback_passes(tmp_path: Path) -> None:
    algo = tmp_path / "algo.py"
    algo.write_text(
        "from nexa_backtest.algo import SimpleAlgo\n"
        "from nexa_backtest.context import TradingContext\n\n"
        "class LookbackAlgo(SimpleAlgo):\n"
        "    def on_bar(self, ctx: TradingContext) -> None:\n"
        "        history = ctx.get_signal_history('forecast', lookback=4)\n"
    )
    result = LookaheadCheck().run(str(algo))
    assert result.status == "pass"


def test_valid_algo_passes() -> None:
    result = LookaheadCheck().run(fixture_path("valid_simple_algo.py"))
    assert result.status == "pass"


def test_sort_iloc_warns(tmp_path: Path) -> None:
    algo = tmp_path / "algo.py"
    algo.write_text(
        "from nexa_backtest.algo import SimpleAlgo\n"
        "from nexa_backtest.context import TradingContext\n"
        "import pandas as pd\n\n"
        "class SortAlgo(SimpleAlgo):\n"
        "    def on_bar(self, ctx: TradingContext) -> None:\n"
        "        df = pd.DataFrame({'ts': [1, 2], 'v': [10, 20]})\n"
        "        val = df.sort_values('ts').iloc[1]\n"
    )
    result = LookaheadCheck().run(str(algo))
    assert result.status == "warn"
    assert any("sort_values" in m for m in result.messages)


def test_file_not_found_is_error() -> None:
    result = LookaheadCheck().run("/nonexistent/path/algo.py")
    assert result.status == "fail"
    assert any("cannot read" in m.lower() for m in result.messages)


def test_syntax_error_is_skip(tmp_path: Path) -> None:
    algo = tmp_path / "algo.py"
    algo.write_text("def broken(\n")
    result = LookaheadCheck().run(str(algo))
    assert result.status == "skip"
    assert any("syntax" in m.lower() for m in result.messages)


def test_shift_periods_keyword_warns(tmp_path: Path) -> None:
    # df.shift(periods=-1) — exercises the keyword-argument path in _extract_negative_int.
    algo = tmp_path / "algo.py"
    algo.write_text(
        "from nexa_backtest.algo import SimpleAlgo\n"
        "from nexa_backtest.context import TradingContext\n"
        "import pandas as pd\n\n"
        "class KwShiftAlgo(SimpleAlgo):\n"
        "    def on_bar(self, ctx: TradingContext) -> None:\n"
        "        df = pd.DataFrame({'x': [1, 2, 3]})\n"
        "        future = df['x'].shift(periods=-2)\n"
    )
    result = LookaheadCheck().run(str(algo))
    assert result.status == "warn"
    assert any("shift" in m.lower() for m in result.messages)


def test_signal_history_positional_lookback_warns(tmp_path: Path) -> None:
    # ctx.get_signal_history('name', 100) — positional arg, exercises lines 166-168.
    algo = tmp_path / "algo.py"
    algo.write_text(
        "from nexa_backtest.algo import SimpleAlgo\n"
        "from nexa_backtest.context import TradingContext\n\n"
        "class PosLookbackAlgo(SimpleAlgo):\n"
        "    def on_bar(self, ctx: TradingContext) -> None:\n"
        "        history = ctx.get_signal_history('forecast', 100)\n"
    )
    result = LookaheadCheck().run(str(algo))
    assert result.status == "warn"
    assert any("lookback" in m.lower() for m in result.messages)


def test_iloc_forward_offset_warns(tmp_path: Path) -> None:
    # df.iloc[i + 2] — exercises _check_iloc_forward (lines 195-196).
    algo = tmp_path / "algo.py"
    algo.write_text(
        "from nexa_backtest.algo import SimpleAlgo\n"
        "from nexa_backtest.context import TradingContext\n"
        "import pandas as pd\n\n"
        "class IlocAlgo(SimpleAlgo):\n"
        "    def on_bar(self, ctx: TradingContext) -> None:\n"
        "        df = pd.DataFrame({'x': range(10)})\n"
        "        i = 3\n"
        "        val = df.iloc[i + 2]\n"
    )
    result = LookaheadCheck().run(str(algo))
    assert result.status == "warn"
    assert any("iloc" in m.lower() for m in result.messages)
