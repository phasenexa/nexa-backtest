"""Tests for Step 3: Interface Compliance."""

from __future__ import annotations

from pathlib import Path

from nexa_backtest.validation.interface_check import InterfaceCheck
from tests.test_validation.conftest import fixture_path


def test_valid_simple_algo_passes() -> None:
    result = InterfaceCheck().run(fixture_path("valid_simple_algo.py"))
    assert result.status == "pass"
    assert not result.messages


def test_valid_async_algo_passes() -> None:
    result = InterfaceCheck().run(fixture_path("valid_async_algo.py"))
    assert result.status == "pass"


def test_no_hooks_is_error() -> None:
    result = InterfaceCheck().run(fixture_path("no_hooks.py"))
    assert result.status == "fail"
    assert any("no hooks" in m.lower() or "hook" in m.lower() for m in result.messages)


def test_unsubscribed_signal_is_warning() -> None:
    result = InterfaceCheck().run(fixture_path("unsubscribed_signal.py"))
    assert result.status == "warn"
    assert any("subscribe" in m.lower() or "subscribed" in m.lower() for m in result.messages)


def test_multiple_algos_is_error() -> None:
    result = InterfaceCheck().run(fixture_path("multiple_algos.py"))
    assert result.status == "fail"
    assert any("multiple" in m.lower() or "ambiguous" in m.lower() for m in result.messages)


def test_async_no_events_is_warning() -> None:
    result = InterfaceCheck().run(fixture_path("async_no_events.py"))
    assert result.status == "warn"
    assert any("events" in m.lower() for m in result.messages)


def test_syntax_error_skips() -> None:
    result = InterfaceCheck().run(fixture_path("syntax_error.py"))
    assert result.status == "skip"


def test_no_nexa_import_warns(tmp_path: Path) -> None:
    algo = tmp_path / "algo.py"
    algo.write_text("class Foo:\n    pass\n")
    result = InterfaceCheck().run(str(algo))
    assert result.status == "warn"
    assert any("nexa_backtest" in m for m in result.messages)


def test_async_non_async_decorator_is_error(tmp_path: Path) -> None:
    algo = tmp_path / "algo.py"
    algo.write_text(
        "from nexa_backtest.algo import algo\n"
        "from nexa_backtest.context import TradingContext\n\n"
        "@algo(name='x', version='1.0.0')\n"
        "def run(ctx: TradingContext) -> None:\n"
        "    pass\n"
    )
    result = InterfaceCheck().run(str(algo))
    assert result.status == "fail"
    assert any("async" in m.lower() for m in result.messages)


def test_algo_wrong_param_count_is_error(tmp_path: Path) -> None:
    algo = tmp_path / "algo.py"
    algo.write_text(
        "from nexa_backtest.algo import algo\n"
        "from nexa_backtest.context import TradingContext\n\n"
        "@algo(name='x', version='1.0.0')\n"
        "async def run(ctx: TradingContext, extra: int) -> None:\n"
        "    async for event in ctx.events():\n"
        "        pass\n"
    )
    result = InterfaceCheck().run(str(algo))
    assert result.status == "fail"
    assert any("one argument" in m.lower() or "parameter" in m.lower() for m in result.messages)


def test_file_not_found_is_error() -> None:
    result = InterfaceCheck().run("/nonexistent/path/algo.py")
    assert result.status == "fail"
    assert any("cannot read" in m.lower() for m in result.messages)


def test_direct_import_nexa_backtest_is_recognised(tmp_path: Path) -> None:
    # `import nexa_backtest` (not `from nexa_backtest import ...`) must not warn.
    algo = tmp_path / "algo.py"
    algo.write_text(
        "import nexa_backtest\n"
        "from nexa_backtest.algo import SimpleAlgo\n"
        "from nexa_backtest.context import TradingContext\n\n"
        "class A(SimpleAlgo):\n"
        "    def on_setup(self, ctx: TradingContext) -> None:\n"
        "        pass\n"
    )
    result = InterfaceCheck().run(str(algo))
    # Should not warn about missing nexa_backtest import.
    assert not any("no import" in m.lower() for m in result.messages)


def test_dotted_base_class_recognised(tmp_path: Path) -> None:
    # class Foo(algo_module.SimpleAlgo) — covers the ast.Attribute branch in _inherits_from.
    algo = tmp_path / "algo.py"
    algo.write_text(
        "from nexa_backtest import algo as algo_module\n\n"
        "class MyAlgo(algo_module.SimpleAlgo):\n"
        "    def on_setup(self, ctx) -> None:  # type: ignore[override]\n"
        "        pass\n"
    )
    result = InterfaceCheck().run(str(algo))
    # The dotted base should be recognised as a SimpleAlgo — no "no hooks" error.
    assert not any("no hooks" in m.lower() for m in result.messages)


def test_hook_missing_parameter_is_error(tmp_path: Path) -> None:
    # on_fill expects (self, ctx, fill) but we only provide (self, ctx).
    algo = tmp_path / "algo.py"
    algo.write_text(
        "from nexa_backtest.algo import SimpleAlgo\n"
        "from nexa_backtest.context import TradingContext\n\n"
        "class BadSig(SimpleAlgo):\n"
        "    def on_fill(self, ctx: TradingContext) -> None:\n"
        "        pass\n"
    )
    result = InterfaceCheck().run(str(algo))
    assert result.status == "fail"
    assert any("missing" in m.lower() or "parameter" in m.lower() for m in result.messages)


def test_module_algo_decorator_recognised(tmp_path: Path) -> None:
    # @nexa_backtest.algo(name=...) — covers ast.Attribute branch in _has_algo_decorator.
    algo = tmp_path / "algo.py"
    algo.write_text(
        "import nexa_backtest\n"
        "from nexa_backtest.context import TradingContext\n\n"
        "@nexa_backtest.algo(name='x', version='1.0.0')\n"
        "async def run(ctx: TradingContext) -> None:\n"
        "    async for event in ctx.events():\n"
        "        pass\n"
    )
    result = InterfaceCheck().run(str(algo))
    # Should pass (or warn at most) — not error about "not async".
    assert not any("must be async" in m.lower() for m in result.messages)


def test_non_hook_method_in_simple_algo_skipped(tmp_path: Path) -> None:
    # A helper method (not a hook) in a SimpleAlgo subclass — covers line 175 (continue branch).
    algo = tmp_path / "algo.py"
    algo.write_text(
        "from nexa_backtest.algo import SimpleAlgo\n"
        "from nexa_backtest.context import TradingContext\n\n"
        "class HelperAlgo(SimpleAlgo):\n"
        "    def on_setup(self, ctx: TradingContext) -> None:\n"
        "        self._init_data()\n\n"
        "    def _init_data(self) -> None:\n"
        "        pass\n"
    )
    result = InterfaceCheck().run(str(algo))
    # Should pass — helper method should not cause errors.
    assert result.status == "pass"
    assert not result.messages


def test_bare_algo_decorator_recognised(tmp_path: Path) -> None:
    # @algo without parentheses — covers the ast.Name return True path (line 268).
    algo = tmp_path / "algo.py"
    algo.write_text(
        "from nexa_backtest.algo import algo\n"
        "from nexa_backtest.context import TradingContext\n\n"
        "@algo\n"
        "async def run(ctx: TradingContext) -> None:\n"
        "    async for event in ctx.events():\n"
        "        pass\n"
    )
    result = InterfaceCheck().run(str(algo))
    # Bare @algo is recognised — should not error about "must be async".
    assert not any("must be async" in m.lower() for m in result.messages)
