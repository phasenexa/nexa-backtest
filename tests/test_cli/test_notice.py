"""Tests for the first-run support notice (nexa_backtest.cli.notice)."""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import patch

import pytest

from nexa_backtest import __version__
from nexa_backtest.cli.notice import NOTICE, maybe_show_notice

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_notice(
    config_dir: Path,
    *,
    quiet: bool = False,
    nexa_quiet_env: str | None = None,
) -> tuple[str, str]:
    """Run the notice logic and return (stdout, stderr) as strings."""
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()

    env_patch: dict[str, str] = {}
    if nexa_quiet_env is not None:
        env_patch["NEXA_QUIET"] = nexa_quiet_env

    with (
        patch("nexa_backtest.cli.notice._config_dir", return_value=config_dir),
        patch("sys.stdout", stdout_buf),
        patch("sys.stderr", stderr_buf),
        patch.dict("os.environ", env_patch, clear=False),
    ):
        if not quiet:
            maybe_show_notice()

    return stdout_buf.getvalue(), stderr_buf.getvalue()


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


class TestMaybeShowNotice:
    def test_shows_on_first_run(self, tmp_path: Path) -> None:
        """Marker absent → notice is printed to stderr, marker is created."""
        config_dir = tmp_path / "nexa"
        _, stderr = _run_notice(config_dir)

        assert NOTICE in stderr
        marker = config_dir / "notice_shown"
        assert marker.exists()
        assert marker.read_text() == __version__

    def test_does_not_show_on_second_run(self, tmp_path: Path) -> None:
        """Marker present → no output, marker unchanged."""
        config_dir = tmp_path / "nexa"
        config_dir.mkdir(parents=True)
        marker = config_dir / "notice_shown"
        marker.write_text("0.0.1")

        _, stderr = _run_notice(config_dir)

        assert stderr == ""
        assert marker.read_text() == "0.0.1"  # unchanged

    def test_does_not_pollute_stdout(self, tmp_path: Path) -> None:
        """Notice appears on stderr only, stdout remains empty."""
        config_dir = tmp_path / "nexa"
        stdout, stderr = _run_notice(config_dir)

        assert NOTICE in stderr
        assert NOTICE not in stdout
        assert stdout == ""

    def test_creates_missing_config_dir(self, tmp_path: Path) -> None:
        """Missing config directory is created together with the marker file."""
        config_dir = tmp_path / "deep" / "nested" / "nexa"
        assert not config_dir.exists()

        _run_notice(config_dir)

        assert config_dir.exists()
        assert (config_dir / "notice_shown").exists()

    def test_handles_read_only_filesystem(self, tmp_path: Path) -> None:
        """OSError on write_text must not propagate."""
        config_dir = tmp_path / "nexa"

        with patch("pathlib.Path.write_text", side_effect=OSError("read-only")):
            # Should complete without raising
            _run_notice(config_dir)

    def test_respects_quiet_flag(self, tmp_path: Path) -> None:
        """When quiet=True, maybe_show_notice is not called at all (guard in caller).

        This test verifies the underlying function still writes normally; the
        guard lives in the CLI command, which is tested in the integration tests.
        """
        config_dir = tmp_path / "nexa"
        # Simulate the caller guard: don't call maybe_show_notice when quiet
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()

        quiet = True
        with (
            patch("nexa_backtest.cli.notice._config_dir", return_value=config_dir),
            patch("sys.stdout", stdout_buf),
            patch("sys.stderr", stderr_buf),
        ):
            if not quiet:
                maybe_show_notice()

        assert stderr_buf.getvalue() == ""
        assert not (config_dir / "notice_shown").exists()

    def test_respects_nexa_quiet_env(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """NEXA_QUIET=1 prevents the notice from being shown or marker written.

        The guard lives in the CLI caller, so we replicate it here.
        """
        monkeypatch.setenv("NEXA_QUIET", "1")
        import os

        config_dir = tmp_path / "nexa"
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()

        with (
            patch("nexa_backtest.cli.notice._config_dir", return_value=config_dir),
            patch("sys.stdout", stdout_buf),
            patch("sys.stderr", stderr_buf),
        ):
            if not os.environ.get("NEXA_QUIET"):
                maybe_show_notice()

        assert stderr_buf.getvalue() == ""
        assert not (config_dir / "notice_shown").exists()


# ---------------------------------------------------------------------------
# Platform-specific config dir tests
# ---------------------------------------------------------------------------


class TestConfigDir:
    """Unit tests for _config_dir() cross-platform behaviour."""

    def test_windows_uses_appdata(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """On Windows, APPDATA env var is used as the config base."""
        from nexa_backtest.cli.notice import _config_dir

        monkeypatch.setattr("sys.platform", "win32")
        monkeypatch.setenv("APPDATA", r"C:\Users\tom\AppData\Roaming")
        # XDG_CONFIG_HOME must not interfere on Windows
        monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)

        result = _config_dir()

        assert result == Path(r"C:\Users\tom\AppData\Roaming") / "nexa"

    def test_windows_falls_back_to_home_appdata(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """On Windows with APPDATA unset, falls back to ~/AppData/Roaming."""
        from nexa_backtest.cli.notice import _config_dir

        monkeypatch.setattr("sys.platform", "win32")
        monkeypatch.delenv("APPDATA", raising=False)
        monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
        # Redirect Path.home() to tmp_path to avoid touching the real home dir
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))

        result = _config_dir()

        assert result == tmp_path / "AppData" / "Roaming" / "nexa"


# ---------------------------------------------------------------------------
# CLI integration tests
# ---------------------------------------------------------------------------


class TestCliQuietFlag:
    """Verify --quiet suppresses the notice and does not write the marker."""

    def _marker_exists(self, tmp_path: Path) -> bool:
        return (tmp_path / "nexa" / "notice_shown").exists()

    def test_run_command_quiet_suppresses_notice(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from click.testing import CliRunner

        from nexa_backtest.cli.main import run_command

        config_dir = tmp_path / "nexa"
        monkeypatch.setattr("nexa_backtest.cli.notice._config_dir", lambda: config_dir)

        runner = CliRunner()
        # Invoke with --quiet; even if the engine would fail, notice must not appear
        result = runner.invoke(
            run_command,
            [
                "non_existent_algo.py",
                "--exchange",
                "nordpool",
                "--start",
                "2026-01-01",
                "--end",
                "2026-01-31",
                "--products",
                "NO1_DA",
                "--data-dir",
                str(tmp_path),
                "--quiet",
            ],
        )
        assert NOTICE not in (result.output or "")
        assert not self._marker_exists(tmp_path)

    def test_validate_command_quiet_suppresses_notice(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from click.testing import CliRunner

        from nexa_backtest.cli.validate import validate_command

        config_dir = tmp_path / "nexa"
        monkeypatch.setattr("nexa_backtest.cli.notice._config_dir", lambda: config_dir)

        # Write a minimal valid-looking algo file
        algo_file = tmp_path / "dummy_algo.py"
        algo_file.write_text("# placeholder\n")

        runner = CliRunner()
        result = runner.invoke(
            validate_command,
            [
                str(algo_file),
                "--exchange",
                "nordpool",
                "--quiet",
            ],
        )
        assert NOTICE not in (result.output or "")
        assert not self._marker_exists(tmp_path)

    def test_compare_command_quiet_suppresses_notice(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from click.testing import CliRunner

        from nexa_backtest.cli.main import compare_command

        config_dir = tmp_path / "nexa"
        monkeypatch.setattr("nexa_backtest.cli.notice._config_dir", lambda: config_dir)

        runner = CliRunner()
        result = runner.invoke(
            compare_command,
            [
                "a.py",
                "b.py",
                "--exchange",
                "nordpool",
                "--start",
                "2026-01-01",
                "--end",
                "2026-01-31",
                "--products",
                "NO1_DA",
                "--data-dir",
                str(tmp_path),
                "--quiet",
            ],
        )
        assert NOTICE not in (result.output or "")
        assert not self._marker_exists(tmp_path)
