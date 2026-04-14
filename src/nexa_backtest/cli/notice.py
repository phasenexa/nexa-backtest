"""First-run support notice for the nexa CLI."""

import os
import sys
from pathlib import Path

from nexa_backtest import __version__

NOTICE = """\

--------------------------------------------------------------------------------
  Thanks for using nexa-backtest.
  Support tiers & priority fixes: https://github.com/phasenexa/nexa-backtest/blob/main/SUPPORT.md
  Community & questions:          https://github.com/phasenexa/nexa-backtest/discussions
--------------------------------------------------------------------------------"""


def _config_dir() -> Path:
    """Return the nexa config directory.

    On Windows uses ``%APPDATA%\\nexa`` (roaming profile), falling back to
    ``~/AppData/Roaming/nexa``. On all other platforms uses
    ``$XDG_CONFIG_HOME/nexa``, falling back to ``~/.config/nexa``.
    """
    if sys.platform == "win32":
        base = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
    else:
        base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "nexa"


def _marker_path() -> Path:
    return _config_dir() / "notice_shown"


def maybe_show_notice() -> None:
    """Show the support notice once, then never again.

    Prints the notice to stderr on first invocation of any nexa CLI command.
    After displaying, writes a marker file containing the current version so
    the notice is not shown again. On filesystem errors, silently does nothing.
    """
    try:
        if _marker_path().exists():
            return

        print(NOTICE, file=sys.stderr)

        _config_dir().mkdir(parents=True, exist_ok=True)
        _marker_path().write_text(__version__)
    except OSError:
        pass
