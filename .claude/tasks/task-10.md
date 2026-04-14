# Task 10: CLI First-Run Support Notice

## Objective

Add a one-time, non-intrusive notice to the `nexa` CLI that informs users about
support options and community resources on first run. This is the primary
mechanism for reaching users who install via `pip` and never read the full
README.

## Requirements

### Core behaviour

1. On the **first** invocation of any `nexa` subcommand (`run`, `compare`,
   `validate`), print a short notice to stderr after the command output.
2. The notice must appear **after** the command's own output, not before. The
   user ran a command and expects its results first.
3. After displaying once, write a marker file so it never displays again.
4. The notice must print to **stderr**, not stdout. This ensures it doesn't
   pollute piped output (e.g. `nexa run algo.py --output results.json`).
5. The notice must be suppressible via `--quiet` flag or `NEXA_QUIET=1`
   environment variable.

### Notice content

```text
--------------------------------------------------------------------------------
  Thanks for using nexa-backtest.
  Support tiers & priority fixes: https://github.com/phasenexa/nexa-backtest/blob/main/SUPPORT.md
  Community & questions:          https://github.com/phasenexa/nexa-backtest/discussions
--------------------------------------------------------------------------------
```

Keep it to exactly this. No emoji, no sales language, no version check, no
telemetry. Two links and a thank you.

### Marker file

- Location: `~/.config/nexa/notice_shown`
- Create parent directories if they don't exist.
- The file should contain the version that displayed the notice (e.g.
  `0.6.0b1`) for future reference, but the presence of the file is what
  matters, not the content.
- Respect `XDG_CONFIG_HOME` if set. Fall back to `~/.config`.

### Quiet mode

- `--quiet` / `-q` flag on all subcommands suppresses the notice (and any
  other non-essential output).
- `NEXA_QUIET=1` environment variable does the same.
- Quiet mode does **not** write the marker file. The user might be running in
  CI where they always pass `--quiet`. They should still see the notice the
  first time they run interactively.

## Implementation

### `src/nexa_backtest/cli/notice.py`

New module. No external dependencies.

```python
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
    """Return the nexa config directory, respecting XDG_CONFIG_HOME."""
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "nexa"


def _marker_path() -> Path:
    return _config_dir() / "notice_shown"


def maybe_show_notice() -> None:
    """Show the support notice once, then never again."""
    try:
        if _marker_path().exists():
            return

        print(NOTICE, file=sys.stderr)

        _config_dir().mkdir(parents=True, exist_ok=True)
        _marker_path().write_text(__version__)
    except OSError:
        # Silently skip on permission errors or read-only filesystems.
        pass
```

### Hooking into `src/nexa_backtest/cli/main.py`

Add `--quiet` / `-q` to both `run_command` and `compare_command`. Call
`maybe_show_notice()` at the end of each, after all output has been written.

For `run_command`:

```python
@cli.command("run")
# ... existing options ...
@click.option("--quiet", "-q", is_flag=True, default=False,
              help="Suppress non-essential output including the first-run notice.")
def run_command(..., quiet: bool) -> None:
    ...
    click.echo(result.summary())
    # ... output file writing ...

    if not quiet and not os.environ.get("NEXA_QUIET"):
        from nexa_backtest.cli.notice import maybe_show_notice
        maybe_show_notice()
```

For `compare_command`: same pattern — add `quiet: bool` parameter and call at
the end after `click.echo(comparison.summary())` and any output file writing.

### Hooking into `src/nexa_backtest/cli/validate.py`

`validate_command` calls `sys.exit()` when validation fails, so the notice must
be printed **before** those exit calls. Add `--quiet` / `-q` and insert the
notice call before the `sys.exit` block at the bottom of the function:

```python
@click.command("validate")
# ... existing options ...
@click.option("--quiet", "-q", is_flag=True, default=False,
              help="Suppress non-essential output including the first-run notice.")
def validate_command(..., quiet: bool) -> None:
    ...
    # existing output (result.summary() or JSON) already printed above

    if not quiet and not os.environ.get("NEXA_QUIET"):
        from nexa_backtest.cli.notice import maybe_show_notice
        maybe_show_notice()

    # Exit codes — must come after the notice
    if not result.passed:
        if strict and result.warning_count > 0 and result.error_count == 0:
            sys.exit(2)
        sys.exit(1)
```

The `import os` is already in `main.py`; add it to `validate.py` if not
present.

## Testing

Add `tests/test_cli_notice.py`. Use the `tmp_path` fixture to isolate the
config directory; monkeypatch `nexa_backtest.cli.notice._config_dir` to return
a path under `tmp_path`.

Test scenarios:

1. **Shows on first run**: marker absent → notice printed to stderr, marker
   file written containing the current version string.
2. **Does not show on second run**: marker present → no output, marker
   unchanged.
3. **Respects `--quiet`**: marker absent, call `maybe_show_notice()` with the
   quiet guard in place → no output, marker **not** written.
4. **Respects `NEXA_QUIET`**: same as above with the env var set to `"1"`.
5. **Does not pollute stdout**: capture stdout and stderr separately; confirm
   notice appears only on stderr.
6. **Creates missing config dir**: delete the tmp dir and run → dir and marker
   are created.
7. **Handles read-only filesystem**: monkeypatch `Path.write_text` to raise
   `OSError` → no exception propagates.

For the CLI integration, use `click.testing.CliRunner` to invoke `run_command`,
`compare_command`, and `validate_command` with the `--quiet` flag and confirm
the notice is absent. At minimum test one command; testing all three is
preferred.

## Acceptance criteria

- [ ] Notice displays once on first CLI invocation, after command output
- [ ] Notice prints to stderr only
- [ ] Marker file written to `~/.config/nexa/notice_shown` (or XDG equivalent)
- [ ] `--quiet` / `-q` and `NEXA_QUIET=1` suppress the notice without writing marker
- [ ] `validate_command` shows notice before `sys.exit()` calls
- [ ] No crash on filesystem errors (permissions, read-only, etc.)
- [ ] Tests cover all 7 scenarios above
- [ ] `make ci` passes
- [ ] No external dependencies added
