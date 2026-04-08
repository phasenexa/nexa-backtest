# Task 06: Validation Pipeline

## Goal

Build the `nexa validate` CLI command that catches bugs, type errors,
unsupported exchange features, look-ahead bias, and unsafe resource
usage before the algo runs. Six checks, one command, clear output.

This is the feature that saves customers from wasting a 10-minute
IDC backtest on an algo that was never going to work. It also builds
trust: if the validator passes, the customer can be reasonably
confident the algo will run without surprises.

---

## What to build

### 1. `validation/runner.py` - Orchestrator

Runs all six validation steps in sequence and reports results:

```python
class ValidationRunner:
    """Orchestrates the validation pipeline."""

    def __init__(
        self,
        algo_path: str,
        exchange: str,
        strict: bool = False,  # Treat warnings as errors
    ) -> None: ...

    def run(self) -> ValidationResult: ...

@dataclass
class ValidationResult:
    steps: list[StepResult]

    @property
    def passed(self) -> bool:
        """True if no errors (warnings are ok unless strict)."""

    @property
    def error_count(self) -> int: ...

    @property
    def warning_count(self) -> int: ...

    def summary(self) -> str:
        """Human-readable summary for CLI output."""

@dataclass
class StepResult:
    name: str
    status: str          # "pass", "fail", "warn", "skip"
    messages: list[str]  # Individual findings
    duration_ms: int     # How long the step took
```

The runner should handle each step independently. If step 2 fails,
steps 3-6 still run (unless step 1 fails with syntax errors, in
which case steps 2-6 are skipped because the file can't be parsed).

### 2. `validation/ruff_check.py` - Step 1: Syntax and Style

Run ruff against the algo file. This catches syntax errors, unused
imports, undefined variables, and style issues.

```python
class RuffCheck:
    def run(self, algo_path: str) -> StepResult: ...
```

Implementation:
- Shell out to `ruff check {algo_path}` with `--output-format json`
- Parse the JSON output
- Syntax errors and undefined names are errors
- Unused imports and style issues are warnings
- If ruff is not installed, skip the step with a message suggesting
  `pip install ruff`

Use a Phase Nexa ruff config that's strict but not obnoxious. The
config should be bundled with the package (in the package data or
as a default dict) so the customer doesn't need their own ruff config:

```toml
[tool.ruff]
target-version = "py311"
line-length = 120

[tool.ruff.lint]
select = ["E", "F", "W", "I", "N", "UP", "B", "SIM"]
# E = pycodestyle errors
# F = pyflakes (includes syntax, undefined, unused)
# W = pycodestyle warnings
# I = isort
# N = naming conventions
# UP = pyupgrade
# B = bugbear
# SIM = simplify
```

### 3. `validation/mypy_check.py` - Step 2: Type Safety

Run mypy in strict mode against the algo file. This verifies the
algo satisfies the `TradingContext` protocol and uses types correctly.

```python
class MypyCheck:
    def run(self, algo_path: str) -> StepResult: ...
```

Implementation:
- Shell out to `mypy --strict {algo_path}` with `--no-error-summary`
  and `--output json` (or parse the text output)
- Type errors are errors
- Notes and hints are warnings
- If mypy is not installed, skip with a message

For mypy to understand `TradingContext`, the customer needs
`nexa-backtest` installed (which they will, since they're importing
from it). Mypy resolves the protocol from the installed package.

One subtlety: the customer's algo file might not have a
`py.typed` marker or might import from untyped third-party libraries.
Handle gracefully: report mypy findings from the algo file itself,
but don't fail on missing stubs from third-party deps. Use
`--ignore-missing-imports` for third-party modules only, not for
`nexa_backtest` imports.

### 4. `validation/interface_check.py` - Step 3: Interface Compliance

Custom AST analysis that checks whether the algo correctly implements
the required interface. This catches structural issues that ruff and
mypy might miss.

```python
class InterfaceCheck:
    def run(self, algo_path: str) -> StepResult: ...
```

**Checks for SimpleAlgo subclasses:**

- Does the class subclass `SimpleAlgo`?
- Does it implement at least one hook? (An algo with no hooks is
  useless, likely a mistake)
- Does `on_setup` call `subscribe_signal()` for any signals
  referenced by `ctx.get_signal()` elsewhere? (Common mistake: using
  a signal without subscribing)
- Do hook methods have the correct signature? (e.g., `on_fill` must
  accept `ctx` and `fill`)

**Checks for @algo decorated functions:**

- Is the function async?
- Does it accept exactly one argument?
- Does it contain `async for event in ctx.events()`? (An @algo that
  never consumes events is likely a mistake)

**Checks for both:**

- Are Order constructions valid? (e.g., `Order.buy()` must have
  `product`, `volume_mw`, `price_eur`)
- Are there multiple algo definitions in one file? (Ambiguous, flag
  as error)
- Does the file import from `nexa_backtest`? (If not, it's probably
  not an algo file)

Implementation: use Python's `ast` module to parse the file and walk
the tree. No execution of the customer's code. This is a static check.

### 5. `validation/feature_check.py` - Step 4: Exchange Feature Compatibility

Cross-reference the algo's order types and parameters against the
target exchange's `ExchangeCapabilities`.

```python
class FeatureCheck:
    def run(self, algo_path: str, exchange: str) -> StepResult: ...
```

**How it works:**

1. Parse the algo's AST
2. Find all `Order.buy()`, `Order.sell()`, `Order.market()`,
   `Order.block_bid()`, etc. call sites
3. Extract the parameters used (e.g., `exclusive_group=...`)
4. Load the `ExchangeCapabilities` for the target exchange
5. Check each order construction against the capabilities

**Examples of what this catches:**

- `Order.block_bid()` used, but target exchange doesn't support
  block bids in the requested market type
- `exclusive_group` parameter used, but target exchange doesn't
  support exclusive groups
- Volume below exchange minimum or above exchange maximum
- Price outside exchange price limits (if hardcoded in the algo)

**Limitations (document clearly):**

- Can only check statically visible values. If the algo computes
  volume dynamically (`volume_mw=calculate_volume()`), the check
  can't validate the result.
- Can only check calls that use keyword arguments. `Order.buy("X",
  10, 50.0)` is harder to validate than `Order.buy(product="X",
  volume_mw=10, price_eur=50.0)`.
- Flag these as "unable to validate" warnings, not errors.

### 6. `validation/lookahead_check.py` - Step 5: Look-Ahead Bias Detection

Heuristic static analysis that flags patterns likely to cause
look-ahead bias.

```python
class LookaheadCheck:
    def run(self, algo_path: str) -> StepResult: ...
```

**Patterns to detect:**

1. **Direct DataFrame indexing with future timestamps:**
   ```python
   # Flag: indexing with a timestamp that could be in the future
   df.loc[some_future_time]
   df.iloc[current_index + N]  # where N > 0
   ```

2. **Signal access without publication_offset consideration:**
   ```python
   # Flag: get_signal_history() with a large lookback
   ctx.get_signal_history("forecast", lookback=96)
   # Warn: lookback of 96 covers 24 hours. Verify the signal's
   # publication_offset supports this range.
   ```

3. **Pandas shift with negative values:**
   ```python
   # Flag: shift(-1) looks forward in time
   df["price"].shift(-1)
   ```

4. **Sorting by timestamp and accessing later rows:**
   ```python
   # Flag: sort_values + iloc that might access future data
   df.sort_values("timestamp").iloc[i+1]
   ```

**Important caveats (document in output):**

- This is heuristic, not comprehensive. It catches common patterns
  but a determined person can still introduce bias in ways that
  static analysis cannot detect.
- False positives are possible. The check flags suspicious patterns,
  not guaranteed violations.
- For this reason, all findings from this step are **warnings**, not
  errors. Even in `--strict` mode, these are warnings. The customer
  must assess whether the flagged pattern is actually problematic.

### 7. `validation/resource_check.py` - Step 6: Resource Safety

Flags operations that would behave differently in backtest vs
live mode, or that could cause the backtest to hang/fail.

```python
class ResourceCheck:
    def run(self, algo_path: str) -> StepResult: ...
```

**Patterns to detect:**

1. **`time.sleep()` or `asyncio.sleep()`** - pauses the real clock,
   not the simulated one. The backtest will actually wait. Flag as
   error. Suggest using `ctx.wait()` for simulated delays.

2. **File I/O in the hot path** - `open()`, `Path.read_text()`,
   `pd.read_csv()`, etc. inside hook methods or the event loop.
   These should happen in `on_setup`, not on every tick. Flag as
   warning.

3. **Network calls** - `requests.get()`, `urllib.request.urlopen()`,
   `httpx`, `aiohttp` usage. These won't work in backtest mode
   (no real network for simulated time). Flag as error.

4. **`datetime.now()` or `datetime.utcnow()`** - should use
   `ctx.now()` instead. Using wall-clock time in a backtest gives
   wrong results. Flag as error.

5. **Threading** - `threading.Thread`, `concurrent.futures` in the
   algo. Not safe with the simulated clock. Flag as warning.

6. **Global mutable state** - module-level mutable variables that
   could leak state between backtest runs. Flag as warning.

Implementation: AST analysis, checking for specific function calls
and imports.

### 8. `cli/validate.py` - CLI Command

Add `nexa validate` as a subcommand:

```bash
$ nexa validate my_algo.py --exchange nordpool

Validating my_algo.py against Nord Pool...

Step 1/6: Syntax & Style (ruff)        [PASS]
Step 2/6: Type Safety (mypy)           [PASS]  (2.3s)
Step 3/6: Interface Compliance         [PASS]
Step 4/6: Exchange Features            [PASS]
Step 5/6: Look-Ahead Bias             [WARN]
  Line 45: get_signal_history() with lookback=96 covers 24 hours.
  Verify the signal's publication_offset supports this range
  without leaking future data.
Step 6/6: Resource Safety              [PASS]

Result: PASSED (1 warning)
Your algo is ready to run against Nord Pool.
```

```bash
$ nexa validate my_algo.py --exchange epex_spot --strict

Validating my_algo.py against EPEX SPOT...

Step 1/6: Syntax & Style (ruff)        [PASS]
Step 2/6: Type Safety (mypy)           [FAIL]
  my_algo.py:23: error: Argument "volume_mw" to "buy" has
  incompatible type "float"; expected "Decimal"  [arg-type]
Step 3/6: Interface Compliance         [PASS]
Step 4/6: Exchange Features            [FAIL]
  Line 42: Order.block_bid() used, but EPEX SPOT continuous
  does not support block bids.
Step 5/6: Look-Ahead Bias             [WARN -> FAIL (strict)]
  Line 45: get_signal_history() with lookback=96...
Step 6/6: Resource Safety              [PASS]

Result: FAILED (2 errors, 1 warning treated as error)
```

Flags:
- `--exchange` (required): target exchange for feature compatibility
- `--strict`: treat warnings as errors
- `--skip`: comma-separated list of steps to skip (e.g.,
  `--skip ruff,mypy` if the customer has their own linting setup)
- `--json`: output results as JSON instead of human-readable text

Exit codes:
- 0: passed (no errors, warnings ok)
- 1: failed (errors found)
- 2: failed in strict mode (warnings treated as errors)

### 9. Integration with `nexa run`

Add a `--validate` flag to `nexa run` that runs the validation
pipeline before starting the backtest:

```bash
# Validate first, then run if it passes
nexa run my_algo.py --exchange nordpool --validate \
    --start 2026-03-01 --end 2026-03-31

# Validate, treat warnings as errors
nexa run my_algo.py --exchange nordpool --validate --strict \
    --start 2026-03-01 --end 2026-03-31
```

If validation fails, the backtest does not start. Print the
validation output and exit with a non-zero code.

---

## Tests

### Step tests (unit)

1. **RuffCheck**: pass a file with a syntax error, verify error.
   Pass a file with an unused import, verify warning. Pass a clean
   file, verify pass.

2. **MypyCheck**: pass a file with a type error (float instead of
   Decimal), verify error. Pass a file with correct types, verify
   pass. Pass a file that imports from an untyped library, verify
   it doesn't fail on the missing stubs.

3. **InterfaceCheck - SimpleAlgo**: pass a valid SimpleAlgo, verify
   pass. Pass a SimpleAlgo with no hooks, verify error. Pass a
   SimpleAlgo that calls `ctx.get_signal()` without subscribing,
   verify warning. Pass a file with two SimpleAlgo subclasses,
   verify error.

4. **InterfaceCheck - @algo**: pass a valid @algo function, verify
   pass. Pass a non-async function with @algo, verify error. Pass
   an @algo that never calls `ctx.events()`, verify warning.

5. **FeatureCheck**: pass an algo that uses `Order.block_bid()` and
   validate against Nord Pool DA (supports it) and EPEX continuous
   (doesn't). Verify pass for Nord Pool, fail for EPEX.

6. **LookaheadCheck**: pass an algo with `df.shift(-1)`, verify
   warning. Pass an algo with `get_signal_history(lookback=96)`,
   verify warning. Pass a clean algo, verify pass.

7. **ResourceCheck**: pass an algo with `time.sleep(1)`, verify
   error. Pass an algo with `datetime.now()`, verify error. Pass
   an algo with `open()` inside `on_bar`, verify warning. Pass an
   algo with `open()` inside `on_setup` only, verify pass (setup
   is fine for file I/O).

### Integration tests

8. **Full pipeline**: run all 6 steps against a known-good algo,
   verify all pass. Run against a deliberately broken algo (one
   issue per step), verify each step catches its issue.

9. **CLI validate**: use CliRunner to test `nexa validate` with
   a good algo and a bad algo. Verify exit codes. Test `--strict`
   flag. Test `--skip` flag. Test `--json` output.

10. **CLI run --validate**: verify that `nexa run --validate` blocks
    execution when validation fails, and proceeds when it passes.

11. **Step isolation**: break step 1 (syntax error). Verify steps
    2-6 are skipped (can't parse the file). Break step 2 (type
    error). Verify steps 3-6 still run.

---

## Test Fixture Algos

Create a set of small algo files specifically for validation testing:

```
tests/fixtures/validation/
    valid_simple_algo.py        # Clean, passes everything
    valid_async_algo.py         # Clean @algo, passes everything
    syntax_error.py             # Broken syntax
    type_error.py               # float instead of Decimal
    no_hooks.py                 # SimpleAlgo with no hooks implemented
    unsupported_feature.py      # Uses block_bid()
    lookahead_bias.py           # Uses df.shift(-1)
    resource_unsafe.py          # Uses time.sleep()
    datetime_now.py             # Uses datetime.now() instead of ctx.now()
    unsubscribed_signal.py      # Uses signal without subscribing
    multiple_algos.py           # Two SimpleAlgo subclasses
    file_io_in_setup.py         # open() in on_setup (ok)
    file_io_in_bar.py           # open() in on_bar (warning)
```

---

## What NOT to build

- Auto-fixing (ruff --fix or similar). Validate only, don't modify.
- Custom ruff rules for energy trading patterns
- Runtime validation (checking during backtest execution). This is
  all pre-run static analysis.
- ML model validation (that's task 07)
- Security scanning of compiled algo binaries
- Any changes to matching engines, data loading, or signals

---

## Acceptance criteria

1. `make ci` passes
2. `nexa validate` runs all 6 steps and produces clear output
3. Each step catches its target issues (verified by fixture algos)
4. Steps are independent: later steps run even if earlier ones fail
   (except step 1 syntax errors, which skip remaining steps)
5. `--strict` treats warnings as errors
6. `--skip` allows skipping specific steps
7. `--json` produces machine-readable output
8. `nexa run --validate` blocks on validation failure
9. Exit codes are correct (0 = pass, 1 = fail, 2 = strict fail)
10. All new public API has Google-style docstrings
