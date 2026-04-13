# Task 09: Algo Compilation and IP Protection

## Goal

Let customers compile their Python algo to a native binary so they
can use the hosted backtest service without sharing source code. The
compiled binary imports and runs exactly like the original Python
file. The engine doesn't know or care whether it's loading source or
a compiled module.

After this task, a customer can: write an algo, compile it with
`nexa compile`, verify the compiled version produces identical results,
and upload the binary knowing their IP is protected.

---

## Context: Why This Matters

A profitable trading algo is worth real money. If a customer has to
upload their source code to a hosted platform, they won't do it. Even
with contractual protections, the risk is too high. Compilation solves
this: the customer compiles locally, uploads the binary, and we run it
without ever having access to the source.

This is specifically for the hosted backtest service (stage 4). Self-hosted
customers don't need compilation because the code never leaves their
machine. But compilation is a prerequisite for the hosted service to be
commercially viable.

---

## What to build

### 1. `compile/nuitka_compiler.py` - Nuitka Compilation (Primary)

Nuitka compiles Python to C, then to a native shared library. The
output is a `.so` (Linux) or `.pyd` (Windows) that Python's import
system loads transparently.

```python
class NuitkaCompiler:
    """Compile a Python algo to a native shared library using Nuitka.

    The compiled module exports the same classes/functions as the
    source file. The BacktestEngine imports and uses it identically.

    Args:
        algo_path: Path to the .py source file.
        output_dir: Directory for the compiled output.
        python_version: Target Python version (default: current).
        include_packages: Additional packages to include in the
            compilation (e.g., numpy, pydantic). nexa_backtest
            is always included.
    """

    def compile(
        self,
        algo_path: str | Path,
        output_dir: str | Path | None = None,
        include_packages: list[str] | None = None,
    ) -> CompilationResult: ...

@dataclass(frozen=True)
class CompilationResult:
    success: bool
    output_path: Path | None     # Path to compiled .so/.pyd
    source_path: Path            # Original .py path
    compiler: str                # "nuitka" or "cython"
    compile_time_seconds: float
    output_size_bytes: int | None
    errors: list[str]
    warnings: list[str]
```

Implementation:

- Shell out to `nuitka --module {algo_path}` with appropriate flags
- Key Nuitka flags:

  ```text
  --module                        # Compile as importable module
  --include-package=nexa_backtest # Include framework types
  --python-flag=no_site           # Minimal Python overhead
  --remove-output                 # Clean up build artifacts
  --output-dir={output_dir}       # Where to put the .so
  ```

- If `output_dir` is not specified, put the output next to the source
  file
- The output filename follows Python's convention:
  `my_algo.cpython-311-x86_64-linux-gnu.so`
- Verify Nuitka is installed before attempting compilation. If not,
  raise `CompilerNotFoundError` with install instructions:
  `pip install nuitka`

### 2. `compile/cython_compiler.py` - Cython Compilation (Secondary)

Cython transpiles Python to C, then compiles to a shared library.
Faster compilation than Nuitka but less thorough protection.

```python
class CythonCompiler:
    """Compile a Python algo using Cython.

    Transpiles .py to .c via Cython, then compiles to .so with gcc.
    Faster than Nuitka but less thorough obfuscation. Some dynamic
    Python patterns may not compile cleanly.

    Args:
        algo_path: Path to the .py source file.
        output_dir: Directory for the compiled output.
    """

    def compile(
        self,
        algo_path: str | Path,
        output_dir: str | Path | None = None,
    ) -> CompilationResult: ...
```

Implementation:

- Write a temporary `setup.py` that uses `Cython.Build.cythonize`
- Shell out to `python setup.py build_ext --inplace`
- Clean up the temporary setup.py and build artifacts
- The output is a `.so` file, same convention as Nuitka
- Verify Cython is installed. If not, raise `CompilerNotFoundError`

### 3. `compile/base.py` - Compiler Protocol

```python
class AlgoCompiler(Protocol):
    """Protocol for algo compilers."""

    def compile(
        self,
        algo_path: str | Path,
        output_dir: str | Path | None = None,
    ) -> CompilationResult: ...

def get_compiler(name: str) -> AlgoCompiler:
    """Get a compiler by name.

    Args:
        name: "nuitka" or "cython"

    Returns:
        The compiler instance.

    Raises:
        CompilerNotFoundError: if the compiler tool is not installed.
    """
```

### 4. `compile/loader.py` - Compiled Module Loader

Load a compiled `.so`/`.pyd` file and extract the algo from it.
This is what the engine uses when given a compiled binary instead
of a source file.

```python
class CompiledAlgoLoader:
    """Load an algo from a compiled shared library.

    Uses importlib to load the .so/.pyd as a Python module, then
    finds the SimpleAlgo subclass or @algo function, same as the
    source loader does with .py files.
    """

    def load(self, path: str | Path) -> SimpleAlgo | AsyncAlgoFunction:
        """Load and return the algo from a compiled module.

        The module is imported using importlib.util.spec_from_file_location.
        Then the same discovery logic from the CLI runs: find a
        SimpleAlgo subclass or @algo-decorated function.

        Raises:
            AlgoLoadError: if no algo is found in the module.
            AlgoLoadError: if multiple algos are found.
        """
```

The loader should work transparently with the existing engine.
The engine doesn't need changes - it already accepts algo instances.
The loader is used by the CLI to turn a file path into an algo
instance.

Update the CLI's algo loading logic to handle both `.py` and
`.so`/`.pyd` files:

```python
# In cli/run.py - algo discovery:
def load_algo(path: str) -> SimpleAlgo | AsyncAlgoFunction:
    if path.endswith(".py"):
        return load_source_algo(path)      # Existing logic
    elif path.endswith(".so") or path.endswith(".pyd"):
        return CompiledAlgoLoader().load(path)
    else:
        raise AlgoLoadError(f"Unsupported file type: {path}")
```

### 5. `cli/compile.py` - CLI Command

```bash
# Compile with Nuitka (default)
$ nexa compile my_algo.py

Compiling my_algo.py with Nuitka...
Output: my_algo.cpython-311-x86_64-linux-gnu.so (284 KB)
Compile time: 12.3s

# Compile with Cython
$ nexa compile my_algo.py --compiler cython

# Specify output directory
$ nexa compile my_algo.py --output-dir ./compiled/

# Validate before compiling
$ nexa compile my_algo.py --validate --exchange nordpool
```

Flags:

- `--compiler`: "nuitka" (default) or "cython"
- `--output-dir`: where to put the compiled binary
- `--validate`: run `nexa validate` before compiling. If validation
  fails, don't compile.
- `--exchange`: required if `--validate` is used

The `nexa run` command should also accept compiled files:

```bash
# Run a compiled algo (works identically to source)
nexa run my_algo.so --exchange nordpool --start 2026-03-01 --end 2026-03-31
```

### 6. Verification Command

Add `nexa compile --verify` that compiles the algo, runs the same
backtest with both source and compiled versions, and confirms
identical results:

```bash
$ nexa compile my_algo.py --verify \
    --exchange nordpool \
    --start 2026-03-01 \
    --end 2026-03-31 \
    --products NO1_DA \
    --data-dir ./data

Compiling my_algo.py with Nuitka...
Output: my_algo.cpython-311-x86_64-linux-gnu.so (284 KB)

Verifying compiled algo produces identical results...
  Source PnL:    +12,340.50 EUR
  Compiled PnL:  +12,340.50 EUR
  Fills match:   186/186
  MATCH: Compiled algo produces identical results.
```

This is the trust builder. If the customer sees that the compiled
version produces exactly the same output, they know compilation
didn't break anything.

Implementation:

1. Compile the algo
2. Run BacktestEngine with the source algo
3. Run BacktestEngine with the compiled algo
4. Compare: total_pnl, trade count, each fill (price, volume,
   timestamp)
5. Report match or mismatch

### 7. Jupyter Notebook

Create `notebooks/compilation_demo.ipynb`:

```markdown
# Algo Compilation Demo

This notebook demonstrates the full compilation workflow:
1. Write a trading algo
2. Backtest it (source)
3. Compile it to a native binary
4. Backtest it (compiled)
5. Verify identical results
6. Inspect what the compiled output looks like

## Why compile?
- Protect your IP when using hosted backtest services
- The compiled binary runs identically to source
- No source code is included in the output
```

The notebook should walk through:

1. **Define a SimpleAlgo** inline in the notebook
2. **Save it to a .py file** using `%%writefile`
3. **Run a backtest with the source file** and show the summary
4. **Compile it** using `NuitkaCompiler` (Python API, not CLI)
5. **Inspect the output**: show the .so file size, show that
   `strings my_algo.so | grep "def on_"` returns nothing (source
   is not in the binary)
6. **Run the same backtest with the compiled file** and show the
   summary
7. **Compare results** side by side, proving they're identical
8. **Repeat with an @algo function** to prove both API levels work

Also create `notebooks/compilation_demo_at_algo.ipynb` or include
the @algo example as a second section in the same notebook.

### 8. Example Scripts

Create `examples/compile_and_run.py`:

```python
"""
Example: compile an algo and verify identical backtest results.

Usage:
    python examples/compile_and_run.py
"""
from pathlib import Path
from nexa_backtest.compile import NuitkaCompiler
from nexa_backtest import BacktestEngine

# 1. Compile
compiler = NuitkaCompiler()
result = compiler.compile("examples/simple_da_algo.py", output_dir="./compiled")
print(f"Compiled to: {result.output_path}")

# 2. Load both versions
from examples.simple_da_algo import MyAlgo as SourceAlgo
from nexa_backtest.compile.loader import CompiledAlgoLoader
compiled_algo = CompiledAlgoLoader().load(result.output_path)

# 3. Run both
source_result = BacktestEngine(
    algo=SourceAlgo(), exchange="nordpool", ...
).run()
compiled_result = BacktestEngine(
    algo=compiled_algo, exchange="nordpool", ...
).run()

# 4. Compare
assert source_result.total_pnl == compiled_result.total_pnl
print("Results match!")
```

---

## Edge Cases and Gotchas

**Algo imports third-party libraries:**
If the algo imports numpy, pandas, sklearn, etc., those packages must
be available at runtime on the machine that runs the compiled binary.
Nuitka's `--include-package` bundles them into the binary, but this
increases file size significantly. By default, don't bundle third-party
packages. Document that the runtime environment must have the same
packages installed.

**Platform mismatch:**
A `.so` compiled on Linux x86_64 won't run on macOS arm64. The
compiled binary is platform-specific. Document this clearly. For the
hosted service, specify the target platform (Linux x86_64) and tell
customers to compile on a matching platform or use a Docker build
environment.

**Python version mismatch:**
The compiled binary targets a specific CPython version (encoded in the
filename). Running it on a different version will fail to import.
Document this. The hosted service must pin a Python version.

**`@algo` with closures:**
Async functions with closures (capturing variables from outer scope)
should compile fine with Nuitka but may fail with Cython. Test this
explicitly. If Cython can't handle it, document it as a Nuitka-only
pattern.

**Dynamic imports in the algo:**
If the algo does `importlib.import_module("something")` at runtime,
the compiled version may not include that module. Flag this in the
resource safety check (task 06) as a warning.

---

## Tests

1. **NuitkaCompiler - SimpleAlgo**: compile a simple algo, verify
   the .so file exists and has non-zero size. Verify
   CompilationResult fields are populated.

2. **NuitkaCompiler - @algo**: compile an @algo function, verify
   compilation succeeds.

3. **CythonCompiler - SimpleAlgo**: same as test 1 but with Cython.

4. **CompiledAlgoLoader - SimpleAlgo**: compile, then load via
   CompiledAlgoLoader. Verify the returned object is a SimpleAlgo
   subclass with the expected hooks.

5. **CompiledAlgoLoader - @algo**: compile, then load. Verify the
   returned object is an async function with the @algo metadata.

6. **Identical results - DA**: run the same backtest with source and
   compiled versions of a DA algo. Verify total_pnl, trade count,
   and every fill matches exactly.

7. **Identical results - IDC**: same test with an IDC algo against
   IDC fixture data.

8. **Identical results - with signals**: same test with an algo that
   uses signals.

9. **Identical results - with ML model**: same test with an algo that
   calls ctx.predict().

10. **CLI nexa compile**: use CliRunner. Verify .so is created.
    Test --compiler flag. Test --output-dir flag.

11. **CLI nexa compile --verify**: verify the full compile-and-compare
    workflow runs and reports match.

12. **CLI nexa run with .so**: verify `nexa run my_algo.so` works.

13. **Compiler not found**: attempt compilation when Nuitka is not
    installed. Verify CompilerNotFoundError with install instructions.

14. **Source not recoverable**: compile an algo, read the .so file
    as bytes, verify the source code strings (function names,
    comments, docstrings) are not present in the binary. Use
    `strings` or byte search.

15. **CLI nexa compile --validate**: verify validation runs before
    compilation. If validation fails, verify compilation is skipped.

### Test environment note

Nuitka and Cython compilation requires a C compiler (gcc/clang).
Tests that invoke compilation should be marked with
`@pytest.mark.compilation` so they can be skipped in CI environments
that don't have a C compiler. Include a CI job that installs gcc and
runs these tests separately.

```python
@pytest.mark.compilation
def test_nuitka_simple_algo():
    ...
```

Add to `pyproject.toml`:

```toml
[tool.pytest.ini_options]
markers = [
    "compilation: tests that require C compiler and Nuitka/Cython",
]
```

---

## Dependencies

Add to optional extras in `pyproject.toml`:

```toml
[project.optional-dependencies]
compile = [
    "nuitka>=2.0",
]
compile-cython = [
    "cython>=3.0",
]
```

Nuitka and Cython are optional. The core library doesn't need them.
Only customers who want to compile their algos install them.

---

## What NOT to build

- Container isolation (Docker + gRPC). That's a separate task with
  its own protocol definition and orchestration.
- Cross-compilation (compile on macOS for Linux). Use Docker for this.
- Obfuscation beyond compilation (symbol stripping, string encryption).
  Nuitka's default output is sufficient.
- Hosted compilation service (compile in the cloud). Customers compile
  locally.
- Binary signing or verification (tamper detection)
- Compiled algo marketplace or sharing

---

## Acceptance criteria

1. `make ci` passes (compilation tests may need a separate CI job)
2. `nexa compile` produces a .so/.pyd from a .py algo using Nuitka
3. `nexa compile --compiler cython` works as an alternative
4. `nexa run my_algo.so` runs the compiled algo identically to source
5. `nexa compile --verify` proves identical results automatically
6. Both SimpleAlgo and @algo compile and run correctly
7. The Jupyter notebook demonstrates the full workflow and proves
   source code is not in the binary
8. CompiledAlgoLoader handles both API levels transparently
9. Source code is not recoverable from the compiled binary
10. All new public API has Google-style docstrings
