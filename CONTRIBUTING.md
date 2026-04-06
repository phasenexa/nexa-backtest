# Contributing to nexa-backtest

Thanks for your interest in contributing. This document covers how to get set up,
the conventions we follow, and the process for getting changes merged.

## Getting started

### Prerequisites

- Python 3.11+
- Make (for common tasks)
- A GitHub account

### Setup

```bash
# Clone the repo
git clone https://github.com/phasenexa/nexa-backtest.git
cd nexa-backtest

# Create a virtual environment and install dev dependencies
python -m venv .venv
source .venv/bin/activate
make install

# Verify everything works
make ci
```

### Project structure

```
src/nexa_backtest/    # library source
tests/                # test suite
tests/fixtures/       # sample Parquet data files
examples/             # Jupyter notebooks
```

See `CLAUDE.md` for a full breakdown of the code layout and domain context.

## Development workflow

We use **trunk-based development**. The `main` branch is protected and all changes
go through pull requests.

### 1. Create a feature branch

```bash
git checkout main && git pull
git checkout -b feat/your-feature-name
```

Branch naming conventions:

| Prefix       | Use for                        |
|--------------|--------------------------------|
| `feat/`      | New features                   |
| `fix/`       | Bug fixes                      |
| `refactor/`  | Refactoring (no new behaviour) |
| `docs/`      | Documentation updates          |
| `test/`      | Test improvements              |
| `chore/`     | Maintenance (deps, config)     |

### 2. Make your changes

Write code, write tests. See the code style section below. Commit as you go
with focused, atomic commits.

### 3. Run the checks

Before opening a PR, run the full check suite locally:

```bash
# Everything in one command
make ci

# Or individually:
make lint       # ruff check + format check
make typecheck  # mypy strict
make test       # pytest with coverage
```

### 4. Open a pull request

```bash
git push -u origin feat/your-feature-name
gh pr create --title "feat: short description" --body "Why this change is needed."
```

PR requirements:

- Clear title describing the change
- Description explaining the motivation
- All CI checks pass
- Code coverage meets or exceeds 80%
- At least 1 approving review (when branch protection is enabled)

### 5. After merge

Delete your feature branch. CI handles the rest.

## Code style

### Python conventions

- **Python 3.11+** with type hints on all public API
- **Pydantic v2** for data models
- **Ruff** for linting and formatting (no black, no isort, ruff handles both)
- **mypy** in strict mode for type checking
- **Google-style docstrings** on all public functions and classes
- Prefer functions over classes where a function will do
- UK English in documentation and comments (unless using established energy
  trading terminology)

### Data handling

- **`decimal.Decimal`** for all prices and volumes in the public API. Convert
  to `float` only at the serialisation boundary (e.g. writing Parquet).
- **Timezone-aware datetimes only**. Never naive. Use `zoneinfo.ZoneInfo`.
- **pandas DataFrames** for tabular output (optional dependency).

### Testing

- **pytest** with descriptive test names
- Use sample Parquet fixtures in `tests/fixtures/` for data loading tests
- Aim for >80% coverage, but prioritise meaningful tests over chasing the number
- Model round-trip tests: build types, serialise, deserialise, compare
- Matching engine tests: known scenarios with expected fill outcomes

### Dependencies

Keep them minimal. Every dependency is a maintenance burden. If the standard library
can do it, use the standard library.

## Domain notes for contributors

If you are new to energy trading:

- **MTU** = Market Time Unit. 15 minutes since Sept 2025.
- **DA** = Day-Ahead auction. Bids submitted before gate closure, cleared at a
  single price per zone per MTU.
- **IDC** = Intraday Continuous. Limit order book, price-time priority, trading
  up to minutes before delivery.
- **Gate closure** = deadline after which no more orders can be submitted for a
  given product.
- **VWAP** = Volume-Weighted Average Price. The natural benchmark for execution
  quality in energy markets.
- **Clearing price** = the single price at which a DA auction settles. All accepted
  bids transact at this price.

The `CLAUDE.md` file has more detailed domain context.
