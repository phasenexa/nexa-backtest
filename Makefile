.PHONY: install lint typecheck test ci test-notebooks execute-notebooks clear-notice

install:
	poetry install

lint:
	poetry run ruff check src tests
	poetry run ruff format --check src tests

typecheck:
	poetry run mypy src

test:
	poetry run pytest --cov=src/nexa_backtest --cov-report=term-missing

ci: lint typecheck test test-notebooks

test-notebooks:
	poetry run jupyter nbconvert --to notebook --execute notebooks/*.ipynb --output-dir /tmp/

execute-notebooks:
	poetry run jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb

clear-notice:
	rm -f ~/.config/nexa/notice_shown 2>/dev/null
