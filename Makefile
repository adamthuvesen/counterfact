.PHONY: install lint format typecheck test cov ci

install:
	uv pip install -e ".[dev]"

lint:
	ruff check src tests
	ruff format --check src tests

format:
	ruff format src tests

typecheck:
	mypy src/counterfact

test:
	pytest

cov:
	pytest --cov --cov-report=term-missing

ci: lint typecheck cov
