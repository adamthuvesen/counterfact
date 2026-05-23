.PHONY: install lint format typecheck test cov ci

install:
	uv pip install -e ".[dev]"

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff format .

typecheck:
	uv run mypy src/counterfact

test:
	uv run pytest

cov:
	uv run pytest --cov --cov-report=term-missing --cov-fail-under=80

ci: lint typecheck cov
