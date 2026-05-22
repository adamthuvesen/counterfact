.PHONY: install lint format typecheck test cov ci

install:
	uv pip install -e ".[dev]"

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff format .

typecheck:
	mypy src/counterfact

test:
	uv run pytest

cov:
	pytest --cov --cov-report=term-missing

ci: lint typecheck cov
