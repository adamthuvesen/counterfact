.PHONY: install lint test

install:
	uv pip install -e ".[dev]"

lint:
	ruff check src tests
	ruff format --check src tests

format:
	ruff format src tests

test:
	pytest

ci: lint test
