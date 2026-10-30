UV = uv run --locked --extra dev --extra bench

.PHONY: install lint format typecheck imports check cov generated check-generated check-wheel check-container verify

install:
	uv sync --locked --extra dev --extra bench

lint:
	$(UV) ruff check .
	$(UV) ruff format --check .

format:
	$(UV) ruff format .

typecheck:
	$(UV) mypy

imports:
	$(UV) lint-imports

check: lint typecheck imports
	$(UV) pytest -n auto --dist worksteal -m "not container and not slow" --ignore=tests/acceptance

cov:
	$(UV) pytest -n auto --durations=20 --cov --cov-config=$(CURDIR)/pyproject.toml --cov-report=xml --cov-report=term-missing --cov-fail-under=80

generated:
	$(UV) python scripts/build_demo_notebook.py
	$(UV) python scripts/build_benchmark_docs.py
	$(UV) python scripts/build_figures.py
	$(UV) python scripts/build_routed_results.py

check-generated:
	$(UV) python scripts/build_demo_notebook.py --check
	$(UV) python scripts/build_benchmark_docs.py --check
	$(UV) python scripts/build_figures.py --check
	$(UV) python scripts/build_routed_results.py --check

check-wheel:
	$(UV) python scripts/check_wheel.py

check-container:
	$(UV) python scripts/check_container.py --build

verify: check cov check-generated check-wheel check-container
