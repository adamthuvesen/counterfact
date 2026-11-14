UV = uv run --locked --extra dev --extra bench

.PHONY: install lint format typecheck imports check cov generated check-generated check-wheel check-container fix slow verify

install:
	uv sync --locked --extra dev --extra bench

lint:
	$(UV) ruff check . .agents/skills/verify-counterfact/verify
	$(UV) ruff format --check . .agents/skills/verify-counterfact/verify

fix:
	$(UV) ruff check --fix . .agents/skills/verify-counterfact/verify
	$(UV) ruff format . .agents/skills/verify-counterfact/verify

format:
	$(UV) ruff format . .agents/skills/verify-counterfact/verify

typecheck:
	$(UV) basedpyright
	$(UV) basedpyright --project .agents/skills/verify-counterfact

imports:
	$(UV) lint-imports

check: lint typecheck imports
	$(UV) pytest -n auto --dist worksteal -m "not container and not slow"

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

VERIFY_APP := .agents/skills/verify-counterfact/verify

# Drive the real CLI: no ARGS = drives for changed files; ARGS=all|list|control|<drive>.
verify:
	$(VERIFY_APP) $(ARGS)

slow: lint typecheck imports cov check-generated check-wheel check-container
	$(VERIFY_APP) all
	$(VERIFY_APP) control
