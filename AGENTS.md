# AGENTS.md

Repo-local instructions for AI coding agents working in `counter`.

`counter` is a Python research library for causal attribution over LLM-agent
decision traces. Its product taste is simple: make causal claims only when the
logged data, graph, support, and assumptions can actually support them. A useful
`unidentified` result is better than a confident fake counterfactual.

## Working Contract

- Start from repo truth: read `README.md`, `pyproject.toml`, nearby tests, and
  the module you are touching before inventing patterns.
- Prefer the smallest coherent change. This repo is intentionally sharp and
  narrow; do not turn it into a platform.
- Preserve causal honesty. Never paper over missing support, single-class
  outcomes, or replay-only interventions with optimistic estimates.
- Keep public APIs typed, explicit, and inspectable. Favor strict Pydantic
  schemas and structured result objects over strings that consumers must parse.
- Do not add backward-compatibility shims, dual paths, or deprecated interfaces
  unless a concrete caller is named.
- Do not add new runtime dependencies without asking first. The acceptance gate
  explicitly rejects broad causal/agent frameworks such as `dowhy`, `causalml`,
  `pyro`, `langchain`, `langgraph`, and `networkx`.

## Repo Map

- `src/counter/schema/` - strict trace schema. This is the producer/consumer
  contract for CounterBench and external adapters.
- `src/counter/dag/` - inspectable per-trace graph construction.
- `src/counter/outcome/` - transparent outcome modeling.
- `src/counter/intervene/` - intervention API, identifiability labels,
  sensitivity/bounds, and structured `next_step` guidance.
- `src/counter/attribute/` - failure attribution ranking.
- `src/counter/baselines.py` - descriptive baselines such as pass-rate tables.
- `bench/synthetic/` - deterministic SCM benchmark. Safe for CI and local work.
- `bench/real/coding_agent/` - real-agent trace harness. Can call external LLMs
  and spend money; treat it as gated infrastructure.
- `bench/real/runs_v1/` - committed real pilot corpus used by the demo story.
- `notebooks/demo.ipynb` and `docs/demo-excerpt.md` - naive-vs-honest demo
  surface. Keep these aligned when demo behavior changes.
- `tests/` - unit and acceptance coverage. Acceptance tests encode the v0
  product contract; read them before weakening behavior.

## Local Commands

Install:

```bash
uv pip install -e ".[dev]"
```

Main validation:

```bash
uv run ruff check .
uv run pytest
```

Makefile equivalents:

```bash
make lint
make test
make ci
```

Useful local demos:

```bash
uv run counter demo
uv run counter demo --runs-dir /tmp/missing --synthetic-n 500 --target sonnet
uv run counter bench synthetic --n 500 --seed 42 --output-dir /tmp/counter-syn
```

## Safety Rules For Real-Agent Benchmarks

- `uv run counter bench real ...` can make external LLM API calls and incur USD
  spend. Do not run it casually as a validation step.
- The harness is intentionally protected by `.counter/approved`. Do not create
  that marker for the user, do not bypass the first-run gate, and do not commit
  `.counter/` artifacts.
- If real traces must be generated, ask first and state the exact command,
  fixture set, output directory, and budget cap.
- Keep budget behavior conservative. `BudgetTracker` halts at 80% of the cap by
  design; do not relax that without an explicit requirement.
- Never commit secrets or provider credentials. Use environment variables loaded
  from the user's secret manager, not hardcoded values or `.env` files.
- Treat `bench/real/runs/`, `bench/synthetic/_out/`, checkpoints, and ad hoc
  generated corpora as local artifacts unless the task explicitly says to
  curate and commit a corpus.

## Causal And Statistical Invariants

- `CausalEstimate.identifiability` must be one of `identified`, `bounded`, or
  `unidentified`, and the rest of the object must make that label defensible.
- Single-class real corpora are not model-fit inputs. Surface the degenerate
  case as `unidentified` with a concrete `NextStep`.
- Prediction uncertainty and identifiability uncertainty are different. Do not
  blur bootstrap CIs, sensitivity bounds, support gaps, and replay requirements.
- If a query needs a prompt rewrite, hidden state change, or unavailable arm,
  return an honest replay/support next step rather than an estimated effect.
- `pass_rate_by_arm()` is descriptive, not causal. Keep that distinction visible
  in demos, docs, tests, and CLI output.
- Synthetic SCM tests should remain deterministic by seed and recover the known
  headline effect within the acceptance tolerance.

## Python Style

- Python 3.11+; use `from __future__ import annotations` in Python modules.
- Use `pathlib.Path` for filesystem work.
- Type public functions, dataclasses, and Pydantic models.
- Keep Pydantic models strict with `extra="forbid"` unless there is a concrete
  schema-evolution reason not to.
- Prefer explicit domain names: `decision_type`, `chosen_action`,
  `identifiability`, `outcome_delta`, `next_step`.
- Comments should explain the causal/statistical reason or operational guardrail,
  not narrate obvious code.
- Match nearby code before introducing helpers. Avoid single-use abstractions.
- Keep imports clean and sorted by Ruff.

## Testing Expectations

- For schema changes, add or update tests under `tests/unit/test_trace_schema.py`.
- For DAG behavior, update `tests/unit/test_dag.py`.
- For intervention semantics, update `tests/unit/test_causal_engine.py` and
  `tests/unit/test_next_step.py` as appropriate.
- For demo/CLI behavior, update `tests/unit/test_cli_demo.py`,
  `tests/acceptance/test_demo_executes.py`, and `docs/demo-excerpt.md` if output
  changes.
- For benchmark harness behavior, add tests around injected clients or fixture
  resolution; do not rely on live provider calls.
- After renames or public API changes, grep the whole repo for old names and
  update notebooks/docs/tests together.

## Documentation Rules

- Keep `README.md` aligned with the real CLI and API, not aspirational features.
- Do not claim `counter` supports Pearl L3 structural counterfactuals, DAG
  learning, provider replay guarantees, an observability UI, or calibrated
  universal success probabilities unless those are actually implemented.
- When changing demo behavior, update `docs/demo-excerpt.md` and rebuild or
  inspect `notebooks/demo.ipynb` when relevant.
- Write docs in the same voice as the README: concise, honest, a little sharp,
  and allergic to fake certainty.

## Git Hygiene

- Check `git status --short --branch` before and after edits.
- Stage only intended files. Do not use broad `git add .` when generated traces,
  notebooks, caches, or runtime artifacts may be present.
- Do not push, create PRs, or merge without explicit user approval in the thread.
- Do not commit to `main` without explicit permission.
- Never include AI attribution in commit messages.
