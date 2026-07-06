# AGENTS.md — counterfact

`counterfact` is a Python research library for understanding agent decision traces through counterfactual questions: if the agent had called a different model, used a different tool, retried, or stopped later, would it have been more likely to complete the task? Its taste is simple: make causal claims only when the logged data, graph, support, and assumptions can actually support them. A useful `unidentified` result is better than a confident fake counterfactual.

User-level guidance (tone, principles, git etiquette, Python defaults) lives in `~/.claude/CLAUDE.md` and `~/dotfiles/agents/AGENTS.md` and is *not* duplicated here. This file is for project-specific facts.

## Layout

```
src/counterfact/
├── schema/      strict trace schema — producer/consumer contract for CounterBench and adapters
├── dag/         inspectable per-trace graph construction
├── outcome/     transparent outcome modeling
├── intervene/   intervention API, identifiability labels, sensitivity/bounds, next_step guidance
├── attribute/   failure attribution ranking
├── explain/     per-trace narrative: ExplainReport model + stdlib HTML renderer (`counterfact explain`)
├── adapters/    SDK adapters and live tracers
└── baselines.py descriptive baselines such as pass-rate tables

bench/synthetic/  deterministic SCM benchmark — safe for CI and local work
bench/real/       real-agent harness (gated, can spend money) + committed demo/anchor corpora
notebooks/        demo.ipynb (keep aligned with docs/demo-excerpt.md)
docs/             deeper subsystem docs — see Index
```

## Quickstart

```bash
uv pip install -e ".[dev]"   # install
uv run ruff check .          # lint
uv run ruff format --check src tests
uv run mypy src/counterfact  # types
uv run pytest                # tests (CI gates at --cov-fail-under=80)
make ci                      # lint + test in one shot
```

Local demos:

```bash
uv run counterfact demo --confound --synthetic-n 1000 --seed 42  # canonical showcase
uv run counterfact demo                                           # real-trace smoke test
uv run counterfact bench synthetic --n 500 --seed 42 --output-dir /tmp/counterfact-syn
```

## Critical Conventions

- **Causal honesty is the product.** Never paper over missing support, single-class outcomes, or replay-only interventions with optimistic estimates. The full guarantees live in [docs/invariants.md](docs/invariants.md) — read it before touching `intervene/`, `outcome/`, or `explain/`.
- **Smallest coherent change.** This repo is intentionally sharp and narrow; do not turn it into a platform. No backward-compat shims, dual paths, or deprecated interfaces unless a concrete caller is named.
- **Public APIs stay typed and inspectable.** Favor strict Pydantic schemas (`extra="forbid"`) and structured result objects over strings consumers must parse. Use explicit domain names: `decision_type`, `chosen_action`, `identifiability`, `outcome_delta`, `next_step`.
- **No new runtime dependencies without asking.** The acceptance gate rejects broad causal/agent frameworks — `dowhy`, `causalml`, `pyro`, `langchain`, `langgraph`, `networkx`.
- **`pass_rate_by_arm()` is descriptive, not causal.** Keep that distinction visible in demos, docs, tests, and CLI output.
- **Never commit secrets, `.env`, provider credentials, AI-attribution lines, or `.counterfact/` artifacts.**

## Safety Rules For Real-Agent Benchmarks

- `uv run counterfact bench real ...` can make external LLM API calls and incur USD spend. Do not run it casually as a validation step.
- The harness is protected by `.counterfact/approved`. Do not create that marker for the user, bypass the first-run gate, or commit `.counterfact/` artifacts.
- If real traces must be generated, ask first and state the exact command, fixture set, output directory, and budget cap.
- Keep budget behavior conservative. `BudgetTracker` halts at 80% of the cap by design; do not relax that without an explicit requirement.
- Treat `bench/real/pilot/`, `bench/real/pilot_*`, `bench/synthetic/_out/`, checkpoints, and ad hoc generated corpora as local artifacts unless the task explicitly says to curate and commit a corpus.

## Read The Docs First

Before editing a subsystem, read the matching `docs/*.md`:

- **Task routing, sync points, validation ladder** → [repo-context.md](docs/repo-context.md)
- **Causal/statistical invariants** → [invariants.md](docs/invariants.md)
- **System architecture** → [architecture.md](docs/architecture.md)
- **Naive-vs-honest demo surface** → [demo-excerpt.md](docs/demo-excerpt.md) (keep aligned with `notebooks/demo.ipynb`)

[repo-context.md](docs/repo-context.md) owns which tests and corpus checks to run for a given change; pick the narrowest meaningful set there before falling back to full `uv run pytest`. Keep `README.md`, docs, and `notebooks/demo.ipynb` aligned with the real CLI and API, not aspirational features. Do not claim Pearl L3 structural counterfactuals, DAG learning, provider replay guarantees, an observability UI, or calibrated universal success probabilities unless they are actually implemented. If a doc disagrees with code, fix the doc in the same change.

## Index

Start in [repo-context.md](docs/repo-context.md) for the task-routing map, then follow the subsystem docs above.
