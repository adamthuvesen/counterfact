# counterfact

> Causal attribution for LLM-agent traces, with the rare good manners to say when the data cannot support the counterfactual.

`counterfact` is a small Python research library for asking which agent decisions plausibly changed an outcome. It ingests typed decision traces, builds an inspectable per-trace DAG, fits a transparent outcome model, and answers intervention-style questions with an explicit identifiability label:

- `identified` - estimable under the declared graph, support, and assumptions
- `bounded` - not point-identified, but sensitivity bounds are available
- `unidentified` - unsupported by the traces without stronger assumptions or replay

The point is not to turn trace inspection into a confident-looking probability. The point is to keep the causal claim honest.

## Quickstart

```bash
uv pip install -e ".[dev]"
uv run pytest
uv run counterfact demo
```

The demo command is local-only. It uses the committed `bench/real/runs_v1/` corpus when present and falls back to synthetic SCM traces when it is not. It does not call the real-agent LLM harness or require provider credentials.

Example output:

```text
counterfact demo: naive vs honest
data: bench/real/runs_v1
outcomes: 30 pass / 0 fail

pass_rate_by_arm(model_call)
arm              n  pass  rate    95% CI
large           28    28 1.000  [0.879, 1.000]
small            2     2 1.000  [0.342, 1.000]

intervene(model_call -> large)
identifiability: unidentified
reason: real corpus is causally degenerate: every trace has Outcome.value=True; no outcome variation exists for an outcome model or back-door adjustment to leverage
next_step: broaden_arm_support - Collect or construct traces with both pass and fail outcomes before estimating decision-level effects on the real corpus.
```

See [docs/demo-excerpt.md](docs/demo-excerpt.md) for the rendered notebook-style excerpt. For how the `csv_dedupe` corpus in `bench/real/runs_v1/` was piloted, see [docs/pilot-csv-dedupe.md](docs/pilot-csv-dedupe.md).

## Why This Matters

Most agent-debugging tools can show you the trace. Some can score the final outcome. Very few can say, with discipline, whether a decision-level causal question is actually identifiable from the data you logged.

That matters because LLM-agent traces invite fake counterfactuals:

> If the agent had used a different model, retried once, or called another tool, would the task have passed?

Sometimes `counterfact` can estimate that. Sometimes it can only bound it. Sometimes the right answer is: no, this corpus does not contain the variation needed to make that claim. That refusal is the product taste.

## What Ships in v0

- Native Pydantic trace schema: `Run`, `Step`, `Decision`, `Observation`, `Outcome`, strict JSON validation
- Decision taxonomy: `plan_step`, `model_call`, `tool_call`, `memory_read`, `retry`, `termination`
- Hand-built per-trace DAG builder
- Logistic-regression outcome model with bootstrap uncertainty
- `intervene()` returning `identified | bounded | unidentified`, assumptions, warnings, bounds, outcome deltas, and structured `next_step`
- E-value sensitivity bounds
- `attribute_failure()` decision ranking
- `pass_rate_by_arm()` naive baseline table
- `power_analysis()` rough sample-size guidance
- Synthetic SCM benchmark with a known treatment effect
- Real-agent coding harness with randomized decisions, budget gate, hidden/public fixture support, and a committed 30-trace `runs_v1` pilot corpus
- Demo notebook and acceptance tests around the naive-vs-honest story

### NextStep payload contract

Refusals are first-class. Every `CausalEstimate.next_step` carries a structured payload alongside `human_text`:

- `broaden_arm_support` → `arm_name`, `missing_strata`, `observed_arms`, `missing_arms`
- `increase_n` → `current_n`, `estimated_required_n`, `target_ci_width`, `power_method` (`binomial_wald_two_arm` | `inline_scaling` | `degenerate`), `arm_breakdown`
- `replay_required` → `intervention_target`, `replay_inputs_required`, `note`
- `add_arm_randomization` → `arm_name`, `current_policy`
- `none` → empty

When the harness can generate the missing data, `payload["suggested_command"]` carries a copy-pasteable `uv run counterfact bench …` invocation. The full schema lives in `openspec/specs/causal-engine/spec.md`.

## Python API

```python
from bench.synthetic import generate_traces
from counterfact import build_dag, fit_outcome_model, intervene, pass_rate_by_arm
from counterfact.schema import Run

runs = [Run.model_validate(trace) for trace in generate_traces(n=500, seed=42)]

print(pass_rate_by_arm(runs, "model_call"))

model = fit_outcome_model(runs, n_bootstrap=200, seed=42)
estimate = intervene(
    dag=build_dag(runs[0]),
    model=model,
    step=2,
    intervention={"model_choice": "sonnet"},
)

print(estimate.identifiability)
print(estimate.assumptions)
print(estimate.next_step)
```

The committed real corpus currently has one outcome class, so the CLI demo intentionally surfaces the degenerate case instead of fitting logistic regression. Use the synthetic SCM, as above, or a mixed-outcome real corpus for identified estimates.

## CLI

```bash
# Synthetic SCM traces, deterministic and no LLM calls
uv run counterfact bench synthetic --n 500 --seed 42 --output-dir /tmp/counterfact-syn

# Local showcase demo
uv run counterfact demo

# Force synthetic fallback for the demo
uv run counterfact demo --runs-dir /tmp/missing --synthetic-n 500 --target sonnet
```

The real-agent harness is available behind an explicit budget and first-run approval gate:

```bash
uv run counterfact bench real --n 30 --budget-cap 50 --fixture-set hidden_v1
```

That command can call external LLM APIs. The showcase demo and CI never run it.

## Validation

```bash
uv run ruff check .
uv run pytest
```

CI and a clean checkout should pass all non-skipped tests (`ruff check` + full `pytest`). A small number of tests skip unless optional human labels are present.

## What v0 Does Not Claim

- No Pearl L3 structural counterfactual for arbitrary prompt changes
- No DAG learning
- No multi-agent or token-level graph
- No calibrated universal `P(success)` story
- No hidden "LLM quality" latent node
- No observability UI
- No provider-specific replay guarantee

`counterfact` is deliberately a research artifact, not a polished platform. Its useful edge is narrower and sharper: causal attribution for logged agent decisions, with identifiability visible in the API.
