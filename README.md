# counterfact

`counterfact` is a small research library for understanding agent decision traces through counterfactual questions: if the agent had called a different model, used a different tool, retried, or stopped later, would it have been more likely to complete the task?

The answer is always labelled:

- `identified` - the logged data supports a point estimate under the graph, support, and assumptions.
- `bounded` - the data does not support a point estimate, but sensitivity bounds are available.
- `unidentified` - the corpus does not support the counterfactual without more data, stronger assumptions, or replay.

## Install

```bash
uv pip install -e ".[dev]"
```

Requires Python 3.11+.

## First Run

Run the local demo:

```bash
uv run counterfact demo --confound --synthetic-n 1000 --seed 42
```

This uses a deterministic synthetic corpus. It does not call external LLMs.

You should see three things:

1. A descriptive pass-rate table.
2. A causal intervention estimate.
3. A contrast between the naive table and the adjusted causal estimate.

Example shape:

```text
counterfact demo: naive vs honest
data: synthetic SCM (confounded, n=1000, seed=42)

pass_rate_by_arm(model_call)
arm              n  pass  rate    95% CI
haiku          581   212 0.365  [0.327, 0.405]
sonnet         419   302 0.721  [0.676, 0.762]

intervene(model_call -> sonnet)
identifiability: identified
outcome_delta: 0.663 [...]
next_step: none - CI width ...; no further action required.
naive_vs_causal_contrast: naive arm gap = +0.356; causal arm gap ... = +0.251; ...
```

The pass-rate table is descriptive. It says what happened in the logged corpus.

The intervention estimate is causal. It asks what the model predicts under a declared intervention, using the observed graph, support, and assumptions.

Those are different claims.

## What It Does

`counterfact` provides:

- A strict Pydantic trace schema for agent runs.
- A per-trace DAG builder.
- A transparent outcome model.
- A causal intervention API.
- Identifiability labels on every estimate.
- Structured `next_step` guidance when the corpus is not enough.
- Descriptive baselines such as `pass_rate_by_arm()`.
- Failure attribution ranking.
- Synthetic and real-agent benchmark harnesses.
- A small HTML report renderer for single traces.

## Core Concepts

### Trace

A trace is a `Run`.

It contains:

- steps
- decisions
- observations
- one final outcome

The schema lives under `src/counterfact/schema/`.

### Decision

A decision is a logged choice the agent made.

Common decision types:

- `model_call`
- `tool_call`
- `retry`
- `plan_step`
- `memory_read`
- `termination`

The demo focuses on `model_call`, `tool_call`, and `retry` because those have clear intervention arms.

### DAG

`build_dag()` builds an inspectable graph for one trace.

The graph is not learned from data. It is built from the trace structure so the causal assumptions stay visible.

### Outcome Model

`fit_outcome_model()` fits a simple binary outcome model over a corpus.

It is intentionally transparent. If the corpus has only one outcome class, the engine refuses to fit and returns `unidentified`.

### Intervention

`intervene()` asks what changes under a proposed decision edit.

Example:

```python
intervene(
    dag=dag,
    model=model,
    step=2,
    intervention={"model_choice": "sonnet"},
)
```

The result is a `CausalEstimate`.

Important fields:

- `identifiability`
- `outcome_delta`
- `assumptions`
- `warnings`
- `bounds`
- `next_step`

## Python API

```python
from bench.synthetic import generate_traces
from counterfact import build_dag, fit_outcome_model, intervene, pass_rate_by_arm
from counterfact.schema import Run

runs = [Run.model_validate(trace) for trace in generate_traces(n=500, seed=42)]

table = pass_rate_by_arm(runs, "model_call")
print(table)

model = fit_outcome_model(runs, n_bootstrap=200, seed=42)
dag = build_dag(runs[0])

estimate = intervene(
    dag=dag,
    model=model,
    step=2,
    intervention={"model_choice": "sonnet"},
)

print(estimate.identifiability)
print(estimate.outcome_delta)
print(estimate.next_step)
```

## CLI

### Demo

Confounded synthetic showcase:

```bash
uv run counterfact demo --confound --synthetic-n 1000 --seed 42
```

Real-trace smoke test:

```bash
uv run counterfact demo
```

The default demo reads the committed smoke-test corpus at `bench/real/smoke_mixed_outcome/`.

This command does not run the real-agent harness and does not call external LLMs.

Single-class refusal branch:

```bash
uv run counterfact demo --runs-dir bench/real/single_class_refusal
```

This corpus has only one outcome class. The expected result is `identifiability: unidentified`.

Synthetic fallback:

```bash
uv run counterfact demo --runs-dir /tmp/missing --synthetic-n 500 --target sonnet
```

### Generate Synthetic Traces

```bash
uv run counterfact bench synthetic --n 500 --seed 42 --output-dir /tmp/counterfact-syn
```

This is deterministic and local.

### Explain One Trace

```bash
uv run counterfact explain bench/real/smoke_mixed_outcome/real-streaming_watermark_dedupe-000000.json \
  --runs-dir bench/real/smoke_mixed_outcome
```

This writes a self-contained HTML report.

The report includes:

- the descriptive `pass_rate_by_arm()` table
- the per-trace DAG
- ranked decision attribution
- one `CausalEstimate` card per ranked decision

For `unidentified` estimates, numeric outcome deltas are hidden.

### Analyze A Corpus

```bash
uv run counterfact analyze corpus bench/real/smoke_mixed_outcome
```

The analyzer checks whether a corpus is ready to be promoted as a committed real-trace corpus.

The default rubric expects:

- pass rate between 0.3 and 0.7
- at least two supported arms for some randomized decision type
- at least five traces per supported arm
- at least one decision type where `intervene()` returns `identified`
- mixed pass/fail outcomes for both real-agent model arms (`small` and `large`)

The analyzer reports. It does not promote or rename anything.

The pilot analyzer gives a different signal: whether a hard corpus is a good showcase. A showcase corpus should be dominated by hidden semantic failures, not patch-format failures.

## Real-Agent Benchmarks

The real benchmark harness can call external LLM APIs and spend money. The local demo, synthetic benchmark, analyzer, and tests do not require provider credentials.

Example pilot:

```bash
uv run counterfact bench real --n 60 \
  --fixture-set stateful_calibration \
  --model-epsilon 1.0 \
  --tool-epsilon 0.0 \
  --retry-epsilon 0.0 \
  --budget-cap 10 \
  --output-dir bench/real/pilot_stateful_calibration_balanced
```

Do not run this as a routine test.

See `bench/real/README.md` for corpus promotion conventions and fixture sets.

## Development

```bash
uv run ruff check .
uv run pytest
```
