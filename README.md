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
uv run counterfact demo --confound
```

The headline demo is the **confounded synthetic showcase**. It generates a deterministic synthetic corpus where `model_choice` is biased by `tool_choice` (a textbook back-door confounding scenario), runs the descriptive `pass_rate_by_arm` baseline alongside the engine's g-formula adjustment, and surfaces the gap between them on one labelled line. The synthetic SCM is the right surface to teach the project's product stance — it has ground truth, deterministic seeds, and a controllable causal structure where the marginal table and the causal estimate disagree on purpose.

Example output:

```text
counterfact demo: naive vs honest
data: synthetic SCM (confounded, n=1000, seed=42)
outcomes: 514 pass / 486 fail

pass_rate_by_arm(model_call)
arm              n  pass  rate    95% CI
haiku          581   212 0.365  [0.327, 0.405]
sonnet         419   302 0.721  [0.676, 0.762]

intervene(model_call -> sonnet)
identifiability: identified
outcome_delta: 0.663 [0.619, 0.704]
next_step: none - CI width 0.085 ≤ 0.10; no further action required.
naive_vs_causal_contrast: naive arm gap = +0.356; causal arm gap (do-calculus, g-formula) = +0.251; the marginal table overstates what the corpus supports — see DAG and assumptions.
```

Without `--confound`, `counterfact demo` is the **messy real-trace smoke test**: it points at the committed `bench/real/runs_v2/` corpus (30 mixed-outcome `date_window` traces from the real-agent harness) and runs the same flow. It falls back to `runs_single_class/` (the single-class regression anchor — three traces kept on disk to keep the engine's "honest refusal" branch exercised) and then to an unconfounded synthetic corpus when neither real corpus is present. It does not call the real-agent LLM harness or require provider credentials.

Pointing the demo at `runs_single_class` (`uv run counterfact demo --runs-dir bench/real/runs_single_class`) reproduces the "honest refusal" branch — the engine refuses to fit a single-class outcome model and emits an `unidentified` verdict with `next_step: broaden_arm_support`. All three branches (confounded showcase, real-trace smoke test, single-class refusal) are intended.

See [docs/demo-excerpt.md](docs/demo-excerpt.md) for the rendered notebook-style excerpt and [bench/real/README.md](bench/real/README.md) for the corpus-promotion convention.

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
- Real-agent coding harness with randomized decisions, budget gate, hidden/public fixture support, and two committed pilot corpora: `runs_single_class` (3 csv_dedupe traces, single-class regression anchor) and `runs_v2` (30 date_window traces, mixed outcomes, default for the demo)
- Demo notebook and acceptance tests around the naive-vs-honest story

### NextStep payload contract

Refusals are first-class. Every `CausalEstimate.next_step` carries a structured payload alongside `human_text`:

- `broaden_arm_support` → `arm_name`, `missing_strata`, `observed_arms`, `missing_arms`
- `increase_n` → `current_n`, `estimated_required_n`, `target_ci_width`, `power_method` (`binomial_wald_two_arm` | `inline_scaling` | `degenerate`), `arm_breakdown`
- `replay_required` → `intervention_target`, `replay_inputs_required`, `note`
- `add_arm_randomization` → `arm_name`, `current_policy`
- `none` → empty

When the harness can generate the missing data, `payload["suggested_command"]` carries a copy-pasteable `uv run counterfact bench …` invocation. The full schema lives in `openspec/specs/causal-engine/spec.md`.

### Corpus readiness

Before any new real corpus is committed under `bench/real/runs_v2/`, run the no-spend analyzer to score it against the promotion rubric:

```bash
uv run counterfact analyze corpus bench/real/runs_pilot_<YYYY-MM-DD>/
```

The rubric (`src/counterfact/corpus_analyzer/rubric.py`) requires pass rate in [0.3, 0.7], ≥2 arms with `n ≥ 5` for some randomized decision type, and ≥1 decision type where `intervene()` returns `identified`. Promotion to `runs_v2/` is a deliberate human `mv` — see `bench/real/README.md` for the convention.

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

The default committed real corpus (`runs_v2`) has mixed outcomes, so the CLI demo fits the outcome model and returns an `identified` estimate. The `runs_single_class` corpus is the single-class regression anchor — three traces kept on disk to keep the engine's "honest refusal" path exercised; point the demo at it explicitly to see that branch.

## CLI

```bash
# Synthetic SCM traces, deterministic and no LLM calls
uv run counterfact bench synthetic --n 500 --seed 42 --output-dir /tmp/counterfact-syn

# Headline showcase: confounded synthetic naive-vs-causal demo (no LLM calls)
uv run counterfact demo --confound --synthetic-n 1000 --seed 42

# Real-trace smoke test (defaults to bench/real/runs_v2)
uv run counterfact demo

# Force synthetic fallback for the demo (uniform, no confounding)
uv run counterfact demo --runs-dir /tmp/missing --synthetic-n 500 --target sonnet

# Per-trace HTML report grounded in CausalEstimate
uv run counterfact explain bench/real/runs_v2/real-date_window-000000.json --runs-dir bench/real/runs_v2
```

`counterfact explain` writes a self-contained HTML file (no JS, no CDN, no
new runtime deps) that pairs the descriptive `pass_rate_by_arm` baseline
with one CausalEstimate card per ranked decision. Numeric estimates are
suppressed for `unidentified` cards; `next_step` and any
`suggested_command` are surfaced verbatim.

The real-agent harness is available behind an explicit budget and first-run approval gate:

```bash
uv run counterfact bench real --n 30 --budget-cap 50 --fixture-set hard_hidden_v1
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
