# Using counterfact

This page covers getting traces in, every CLI command, and the core Python objects. Start with the [README](../README.md) for what `counterfact` is and the [technical report](report.md) for how it works.

## Getting traces in

You can capture traces live while your agent runs, or ingest a trace dump after the fact. Both paths share one mapping, so `corpus/<id>.json` is byte-equivalent whichever way you produced it.

```text
your agent run ─┬─► live tracer  ─┐
                │                  ├─► corpus/  ─► counterfact diagnose
                └─► trace dump  ──►│
                    ingest <sdk>  ─┘
```

| Adapter | Source format | Live helper |
| --- | --- | --- |
| `claude-agent-sdk` | JSONL of Claude Agent SDK message dicts | `ClaudeAgentTracer` |
| `openai-agents` | JSON trace export from OpenAI Agents SDK | `CounterfactSpanProcessor` |
| `generic-jsonl` | any JSONL with a user-supplied mapping | none |

### Claude Agent SDK

Offline ingest:

```bash
uv run counterfact ingest claude-agent-sdk traces.jsonl --output-dir corpus/
uv run counterfact diagnose corpus/<session-id>.json --runs-dir corpus/
```

`traces.jsonl` is one session per line, each line either a JSON object `{"messages": [...]}` or a JSON array of message dicts. There is no mapping file: counterfact reads `session_id`, `ToolUseBlock`, `AssistantMessage.model`, and `ResultMessage.is_error` directly.

Live tracing wraps the message stream:

```python
from pathlib import Path
from claude_agent_sdk import query
from counterfact.tracing import ClaudeAgentTracer

async with ClaudeAgentTracer(output_dir=Path("corpus/")) as tracer:
    async for msg in query(prompt="..."):
        tracer.observe(msg)
# corpus/<session_id>.json is written on exit.
```

### OpenAI Agents SDK

Offline ingest:

```bash
uv run counterfact ingest openai-agents trace.json --output-dir corpus/ --outcome pass
uv run counterfact diagnose corpus/<trace-id>.json --runs-dir corpus/
```

`--outcome pass|fail` is required when the trace has neither a root error nor a `counterfact.outcome` marker span. counterfact never infers pass/fail from "no error". That would manufacture a confidence the data does not support.

Live tracing registers the processor:

```python
from pathlib import Path

from agents import add_trace_processor
from counterfact.tracing import CounterfactSpanProcessor

add_trace_processor(
    CounterfactSpanProcessor(
        output_dir=Path("corpus/"),
        outcome_provider=my_evaluator,  # optional; returns True/False/None per trace
    )
)
```

## CLI

### Diagnose A Failed Trace

```bash
uv run counterfact diagnose bench/real/smoke_mixed_outcome/real-streaming_watermark_dedupe-000000.json \
  --runs-dir bench/real/smoke_mixed_outcome \
  --top-k 3
```

Use `--json` to emit the reusable diagnosis artifact. Numeric effects appear only when the underlying `CausalEstimate` is identified or bounded; unsupported entries carry concrete `next_step` guidance instead.

Use `--html report.html` to write a self-contained diagnosis report with the same ranked decisions, trace context, and support diagnostics, with numeric effects hidden for `unidentified` decisions. `--json --html report.html` keeps stdout as JSON and writes the HTML path notification to stderr.

### Compare Two Traces

```bash
uv run counterfact compare pass.json fail.json
```

This is a descriptive trace diff: outcome, steps, decision actions, and observation counts. Add `--runs-dir <corpus> --focal right` to overlay a corpus-backed diagnosis for one side. Without a corpus, `compare` does not make causal claims.

### Explain A Trace

```bash
uv run counterfact explain bench/real/smoke_mixed_outcome/real-streaming_watermark_dedupe-000000.json \
  --runs-dir bench/real/smoke_mixed_outcome \
  --output report.html
```

This writes a self-contained HTML report and prints its absolute path. The report contains a trace timeline, decision cards, the per-trace DAG, support/replay warnings, and ranked causal estimates. Numeric outcome deltas are hidden for `unidentified` estimates.

### Intervene On A Decision

```bash
uv run counterfact intervene bench/real/smoke_mixed_outcome/real-streaming_watermark_dedupe-000000.json \
  --runs-dir bench/real/smoke_mixed_outcome \
  --decision-id d-000000-model-1 \
  --set model_choice=small
```

This trace chose `large`, so the answer is the contrast `E[Y | do(small)] - E[Y | do(large)]`. On the committed corpus it is `identified` at -0.656 [-0.792, -0.519], with g-formula and IPW cross-checks. Pass `--baseline <arm>` to contrast against an arm other than the logged one.

Decision IDs are listed in the trace timeline rendered by `counterfact explain`.

Use `--json` or `--output estimate.json` to emit the reusable `CausalEstimate` artifact.

Prefer `--decision-id` for precise targeting. `--step <n>` is accepted only when the step contains exactly one decision.

Useful trace-level questions:

- wrong model choice: `--set model_choice=small`
- logged temperature choice: `--set temperature=0.7`. Temperature is logged as metadata, not as a randomized arm, so this returns `bounded` with Manski bounds
- bad tool choice: `--set tool_choice=run_tests`
- missed retry: `--set retry_policy=retry_once`
- stopped too early: inspect the termination decision in `counterfact explain`
- unsupported intervention: prompt or hidden-state edits return support/replay guidance instead of fake estimates

### Analyze A Corpus

```bash
uv run counterfact analyze corpus bench/real/smoke_mixed_outcome
```

The analyzer checks whether a trace corpus has enough support for counterfactual trace diagnosis and intervention estimates. It reports; it does not promote, score, rank, or rename anything.

### Ingest Generic JSONL

```bash
uv run counterfact ingest generic-jsonl source.jsonl \
  --mapping mapping.json \
  --output-dir traces/
```

The mapping file is explicit: `fields` maps native target paths to source paths, and `defaults` fills safe constants.

```json
{
  "fields": {
    "run_id": "id",
    "steps": "steps",
    "outcome.value": "passed"
  },
  "defaults": {
    "schema_version": "0.1.0",
    "outcome.kind": "binary",
    "outcome.verifier": "imported"
  }
}
```

The command writes native `Run` JSON files plus `ingest-receipt.json` with warnings about dropped fields and missing randomization metadata. It does not loosen the native trace schema.

### Export To Parquet

```bash
uv run counterfact export-runs bench/real/smoke_mixed_outcome \
  --to runrecord-parquet \
  --output /tmp/counterfact-runs.parquet
```

This writes one parquet row per run (agent identity, outcome, cost, tokens, provenance) plus a receipt that records how each column was derived. Use it to load a corpus into your own analysis tools. `counterfact` does no population-level scoring itself.

### Evaluate A Policy

```bash
uv run counterfact evaluate-policy corpus/ --policy sonnet_retry_on_fail.json --observation-key tests_passed
```

`sonnet_retry_on_fail.json` declares what the agent should do at each decision type it controls. Every other decision stays as logged:

```json
{
  "name": "sonnet_retry_on_fail",
  "rules": [
    {"decision_type": "model_call", "action": "sonnet"},
    {
      "decision_type": "retry",
      "when": {"source": "observation", "key": "tests_passed"},
      "cases": {"true": "no_retry", "false": "retry_once"}
    }
  ]
}
```

The answer is V(policy) - V(logged), or V(policy) - V(baseline) with `--baseline other.json`. It comes from sequential doubly robust estimation over the logged decision sequence, with ICE g-formula and self-normalized importance sampling as cross-checks. It is `identified` only when every decision the policy controls logs a propensity that gives the policy's action positive probability. The output also prints what filtering the logs to matching runs would say, labeled as descriptive. On the benchmark that number gets both retry policies backwards; see [report §5](report.md#policies).

### Benchmark The Estimator

```bash
uv run counterfact bench suite --seeds 500 --markdown docs/benchmark-results.md
```

This scores the naive table, an adjust-everything g-formula, and the engine on every scenario in `bench/synthetic/scenarios.py`. It is deterministic and local. See [report §5](report.md#5-results) for the results.

### Generate Synthetic Traces

```bash
uv run counterfact bench synthetic --n 500 --seed 42 --output-dir /tmp/counterfact-syn
```

This is deterministic and local.

## Python API

```python
from bench.synthetic import generate_traces
from counterfact import build_dag, fit_outcome_model, intervene, pass_rate_by_arm
from counterfact.schema import Run

runs = [Run.model_validate(trace) for trace in generate_traces(n=500, seed=42)]

table = pass_rate_by_arm(runs, "model_call")
model = fit_outcome_model(runs, seed=42)
dag = build_dag(runs[0])

estimate = intervene(
    dag=dag,
    model=model,
    step=2,
    intervention={"model_choice": "sonnet"},
    baseline="haiku",
)

print(table)
print(estimate.identifiability)
print(estimate.outcome_delta)
print(estimate.next_step)
```

## Concepts

- `Run`: one agent trace with steps, decisions, observations, and a final outcome.
- `Decision`: a logged choice such as `model_call`, `tool_call`, `retry`, `plan_step`, `memory_read`, or `termination`.
- `build_dag()`: builds an inspectable per-trace graph from logged structure. The graph is not learned from data.
- `fit_outcome_model()`: validates a corpus and fits a transparent binary outcome model over it. Single-class corpora return `unidentified` instead of pretending to fit.
- `intervene()`: estimates `E[Y | do(target)] - E[Y | do(baseline)]` for one decision and returns a structured `CausalEstimate`. The baseline defaults to the arm the trace chose.
- `evaluate_policy()`: estimates how success would change if the agent followed a `Policy` at the decision types it controls, and returns a `PolicyEstimate` with the same identifiability labels.
- `pass_rate_by_arm()`: descriptive baseline, not a causal claim.

## Real-Agent Trace Corpora

The real-agent harness can call external LLM APIs and spend money. The local demo, synthetic generator, analyzer, and tests do not require provider credentials.

Do not run real-agent benchmarks as routine validation. `counterfact bench real` refuses to start until `.counterfact/approved` contains a JSON approval receipt matching the command's trace count, budget cap, output directory, fixture set, model map, and randomization config. See [`bench/real/README.md`](../bench/real/README.md) for fixture sets, the routed-trial design, and budgeted commands.

