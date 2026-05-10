# counterfact

`counterfact` is a small research library for understanding agent decision traces through counterfactual questions.

Given a decision the agent actually logged, such as a model call, tool call, retry, or stop decision, it asks:

> If the agent had done X instead of Y, would the task have been more likely to succeed?

Every answer is labelled:

- `identified` - the trace corpus supports a point estimate under the graph, support, and assumptions.
- `bounded` - a point estimate is not supported, but sensitivity bounds are available.
- `unidentified` - the corpus cannot support the counterfactual without more data, stronger assumptions, or replay.

## Install

```bash
uv pip install -e ".[dev]"
```

Requires Python 3.11+.

The published wheel ships only the core causal-attribution library. The
real-agent harness under `bench/` is excluded from the wheel; if you need
`counterfact bench real`, install the `bench` extra (`pip install
"counterfact[bench]"`, which pulls `litellm`) or use an editable dev
install from a checkout.

## Quickstart

Drop in your agent traces, run diagnose. If you already use the Claude
Agent SDK, the path is one command:

```bash
uv run counterfact ingest claude-agent-sdk traces.jsonl --output-dir corpus/
uv run counterfact diagnose corpus/<session-id>.json --runs-dir corpus/
```

`traces.jsonl` is one session per line, each line either a JSON object
`{"messages": [...]}` or a JSON array of message dicts. There is no
mapping file: counterfact reads `session_id`, `ToolUseBlock`,
`AssistantMessage.model`, and `ResultMessage.is_error` directly.

For live tracing while your agent runs, wrap the message stream:

```python
from pathlib import Path
from claude_agent_sdk import query
from counterfact.tracing import ClaudeAgentTracer

async with ClaudeAgentTracer(output_dir=Path("corpus/")) as tracer:
    async for msg in query(prompt="..."):
        tracer.observe(msg)
# corpus/<session_id>.json is written on exit.
```

The same offline-vs-live mapping is used for both — byte-equivalent output.

If you use the OpenAI Agents SDK instead, the path is symmetric:

```bash
uv run counterfact ingest openai-agents trace.json --output-dir corpus/ --outcome pass
uv run counterfact diagnose corpus/<trace-id>.json --runs-dir corpus/
```

`--outcome pass|fail` is required when the trace has neither a root error
nor a `counterfact.outcome` marker span. counterfact never infers
pass/fail from "no error" — that would manufacture a confidence the data
does not support.

For live tracing, register the processor:

```python
from agents import add_trace_processor
from counterfact.tracing import CounterfactSpanProcessor

add_trace_processor(
    CounterfactSpanProcessor(
        output_dir=Path("corpus/"),
        outcome_provider=my_evaluator,  # optional; returns True/False/None per trace
    )
)
```

| Adapter | Source format | Live helper |
| --- | --- | --- |
| `claude-agent-sdk` | JSONL of Claude Agent SDK message dicts | `ClaudeAgentTracer` |
| `openai-agents` | JSON trace export from OpenAI Agents SDK | `CounterfactSpanProcessor` |
| `generic-jsonl` | any JSONL with a user-supplied mapping | — |

The full loop is the same in either direction:

```text
your agent run ─┬─► live tracer  ─┐
                │                  ├─► corpus/  ─► counterfact diagnose
                └─► trace dump  ──►│
                    ingest <sdk>  ─┘
```

Live tracing and offline ingest share one mapping, so `corpus/<id>.json` is
byte-equivalent whether you instrumented the run or imported its dump.

Diagnose a failed trace against its corpus and write a shareable HTML report:

```bash
uv run counterfact diagnose bench/real/smoke_mixed_outcome/real-streaming_watermark_dedupe-000000.json \
  --runs-dir bench/real/smoke_mixed_outcome \
  --html /tmp/counterfact-diagnosis.html
```

`diagnose` ranks likely load-bearing decisions, shows which counterfactual
questions the corpus can honestly support, and gives the next data-collection
step when the answer is `unidentified`.

Run the deterministic local demo:

```bash
uv run counterfact demo --confound --synthetic-n 1000 --seed 42
```

It prints a descriptive pass-rate table, a causal intervention estimate, and the contrast between the two. The table says what happened in the logged corpus. The intervention estimate asks what the model predicts under a declared decision edit. Those are different claims.

## CLI

### Diagnose A Failed Trace

```bash
uv run counterfact diagnose bench/real/smoke_mixed_outcome/real-streaming_watermark_dedupe-000000.json \
  --runs-dir bench/real/smoke_mixed_outcome \
  --top-k 3
```

Use `--json` to emit the reusable diagnosis artifact. Numeric effects appear
only when the underlying `CausalEstimate` is identified or bounded; unsupported
entries carry concrete `next_step` guidance instead.

Use `--html report.html` to write a self-contained diagnosis report with the
same ranked decisions, trace context, support diagnostics, and unidentified
hide rules. `--json --html report.html` keeps stdout as JSON and writes the
HTML path notification to stderr.

### Compare Two Traces

```bash
uv run counterfact compare pass.json fail.json
```

This is a descriptive trace diff: outcome, steps, decision actions, and
observation counts. Add `--runs-dir <corpus> --focal right` to overlay a
corpus-backed diagnosis for one side. Without a corpus, `compare` does not make
causal claims.

### Explain A Trace

```bash
uv run counterfact explain bench/real/smoke_mixed_outcome/real-streaming_watermark_dedupe-000000.json \
  --runs-dir bench/real/smoke_mixed_outcome \
  --output report.html
```

This writes a self-contained HTML report with a trace timeline, decision cards, the per-trace DAG, support/replay warnings, and ranked causal estimates. Numeric outcome deltas are hidden for `unidentified` estimates.

### Intervene On A Decision

```bash
uv run counterfact intervene bench/real/smoke_mixed_outcome/real-streaming_watermark_dedupe-000000.json \
  --runs-dir bench/real/smoke_mixed_outcome \
  --decision-id <decision-id> \
  --set model_choice=large
```

Decision IDs are listed in the trace timeline rendered by `counterfact explain`.

Use `--json` or `--output estimate.json` to emit the reusable `CausalEstimate` artifact.

Prefer `--decision-id` for precise targeting. `--step <n>` is accepted only when the step contains exactly one decision.

Useful trace-level questions:

- wrong model choice: `--set model_choice=large`
- bad tool choice: `--set tool_choice=run_tests`
- missed retry: `--set retry_policy=retry_once`
- stopped too early: inspect the termination decision in `counterfact explain`
- unsupported intervention: prompt or hidden-state edits return support/replay guidance instead of fake estimates

### Analyze A Corpus

```bash
uv run counterfact analyze corpus bench/real/smoke_mixed_outcome
```

The analyzer checks whether a trace corpus has enough support for
counterfactual trace diagnosis and intervention estimates. It reports; it does
not promote, score, rank, or rename anything.

### Ingest Generic JSONL

```bash
uv run counterfact ingest generic-jsonl source.jsonl \
  --mapping mapping.json \
  --output-dir traces/
```

The mapping file is explicit: `fields` maps native target paths to source paths,
and `defaults` fills safe constants.

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

The command writes native `Run` JSON files plus `ingest-receipt.json` with
warnings about dropped fields and missing randomization metadata. It does not
loosen the native trace schema.

### Export To RunRecord Parquet

```bash
uv run counterfact export-runs bench/real/smoke_mixed_outcome \
  --to runrecord-parquet \
  --output /tmp/counterfact-runs.parquet
```

This writes a RunRecord-shaped parquet (one row per run; columns for agent
identity, outcome, cost, tokens, provenance) plus a receipt documenting how
each field was derived. The output is consumable by any downstream tool that
reads RunRecord-style parquet — population-level benchmark audits, dashboards,
or your own analysis. `counterfact` itself does not render verdicts at the
population level; the export is the boundary.

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
model = fit_outcome_model(runs, n_bootstrap=200, seed=42)
dag = build_dag(runs[0])

estimate = intervene(
    dag=dag,
    model=model,
    step=2,
    intervention={"model_choice": "sonnet"},
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
- `fit_outcome_model()`: fits a transparent binary outcome model over a corpus. Single-class corpora return `unidentified` instead of pretending to fit.
- `intervene()`: estimates a proposed decision edit and returns a structured `CausalEstimate`.
- `pass_rate_by_arm()`: descriptive baseline, not a causal claim.

## Real-Agent Trace Corpora

The real-agent harness can call external LLM APIs and spend money. The local demo, synthetic generator, analyzer, and tests do not require provider credentials.

Do not run real-agent benchmarks as routine validation. See `bench/real/README.md` for corpus promotion conventions, fixture sets, and budgeted pilot commands.

## Development

```bash
uv run ruff check .
uv run pytest
```
