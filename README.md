# counterfact

![License](https://img.shields.io/github/license/adamthuvesen/counterfact) ![Python](https://img.shields.io/badge/python-3.13%2B-blue)

`counterfact` is a small research library for understanding agent decision traces through counterfactual questions.

Given a decision the agent actually logged, such as a model call, tool call, retry, or stop decision, it asks:

> If the agent had done X instead of Y, would the task have been more likely to succeed?

It exists because naive pass-rate analysis on agent traces is misleading: logged decisions are confounded by upstream choices, so "runs that used model B passed more often" is not the same claim as "switching to model B raises the pass rate." `counterfact` keeps those two statements distinct, and refuses to fabricate the causal one when the data cannot support it.

Every answer carries an identifiability label. Read that label before you read the point estimate:

- `identified` - the corpus logs randomized assignment (propensities) with positive probability for both arms. You get a cross-fitted AIPW contrast with a 95% CI, plus g-formula and IPW cross-checks.
- `bounded` - both arms have support, but the corpus does not log randomization, so a point estimate would assume no unmeasured confounding. You get Manski no-assumption bounds, or `None` when the data cannot support any.
- `unidentified` - the corpus cannot answer the question without more data, stronger assumptions, or replay. You get a concrete `next_step` instead of a number.

For the design philosophy and a reading guide through the source, see [`docs/architecture.md`](docs/architecture.md).

## Install

```bash
uv pip install -e ".[dev]"
```

Requires Python 3.13+ (the floor follows [SPEC 0](https://scientific-python.org/specs/spec-0000/), which numpy and scikit-learn use).

The published wheel ships only the core causal-attribution library. The
real-agent harness under `bench/` is excluded from the wheel; if you need
`counterfact bench real`, install the `bench` extra (`pip install
"counterfact[bench]"`, which pulls `litellm`) or use an editable dev
install from a checkout.

## Quickstart

The fastest way to see the library at work is the deterministic local demo. It needs no traces and no API keys:

```bash
uv run counterfact demo --confound --synthetic-n 1000 --seed 42
```

Expected output:

```text
counterfact demo: naive vs honest
data: synthetic SCM (confounded, n=1000, seed=42)
outcomes: 514 pass / 486 fail

pass_rate_by_arm(model_call)
arm              n  pass  rate    95% CI
haiku          581   212 0.365  [0.327, 0.405]
sonnet         419   302 0.721  [0.676, 0.762]

intervene(model_call: haiku -> sonnet)
identifiability: identified
outcome_delta: +0.255 [+0.179, +0.330]  (AIPW, cross-fit)
cross-check: g-formula +0.251, IPW +0.251; min propensity 0.20/0.20
next_step: increase_n - CI width 0.150 > 0.10; ~1351 traces would tighten it.
suggested_command: uv run counterfact bench real --n 1351 --fixture-set broad_calibration
naive_vs_causal_contrast: naive arm gap = +0.356; causal arm gap (AIPW) = +0.255; the marginal table overstates what the corpus supports — see DAG and assumptions.
```

The table says what happened in the logged corpus. The intervention estimate is the contrast `E[Y | do(sonnet)] - E[Y | do(haiku)]`. It uses the logged propensities and adjusts only for decisions made before `model_call`. Those are different claims, and the demo shows them disagreeing on purpose. The truth under the synthetic SCM is +0.233.

Identification is checked against the data, not declared. Remove the confounder and the logged propensities from the same corpus, and `intervene` returns `bounded` with Manski bounds of [-0.36, +0.64] instead of a confident wrong number. `tests/acceptance/test_scm_recovery_confounded.py` pins that behavior.

To run `counterfact` on your own agent, you need a corpus of traces. Drop them in, then diagnose one run against the corpus:

```bash
uv run counterfact diagnose corpus/<session-id>.json --runs-dir corpus/
```

`diagnose` ranks likely load-bearing decisions, shows which counterfactual
questions the corpus can honestly support, and gives the next data-collection
step when the answer is `unidentified`. Add `--html report.html` to write a
self-contained report.

The public smoke demo uses the committed real corpus:

```bash
uv run counterfact demo
```

It reads `bench/real/smoke_mixed_outcome` and makes no provider calls. If that corpus is missing, the command exits instead of quietly switching evidence bases. Pass `--synthetic-fallback` only when you explicitly want synthetic data. Treat this as a smoke test, not a general benchmark claim.

## Getting traces in

You can capture traces live while your agent runs, or ingest a trace dump after
the fact. Both paths share one mapping, so `corpus/<id>.json` is byte-equivalent
whichever way you produced it.

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

`traces.jsonl` is one session per line, each line either a JSON object
`{"messages": [...]}` or a JSON array of message dicts. There is no mapping file:
counterfact reads `session_id`, `ToolUseBlock`, `AssistantMessage.model`, and
`ResultMessage.is_error` directly.

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

`--outcome pass|fail` is required when the trace has neither a root error nor a
`counterfact.outcome` marker span. counterfact never infers pass/fail from "no
error". That would manufacture a confidence the data does not support.

Live tracing registers the processor:

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

This writes a self-contained HTML report and prints the resolved report path, for example `/private/tmp/counterfact-report.html` when `--output /tmp/counterfact-report.html` is used. The report contains a trace timeline, decision cards, the per-trace DAG, support/replay warnings, and ranked causal estimates. Numeric outcome deltas are hidden for `unidentified` estimates.

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
reads RunRecord-style parquet: population-level benchmark audits, dashboards,
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
- `pass_rate_by_arm()`: descriptive baseline, not a causal claim.

## Real-Agent Trace Corpora

The real-agent harness can call external LLM APIs and spend money. The local demo, synthetic generator, analyzer, and tests do not require provider credentials.

Do not run real-agent benchmarks as routine validation. `counterfact bench real` refuses to start until `.counterfact/approved` contains a JSON approval receipt matching the command's trace count, budget cap, output directory, fixture set, model map, and randomization config. See `bench/real/README.md` for corpus promotion conventions, fixture sets, and budgeted pilot commands.

## Development

```bash
uv sync --extra dev
uv run pre-commit install
uv run ruff check .
uv run pytest
```
