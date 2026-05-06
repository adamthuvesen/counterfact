# counterfact

`counterfact` helps you find where an agent run went wrong.

Given a decision the agent actually logged, such as a model call, tool call, retry, or stop decision, it asks:

> If the agent had done X instead of Y, would the task have been more likely to succeed?

Every answer is labelled:

- `identified` - the trace corpus supports a point estimate under the graph, support, and assumptions.
- `bounded` - a point estimate is not supported, but sensitivity bounds are available.
- `unidentified` - the corpus cannot support the counterfactual without more data, stronger assumptions, or replay.

The useful feature is not confidence. It is knowing when confidence would be fake.

## Install

```bash
uv pip install -e ".[dev]"
```

Requires Python 3.11+.

## Quickstart

Diagnose a failed trace against its corpus and write a shareable HTML report:

```bash
uv run counterfact diagnose bench/real/smoke_mixed_outcome/real-streaming_watermark_dedupe-000000.json \
  --runs-dir bench/real/smoke_mixed_outcome \
  --html /tmp/counterfact-diagnosis.html
```

`diagnose` ranks likely load-bearing decisions, shows which counterfactual
questions the corpus can honestly support, and gives the next data-collection
step when the answer is `unidentified`.

For a faster tour of common failure shapes, use the committed trace-forensics gallery:

```bash
uv run counterfact diagnose examples/trace-forensics/runs/syn-000000.json \
  --runs-dir examples/trace-forensics/runs \
  --decision-type model_call \
  --html /tmp/counterfact-wrong-model.html
```

The gallery covers wrong model choice, bad tool choice, missed retry, stopped
too early, unsupported counterfactuals, single-class support refusal, and
pass/fail trace diffs. It is illustrative, not benchmark evidence.

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

### Export To eval-audit

```bash
uv run counterfact export-runs bench/real/smoke_mixed_outcome \
  --to eval-audit-parquet \
  --output /tmp/counterfact-runs.parquet
```

This writes an `eval-audit` RunRecord-shaped parquet and a receipt. Use it when
you want to take a trace corpus into a separate population-level benchmark
audit:

```bash
eval-audit validate /tmp/counterfact-runs.parquet study.yaml
```

`counterfact` does not render model-switch verdicts; `eval-audit` owns that
question.

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
