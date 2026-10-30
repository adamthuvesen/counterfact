# counterfact

![License](https://img.shields.io/github/license/adamthuvesen/counterfact) ![Python](https://img.shields.io/badge/python-3.13%2B-blue)

`counterfact` answers counterfactual questions about LLM agent traces:

> If the agent had used a different model, tool, or retry rule, would the task have been more likely to succeed?

The usual answer is a pass-rate table split by the decision. That table is confounded. Agents send hard tasks to strong models and retry after failures, so the runs that took an action differ from the runs that did not. `counterfact` estimates the causal effect instead, and it refuses when the logs cannot support one.

Every answer carries a label, checked against the data:

- `identified`: the logs record how each decision was randomized. You get a doubly robust estimate with a 95% CI.
- `bounded`: the logs do not show how the options were assigned, so a point estimate would be a guess. You get no-assumption bounds when the data support them, and no interval when they do not.
- `unidentified`: the logs cannot answer the question. You get the concrete next step, such as which runs to collect.

![Single-decision bias by estimator](https://raw.githubusercontent.com/adamthuvesen/counterfact/main/docs/figures/single-decision-bias.svg)

The benchmark compares three estimators on causal models with exact true effects. It tests confounding, downstream adjustment, missing propensities, and whole policies. The [generated results](docs/benchmark-results.md) show estimation error, interval coverage, and refusals.

The committed real-agent experiment compares a confounded router with an independent randomized arm. Its [current reanalysis](bench/real/routed_trial/RESULTS.md) shows agreement within sampling uncertainty. The [technical report](https://github.com/adamthuvesen/counterfact/blob/main/docs/report.md) covers the method, the results, and the limits.

## Install

```bash
uv sync --locked --extra dev --extra bench
```

Requires Python 3.13 or later. The wheel ships only the library. The real-agent harness under `bench/` needs a checkout and the `bench` extra.

## Quickstart

The demo runs on synthetic traces with a known true effect. It needs no API keys:

```bash
uv run --locked counterfact demo --confound --synthetic-n 1000 --seed 42
```

The demo prints a descriptive pass-rate table, an identified causal contrast, and concrete next-step guidance. The synthetic model provides a known truth for comparison. [The demo guide](docs/demo-excerpt.md) explains the results and the confounding example.

## Use it on your agent

Capture traces live with `ClaudeAgentTracer` (Claude Agent SDK) or `CounterfactSpanProcessor` (OpenAI Agents SDK), or ingest an export:

```bash
uv run --locked counterfact ingest claude-agent-sdk traces.jsonl --output-dir corpus/
```

Then ask questions of the corpus:

```bash
# Rank the decisions most likely to explain a failed run
uv run --locked counterfact diagnose corpus/<run-id>.json --runs-dir corpus/ --html report.html

# Estimate the effect of one decision
uv run --locked counterfact intervene corpus/<run-id>.json --runs-dir corpus/ \
  --decision-id <decision-id> --set model_choice=small

# Evaluate a whole policy, such as "use sonnet and retry only when tests fail"
uv run --locked counterfact evaluate-policy corpus/ --policy policy.json --observation-key tests_passed
```

From Python, use `AnalysisCorpus` with `intervene` for a focal decision or `contrast` for a first-occurrence total effect. `evaluate_policy` handles sequential rules. `fit_outcome_model` remains an optional descriptive fit. [docs/usage.md](docs/usage.md) documents the analysis commands, the trace adapters, the policy format, and the core objects.

## Documentation

- [Technical report](docs/report.md): method, benchmark, results, and limits.
- [Usage](docs/usage.md): CLI reference, adapters, and core objects.
- [Architecture](docs/architecture.md): how the code is organized.
- [Invariants](docs/invariants.md): the guarantees every change must keep.
- [Migration](docs/migration.md): analysis snapshots, corpus recovery, and versioned exports.
- [Real-agent harness](bench/real/README.md): a budgeted coding-agent benchmark with its own randomized trial.

## Development

```bash
make check
make verify
```

`make check` runs the fast offline checks. `make verify` adds full-suite coverage, generated artifacts, installed-wheel consumers, and Docker execution checks. The Docker verifier requires a running Linux-container engine and its pinned image. See the [real-agent guide](bench/real/README.md).

Synthetic benchmarks require a checkout and make no provider calls:

```bash
uv run --locked counterfact bench suite --seeds 500
```
