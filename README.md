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

The benchmark scores three estimators on eight causal models built to break them, with an exact true effect for each. The naive pass-rate gap and an adjust-everything g-formula are off by up to 0.31. `counterfact` has a mean bias within 0.01, or it refuses. For whole policies over several decisions, such as "retry only when the tests fail", filtering the logs gets the answer backwards and `counterfact` stays unbiased.

On 800 real coding-agent traces with a confounded model router, the naive gap says the large model adds 0.05 to the pass rate. A randomized trial in the same run says 0.25. `counterfact`, given only the routed runs, says 0.32, with a 95% CI of [0.20, 0.45] that contains the truth. The [technical report](https://github.com/adamthuvesen/counterfact/blob/main/docs/report.md) covers the method, the results, and the limits.

## Install

```bash
uv pip install -e ".[dev]"
```

Requires Python 3.13 or later. The wheel ships only the library. The real-agent harness under `bench/` needs a checkout and the `bench` extra.

## Quickstart

The demo runs on synthetic traces with a known true effect. It needs no API keys:

```bash
uv run counterfact demo --confound --synthetic-n 1000 --seed 42
```

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
next_step: increase_n - CI width 0.150 > 0.10; ~2264 traces would tighten it.
naive_vs_causal_contrast: naive arm gap = +0.356; causal arm gap (AIPW) = +0.255; the marginal table overstates what the corpus supports — see DAG and assumptions.
```

The table reports a gap of +0.356 between the models. The true effect is +0.233. The causal estimate is +0.255, and its CI contains the truth. The table is descriptive. It says what happened in the logs, not what switching models would do.

## Use it on your agent

Capture traces live with `ClaudeAgentTracer` (Claude Agent SDK) or `CounterfactSpanProcessor` (OpenAI Agents SDK), or ingest an export:

```bash
uv run counterfact ingest claude-agent-sdk traces.jsonl --output-dir corpus/
```

Then ask questions of the corpus:

```bash
# Rank the decisions most likely to explain a failed run
uv run counterfact diagnose corpus/<run-id>.json --runs-dir corpus/ --html report.html

# Estimate the effect of one decision
uv run counterfact intervene corpus/<run-id>.json --runs-dir corpus/ \
  --decision-id <decision-id> --set model_choice=small

# Evaluate a whole policy, such as "use sonnet and retry only when tests fail"
uv run counterfact evaluate-policy corpus/ --policy policy.json --observation-key tests_passed
```

The same functions are available from Python: `intervene`, `evaluate_policy`, `fit_outcome_model`, and `pass_rate_by_arm`. [docs/usage.md](docs/usage.md) documents the analysis commands, the trace adapters, the policy format, and the core objects.

## Documentation

- [Technical report](docs/report.md): method, benchmark, results, and limits.
- [Usage](docs/usage.md): CLI reference, adapters, and core objects.
- [Architecture](docs/architecture.md): how the code is organized.
- [Invariants](docs/invariants.md): the guarantees every change must keep.
- [Real-agent harness](bench/real/README.md): a budgeted coding-agent benchmark with its own ground truth.

## Development

```bash
uv sync --extra dev
uv run pre-commit install
make ci
```

`make ci` runs lint, type checks, and tests with coverage. The benchmark suite runs locally and makes no API calls:

```bash
uv run counterfact bench suite --seeds 500
```
