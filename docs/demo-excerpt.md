# Demo Excerpt: Naive vs Honest

This excerpt mirrors the v0 notebook story in a compact GitHub-friendly form.
The real `runs_v1` pilot corpus is useful because it makes the central point
visible: a naive table can look decisive while the causal answer is still
unsupported.

## Naive Baseline

`pass_rate_by_arm(model_call)` on the committed real-agent pilot:

| arm | n | pass | pass rate | 95% CI |
| --- | ---: | ---: | ---: | --- |
| large | 28 | 28 | 1.000 | [0.879, 1.000] |
| small | 2 | 2 | 1.000 | [0.342, 1.000] |

The tempting read is that every model arm worked. That is descriptively true,
but it is not a causal attribution.

## Honest Causal Verdict

`intervene(model_call -> large)` on the same corpus:

| field | value |
| --- | --- |
| identifiability | `unidentified` |
| reason | every trace has `Outcome.value=True`; there is no outcome variation for an outcome model or back-door adjustment |
| warning | `fit_outcome_model` is intentionally skipped for single-class real corpora |
| next step | `broaden_arm_support`: collect or construct traces with both pass and fail outcomes before estimating decision-level effects |
| suggested command | `uv run counterfact bench real --n 30 --fixture-set hidden_v1 --model-greedy large --model-epsilon 0.5` |

That is the product stance: `counterfact` does not launder a degenerate corpus into
a fake probability. It returns a useful no, plus the next data collection step — and,
when the harness can produce that data, the exact command to do it.
