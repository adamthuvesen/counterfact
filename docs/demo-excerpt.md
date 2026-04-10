# Demo Excerpt: Naive vs Honest

This excerpt mirrors the v0 notebook story in a compact GitHub-friendly form
on the canonical `runs_v2` corpus (30 `date_window` traces, mixed outcomes
from inverted-greedy randomization). The earlier `runs_v1` corpus
(single-class) is still committed as a regression anchor for the engine's
"honest refusal" branch — point the CLI at it with
`--runs-dir bench/real/runs_v1` to see that branch instead.

## Naive Baseline

`pass_rate_by_arm(model_call)` on the committed `runs_v2` corpus:

| arm | n | pass | pass rate | 95% CI |
| --- | ---: | ---: | ---: | --- |
| large | 8 | 8 | 1.000 | [0.676, 1.000] |
| small | 22 | 6 | 0.273 | [0.132, 0.482] |

The descriptive read is clear: the small arm fails most of the time, the
large arm is almost always right. But a marginal table cannot tell you
whether that gap is causal — the agent's other randomized decisions covary
with the model arm, and the per-arm CIs alone do not adjust for that.

## Honest Causal Verdict

`intervene(model_call -> small)` on the same corpus:

| field | value |
| --- | --- |
| identifiability | `identified` |
| outcome_delta | `0.332 [0.179, 0.493]` |
| next step | `increase_n`: CI width 0.314 > 0.10; ~416 traces would tighten it |
| suggested command | `uv run counterfact bench real --n 416 --fixture-set hard_hidden_v1` |

That is the product stance: when the corpus supports the question,
`counterfact` returns an estimate with a bootstrap CI *and* names the
concrete sample size that would shrink it. When the corpus does not (the
`runs_v1` branch), the same engine refuses to fit a one-class outcome model
and returns `unidentified` with `next_step.action = broaden_arm_support`.

The naive table and the honest verdict do not contradict each other here —
the headline is that `counterfact` produces a labeled estimate alongside the
descriptive baseline so consumers can tell which one to trust.
