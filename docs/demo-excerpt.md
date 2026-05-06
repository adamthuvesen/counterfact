# Demo Excerpt: Naive vs Causal

The canonical `counterfact` demo is the **confounded synthetic showcase** —
a deterministic, no-spend corpus where the descriptive `pass_rate_by_arm`
baseline overstates what the data supports and the engine's g-formula
adjustment recovers the true do-calculus arm gap. The synthetic SCM is the
right surface to teach the project's product stance: ground truth, controlled
confounding, deterministic seeds, and a structure where the marginal table
and the causal estimate disagree on purpose.

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

## What the contrast says

The naive `pass_rate_by_arm` table reports a **+0.356** arm gap between
sonnet and haiku. The engine's g-formula adjustment, working off the same
corpus, reports a **+0.251** causal arm gap — and that number matches the
SCM's known do-calculus headline (`HEADLINE_TRUE_EFFECT ≈ 0.233`) within the
project's `±0.05` recovery tolerance. The marginal table is descriptive and
limited; it does not, by itself, license the causal claim because
`tool_choice` is a back-door confounder for the `model_choice → outcome`
relationship and the descriptive view does not adjust for it. The engine
does, because `tool_choice` is in the outcome model's feature set and the
g-formula marginalizes over the empirical `(tool, retry)` distribution.

That is the product stance in one comparison: when the corpus supports the
question and the back-door variables are observed, `counterfact` returns a
labelled estimate with a bootstrap CI; when the descriptive view and the
causal estimate disagree, the disagreement is the diagnostic.

## The real-trace smoke test

Without `--confound`, `counterfact demo` runs the same flow against the
committed real corpus, `bench/real/runs_v2/` (30 `date_window` traces,
mixed-outcome from inverted-greedy randomization). It is intentionally
small — a smoke test that the engine works on real-agent traces, not the
statistical headline. The CI is wide and the naive arm gap and the engine's
estimate happen to point the same direction; the report is honest about that
shape.

| arm | n | pass | pass rate | 95% CI |
| --- | ---: | ---: | ---: | --- |
| large | 8 | 8 | 1.000 | [0.676, 1.000] |
| small | 22 | 6 | 0.273 | [0.132, 0.482] |

`intervene(model_call -> small)` on the same corpus returns
`identifiability=identified` with `outcome_delta=0.332 [0.179, 0.493]` and a
`next_step.action="increase_n"` recommending ~416 traces to tighten the CI.

## The single-class regression anchor

A small companion corpus, `runs_single_class` (3 single-class traces from
the original `csv_dedupe` pilot), is committed as the regression anchor for
the engine's "honest refusal" branch. Pointed at it
(`--runs-dir bench/real/runs_single_class`), the same engine refuses to fit
a one-class outcome model and returns `unidentified` with
`next_step.action=broaden_arm_support`. That refusal is itself the feature.

## Per-Trace HTML Report

For a single trace, `counterfact explain <run-json>` renders a
self-contained HTML report that mirrors this naive-vs-causal framing: the
descriptive `pass_rate_by_arm` table at the top, the per-trace DAG inline
as SVG with the top-attributed decision highlighted, and one
`CausalEstimate` card per ranked decision below. Cards are colour-coded by
`IdentifiabilityStatus`, and numeric `outcome_delta` blocks are
structurally suppressed when a card is `unidentified` so the report cannot
emit an estimate the data does not support. Pointing it at a single-class
run (e.g. anything in `runs_single_class`) renders exactly one
`unidentified` card with the `broaden_arm_support` next-step callout and no
point estimate anywhere on the page.
