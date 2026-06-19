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
outcome_delta: 0.663 [0.617, 0.707]
next_step: none - CI width 0.090 ≤ 0.10; no further action required.
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
committed real corpus, `bench/real/smoke_mixed_outcome/` (120
`streaming_watermark_dedupe` traces with mixed pass/fail outcomes in both model
arms). It is intentionally small — a smoke test that the engine works on
real-agent traces, not the statistical headline. The failures are hidden
stateful-semantic misses, not patch-format misses, and the report is honest
about the remaining CI width.

| arm | n | pass | pass rate | 95% CI |
| --- | ---: | ---: | ---: | --- |
| large | 65 | 50 | 0.769 | [0.654, 0.855] |
| small | 55 | 6 | 0.109 | [0.051, 0.218] |

`intervene(model_call -> large)` on the same corpus returns
`identifiability=identified` with `outcome_delta=0.747 [0.630, 0.831]` and a
`next_step.action="increase_n"` recommending ~830 traces to tighten the CI.

## The single-class regression anchor

A small companion corpus, `single_class_refusal` (3 single-class traces from
the original `csv_dedupe` pilot), is committed as the regression anchor for
the engine's "honest refusal" branch. Pointed at it
(`--runs-dir bench/real/single_class_refusal`), the same engine refuses to fit
a one-class outcome model and returns `unidentified` with
`next_step.action=broaden_arm_support`. That refusal is itself the feature.

## Trace forensics

For a single trace, start with `counterfact diagnose <run-json> --html report.html`.
It writes a diagnosis-first HTML report that answers the practical question:
where did this run most plausibly go wrong, and can the corpus honestly support
that counterfactual? The report includes the same descriptive
`pass_rate_by_arm` table, per-trace DAG, trace timeline, decision cards with
support/replay warnings, and `CausalEstimate` cards used by `counterfact explain`.
Cards are colour-coded by
`IdentifiabilityStatus`, and numeric `outcome_delta` blocks are structurally
suppressed when a card is `unidentified` so the report cannot emit an estimate
the data does not support. Pointing it at a single-class run (e.g. anything in
`single_class_refusal`) renders exactly one `unidentified` card with the
`broaden_arm_support` next-step callout and no point estimate anywhere on the
page.

For a reusable artifact, `counterfact intervene <run-json> --decision-id <id>
--set model_choice=large --json` emits the raw `CausalEstimate` JSON for one
decision edit. That is the trace-forensics loop: use `diagnose` to find the
candidate decision, use `intervene` to ask a precise causal question, then read support,
replay, or missing-arm diagnostics when the trace corpus cannot answer it.
The same pattern covers wrong model choice, bad tool choice, missed retry,
stopped too early, and unsupported intervention scenarios.
