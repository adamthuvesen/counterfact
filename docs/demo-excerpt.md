# Demo Excerpt: Naive vs Causal

The canonical `counterfact` demo is the **confounded synthetic showcase**:
a deterministic, no-spend corpus where the descriptive `pass_rate_by_arm`
baseline overstates what the data supports and the engine's AIPW estimate
recovers the true do-calculus arm gap. The synthetic SCM is the right surface
to teach the project's product stance: ground truth, controlled confounding,
deterministic seeds, and a structure where the marginal table and the causal
estimate disagree on purpose.

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

## What the contrast says

The naive `pass_rate_by_arm` table reports a **+0.356** arm gap between sonnet and haiku. The engine reports a causal contrast of **+0.255 [+0.179, +0.330]**, and its CI contains the SCM's known do-calculus truth (`HEADLINE_TRUE_EFFECT ≈ 0.233`). The naive gap is outside that CI.

The table is descriptive. It does not license the causal claim, because `tool_choice` is a back-door confounder for `model_choice → outcome`. The engine can answer because the assignment policy logs its propensities: every run records the probability of the model it got. The engine checks that on the data before it reports `identified`. It then combines an outcome model over the pre-treatment `tool_choice` with inverse-propensity weighting (AIPW). The g-formula and IPW cross-checks agree (+0.251, +0.251), so neither model is visibly misspecified.

## Hide the confounder

Drop the `tool_call` decision and every logged propensity from the same corpus. That is what most production agent traces look like. The engine finds no logged randomization and returns `bounded` with Manski bounds of about [-0.36, +0.64]. Before data-checked identification, this setup returned `identified` with a biased gap (0.312 against a truth of 0.233 at n=2000, seed=1). The bounds contain the truth because they assume nothing. `notebooks/demo.ipynb` section 2c runs this case, and `tests/acceptance/test_scm_recovery_confounded.py` pins it.

## The real-trace smoke test

Without `--confound`, `counterfact demo` runs the same flow against the committed real corpus, `bench/real/smoke_mixed_outcome/` (120 `streaming_watermark_dedupe` traces with mixed pass/fail outcomes in both model arms). It is a smoke test that the engine works on real-agent traces, rather than a statistical headline. The failures are hidden stateful-semantic misses rather than patch-format misses, and the report is honest about the remaining CI width.

| arm | n | pass | pass rate | 95% CI |
| --- | ---: | ---: | ---: | --- |
| large | 65 | 50 | 0.769 | [0.654, 0.855] |
| small | 55 | 6 | 0.109 | [0.051, 0.218] |

`intervene(model_call: small -> large)` on the same corpus returns
`identifiability=identified` with `outcome_delta=+0.656 [+0.519, +0.792]` and a
`next_step.action="increase_n"` recommending ~830 traces to tighten the CI.
The corpus assigns the model by a 50/50 coin flip, so the naive gap (+0.660)
is itself unbiased here, and the AIPW contrast agrees with it.

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
