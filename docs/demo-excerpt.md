# Demo: Naive vs Causal

`counterfact demo` shows a descriptive pass-rate table and a causal estimate answering the same question. `notebooks/demo.ipynb` runs the same cases step by step.

## The confounded synthetic corpus

```bash
uv run counterfact demo --confound --synthetic-n 1000 --seed 42
```

The output is in the [README quickstart](../README.md#quickstart). The corpus comes from a structural causal model with a known true effect of +0.233 for sonnet against haiku.

The naive `pass_rate_by_arm` table reports a **+0.356** gap. The engine reports a causal contrast of **+0.255 [+0.179, +0.330]**, and its CI contains the truth. The naive gap is outside that CI.

The table is descriptive. `tool_choice` is a back-door confounder for `model_choice → outcome`, so the table does not support the causal claim. The engine can answer because every run logs the probability of the model it got. The engine checks that on the data before it reports `identified`. It then combines an outcome model over the pre-treatment `tool_choice` with inverse-propensity weighting (AIPW). The g-formula and IPW cross-checks agree (+0.251, +0.251).

## Hide the confounder

Drop the `tool_call` decision and every logged propensity from the same corpus. Most production agent traces look like this. The engine finds no logged randomization and returns `bounded` with Manski bounds of about [-0.33, +0.67]. The bounds contain the truth because they assume nothing. Notebook section 2c runs this case, and `tests/acceptance/test_scm_recovery_confounded.py` pins it.

## The real-trace corpus

```bash
uv run counterfact demo
```

Without `--confound`, the demo reads the committed real corpus, `bench/real/smoke_mixed_outcome/`: 120 `streaming_watermark_dedupe` traces with passes and failures in both model arms. Every failure is a valid patch that misses a hidden-test behavior, not a malformed patch. This is a check that the engine works on real traces, not a general benchmark result.

| arm | n | pass | pass rate | 95% CI |
| --- | ---: | ---: | ---: | --- |
| large | 65 | 50 | 0.769 | [0.654, 0.855] |
| small | 55 | 6 | 0.109 | [0.051, 0.218] |

`intervene(model_call: small -> large)` returns `identified` with `outcome_delta=+0.656 [+0.519, +0.792]`, and a `next_step` of `increase_n` that suggests about 896 traces to tighten the CI. The corpus assigns the model by a 50/50 coin flip, so the naive gap (+0.660) is unbiased here, and the AIPW contrast agrees with it.

## The single-class corpus

`bench/real/single_class_refusal/` holds three `csv_dedupe` traces that all pass. Pointed at it with `--runs-dir bench/real/single_class_refusal`, the engine refuses to fit a one-class outcome model. It returns `unidentified` with `next_step.action=broaden_arm_support`.

For per-trace analysis with `diagnose`, `explain`, and `intervene`, see [usage.md](usage.md).
