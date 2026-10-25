# Routed trial: 800 real coding-agent traces

The harness ran a coding agent on four fixtures and randomized each run into one of two logged designs for its first `model_call`:

- `trial`: P(large) = 0.5 on every fixture. Its difference in pass rates is an unbiased estimate of the true effect.
- `fixture_router`: P(large) is 0.85 on `streaming_watermark_dedupe`, 0.7 on `date_window`, 0.3 on `rate_limit`, and 0.15 on `csv-stats`, so hard fixtures mostly get the large model.

Models: `claude-haiku-4-5-20251001` (small) and `claude-sonnet-4-6` (large). Run on 2026-10-25 for $6.52.

## Results

```bash
uv run python -m bench.real.analyze_routed_trial bench/real/routed_trial
```

Trial runs: 402. Routed runs: 398.

| Estimate | Source | Effect of large vs small | 95% interval |
| --- | --- | ---: | --- |
| Ground truth | trial, difference in means | +0.252 | [+0.184, +0.320] |
| Naive | routed, pass-rate gap | +0.053 | — |
| Engine (identified) | routed, AIPW | +0.324 | [+0.197, +0.451] |
| Engine (bounded) | routed, propensities removed | — | [-0.477, +0.523] (Manski) |

Engine minus truth: +0.072 (z = +0.98, independent samples). Naive minus truth: -0.199.

The naive gap is too small because the router sends the large model to the hard fixtures, where every arm passes less often. The engine's interval contains the trial truth. With the propensities removed, the engine returns `bounded`.

## Pass counts by fixture

| Fixture | Trial, small | Trial, large | Routed, small | Routed, large |
| --- | ---: | ---: | ---: | ---: |
| `streaming_watermark_dedupe` | 13/45 | 44/52 | 5/21 | 61/82 |
| `date_window` | 19/48 | 52/52 | 17/33 | 67/67 |
| `rate_limit` | 56/56 | 39/39 | 68/68 | 37/37 |
| `csv-stats` | 59/59 | 51/51 | 79/79 | 11/11 |

Both models pass `rate_limit` and `csv-stats` on every run, so the effect comes from the two harder fixtures.

`tests/acceptance/test_routed_trial_corpus.py` checks these conclusions on this corpus.
