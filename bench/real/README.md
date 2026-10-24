# Real-agent harness

`counterfact bench real` runs a coding agent on small Python repair tasks and writes one trace per run. The agent's model, tool, and retry choices are randomized with logged propensities, so the traces support causal estimates. The tasks are described in [coding_agent/fixtures/README.md](coding_agent/fixtures/README.md).

The harness calls LLM APIs and costs money. It refuses to start until `.counterfact/approved` holds a JSON approval receipt that matches the exact command: trace count, budget cap, output directory, fixture set, model map, and randomization config. If the receipt is missing or stale, the command prints the JSON to write. The run halts at 80% of its budget cap.

## Contents

| Name | Kind | Purpose |
| --- | --- | --- |
| `smoke_mixed_outcome/` | corpus | The real-trace demo corpus. |
| `single_class_refusal/` | corpus | Checks that the engine refuses a corpus with one outcome class. |
| `broad_calibration` | fixture set | `date_window`, `rate_limit`, `version_range`. |
| `stateful_calibration` | fixture set | `streaming_watermark_dedupe`, a harder stateful task. |

- **`smoke_mixed_outcome/`** holds 120 `streaming_watermark_dedupe` traces from the `stateful_calibration` fixture set, with the model randomized 50/50 (`--model-epsilon 1.0 --tool-epsilon 0.0 --retry-epsilon 0.0`). The pass rate is 0.467, and both model arms have passes and failures. All 64 failures are valid patches that miss a hidden-test behavior, not malformed patches. `counterfact demo` reads this corpus.
- **`single_class_refusal/`** holds three `csv_dedupe` traces that all pass. `counterfact analyze corpus` must never accept it, and `intervene` must return `unidentified` on it.

## Run a corpus

```bash
counterfact bench real --n 60 \
  --fixture-set stateful_calibration \
  --model-epsilon 1.0 --tool-epsilon 0.0 --retry-epsilon 0.0 \
  --budget-cap 10 \
  --output-dir bench/real/pilot_stateful
counterfact analyze corpus bench/real/pilot_stateful
python -m bench.real.analyze_pilot bench/real/pilot_stateful
```

`analyze corpus` checks whether the corpus has enough outcome variation and arm support for causal estimates. `analyze_pilot` breaks the failures down by mode. A useful corpus has failures in both model arms that come from task behavior, not from malformed patches or harness errors. Output under `bench/real/pilot_*` is gitignored. Committing a corpus is a manual step.

## Routed trial

`--model-route-by-fixture` turns one command into an experiment with its own ground truth. Each run is randomized into one of two logged designs for the first `model_call`:

- `trial`: P(large) = 0.5 on every fixture. Its difference in pass rates is an unbiased estimate of the true effect.
- `fixture_router`: P(large) is set per fixture. Fixtures differ in difficulty, so a naive comparison of these runs is confounded, as with a production router that escalates hard tasks.

```bash
counterfact bench real --n 800 \
  --fixtures streaming_watermark_dedupe,date_window,rate_limit,csv-stats \
  --model-route-by-fixture streaming_watermark_dedupe=0.85,date_window=0.7,rate_limit=0.3,csv-stats=0.15 \
  --tool-epsilon 0.0 --retry-epsilon 0.0 \
  --budget-cap 18 \
  --output-dir bench/real/pilot_routed_trial
python -m bench.real.analyze_routed_trial bench/real/pilot_routed_trial
```

The analysis compares the naive gap and the engine's AIPW estimate on the routed runs against the trial truth. It then removes the propensities from the routed runs and checks that the engine refuses. The route map must name exactly the run's fixtures, with every probability strictly between 0 and 1. [Report §6](../../docs/report.md#6-real-agent-ground-truth) explains the design.
