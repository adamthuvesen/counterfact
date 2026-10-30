# Real-agent harness

`counterfact bench real` runs a coding agent on fixed Python repair tasks and writes native traces. Model, tool, and retry policies log their assignment probabilities. Analysis still checks support before making causal claims.

This command makes external provider calls and can spend money. Obtain approval for the exact command before running it. `.counterfact/approved` must cover the requested trace count and match the budget, destination, fixtures, model map, and randomization configuration. Agents must never create this receipt for the user.

The wheel excludes `bench/`. Use a checkout with the `bench` extra for every command on this page.

## Execution prerequisites

Generated Python runs in Docker. Fixture files and the Docker runtime probe are checked before credential lookup or client creation. The host performs provider calls.

Build and verify the pinned image without provider calls:

```bash
uv run --locked python scripts/check_container.py --build
```

The complete gate builds the image, reusing Docker's cache, and runs the checks:

```bash
make check-container
```

Use `uv run --locked python scripts/check_container.py` to check an existing image without building it.

The verifier supports native Linux arm64 and amd64 containers. Its image recipe is pinned, and its tag derives from the recipe contents. Preflight checks Python, pytest, user ID, locale, and timezone.

Docker disables container networking, uses a read-only root filesystem, removes capabilities, and runs an unprivileged user. Source and tests enter through read-only mounts. Writable work uses temporary filesystems with resource limits.

Each execution receives a fresh source snapshot. The verifier checks its manifest and file digests before running tests. A byte mismatch produces an infrastructure error. Reference implementations live under `tests/fixtures/coding_agent/references/`, outside runtime fixture sources.

Host output capture stops at its byte limit. Docker logging is disabled. Output-limit failures carry `failure_reason="output_limit"` and remain distinct in pilot analysis.

This boundary limits host exposure. It does not establish the execution environment of historical committed corpora. Candidate code and tests share the pytest workload. A candidate can inspect mounted tests or interfere with verification, so passing tests do not prove resistance to cheating.

Docker startup, cleanup, or verifier failures stop the run as infrastructure errors. These attempts stay under `.checkpoints/attempts/` and do not become completed corpus rows. Task test failures and workload timeouts remain graded outcomes.

## Spend and resume

The budget halts at 80% of the approved cap. Actual provider charges can arrive after a call, so the halt is conservative accounting rather than a guaranteed billing ceiling.

The append-only ledger records a pending call before provider execution and a priced entry after cost becomes known. A call ID prevents double counting between new ledger entries and completed traces. Legacy entries remain conservative additional spend.

An unknown charge stops the run and blocks resume. Reconcile the provider charge before continuing. Missing or malformed ledger records never become zero-cost calls.

One runner owns the output lock. Resume checks fixture, model, seed, and randomization identity. A different configuration requires a new destination. Completed trace files publish atomically. Infrastructure attempts and incomplete calls remain separately visible.

## Corpora and fixtures

| Name | Purpose |
| --- | --- |
| `smoke_mixed_outcome/` | Committed real demo corpus with both outcomes in both model arms |
| `single_class_refusal/` | Single-class anchor that must refuse causal estimation |
| `routed_trial/` | Committed routed experiment with an independent randomized comparison |
| `broad_calibration` | Fixture set containing `date_window`, `rate_limit`, and `version_range` |
| `stateful_calibration` | Fixture set containing `streaming_watermark_dedupe` |

Fixture specifications live in [coding_agent/fixtures/README.md](coding_agent/fixtures/README.md). Corpus notes record historical provenance. [Routed trial results](routed_trial/RESULTS.md) hold the current reanalysis.

## Approved corpus command

After approval, this command runs a stateful corpus:

```bash
uv run --locked counterfact bench real --n 60 \
  --fixture-set stateful_calibration \
  --model-epsilon 1.0 --tool-epsilon 0.0 --retry-epsilon 0.0 \
  --budget-cap 10 \
  --output-dir bench/real/pilot_stateful
```

These inspection commands make no provider calls:

```bash
uv run --locked counterfact analyze corpus bench/real/pilot_stateful
uv run --locked python -m bench.real.analyze_pilot bench/real/pilot_stateful
```

The analyzer reports support readiness. The pilot analysis distinguishes hidden semantic failures, malformed patches, and harness errors. Promotion remains an explicit curation decision. Pilot directories and checkpoints stay local.

## Routed design

`--model-route-by-fixture` randomizes runs between two designs for the first model call:

- `trial` assigns `large` with probability 0.5 on each fixture.
- `fixture_router` uses logged fixture-specific probabilities. Different fixture difficulty can confound the marginal pass-rate gap.

After separate approval, this command reproduces the design with new provider calls:

```bash
uv run --locked counterfact bench real --n 800 \
  --fixtures streaming_watermark_dedupe,date_window,rate_limit,csv-stats \
  --model-route-by-fixture streaming_watermark_dedupe=0.85,date_window=0.7,rate_limit=0.3,csv-stats=0.15 \
  --tool-epsilon 0.0 --retry-epsilon 0.0 \
  --budget-cap 18 \
  --output-dir bench/real/pilot_routed_trial
```

Reanalyze the committed corpus without provider calls:

```bash
uv run --locked python -m bench.real.analyze_routed_trial bench/real/routed_trial
```

The analysis compares a first-occurrence corpus contrast with the trial estimate. Removing logged propensities must remove point identification. The route map must cover exactly the selected fixtures, with probabilities strictly between zero and one. [The report](../../docs/report.md#6-real-agent-trial) explains the statistical limits.
