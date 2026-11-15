# Analysis and corpus format migration

Native `Run` JSON remains schema version `0.1.0`. This change adds analysis snapshots and versions ingest receipts and Parquet exports.

## Analysis APIs

Use `AnalysisCorpus` when a causal query needs validated traces:

```python
from counterfact import AnalysisCorpus, contrast
from counterfact.corpus_io import load_corpus
from pathlib import Path

runs = load_corpus(Path("corpus"))
corpus = AnalysisCorpus(runs, seed=42)

population = contrast(
    corpus=corpus,
    decision_type="model_call",
    intervention={"model_choice": "large"},
    baseline="small",
)
```

`contrast` targets the first occurrence of a decision type. Later occurrences remain downstream. Its result has `query.step=None` and requires an explicit baseline.

`intervene` accepts `model=corpus` for focal analysis. Existing `model=fit_outcome_model(runs)` callers still work. A descriptive global fit is unnecessary for causal analysis.

Different decision types in one step now support explicit decision targeting. Repeated occurrences of the same type still refuse focal localization. Use `contrast` to request the first-occurrence total effect without a focal trace.

Analysis rejects duplicate run IDs, duplicate numeric step indexes, and nonbinary outcomes. Focal CLI analysis also requires matching corpus content, even when the run IDs match. Library `intervene` accepts an external focal trace. That trace selects the question and baseline. The answer remains a population contrast over the supplied corpus.

Results are immutable. Build a replacement with `model_copy(update=...)`, which now validates changes. `identified` results require a distinct named baseline, a point estimate, and estimator diagnostics. `bounded` results may have no interval. Sequential policy intervals use `ci_method="sequential_dr_influence_function"`. If a target or baseline has no matching trajectories, `diagnostics.snis_delta` is null. This replaces the previous invalid JSON NaN behavior. The sequential DR estimand and estimate are unchanged.

Internal outcome and feature helpers moved from `outcome/` to `analysis/`. Public `fit_outcome_model` remains available.

## Ingest publication

Offline ingest creates a new destination. An existing directory produces an error, including an empty directory. Ingest one source batch into one new destination.

Live tracers explicitly append through the shared corpus writer. One writer holds the lock. Duplicate run IDs and existing output filenames produce errors. Receipt counts come from validated committed files.

The writer rejects different source formats and unreceipted legacy corpora during append. Legacy corpora remain readable. Use a new destination for new capture.

Readers validate `ingest-receipt.json` and its run count. They recognize export receipts beside their corresponding artifacts. Native run schemas take precedence over receipt-like filenames. Other JSON files must parse as native runs. A malformed receipt or unexpected JSON file stops the load.

An interrupted batch leaves `.counterfact-write-in-progress`. Readers refuse that corpus. Stop the writer, then roll back the batch:

```python
from pathlib import Path
from counterfact.corpus_io import recover_corpus

recover_corpus(Path("corpus"))
```

Recovery removes only the batch's planned run files and restores the previous receipt. Prior committed runs remain available. The lock prevents recovery while a writer owns the corpus. The writer uses atomic file replacement and a batch marker. This is a local filesystem protocol.

New ingest receipts carry `receipt_version="2"`. Receipts without that field remain readable when their content and counts are valid. `write_corpus` returns a committed receipt copy and leaves the supplied receipt unchanged.

## Parquet export version 3

New export receipts carry `export_version="3"`. The Parquet file now loads with eval-audit's `load_run_records`. A copy of eval-audit's `docs/run-record.schema.json` in `tests/fixtures/` pins the contract.

| Field | Version 3 behavior |
| --- | --- |
| `model_ids`, `model_identity_provenance`, `tokens_*_provenance`, `cost_sources` | Removed as columns. `rerun_metadata` keeps `model_ids`, `cost_sources` (JSON lists), `tokens_in_source`, and `tokens_out_source` |
| `tokens_in_by_model`, `tokens_out_by_model` | A struct keyed by model id. A null field means the model's usage is unknown |
| `tokens_in`, `tokens_out` | `0` when untracked, which is eval-audit's documented value. The token source is `not_available` |
| `outcome_status` | Always `graded`. Runs with nonbinary outcomes are excluded and counted |
| `task_id` | `metadata.extra.task_id`, else `outcome.metadata.fixture_id`. A run with neither stops the export |
| `model_id` | Required. A run without a `model_call` stops the export |
| `timestamp` | A UTC datetime parsed from an ISO 8601 string with an offset |
| `cost_provenance` | An eval-audit value. `rerun_metadata.counterfact_cost_provenance` keeps the counterfact value |

| counterfact cost provenance | eval-audit `cost_provenance` | Reason |
| --- | --- | --- |
| `reconciled` | `reconciled` | Reported and reconstructed costs agree |
| `discrepant` | `as_reported_only` | Reconstruction does not reconcile, so the reported total is the basis. The reconstructed cost exports as null |
| `reported` | `as_reported_only` | Only the reported total exists |
| `reconstructed` | `partial` | No reported total exists to reconcile against, so provenance is incomplete |
| `cost_not_available` | `cost_not_available` | No cost data |

An export with no binary-outcome runs fails. Version 2 wrote an empty file.

## Parquet export version 2

New export receipts carry `export_version="2"` and migration notes. Legacy receipts without the field remain readable.

| Field | Version 2 behavior |
| --- | --- |
| `model_id` | First model-call chosen action, preserving the existing identity even when it names an arm |
| `model_ids` | Distinct logged model identities across the run |
| `model_identity_provenance` | Distinguishes legacy `model_id` from provider identities in `model_ids` |
| `tokens_in`, `tokens_out` | Null when unavailable, replacing fabricated zero counts |
| `tokens_in_by_model`, `tokens_out_by_model` | Lists of `{model_id, tokens}` records, replacing agent-keyed structs |
| `reported_run_total_cost_usd` | Logged run total, or complete provider model totals |
| `reconstructed_per_task_cost_usd` | Unique observed task costs, requiring complete model-call cost coverage |
| `cost_provenance`, `cost_sources` | Identify reported, reconstructed, reconciled, discrepant, or unavailable costs |
| `success`, `outcome_status` | Null and `ungraded` for nonbinary outcomes |

Token totals prefer explicit run totals, then provider model totals, then complete call usage. The exporter never adds provider totals to call totals. Missing model-level tokens remain null.

Harness costs come from `observation.content.cost_usd`. Claude usage comes from result metadata. OpenAI usage comes from model-call decision metadata. Costs remain separate when both reported and reconstructed values exist. Reconstruction associates each model call with one cost observation. An explicit `Decision.metadata.call_id` matches `Observation.content.call_id`. Without that link, only a step containing one decision and one cost observation is unambiguous. Tool-only costs cannot complete model-call coverage. Once coverage is established, reconstruction sums all unique known cost observations, including tool costs. A null cost observation or duplicate cost identity makes the aggregate unknown. Missing or ambiguous model-call associations also produce a null total and a warning. This reports observed task costs without claiming provider billing reconciliation or complete coverage of unlogged tool costs.

Provider model totals must cover every known called model before they become aggregate tokens or cost. Missing models retain null per-model usage. Explicit provider run totals remain usable when the per-model breakdown is incomplete.

The Parquet schema is explicit and stable for empty corpora, multiple models, and missing usage. Existing export destinations produce an error. Choose a new output path when exporting again.

## Real-agent checkpoints

The real-agent harness uses its own append-only spend ledger. New entries have version 2 and a call ID. Completed traces with that call ID are counted once. Legacy ledger entries remain conservative additional spend.

A pending provider call with unknown cost blocks resume. Resolve the actual provider charge before resuming. Do not replace unknown cost with zero. Infrastructure attempts are recorded separately and do not become completed corpus rows.
