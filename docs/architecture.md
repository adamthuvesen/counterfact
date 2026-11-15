# Architecture

`counterfact` separates trace storage, analysis inputs, estimands, and presentation. Start with [usage.md](usage.md) to run an analysis. Use [invariants.md](invariants.md) when changing causal behavior.

## Ownership and dependencies

| Owner | Responsibility | Depends on |
| --- | --- | --- |
| `schema/` | Serialized `Run` contract and pure identity validation | Pydantic |
| `taxonomy/` | Decision specifications, interventions, and causal parent types | Schema |
| `analysis/` | Immutable corpus facts, chronology, outcomes, and feature extraction | Schema and taxonomy |
| `dag/` | Inspectable graph for one logged trace | Schema and taxonomy |
| `intervene/` | Focal interventions, corpus contrasts, identification checks, and estimators | Analysis, DAG, taxonomy, and shared logistic fitting |
| `outcome/` | Optional descriptive logistic fit | Analysis |
| `policy/` | Sequential policy estimands and estimation | Analysis, shared logistic fitting, and result contracts |
| `corpus_io.py` | Native file selection, receipts, publication, and recovery | Schema |
| `adapters/`, `tracing/` | Provider shape mapping and live capture | Schema and corpus IO |
| `runrecord_export.py` | Versioned usage derivation and Parquet export | Schema and corpus IO |
| CLI, diagnosis, explain | User-facing commands and result presentation | Domain APIs |

The schema owns decision-type names. The taxonomy gives each name one decision specification. Analysis owns chronology and feature extraction. Estimators and policy evaluation share the logistic fitter in `outcome/model.py`.

Corpus IO owns filesystem behavior outside the CLI. CLI loaders add stderr messages and exit codes. Analysis uses pure schema validation and never imports filesystem loaders.

## Analysis inputs

`AnalysisCorpus(runs, seed=0)` revalidates and snapshots the supplied traces. It rejects duplicate run IDs and duplicate numeric step indexes. It accepts binary outcomes and preserves the supplied population order.

Nested analysis facts are immutable. `corpus.runs` returns detached `Run` copies. Editing an original run or returned copy cannot change an existing analysis snapshot. DAGs also retain their own trace snapshot.

Chronology follows numeric `step_index`, even when the serialized step list has another order. A decision in the same step as treatment is never an adjustment covariate.

## Separate questions

| API | Question | Localization |
| --- | --- | --- |
| `intervene(dag=..., model=corpus, step=..., decision_id=...)` | Population contrast selected by a focal decision and baseline | Refuses repeated decisions of that type in the focal trace |
| `contrast(corpus=..., decision_type=..., intervention=..., baseline=...)` | Total effect of changing the first occurrence of that type | Keeps later occurrences downstream, with no focal trace |
| `fit_outcome_model(runs)` | Descriptive probability fit over all intervenable decision features | A fitted model is optional for causal queries |
| `evaluate_policy(runs, policy)` | Change the agent's rule at every controlled decision | Uses sequential history and reachable support |

The focal trace selects the decision type and default baseline. The returned contrast remains a population estimate over the supplied corpus. It does not estimate an individualized counterfactual from focal covariates. Library `intervene` permits an external focal trace. CLI and explain inputs require matching corpus content for artifact integrity.

An explicit decision ID resolves different decision types in one step. It cannot localize a repeated decision type. Attribution uses explicit IDs and can therefore evaluate mixed-type steps.

A corpus contrast requires distinct target and baseline arms. It excludes runs that never reach the treatment. Two first occurrences of the same type in one step leave treatment ambiguous and produce a refusal.

## Estimation and results

```text
Run -> AnalysisCorpus -> query-specific treatment and earlier history
                              |
                      identification check
                              |
             AIPW contrast / bounds / structured refusal
                              |
                        CausalEstimate
```

Causal queries construct their own treatment design and nuisance fits. They do not require a preliminary fit over every decision. `intervene` still accepts an `OutcomeModel` and uses its corpus snapshot.

`identified` requires checked assignment probabilities and positive probability for both arms. Missing support produces guidance. Missing randomization can produce Manski bounds. Some `bounded` questions have no justified interval.

Sampling intervals and identification bounds are separate fields. AIPW influence-function intervals remain unbounded by probability limits. Result models validate labels, payloads, interval order, and interval methods. Results are immutable, and `model_copy(update=...)` validates replacements.

## Publication and execution boundaries

[Corpus IO and format migration](migration.md) describes new-directory ingest, locked live append, receipt validation, and interrupted-write recovery.

`bench/` stays outside the wheel. It contains deterministic synthetic benchmarks and the paid real-agent harness. Generated Python executes in Docker with resource limits and no container network. The host performs provider calls. [The harness guide](../bench/real/README.md) describes approval, spend accounting, execution failures, and verifier limits.
