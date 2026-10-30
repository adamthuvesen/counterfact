# Repository context

Use [architecture.md](architecture.md) for ownership and dependency direction. Read [invariants.md](invariants.md) before changing analysis, estimation, or result presentation. [migration.md](migration.md) records compatibility changes.

## Find the owner

| Change | Start here | Focused evidence |
| --- | --- | --- |
| Trace contract or identities | `schema/models.py`, `schema/validation.py` | `test_trace_schema.py`, `test_analysis_contract.py` |
| Analysis snapshots, chronology, features | `analysis/` | `test_analysis_contract.py`, causal-engine tests |
| Decision semantics or graph | `taxonomy/types.py`, `dag/graph.py` | Taxonomy, DAG, and causal-engine tests |
| Focal intervention or corpus contrast | `intervene/api.py`, `intervene/dispatch.py`, `intervene/estimators.py` | Causal-engine, next-step, and ship-gate tests |
| Descriptive outcome fit | `outcome/model.py` | Outcome-model and benchmark tests |
| Policy estimand | `policy/evaluate.py`, `bench/synthetic/sequential.py` | Policy-evaluation and policy-suite tests |
| Corpus files, receipts, export | `corpus_io.py`, `runrecord_export.py` | Corpus-IO, export, and adapter round-trip tests |
| Provider shape mapping | `adapters/`, `tracing/` | Adapter, parity, shape-drift, and live-tracer tests |
| Analysis readiness | `corpus_analyzer/` | Analyzer, CLI-analyze, and analyzer-self-test tests |
| Diagnosis or explanation | `attribute/`, `diagnose.py`, `explain/` | Diagnosis, explain, and HTML-render tests |
| Real-agent execution or spend | `bench/real/coding_agent/` | Runner tests and explicit container checks |
| Derived docs or package boundary | `scripts/`, `Makefile` | Generated checks and installed-wheel checks |

Paths in the owner column are under `src/counterfact/` unless they start with `bench/` or `scripts/`. Find the matching test filename before running it. `make check` defines the fast offline validation command. `make verify` adds full-suite coverage, generated artifacts, wheel consumers, and Docker execution checks.

Run focused checks while editing. Run the shared gates before reporting the whole change complete. The container gate must fail when Docker or the verifier image is unavailable. An offline test run does not prove container behavior.

## Corpus roles

- `bench/real/smoke_mixed_outcome/` is the committed real demo corpus. The default demo reads it without provider calls.
- `bench/real/single_class_refusal/` is the single-class refusal anchor.
- `bench/real/routed_trial/` contains the committed routed experiment. Its analysis compares routed estimates with an independent randomized arm.
- `bench/synthetic/` defines deterministic causal models and known truths. It requires a checkout.

Keep pilot outputs and checkpoints local unless Adam explicitly asks to curate a corpus. The analyzer reports readiness and never promotes or renames a directory.

## Derived artifacts

Use the generation and check targets in `Makefile`. Benchmark tables derive from `docs/benchmark-results.json`. Figures derive from that JSON. The notebook derives from `scripts/build_demo_notebook.py`. Update their owners and regenerate outputs instead of editing generated files.

Documentation explains interpretation and limitations. Keep detailed numeric tables in generated outputs. Link those outputs where a repeated table would require manual synchronization.

## Paid runs

Read [the real-agent guide](../bench/real/README.md) before changing execution, approval, checkpoint identity, or spend accounting. Docker verification is local and makes no model calls. `counterfact bench real` can spend money and requires explicit approval for the command.

The Enforced Rules table in [AGENTS.md](../AGENTS.md) governs `.counterfact/approved`. Never use a paid real-agent run as routine validation. Provider credentials, `.counterfact/`, checkpoints, and ad hoc pilots stay outside commits.
