# Counterfact Agent Context

This file is the deeper routing map for agents. `AGENTS.md` stays short; this file tells you what to read, what to keep synchronized, and what to validate when work touches a specific part of the repo.

## Product Posture

`counterfact` should make agent decision traces understandable through honest counterfactual questions: if a logged decision had been different, would the run have been more likely to complete the task? A descriptive pass-rate table can be decisive-looking and still not support a causal claim. Every change should preserve the distinction between observed associations, model predictions under declared interventions, support gaps, sensitivity bounds, and replay requirements.

## Start Here By Task

| Task touches | Read first | Usually update | Usually validate |
| --- | --- | --- | --- |
| Trace schema | `src/counterfact/schema/models.py`, `tests/unit/test_trace_schema.py` | Schema tests and any fixture JSON that intentionally exercises the contract | `uv run pytest tests/unit/test_trace_schema.py` |
| DAG construction | `src/counterfact/dag/graph.py`, `tests/unit/test_dag.py` | DAG tests and explain fixtures if graph output changes | `uv run pytest tests/unit/test_dag.py` |
| Outcome modeling or intervention semantics | `src/counterfact/outcome/`, `src/counterfact/intervene/`, `tests/unit/test_causal_engine.py`, `tests/unit/test_next_step.py` | Causal engine tests, next-step tests, acceptance gates if labels or bounds change | `uv run pytest tests/unit/test_causal_engine.py tests/unit/test_next_step.py tests/acceptance/test_v0_ship_gate.py` |
| Corpus analyzer | `src/counterfact/corpus_analyzer/`, `tests/unit/test_corpus_analyzer.py`, `tests/unit/test_cli_analyze.py`, `tests/acceptance/test_analyzer_self_test.py` | Analyzer tests and `bench/real/README.md` when promotion rules change | `uv run pytest tests/unit/test_corpus_analyzer.py tests/unit/test_cli_analyze.py tests/acceptance/test_analyzer_self_test.py` |
| Demo CLI | `src/counterfact/cli.py`, `tests/unit/test_cli_demo.py`, `tests/acceptance/test_demo_executes.py`, `docs/demo-excerpt.md` | README, demo excerpt, notebook builder, rebuilt notebook if displayed output changes | `uv run pytest tests/unit/test_cli_demo.py tests/acceptance/test_demo_executes.py` |
| Explain report | `src/counterfact/explain/`, `tests/unit/test_explain_report.py`, `tests/unit/test_explain_render_html.py`, `tests/acceptance/test_explain_cli.py` | Demo excerpt if the narrative or hidden-value policy changes | `uv run pytest tests/unit/test_explain_report.py tests/unit/test_explain_render_html.py tests/acceptance/test_explain_cli.py` |
| Synthetic benchmark | `bench/synthetic/`, `tests/unit/test_corpus_synthetic.py`, `tests/unit/test_scm_confounded.py`, `tests/acceptance/test_scm_recovery*.py` | README/demo docs if headline examples change | `uv run pytest tests/unit/test_corpus_synthetic.py tests/unit/test_scm_confounded.py tests/acceptance/test_scm_recovery.py tests/acceptance/test_scm_recovery_confounded.py` |
| Real harness | `bench/real/coding_agent/`, `bench/real/coding_agent/fixtures/README.md`, `tests/unit/test_corpus_real.py`, `tests/unit/test_corpus_real_hidden.py` | Fixture docs and analyzer docs when fixture sets or failure modes change | `uv run pytest tests/unit/test_corpus_real.py tests/unit/test_corpus_real_hidden.py` |
| SDK adapters and live tracers | `src/counterfact/adapters/`, `src/counterfact/tracing/`, `tests/unit/test_adapters_*.py`, `tests/unit/test_tracing_*.py`, `tests/acceptance/test_ingest_cli.py`, `tests/acceptance/test_adapter_diagnose_round_trip.py`, `tests/fixtures/adapters/` | README quickstart, the adapter table, and the `agent-sdk-adapters` capability spec when subcommands or live helpers change | `uv run pytest tests/unit/test_adapters_claude_agent_sdk.py tests/unit/test_adapters_openai_agents.py tests/unit/test_adapters_shape_drift.py tests/unit/test_tracing_claude_agent.py tests/unit/test_tracing_openai_processor.py tests/acceptance/test_ingest_cli.py tests/acceptance/test_adapter_diagnose_round_trip.py` |
| Committed real corpus | `bench/real/README.md`, `bench/real/smoke_mixed_outcome/PILOT_NOTES.md`, `tests/acceptance/test_analyzer_self_test.py`, `tests/acceptance/test_v0_ship_gate.py` | README, demo excerpt, notebook builder, rebuilt notebook, acceptance thresholds | `uv run counterfact analyze corpus bench/real/smoke_mixed_outcome` and `uv run python -m bench.real.analyze_pilot bench/real/smoke_mixed_outcome` |

## Demo And Corpus Contract

- `uv run counterfact demo` defaults to `bench/real/smoke_mixed_outcome/`. It must read committed traces only and must not import or run the paid real-agent harness.
- `bench/real/smoke_mixed_outcome/` is the promoted real demo corpus: 120 `streaming_watermark_dedupe` traces from the tightened `stateful_calibration` fixture set. The expected shape is mixed pass/fail outcomes in both `small` and `large` model arms, dominant hidden-semantic failures, zero patch-format dominance, and an `identified` demo result.
- `bench/real/single_class_refusal/` is the degenerate anchor. It is intentionally tiny and single-class; it should drive `unidentified`, not model fitting.
- Demo behavior changes must keep `README.md`, `docs/demo-excerpt.md`, `scripts/build_demo_notebook.py`, and `notebooks/demo.ipynb` aligned. Rebuild the notebook through the script rather than hand-editing notebook JSON.
- A strong demo should show the difference between a naive descriptive table and an honest causal estimate. Do not make the README claim the real corpus is a universal statistical headline.

## Real-Agent Benchmark Rules

- Paid runs require a `.counterfact/approved` JSON receipt matching the exact trace count, fixture set, output directory, budget cap, model map, and randomization config.
- Do not create `.counterfact/approved`, bypass the approval receipt, relax budget halting, or commit secrets.
- Ad hoc pilots belong in dated `bench/real/pilot_*` directories and are local artifacts unless the user explicitly asks to curate and commit a corpus.
- Promotion is human-gated. The analyzer reports; it does not rename directories. Use `bench/real/README.md` and `bench/real/coding_agent/fixtures/README.md` for the full promotion convention.
- Showcase quality is not just sample size. The key failure quality signal is runnable hidden-semantic failure, not extraction failure or unparseable patches.

## Validation Ladder

Start narrow, then widen when the change crosses boundaries.

1. Run the focused unit or acceptance tests from the routing table.
2. Run `uv run ruff check .` after Python changes.
3. Run `uv run pytest` before claiming completion for cross-module behavior, public API changes, demo/corpus changes, or anything that affects acceptance tests.
4. For committed corpus changes, also run `uv run counterfact analyze corpus bench/real/smoke_mixed_outcome`, `uv run python -m bench.real.analyze_pilot bench/real/smoke_mixed_outcome`, and `uv run counterfact demo`.

## Generated And Local Artifacts

- Never commit `.counterfact/`, provider credentials, checkpoints, caches, or ad hoc pilot directories unless the user explicitly asks for a curated corpus artifact.
- Keep committed corpus files intentional: `bench/real/smoke_mixed_outcome/` for the promoted demo corpus and `bench/real/single_class_refusal/` for the honest-refusal anchor.
- `bench/real/*/PILOT_NOTES.md` is useful provenance for promoted or reviewed pilots; do not treat it as a substitute for analyzer output.

## Documentation Style

- Prefer short, direct prose with no arbitrary hard wrapping. Markdown is prose, not Python.
- Keep command blocks copy-pasteable.
- Avoid inflated claims. Say what the repo actually does today.
- When removing detail from `README.md`, point to the deeper repo doc that now owns that detail.
