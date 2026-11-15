# AGENTS.md - counterfact

`counterfact` answers causal questions about agent traces. Missing support must remain visible. A useful `unidentified` result is better than a fabricated counterfactual.

Domain terms are in `GLOSSARY.md`. A change that adds, renames, or redefines a term updates it in the same commit.

## Start here

- [docs/repo-context.md](docs/repo-context.md) routes changes to owners and checks.
- [docs/architecture.md](docs/architecture.md) defines dependency direction and estimands.
- [docs/invariants.md](docs/invariants.md) governs causal and statistical behavior.
- [docs/migration.md](docs/migration.md) records analysis and corpus format compatibility.
- [docs/usage.md](docs/usage.md) documents public commands and APIs.

```bash
make install
make check   # fast gate
make fix     # ruff --fix and ruff format
make verify  # drive the real CLI (ARGS=all|list|control|<drive>)
make slow    # full gate: coverage, generated artifacts, wheel, Docker, verify all + control
```

Prove changes with `make verify` (`verify-counterfact` skill at `.agents/skills/verify-counterfact/`).

The Makefile defines every gate. Mark a test `slow` only when it is integration-shaped and a cheaper setup cannot replace it. Spend and approval gates stay in `make check`.

## Enforced Rules

| Rule | Enforced by |
| --- | --- |
| Scientific code does not import presentation, CLI, provider, corpus IO, or bench modules | `pyproject.toml::tool.importlinter.contracts` |
| Domain modules import only downward (cli, compare, diagnose, explain, attribute, intervene, outcome, analysis, taxonomy, schema) | `pyproject.toml::tool.importlinter.contracts` |
| Provider integrations (tracing, adapters) sit above corpus IO above schema | `pyproject.toml::tool.importlinter.contracts` |
| Parse run files only through `corpus_io.load_run` or `load_corpus` | `pyproject.toml::tool.ruff.lint.flake8-tidy-imports.banned-api` |
| Only named owner files import process, network, or paid-model modules | `tests/test_process_imports.py::test_only_owner_files_import_process_network_or_paid_modules` |
| Write run files only through `corpus_io.write_corpus` (exception: `bench/real/coding_agent/runner.py`) | `tests/test_corpus_writes.py::test_run_json_is_not_written_by_hand` |
| Library modules import without `bench/`, `scripts/`, or `tests/` | `tests/test_imports.py::test_library_imports_without_checkout_only_modules` |
| The wheel holds `src/counterfact` only and works outside the checkout | `scripts/check_wheel.py` |
| Generated artifacts match their builders | `Makefile::check-generated` |
| Generated code cannot reach the host or mutate the verifier (`make check-container`) | `tests/acceptance/test_container_executor.py::test_generated_code_cannot_reach_host_or_mutate_verifier` |
| Real-agent spend halts at 80% of the budget | `tests/unit/test_corpus_real.py::test_budget__halt_at_80_percent` |
| Unknown paid-call charges block resume | `tests/unit/test_corpus_real.py::test_unknown_paid_call_cost_blocks_resume` |
| Agents never create or edit `.counterfact/approved`. Stop and ask Adam to approve the spend | `.claude/settings.json::hooks.PreToolUse`, `.codex/hooks.json::hooks.PreToolUse`, `tests/unit/test_approval_guard.py::test_client_hook_denies_with_the_stop_message` |
| The approval receipt must match the budget before any API call | `tests/unit/test_corpus_real.py::test_approval_receipt_must_match_budget_before_any_api_call` |
| Same-step decisions never enter adjustment sets | `tests/unit/test_analysis_contract.py::test_same_step_decisions_are_not_adjustment_features` |
| Repeated decisions of one type refuse focal localization | `tests/unit/test_causal_engine.py::test_intervene__repeated_decision_type_reports_localization_limit` |
| `export-runs` writes exactly the columns, types, and enum values of eval-audit's `RunRecord` (copy of its `docs/run-record.schema.json`) | `tests/unit/test_runrecord_export.py::test_parquet_conforms_to_eval_audit_across_models_and_missing_usage` |
| `verify control` must observe FAIL before a verify pass counts | `.agents/skills/verify-counterfact/verify::control` |
| Tests do not sleep or read the clock, and every skip or xfail gives a reason | `tests/ruff.toml::lint.flake8-tidy-imports.banned-api`, `tests/test_strict_mode.py::test_every_skip_gives_a_reason` |

## Design boundaries

- Keep pure trace validation in `schema/`. Put immutable analysis facts, chronology, and features in `analysis/`.
- Use `AnalysisCorpus` for causal inputs. Keep `fit_outcome_model` descriptive and optional.
- Preserve the distinction between a focal `intervene` query, a first-occurrence `contrast`, and a sequential policy estimand.
- Write native corpora with `corpus_io.write_corpus`. The one exception is `bench/real/coding_agent/runner.py`: it publishes each graded trace at once so resume can count spend. CLI loaders only add user-facing errors.
- Read `docs/invariants.md` before changing estimation, attribution, outcomes, or explanation.
- Keep the project narrow. Name a concrete caller before adding compatibility shims or abstractions.
- Ask before adding runtime dependencies. Broad causal or agent frameworks are outside this library's scope.

The wheel contains `src/counterfact` only. `bench/` and real-agent execution require a checkout.

## Causal honesty

`pass_rate_by_arm()` is descriptive. `identified` requires logged assignment probabilities and checked support. Sampling intervals and identification bounds are separate. A `bounded` result may have no justified interval. AIPW intervals must not be clipped to probability limits.

Mixed decision types in one step can use explicit decision IDs.

## Real-agent work

Read [bench/real/README.md](bench/real/README.md) before touching execution, approvals, or spend accounting. The harness can make paid provider calls. Docker checks do not.

- Obtain approval before generating real traces. State the exact command, fixtures, destination, and budget cap.
- Execute generated code through the Docker verifier. Never add a host-execution fallback.
- Keep infrastructure failures separate from graded task failures.

Never commit credentials, `.env`, `.counterfact/`, checkpoints, or ad hoc pilot outputs. Keep corpus promotion explicit.

## Documentation and artifacts

Use the Makefile's generation targets for derived artifacts. Keep numeric tables in their generated owners and link them from prose. Document interpretation and limitations where judgment is required.

Do not claim structural counterfactuals, learned DAGs, provider replay guarantees, calibrated universal probabilities, or a verifier that prevents candidate cheating. Docker limits host exposure. Candidate code and tests still share the verifier workload.
