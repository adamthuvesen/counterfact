# AGENTS.md - counterfact

`counterfact` answers causal questions about agent traces. Missing support must remain visible. A useful `unidentified` result is better than a fabricated counterfactual.

## Start here

- [docs/repo-context.md](docs/repo-context.md) routes changes to owners and checks.
- [docs/architecture.md](docs/architecture.md) defines dependency direction and estimands.
- [docs/invariants.md](docs/invariants.md) governs causal and statistical behavior.
- [docs/migration.md](docs/migration.md) records analysis and corpus format compatibility.
- [docs/usage.md](docs/usage.md) documents public commands and APIs.

```bash
make install
make ci
make verify
```

`make ci` is the shared offline gate. `make verify` also checks generated artifacts, installed-wheel consumers, and Docker execution. Read the Makefile for their exact commands.

## Design boundaries

- Keep pure trace validation in `schema/`. Put immutable analysis facts, chronology, and features in `analysis/`.
- Use `AnalysisCorpus` for causal inputs. Keep `fit_outcome_model` descriptive and optional.
- Preserve the distinction between a focal `intervene` query, a first-occurrence `contrast`, and a sequential policy estimand.
- Route native corpus reads and writes through `corpus_io.py`. CLI loaders only add user-facing errors.
- Read `docs/invariants.md` before changing estimation, attribution, outcomes, or explanation.
- Keep the project narrow. Name a concrete caller before adding compatibility shims or abstractions.
- Ask before adding runtime dependencies. Broad causal or agent frameworks are outside this library's scope.

The wheel contains `src/counterfact` only. `bench/`, notebook generation, and real-agent execution require a checkout. Do not make installed library commands import checkout-only modules unconditionally.

## Causal honesty

`pass_rate_by_arm()` is descriptive. `identified` requires logged assignment probabilities and checked support. Sampling intervals and identification bounds are separate. A `bounded` result may have no justified interval. AIPW intervals must not be clipped to probability limits.

Mixed decision types in one step can use explicit decision IDs. Repeated decisions of one type still refuse focal localization. Same-step decisions never enter adjustment sets.

## Real-agent work

Read [bench/real/README.md](bench/real/README.md) before touching execution, approvals, or spend accounting. The harness can make paid provider calls. Docker checks do not.

- Obtain approval before generating real traces. State the exact command, fixtures, destination, and budget cap.
- Never create or bypass `.counterfact/approved` for the user.
- Preserve the 80% spend halt and the refusal to resume with unknown charges.
- Execute generated code through the Docker verifier. Never add a host-execution fallback.
- Keep infrastructure failures separate from graded task failures.

Never commit credentials, `.env`, `.counterfact/`, checkpoints, or ad hoc pilot outputs. Keep corpus promotion explicit.

## Documentation and artifacts

Use the Makefile's generation targets for derived artifacts. Keep numeric tables in their generated owners and link them from prose. Document interpretation and limitations where judgment is required.

Do not claim structural counterfactuals, learned DAGs, provider replay guarantees, calibrated universal probabilities, or a verifier that prevents candidate cheating. Docker limits host exposure. Candidate code and tests still share the verifier workload.
