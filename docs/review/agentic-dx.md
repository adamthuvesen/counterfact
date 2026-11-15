# Agentic development reassessment

## Scope and scoring

This review compares commit `16ac10e` with implementation commit `601babf`.
The implementation stays on `adam/refactor-agentic-dx`. It is not published.

Scores express engineering judgment, not a measured performance benchmark.
Each dimension uses the same four-point rubric:

1. Important behavior depends on instructions or unchecked state.
2. Some mechanisms enforce the contract, but demonstrated gaps remain.
3. The design is coherent and checked, but required verification or ownership remains incomplete.
4. Ownership is clear, boundaries reject invalid states, and behavioral checks prove both acceptance and rejection.

Excellent requires 4 in every dimension and no unresolved material finding from independent review.
External-service claims remain separate from local development quality.

| Dimension | Baseline | Baseline evidence | Final |
| --- | ---: | --- | --- |
| Domain ownership | 2 | Chronology and corpus interpretation differ between consumers. | 4 |
| Scientific correctness | 2 | Invalid assignment parameters can produce an identified result. Result labels permit contradictory payloads. | 4 |
| Trust boundaries | 2 | Receipt loading, live malformed outcomes, and exported usage disagree between paths. | 4 |
| State and resources | 1 | Corpus publication can be partial. Generated code executes with host permissions. | 4 |
| Behavioral verification | 3 | 499 tests pass, but coverage scope is contaminated and some gates test constructed substitutes. | 4 |
| Reproducibility | 2 | Local and CI checks differ. Installed-wheel recovery guidance is incorrect. | 4 |
| Discoverability and feedback | 3 | The routing map is useful, but some guidance confuses promotion failure with model eligibility. | 4 |

The final assessment meets the defined excellent local agentic-DX state.
Each score combines enforced ownership with executed acceptance and rejection checks.
Scientific correctness here means enforcement of the stated causal contracts.
It does not mean universal calibration or independently verified causal assumptions in external data.

## Implementation ledger

| Workstream | Required outcome | Status |
| --- | --- | --- |
| Analysis facts | Stable chronology, corpus identity, and mutation isolation | Verified |
| Query and assignment evidence | Explicit first-occurrence contrasts and validated assignment support | Verified |
| Results and presentation | Valid result states, consistent guidance, and boundary-safe trial intervals | Verified |
| Corpus IO and usage | One publication owner, explicit recovery, and honest exports | Verified |
| Generated-code execution | Real container isolation, typed failures, and conservative spend recovery | Verified |
| Development workflow | Shared local/CI checks, type coverage, package smoke tests, and generated-artifact checks | Verified |
| Documentation and independent review | Current navigation, compatibility notes, historical replay, and final reassessment | Verified |

## Historical evidence

- PR #16 and commits `1cad6c1`, `7838fe3`, and `421039e` corrected positivity, temporal adjustment, and sample-size guidance.
- PR #19 and commits `6aa72e0` and `bcb10aa` corrected policy ordering, reachable support, and empty policy evidence.
- PR #18 and commits `f4cce11` and `5186367` corrected routed-trial focal selection and single-class refusal.
- Commits `7cf2630` and `c7901b2` hardened approval and spend accounting.
- PR #16 deferred malformed epsilon validation. PR #18 deferred boundary-safe trial intervals.

Targeted cases introduced by this refactor must be distinguished from replays of these historical failures.

Two historical regression tests were replayed against archived pre-fix source trees:

| Regression | Archived source | Observed historical failure | Refactored source |
| --- | --- | --- | --- |
| A policy controls no observed decision | `bcb10aa^` | Returned `identified` instead of `unidentified` | Passed |
| A same-step decision enters the adjustment set | `1cad6c1^` | Included same-step covariates | Passed |

The test files came from the corresponding fixing commits.
The archives used the current installed test dependencies.
Both historical failures matched the intended regression assertion.
The same test nodes passed against the refactored source:

```bash
uv run --locked pytest -q \
  tests/unit/test_policy_evaluation.py::test_policy_that_never_acts_on_the_corpus_is_unidentified \
  tests/unit/test_causal_engine.py::test_intervene__same_step_decisions_are_not_adjusted_for
```

Chronology permutation, malformed epsilon, result mutation, interrupted publication, and container-boundary checks test current architectural guarantees.
They are not presented as historical incident replays.

## Independent review and iteration

Independent GPT-6.1 Sol reviewers examined the causal core, IO, executor, and development workflow.
The first review found gaps despite passing focused tests:

| Finding | Correction | Verification |
| --- | --- | --- |
| Zero matching policy trajectories produced invalid NaN SNIS diagnostics | Represent unavailable SNIS as null and name the affected policy in warnings | Independent target and baseline failure replays passed |
| Tuple targets retained mutable aliases | Detach nested mutable contents in result construction and public views | Independent caller and returned-view mutation probes passed |
| Set metadata remained mutable inside analysis facts | Freeze sets recursively | Independent source and snapshot mutation probes passed |
| Huge integer epsilon raised an overflow error | Reject out-of-range values before floating-point conversion | Independent contrast and policy refusal probes passed |
| Tool observations could conceal missing model-call costs | Associate model-call costs before declaring reconstruction complete | Independent row and Parquet probes passed |
| Receipt-like run IDs produced unreadable corpora | Distinguish validated native runs from export sidecars | Independent publication and load probes passed |
| Generated code bypassed the verifier output file limit | Bound host streaming capture and disable Docker logs | Independent 2 MiB pipe-bypass probe stopped and removed its container |
| Host cancellation left a generated-code container running | Remove the container in `finally`, preserving cancellation and reporting cleanup failure | Independent interruption probe propagated cancellation and left no container |
| Documentation promised a removed transcript and an incorrect image command | Correct the navigation and distinguish cached builds from existing-image checks | Independent documentation reread passed |

A required Docker parity test also exposed truncated source after an in-place patch.
The exact Docker Desktop cause remains unproven.
Execution now stages a fresh input snapshot and verifies its content digest inside the container.
This prevents changed handoff bytes from being scored as a model failure.

The reviewers independently rescored every delegated dimension at 4/4 after rechecking the fixes.
No demonstrated finding remains unresolved.
The parent integrated those reviews with the complete local verification gates.

## Why checks remain after the architectural changes

| Mechanism | Why ownership or types alone do not suffice |
| --- | --- |
| Import contracts | Python does not enforce module dependency direction. The check protects the declared owners from future reverse imports. |
| Full function annotations and mypy | Runtime boundary validation cannot detect incompatible internal callers before execution. |
| Locked setup and shared Make targets | Types cannot control installed tool versions or ensure CI runs the same commands. Hooks also use `uv.lock`. |
| Generated-artifact checks | Generators own derived content, but consumers read committed artifacts. The check detects an omitted regeneration. |
| Statistical and historical tests | Types cannot prove temporal adjustment, estimator arithmetic, or preservation of measurement semantics. |
| Real Docker and installed-wheel checks | Source structure cannot prove operating-system isolation, cancellation cleanup, or installed-package behavior. |
| Documentation | Ownership navigation, causal assumptions, and external verification limits require explanation and judgment. |
| Independent review | Review examines semantic omissions and adversarial cases that the current automated mechanisms do not express. |

## Measurement preservation

The full 500-seed benchmark ran again through the final public contrast path.
Every scenario and policy summary exactly matched the committed JSON artifact.
The canonical output files had no diff after regeneration.
This checks the repository's synthetic scenarios, not universal estimator calibration.

The committed routed corpus retained its point estimates and causal labels.
The intentional Newcombe-Wilson migration changed the trial lower interval endpoint from 0.184 to 0.183 after rounding.

Five Parquet exports per committed corpus preserved all recorded costs across 920 runs.
The smoke corpus retained USD 1.6807 and the routed corpus retained USD 6.52133.
These values come from trace observations, without provider billing reconciliation.
Unknown token totals now remain null instead of becoming fabricated zeros.

## Verification boundaries

Local checks can establish library behavior, committed-corpus compatibility, package behavior, and the local container execution boundary.
They cannot establish live provider billing or compatibility with every SDK version.
The empty human-label artifact does not establish attribution accuracy.
Container isolation protects the host. It does not prove that generated Python cannot influence a Python-based verifier.

The Docker checks ran Linux arm64 containers on the local Docker daemon.
Linux amd64 execution and hosted GitHub Actions were not run.
No paid provider call or deployment ran. Nothing was pushed.

## Final verification

| Environment | Command | Observed result |
| --- | --- | --- |
| macOS, Python 3.14.9 | `make verify` | 613 offline tests passed, 2 human-label checks skipped, 86.93% branch coverage, 28 Docker tests passed |
| macOS, Python 3.13.17, separate environment | `UV_PYTHON=3.13 UV_PROJECT_ENVIRONMENT=<temporary-env> make verify` | 613 offline tests passed, 2 human-label checks skipped, 86.98% branch coverage, 28 Docker tests passed |
| Both Python environments | Included in `make verify` | Ruff, mypy across 113 source files, two import contracts, all generated-artifact checks, and isolated wheel consumers passed |
| Final tooling and test configuration | Focused tooling, pricing, formatting, typing, import, and generated checks | Passed after the final configuration changes |
| Local Git hooks | `uv run --locked --extra dev --extra bench pre-commit run --all-files` | Locked Ruff check and format hooks passed, then passed again during the implementation commit |
| Local Docker daemon | Container inventory after verification | No verifier container remained |

Coverage includes library code, reusable benchmarks, scripts, and the container verifier source.
The offline coverage result includes the uninstrumented container verifier as uncovered.
Actual container behavior is established by its separate execution gate.
Fixture task inputs are excluded from the coverage denominator.
These percentages are not comparable with the contaminated baseline coverage scope.

The two skipped checks both require the empty human-label artifact.
They remain visible through pytest's skip summary.
Required committed corpora and runtime prerequisites fail instead of silently skipping.

Verification logs were retained locally at `/tmp/counterfact-verify-py314-final.log` and `/tmp/counterfact-verify-py313-final.log`.
Those files are session evidence, not required repository inputs.
The commands and committed checks reproduce the validation path.
