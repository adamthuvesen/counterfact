# Coding-agent fixtures

Each fixture is a small Python repair task: a buggy `src/` file, a `spec.md`,
and either `tests/` (v0 layout) or `tests_public/` + `tests_hidden/` (hidden-
test layout). The agent reads the public tests and the spec, edits the source,
and is graded against the hidden tests.

The product question this directory answers is *not* "can an LLM write code?"
— modern frontier models trivially pass anything that looks like a leetcode
warmup. The question is **"does this fixture produce real outcome variation
when randomized model and tool choices are flipped?"** Without that, the
generated traces are causally degenerate (every trace passes; see
`bench/real/single_class_refusal/`) and `counterfact intervene` correctly refuses to
attribute anything.

## Properties of a useful fixture

A fixture is worth running iff:

- **The prose `spec.md` is naturally lossy.** Real specs have edge cases the
  prose underspecifies — DST transitions, Unicode normalization variants,
  rate-limiter burst tolerances, off-by-one boundaries, locale-dependent
  case-folding. The fixture should ship a spec that names the contract
  honestly without exhaustively encoding every edge case.
- **The hidden tests catch real misreads.** If a smaller model would still
  pass the hidden tests, the fixture is too easy. If the hidden tests check
  a behavior the spec does not name, the fixture is unfair (and useless for
  causal attribution because outcome variation correlates with arbitrary
  reading rather than capability).
- **The bug is plausibly fixable but not obvious.** Multiple correct
  implementations exist; the LLM has to pick between them.
- **The bug surface interacts with arms we can randomize.** Model choice
  (`small` vs `large`), tool choice (`inspect_file`, `run_tests`,
  `search_docs`), and retry policy (`retry_once` vs `no_retry`) should all
  plausibly change the outcome.

## Anti-patterns

- **Leetcode-style toy problems.** The agent one-shots them. No outcome
  variation, no causal signal.
- **Specs precise enough to specify the implementation.** No interpretation
  required ⇒ no model differentiation.
- **Hidden tests that are arbitrary stress tests.** Outcome variation
  becomes "did the LLM happen to think of timezone X" rather than
  "does the larger model handle ambiguity better."
- **Fixtures that depend on external state.** The verifier must be a
  deterministic local pytest invocation.

## Promotion gate

Before any new fixture set ships under `bench/real/smoke_mixed_outcome/`, run:

```bash
counterfact analyze corpus bench/real/pilot_<YYYY-MM-DD>/
```

The analyzer enforces the rubric documented in
`src/counterfact/corpus_analyzer/rubric.py`: pass rate in [0.3, 0.7], at
least two arms with `n >= 5` for some randomized decision type, at least one
decision type where `intervene()` returns `identified`, and mixed pass/fail
outcomes for both real-agent model arms (`small` and `large`) when those arms
are present. Promotion to `smoke_mixed_outcome/` is a deliberate human action
— see `bench/real/README.md` for the convention. The analyzer never
auto-renames anything.

For showcase calibration, also run `python -m bench.real.analyze_pilot` and read
the failure-mode table. A strong showcase should fail because runnable patches
miss hidden semantics, not because the model omitted a parseable patch.

## Existing fixtures

- `date_window/` — broad calibration fixture.
  Public tests cover sorted-window basics; hidden tests cover stated edge
  semantics around inclusive boundaries, unsorted windows, invalid ranges,
  validating every window endpoint, cross-year windows, leap days, and
  malformed dates.
- `rate_limit/` — broad calibration fixture for fixed-window rate limiting.
  Public tests cover basic limits; hidden tests cover inclusive lower bounds,
  unsorted history, future timestamps, duplicate same-second requests, and
  invalid limit/window values.
- `version_range/` — broad calibration fixture for semantic-version ranges.
  Public tests cover simple final-release ranges; hidden tests cover strict
  bounds, prereleases, dot-separated prerelease ordering, malformed versions,
  and malformed constraints.
- `streaming_watermark_dedupe/` — stateful calibration fixture. Public
  tests cover in-order duplicate suppression; hidden tests cover event-time
  watermark advancement, late-event dropping, TTL eviction, checkpoint restore,
  stable emission order, and memory-bounded retained state. It is harder than
  single-function normalization fixtures because correctness depends on state
  transitions across multiple `process()` calls.

Historical/internal fixtures:

- `csv_dedupe/` — lower-bound hidden-test calibration, exposed as `hidden_v1`.
- `unicode_normalize/` — semantic calibration that proved too easy for current
  models, exposed as `very_hard_hidden_v1`.
- `csv-stats/`, `string-utils/`, `date-utils/` — easy v0 fixtures, retained
  for harness-integration testing.
- `regex-anchors/`, `iso-week-dates/`, `agg-with-groups/` — harder v0
  fixtures.

Candidate next fixtures should follow the same rule as `unicode_normalize`:
deterministic local verifier, naturally lossy spec, and hidden tests that stay
derivable from the prose.
