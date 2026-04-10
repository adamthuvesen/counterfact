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
`bench/real/runs_v1/`) and `counterfact intervene` correctly refuses to
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

Before any new fixture set ships under `bench/real/runs_v2/`, run:

```bash
counterfact analyze corpus bench/real/runs_pilot_<YYYY-MM-DD>/
```

The analyzer enforces the rubric documented in
`src/counterfact/corpus_analyzer/rubric.py`: pass rate in [0.3, 0.7], at
least two arms with `n >= 5` for some randomized decision type, and at
least one decision type where `intervene()` returns `identified`. Promotion
to `runs_v2/` is a deliberate human action — see `bench/real/README.md` for
the convention. The analyzer never auto-renames anything.

## Existing fixtures

- `csv_dedupe/` — the canonical hidden-test fixture. Modern models one-shot
  it (see pilot 3); kept as the lower-bound difficulty calibration.
- `csv-stats/`, `string-utils/`, `date-utils/` — easy v0 fixtures, retained
  for harness-integration testing.
- `regex-anchors/`, `iso-week-dates/`, `agg-with-groups/` — harder v0
  fixtures.

Candidate next fixtures (see proposal in
`openspec/changes/.../` follow-on changes): `date_window`, `rate_limit`,
`unicode_normalize`. None of these are authored yet — they are intentionally
left as named placeholders for the next change.
