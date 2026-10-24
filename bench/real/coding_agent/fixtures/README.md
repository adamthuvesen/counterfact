# Coding-agent fixtures

Each fixture is a small Python repair task: a buggy `src/` file, a `spec.md`, and tests. Most fixtures split the tests into `tests_public/`, which the agent sees, and `tests_hidden/`, which grade the run. The simple fixtures have one `tests/` directory.

A fixture is useful only if flipping the randomized model or tool choice changes the outcome. Frontier models pass leetcode-style tasks every time. That gives a corpus with one outcome class, which `counterfact intervene` correctly refuses to analyze (see `bench/real/single_class_refusal/`).

## Properties of a useful fixture

- **The spec is naturally lossy.** Real specs leave edge cases to the reader: DST transitions, Unicode normalization variants, rate-limiter bursts, off-by-one boundaries. The spec should state the contract honestly without encoding every edge case.
- **The hidden tests catch real misreads.** If a smaller model still passes them, the fixture is too easy. If they check a behavior the spec does not name, the fixture is unfair, and the outcome measures luck instead of capability.
- **The fix is plausible but not obvious.** Several correct implementations exist, and the agent has to choose between them.
- **The randomized choices matter.** Model choice (`small` or `large`), tool choice (`inspect_file`, `run_tests`, `search_docs`), and retry policy (`retry_once` or `no_retry`) should each plausibly change the outcome.
- **The verifier is local and deterministic.** Grading is a local pytest run with no external state.

`counterfact analyze corpus <dir>` checks a generated corpus against the rubric in `src/counterfact/corpus_analyzer/rubric.py`. `python -m bench.real.analyze_pilot <dir>` shows whether the failures come from task behavior or from malformed patches.

## Fixtures

Calibration fixtures, used by the demo corpus and the routed trial:

- `date_window/` (`broad_calibration`): hidden tests cover inclusive boundaries, unsorted windows, invalid ranges, cross-year windows, leap days, and malformed dates.
- `rate_limit/` (`broad_calibration`): hidden tests cover inclusive lower bounds, unsorted history, future timestamps, duplicate same-second requests, and invalid limits.
- `version_range/` (`broad_calibration`): hidden tests cover strict bounds, prereleases, prerelease ordering, and malformed versions and constraints.
- `streaming_watermark_dedupe/` (`stateful_calibration`): hidden tests cover event-time watermarks, late-event dropping, TTL eviction, checkpoint restore, stable emission order, and bounded state. Correctness depends on state across several `process()` calls.

Other fixtures:

- `csv_dedupe/` (`hidden_v1`): a lower-bound hidden-test task.
- `unicode_normalize/` (`very_hard_hidden_v1`): semantic normalization. Frontier models pass it consistently.
- `csv-stats/`, `string-utils/`, `date-utils/` (`easy`) and `regex-anchors/`, `iso-week-dates/`, `agg-with-groups/` (`v0`): tasks with one test directory, used for harness integration tests.
