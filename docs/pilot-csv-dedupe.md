# csv_dedupe pilot (Pilot 3) — n=30

These notes explain how the hidden/public harness was exercised on `csv_dedupe`
and why the committed showcase traces live under `bench/real/runs_single_class/`.

**Date:** 2026-05-02  
**Command (historical output dir):**
`counterfact bench real --fixtures csv_dedupe --n 30 --budget-cap 5 --output-dir bench/real/runs_v3_pilot --seed 0 --epsilon 0.2`  
**Observed cost:** $0.1896 ($0.0063 / trace)

## 2×2 contingency (public_pass × hidden_pass)

|             | hidden=T | hidden=F |
|-------------|---------:|---------:|
| **public=T**|       30 |        0 |
| **public=F**|        0 |        0 |

## Per-arm distribution and outcome

| arm          | distribution                                  | all pass? |
|--------------|-----------------------------------------------|-----------|
| `model_choice` | large=28, small=2                          | yes — both haiku draws and all sonnet draws pass |
| `tool_choice`  | inspect_file=25, run_tests=3, search_docs=2 | yes — all three tool branches pass |
| `retry_policy` | retry_once=30, no_retry=0                   | yes — but no variation observed (greedy lock-in, same as Pilot 1) |

## Decision gate (task 6.3)

Generalization-gap cell `(public=T, hidden=F)`: **0** (need ≥3).

**Gate: FAIL.** Do not proceed to step 7 (date_window + rate_limit replication)
under the assumption that "the same harness pattern with another fixture works."
Either move on with a different premise (option b below), or pivot framing
(option c).

## Interpretation

The hidden/public split worked exactly as designed — `tests_hidden/` is genuinely
hidden, the prompt only references `spec.md` and `tests_public/`, the agent
sandbox excludes `tests_hidden/`, and the post-loop hidden eval ran exactly once
per trace. The infrastructure is correct.

What it revealed: both Claude Sonnet 4.6 AND Claude Haiku 4.5 read `spec.md`,
identify the four normalization rules (BOM strip, outer whitespace, case fold,
NFC), and implement all four cleanly — even when the public test suite only
exercises exact-string duplicates. The "agent reads the wrong file" risk
(design.md D2) is mitigated; the "agent infers the spec well enough that hidden
tests also pass" risk (design.md D6) is the live failure mode. Per design.md:

> If the agent can satisfy the full spec from a brief and incomplete public
> tests, the corpus has variation across other arms… either way the corpus is
> useful.

In this case the *other arms* also produce zero variation: even the small-model
+ unusual-tool branches one-shot a fix. So `csv_dedupe` with the current
`spec.md` cannot supply the class balance that internal section 15.2 once asked for.

This is the same fundamental finding as Pilot 1 + Pilot 2 on the v0 hard
fixtures: **single-file Python bug-fixes with a clear written spec are at
ceiling for frontier models, regardless of whether the verifier is visible.**
Hiding the verifier was a real fix to a different problem (the model optimizing
against the test text) — but the binding constraint on the corpus is task
difficulty, not feedback transparency.

## Options (per design.md D6)

1. **(a) Tighten the spec/hidden tests.** Add rules that interact in
   non-obvious ways (e.g. "empty rows collapse to a single empty row,"
   "zero-width joiners count as whitespace"). Risk: drift toward "gotcha"
   tests, undermining the credibility argument. Bounded usefulness — at some
   point the spec is so detailed the model still nails it.
2. **(b) Switch to a fixture where prose-spec is intrinsically lossy.**
   Calendar/timezone semantics (`date_window`) and stateful timing
   (`rate_limit`) are domains where models have measurable error rates *even
   when the rules are written down*. The harness already supports the
   layout; this is the natural next step.
3. **(c) Pivot Path 2's framing.** Use the corpus as a "harness-correctness"
   demo and explicitly drop the class-balance criterion — the demo
   shows that `counterfact` produces honest `unidentified` results when
   intervention support is degenerate. Less compelling but ships sooner.

**Recommendation:** **(b)**. Move on to `date_window` next, with `spec.md`
tightly scoped to date-only semantics (no timezone soup). If `date_window`
also goes 30/30, that is itself a publishable observation about frontier
model capability and we ship under framing (c). Either way, csv_dedupe is
shelved as "implementation works, signal too weak to drive the corpus."

## What to keep from this pilot

- The hidden/public split harness, schema fields, the verifier label
  `pytest_hidden`, and test invariants — none of those need changing.
- `csv_dedupe` stays in the registry as a regression fixture for
  harness-level tests; it drives the qualitative demo via the committed corpus.
- A trimmed subset of these traces lives at **`bench/real/runs_single_class/`**
  (3 keepers — same content as was produced during this pilot, kept as the
  canonical public artifact and as the engine's single-class regression anchor).

## Identifiability-first pivot

This pilot's null result drove the identifiability-first pivot. Three pilots in
a row producing 30/30 pass — across Sonnet 4.6 and Haiku 4.5, with and without
hidden tests — was the empirical basis for retiring v0's class-balance and
CI-width ship-gate criteria. The reframed v0 ships with a smaller, honest corpus
and a demo whose headline is "naive vs honest" rather than "identified effect =
X".

The committed `runs_single_class` traces demonstrate the failure mode the pivot was built
to surface — a causally degenerate corpus where outcome variation is absent and
`intervene()` correctly refuses to invent a causal difference between arms.
