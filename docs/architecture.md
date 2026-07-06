# Architecture

A five-minute orientation for someone evaluating this codebase. The README covers what `counterfact` does and how to drive it; this document explains why it is shaped the way it is and where to read first.

## 1. Why this exists

Naive pass-rate analysis on agent traces is misleading because logged decisions are confounded by upstream choices. Say model B passes 80% of the time and model A passes 60% in a trace corpus. The honest description is that B-runs ended in pass more often. The causal claim, "switching to model B improves pass rate by 20pp", is a different statement and usually a false one. In the corpus that produced those numbers, model B was only ever called when the agent had already chosen the easier tool. The descriptive 80% is real. The causal 20pp is a fiction wearing the same clothes.

`counterfact` exists to keep those two statements distinguishable. It will give you the descriptive number when you ask for one, and a causal estimate when the corpus actually supports one, and an honest "we cannot answer this from this data" when it does not. The library treats that third answer as a first-class output, not a failure mode.

## 2. The insight

Identifiability-first design. Every causal answer carries a label. That label is the load-bearing field.

- `identified` - observed support is sufficient under the per-trace graph and the assumptions the engine declares. You get a point estimate with a bootstrap CI.
- `bounded` - back-door adjustment is the right approach in principle, but the corpus lacks the joint randomization needed to estimate it. You get whatever bounds the data can support, and `None` when it cannot support any. No hardcoded sentinel "E-value" pretending to be a number.
- `unidentified` - the question is structurally unanswerable from this corpus. You get a `next_step` describing what would need to change, such as more data, a randomized arm, or a replay of the original prompt.

The product stance is short: an honest `unidentified` beats a confident wrong number. The CLI, the HTML report, the attribution ranker, and the demos refuse to render numeric effects when the underlying estimate is unidentified.

## 3. Pipeline

```
trace JSON ──► Run (schema/models.py)
                │
                ▼
            DAG (dag/graph.py)              ← decision-type taxonomy
                │
                ▼
        outcome model (outcome/model.py)    ← logistic regression w/ bootstrap
                │
                ▼
        intervention (intervene/api.py + dispatch.py)  ← stance-dispatched g-formula
                │
                ▼
    CausalEstimate (intervene/estimate.py)
        ├── identifiability label
        ├── outcome_delta? (only if identified)
        ├── bounds? (only if observed support)
        └── next_step (always)
```

A trace is parsed into a strict `Run`. The `Run` becomes a per-trace DAG built from logged structure. There is no graph learning; the edges come from a typed taxonomy of decision kinds. A corpus of `Run`s fits a transparent outcome model. An intervention asks "what does the model predict if we edit this decision under back-door adjustment?" and returns a `CausalEstimate` whose label tells you whether the engine answered the question or refused to.

## 4. The honesty contract

The code path next to each invariant enforces it.

- **Bounded path returns `bounds=None` when there is no observed back-door support.** No sentinel E-value, no synthetic prior. Enforced in `src/counterfact/intervene/api.py`.
- **Single-class corpora become `unidentified` with a structured `next_step`.** A logistic regression on no-variation data cannot fit, and saying so is the answer. Enforced in `src/counterfact/outcome/model.py`.
- **`pass_rate_by_arm` returns a `PassRateTable`, not a `CausalEstimate`.** The two are deliberately distinct types so a descriptive table cannot be silently mistaken for a causal claim downstream. Defined in `src/counterfact/baselines.py`.
- **Every `CausalEstimate` has a `next_step`.** Even `identified` ones, where the action might be `none`, carry the field. The shape is uniform so consumers never have to special-case "what do I show the user when there is no number to show?". Defined in `src/counterfact/intervene/estimate.py`.
- **Replay-required interventions return `unidentified` with a `replay_required` next_step.** Editing a prompt or a hidden state is a structural change the corpus cannot answer; the engine refuses to estimate and tells you what replay would be needed instead. Enforced in `src/counterfact/intervene/api.py`.

## 5. Where to read first

Reading order if you have fifteen minutes and want to evaluate the codebase instead of skim it.

1. **`src/counterfact/schema/models.py`** - the producer/consumer contract. Understand `Run`, `Step`, `Decision`, and `Outcome` first; everything else is a transformation over these.
2. **`src/counterfact/taxonomy/types.py`** - the decision-type taxonomy. This drives DAG edges and marks some interventions as valid and others as structurally unanswerable.
3. **`src/counterfact/dag/graph.py`** - how a single trace becomes an inspectable graph. The graph is built from logged structure, not learned.
4. **`src/counterfact/outcome/model.py`** - featurization and the bootstrap fit. Notice the single-class refusal path.
5. **`src/counterfact/intervene/api.py`** - thin `intervene()` entry; **`intervene/dispatch.py`** - stance handlers; **`stats.py`** - shared Wilson/z helpers; **`trace_localization.py`** - duplicate-step honesty; **`cli/`** - decomposed CLI package (`commands/`, `parser.py`, `demo_showcase.py`).
6. **`src/counterfact/intervene/estimate.py`** - the `CausalEstimate` and `NextStep` schemas. The output shape is the contract with every downstream consumer.
7. **`src/counterfact/attribute/failure.py`** - per-decision attribution ranking that powers `counterfact diagnose`.
8. **`tests/acceptance/test_scm_recovery.py`** - the canonical "we recover the synthetic ground truth within tolerance" test. This is the proof the engine is doing what it claims.

## 6. Why bench/ is separate

`bench/` contains two things that do not belong in the published library.

- `bench/synthetic/` - a deterministic structural causal model that generates traces with known ground-truth effects. `tests/acceptance/test_scm_recovery.py` runs against it to check that the engine recovers the truth within tolerance.
- `bench/real/` - a real coding-agent harness that calls actual LLMs (via `litellm`) on fixture coding tasks and writes counterfact-shaped traces. It costs money to run, requires a `.counterfact/approved` marker, and halts at 80% of its budget cap.

`bench/` is excluded from the installable wheel. `pip install counterfact` gives you the library only. A dev install (`pip install -e ".[dev,bench]"`) gets you the harness as well. This separation keeps the published package small and dep-light while the harness stays available for development and benchmarking.

---

For the deeper "what to read when working on X" routing map aimed at AI agents, see [`docs/repo-context.md`](repo-context.md).
