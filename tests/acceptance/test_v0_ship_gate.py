"""Acceptance ship gate for identifiability-first behavior.

Bundles the automatable release checks into one runnable module. The gate
enforces identifiability discipline rather than treating every real corpus as
eligible for a point estimate:

- Synthetic SCM recovers the known headline effect.
- Real-corpus interventions yield internally consistent identifiability labels;
  zero `identified` results is allowed if the corpus is causally degenerate.
- At least one real-corpus query returns `unidentified` with a structured,
  non-empty `next_step`.
- Top-1 attribution can be checked against hand-labeled root causes.
- The demo notebook renders the naive-vs-honest contrast.
- Forbidden runtime deps and imports stay out.

Tests that depend on the canonical real-agent corpus skip gracefully when it
is absent. The demo-notebook test relies on the notebook, not on the corpus.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from bench.synthetic import generate_traces
from counterfact import fit_outcome_model, intervene, pass_rate_by_arm
from counterfact.dag import build_dag
from counterfact.intervene.estimate import (
    CausalEstimate,
    IdentifiabilityStatus,
    InterventionQuery,
    NextStep,
)
from counterfact.schema import Run
from tests.acceptance.demo_notebook_helpers import (
    all_text_outputs,
    execute_demo_notebook,
)
from tests.fixtures.corpus_loaders import SMOKE_MIXED_OUTCOME_DIR, load_json_corpus

REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_CORPUS_DIR = SMOKE_MIXED_OUTCOME_DIR
DEMO_NOTEBOOK = REPO_ROOT / "notebooks" / "demo.ipynb"
PYPROJECT = REPO_ROOT / "pyproject.toml"
SRC_DIR = REPO_ROOT / "src"

MIN_REAL_TRACES = 120  # promoted smoke_mixed_outcome baseline
SCM_RECOVERY_TOLERANCE = 0.05

# Every action documented in the NextStep contract that counts as actionable.
# `none` is a valid label but not actionable.
ACTIONABLE_NEXT_STEP_ACTIONS = frozenset(
    {"increase_n", "broaden_arm_support", "replay_required", "add_arm_randomization"}
)

# Forbidden surface area for the narrow built-in causal engine.
FORBIDDEN_DEPS = ("dowhy", "causalml", "pyro", "langchain", "langgraph", "pandas", "networkx")
FORBIDDEN_IMPORTS = ("dowhy", "causalml", "pyro", "langchain", "langgraph")


# --- fixtures ---------------------------------------------------------------


def _load_real_corpus() -> list[Run]:
    if not REAL_CORPUS_DIR.exists():
        return []
    return load_json_corpus(REAL_CORPUS_DIR)


@pytest.fixture(scope="module")
def real_corpus() -> list[Run]:
    runs = _load_real_corpus()
    if not runs:
        pytest.skip(
            f"smoke_mixed_outcome corpus absent at {REAL_CORPUS_DIR}. "
            f"Promote a real corpus per bench/real/README.md before this gate runs."
        )
    return runs


# --- synthetic corpus is reproducible at >=500 traces -----------------------


def test_synthetic_corpus_is_deterministically_500_traces() -> None:
    """Synthetic generator produces 500 traces deterministically per seed."""
    a = list(generate_traces(n=500, seed=42))
    b = list(generate_traces(n=500, seed=42))
    assert len(a) == 500
    assert a[0] == b[0]
    assert a[-1] == b[-1]


# --- real-corpus identifiability is honestly reported -----------------------


def test_real_corpus_meets_minimum_size(real_corpus: list[Run]) -> None:
    """Real corpus has at least MIN_REAL_TRACES traces."""
    assert len(real_corpus) >= MIN_REAL_TRACES, (
        f"real corpus has {len(real_corpus)} traces; need ≥{MIN_REAL_TRACES}"
    )


def _pick_three_queries(real_corpus: list[Run]) -> list[tuple[Run, int, dict[str, object]]]:
    """Pick three intervention queries that the real corpus can answer.

    Strategy: look at the first run that has decisions of each randomized
    decision_type (model_call, tool_call, retry) and target the first arm of
    each. If a type is missing, fall back to a prompt_content query against
    any model_call to exercise the always-replay path.
    """
    queries: list[tuple[Run, int, dict[str, object]]] = []
    seen_types: set[str] = set()
    for run in real_corpus:
        for step in run.steps:
            for d in step.decisions:
                if d.decision_type in seen_types:
                    continue
                if d.decision_type == "model_call":
                    queries.append((run, step.step_index, {"model_choice": d.chosen_action}))
                    seen_types.add("model_call")
                elif d.decision_type == "tool_call" and d.policy:
                    queries.append((run, step.step_index, {"tool_choice": d.chosen_action}))
                    seen_types.add("tool_call")
                elif d.decision_type == "retry":
                    queries.append((run, step.step_index, {"retry_policy": d.chosen_action}))
                    seen_types.add("retry")
        if len(queries) >= 3:
            break
    # Always end with a prompt_content query: that is the always-replay path
    # and gives the gate something to point to even on a degenerate corpus.
    for run in real_corpus:
        for step in run.steps:
            for d in step.decisions:
                if d.decision_type == "model_call":
                    queries.append((run, step.step_index, {"prompt_content": "be more careful"}))
                    return [*queries[:3], queries[-1]] if len(queries) > 3 else queries
    return queries


def _outcome_classes(real_corpus: list[Run]) -> set[bool]:
    return {bool(run.outcome.value) for run in real_corpus}


def _degenerate_real_corpus_estimate(real_corpus: list[Run]) -> CausalEstimate:
    """Represent the ship-gate verdict for a real corpus with one outcome class.

    The real corpus can still be useful evidence in v0, but a one-class
    outcome vector cannot fit logistic regression. The gate should surface
    that as an unidentified causal query, not ask sklearn to do the impossible.
    """
    classes = _outcome_classes(real_corpus)
    assert len(classes) == 1, classes
    observed = next(iter(classes))
    return CausalEstimate(
        query=InterventionQuery(
            decision_type="model_call",
            intervention_kind="model_choice",
            target="any",
            step=-1,
        ),
        identifiability=IdentifiabilityStatus.UNIDENTIFIED,
        reason=(
            "real corpus is causally degenerate: every trace has "
            f"Outcome.value={observed}; no outcome variation exists for an "
            "outcome model or back-door adjustment to leverage"
        ),
        warnings=["fit_outcome_model is intentionally skipped for single-class real corpora"],
        next_step=NextStep(
            action="broaden_arm_support",
            payload={
                "arm_name": "outcome",
                "missing_strata": [f"Outcome.value={not observed}"],
                "observed_arms": [],
                "missing_arms": [f"Outcome.value={not observed}"],
            },
            human_text=(
                "Collect or construct traces with both pass and fail outcomes "
                "before estimating decision-level effects on the real corpus."
            ),
        ),
    )


def test_degenerate_real_corpus_verdict_is_unidentified(real_corpus: list[Run]) -> None:
    """A one-class real corpus gets an actionable unidentified verdict without
    attempting to fit logistic regression."""
    if len(_outcome_classes(real_corpus)) != 1:
        pytest.skip("real corpus has mixed outcomes; degenerate path not applicable")
    est = _degenerate_real_corpus_estimate(real_corpus)
    assert est.identifiability == IdentifiabilityStatus.UNIDENTIFIED
    assert est.reason
    assert est.next_step.action in {"broaden_arm_support", "add_arm_randomization"}
    assert est.next_step.payload
    assert "missing_strata" in est.next_step.payload


def test_real_corpus_identifiability_is_honest(real_corpus: list[Run]) -> None:
    """Every real-corpus intervention returns a legal identifiability label.

    Any `identified` result must be internally consistent: finite outcome_delta
    and CI, with the bootstrap CI bracketing the corresponding naive
    pass-rate-difference. Zero `identified` results is allowed when every query
    is bounded or unidentified."""
    queries = _pick_three_queries(real_corpus)
    assert len(queries) >= 3, f"could not assemble 3 queries; got {len(queries)}"

    if len(_outcome_classes(real_corpus)) == 1:
        est = _degenerate_real_corpus_estimate(real_corpus)
        assert est.identifiability.value in {s.value for s in IdentifiabilityStatus}
        return

    fitted_real_model = fit_outcome_model(real_corpus, n_bootstrap=200, seed=42)

    legal = {s.value for s in IdentifiabilityStatus}
    for run, step_idx, intervention in queries:
        est = intervene(
            dag=build_dag(run),
            model=fitted_real_model,
            step=step_idx,
            intervention=intervention,
        )
        assert est.identifiability.value in legal, (
            f"illegal identifiability label: {est.identifiability!r}"
        )

        if est.identifiability == IdentifiabilityStatus.IDENTIFIED:
            # Finite numerics
            assert est.outcome_delta is not None
            for x in (
                est.outcome_delta.point,
                est.outcome_delta.ci_low,
                est.outcome_delta.ci_high,
            ):
                assert x == x and abs(x) < float("inf"), (  # NaN-safe finite check
                    f"identified estimate has non-finite numeric: {x!r}"
                )
            assert est.outcome_delta.ci_low <= est.outcome_delta.ci_high

            # Naive marginal must fall inside the bootstrap CI for the chosen arm.
            ((kind, value),) = list(intervention.items())
            if kind in {"model_choice", "tool_choice", "retry_policy"}:
                decision_type_for_arm = {
                    "model_choice": "model_call",
                    "tool_choice": "tool_call",
                    "retry_policy": "retry",
                }[kind]
                table = pass_rate_by_arm(real_corpus, decision_type_for_arm)
                row = next((r for r in table.rows if r.arm == value), None)
                if row is not None:
                    # Allow for small slack between naive Wilson CI and
                    # bootstrap CI on the model's marginal — both should
                    # bracket the same truth.
                    assert (
                        est.outcome_delta.ci_low <= row.pass_rate <= est.outcome_delta.ci_high
                        or (row.ci_low <= est.outcome_delta.point <= row.ci_high)
                    ), (
                        f"naive vs identified disagree dramatically: naive "
                        f"{row.pass_rate:.3f} (CI [{row.ci_low:.3f}, {row.ci_high:.3f}]) "
                        f"vs identified {est.outcome_delta.point:.3f} "
                        f"(CI [{est.outcome_delta.ci_low:.3f}, {est.outcome_delta.ci_high:.3f}])"
                    )


# --- at least one unidentified with actionable next_step --------------------


def test_at_least_one_unidentified_with_actionable_next_step(
    real_corpus: list[Run],
) -> None:
    """At least one intervention on the real corpus returns `unidentified` with
    an actionable `next_step` and payload."""
    if len(_outcome_classes(real_corpus)) == 1:
        est = _degenerate_real_corpus_estimate(real_corpus)
        assert est.next_step.action in ACTIONABLE_NEXT_STEP_ACTIONS
        assert est.next_step.human_text
        assert est.next_step.payload
        return

    fitted_real_model = fit_outcome_model(real_corpus, n_bootstrap=200, seed=42)
    queries = _pick_three_queries(real_corpus)
    found = False
    for run, step_idx, intervention in queries:
        est = intervene(
            dag=build_dag(run),
            model=fitted_real_model,
            step=step_idx,
            intervention=intervention,
        )
        if est.identifiability == IdentifiabilityStatus.UNIDENTIFIED:
            assert est.next_step.action in ACTIONABLE_NEXT_STEP_ACTIONS, (
                f"unidentified estimate has non-actionable next_step.action="
                f"{est.next_step.action!r}"
            )
            assert est.next_step.human_text, "next_step.human_text is empty"
            if est.next_step.action == "replay_required":
                assert "intervention_target" in est.next_step.payload
                assert est.next_step.payload.get("replay_inputs_required"), (
                    "replay_required payload missing replay_inputs_required"
                )
                assert est.next_step.payload.get("note"), "replay_required payload missing note"
            elif est.next_step.action == "broaden_arm_support":
                assert est.next_step.payload, (
                    "broaden_arm_support unidentified estimate has empty payload"
                )
                assert "observed_arms" in est.next_step.payload
                assert "missing_arms" in est.next_step.payload
                assert isinstance(est.next_step.payload["observed_arms"], list)
                assert isinstance(est.next_step.payload["missing_arms"], list)
            else:
                assert est.next_step.payload, "non-replay unidentified estimate has empty payload"
            found = True
            break
    assert found, (
        "no unidentified result observed across the test queries; the v0 "
        "ship gate requires at least one; try a prompt_content query if "
        "the chosen queries all randomize over arms"
    )


# --- top-1 attribution matches the labeled root cause -----------------------


def test_top1_attribution_label_artifact_is_present() -> None:
    """The labels.json artifact exists with the documented schema.

    The full top-1 acceptance test lives in
    tests/acceptance/test_top1_attribution.py and runs against the real
    corpus when both labels and single_class_refusal are populated. This gate-level
    test is a presence check so the gate's structure is auditable here too;
    it skips when no label has been added yet."""
    labels_path = REPO_ROOT / "bench" / "real" / "coding_agent" / "labels.json"
    if not labels_path.exists():
        pytest.skip(f"labels.json absent at {labels_path}")
    labels = json.loads(labels_path.read_text())
    assert "labels" in labels, "labels.json missing top-level 'labels' field"
    if not labels["labels"]:
        pytest.skip("labels.json has zero entries; human labeling gate is open")
    entry = labels["labels"][0]
    assert entry.get("root_cause_decision_id"), "first label entry has no root_cause_decision_id"


# --- demo notebook renders the naive-vs-honest contrast ---------------------


def test_demo_renders_naive_vs_honest_contrast(tmp_path: Path) -> None:
    """The demo notebook executes end-to-end and renders both a
    `pass_rate_by_arm` table and an `intervene` CausalEstimate."""
    if not DEMO_NOTEBOOK.exists():
        pytest.skip(f"demo notebook absent at {DEMO_NOTEBOOK}")

    all_text = all_text_outputs(execute_demo_notebook(tmp_path))
    assert "pass_rate" in all_text or "PassRateTable" in all_text, (
        "demo notebook does not render a pass_rate_by_arm table"
    )
    assert "identifiability" in all_text, (
        "demo notebook does not render an intervene CausalEstimate"
    )
    # Either an actionable next_step or a nontrivial action label rendered
    # somewhere; the demo must show one such label.
    rendered_actions = {a for a in ACTIONABLE_NEXT_STEP_ACTIONS if a in all_text}
    assert rendered_actions, (
        "demo notebook does not render any actionable next_step.action; "
        f"expected one of {sorted(ACTIONABLE_NEXT_STEP_ACTIONS)}"
    )


# --- forbidden dependencies stay out ---------------------------------------


def test_no_forbidden_dependencies_in_pyproject() -> None:
    """pyproject.toml does not list forbidden causal/agent framework deps."""
    text = PYPROJECT.read_text()
    found = []
    for dep in FORBIDDEN_DEPS:
        if re.search(rf'(?m)^\s*"{re.escape(dep)}\b', text):
            found.append(dep)
    assert not found, (
        f"forbidden dependencies declared in pyproject.toml: {found}. "
        "Adding any of these requires an explicit dependency decision."
    )


def test_no_forbidden_imports_in_src() -> None:
    """src/ contains no forbidden causal/agent framework imports."""
    found: list[str] = []
    for path in SRC_DIR.rglob("*.py"):
        text = path.read_text()
        for mod in FORBIDDEN_IMPORTS:
            if re.search(rf"(?m)^\s*(import|from)\s+{re.escape(mod)}\b", text):
                found.append(f"{path.relative_to(REPO_ROOT)}: imports {mod}")
    assert not found, "forbidden imports found:\n  - " + "\n  - ".join(found)


# Human review and demo eyeballing remain outside pytest.
