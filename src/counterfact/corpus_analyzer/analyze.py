"""Corpus-readiness analyzer.

Loads no files itself; takes a list of `Run` objects, computes the diagnostics
the rubric needs, scores them, returns a `CorpusReadinessReport`. Deterministic,
no LLM calls, no filesystem writes.
"""

from __future__ import annotations

from collections.abc import Iterable

from counterfact.baselines import pass_rate_by_arm
from counterfact.corpus_analyzer.report import (
    ArmSupportRow,
    CorpusReadinessReport,
    IdentifiabilityCoverage,
    IdentifiabilityName,
    OutcomeBalance,
    RubricCriterion,
)
from counterfact.corpus_analyzer.rubric import DEFAULT_THRESHOLDS, RubricThresholds
from counterfact.dag import build_dag
from counterfact.errors import (
    InsufficientOutcomeSupportError,
    InvalidInterventionError,
)
from counterfact.intervene import intervene
from counterfact.outcome import fit_outcome_model
from counterfact.outcome.binary import binary_outcome_value
from counterfact.outcome.model import OutcomeModel
from counterfact.schema import Run
from counterfact.taxonomy import DECISION_TYPES, valid_interventions

# Bootstrap draws for the model fit used to probe identifiability coverage.
# Small because the analyzer doesn't need tight CIs — it only reads the
# returned identifiability label per query.
_PROBE_BOOTSTRAP = 50
_PROBE_SEED = 42


def _outcome_balance(runs: list[Run]) -> OutcomeBalance:
    n_pass = sum(1 for r in runs if binary_outcome_value(r))
    n_fail = len(runs) - n_pass
    pass_rate = (n_pass / len(runs)) if runs else 0.0
    return OutcomeBalance(pass_rate=pass_rate, n_pass=n_pass, n_fail=n_fail)


def _arm_support(runs: list[Run]) -> list[ArmSupportRow]:
    rows: list[ArmSupportRow] = []
    for dt in DECISION_TYPES:
        if not valid_interventions(dt):
            continue
        table = pass_rate_by_arm(runs, dt)
        for r in table.rows:
            rows.append(
                ArmSupportRow(
                    decision_type=dt,
                    arm=r.arm,
                    n=r.n,
                    pass_count=r.pass_count,
                    pass_rate=r.pass_rate,
                )
            )
    return rows


def _intervention_kinds_for(decision_type: str) -> tuple[str, ...]:
    """Pick a representative intervention kind per decision type for probing.

    The rubric only cares whether *some* intervention on this decision type
    can return `identified`. The first valid intervention kind in declared
    order is sufficient; deeper coverage is overkill at v0.
    """
    return tuple(sorted(valid_interventions(decision_type)))


def _index_single_decision_steps(
    runs: list[Run],
) -> dict[tuple[str, str], tuple[Run, int]]:
    """Build a (decision_type, chosen_action) -> (run, step_index) index.

    The probe loop hits this lookup in a triple loop; computing it once turns
    the inner search from O(runs * steps) per probe into O(1).
    """
    index: dict[tuple[str, str], tuple[Run, int]] = {}
    for run in runs:
        for step in run.steps:
            if len(step.decisions) != 1:
                continue
            d = step.decisions[0]
            if not d.chosen_action:
                continue
            key = (d.decision_type, d.chosen_action)
            index.setdefault(key, (run, step.step_index))
    return index


def _unfittable_coverage() -> tuple[IdentifiabilityCoverage, set[str]]:
    return (
        IdentifiabilityCoverage(reachable=["unidentified"], unfittable_outcome_model=True),
        set(),
    )


def _fit_probe_model(runs: list[Run]) -> OutcomeModel | None:
    try:
        return fit_outcome_model(runs, n_bootstrap=_PROBE_BOOTSTRAP, seed=_PROBE_SEED)
    except InsufficientOutcomeSupportError:
        return None


def _arms_by_decision_type(arms: list[ArmSupportRow]) -> dict[str, list[str]]:
    by_dt: dict[str, list[str]] = {}
    for row in arms:
        by_dt.setdefault(row.decision_type, []).append(row.arm)
    return by_dt


def _probe_reachable_identifiability(
    *,
    model: OutcomeModel,
    arms_by_dt: dict[str, list[str]],
    step_index: dict[tuple[str, str], tuple[Run, int]],
) -> tuple[set[IdentifiabilityName], set[str]]:
    reachable: set[IdentifiabilityName] = set()
    identified_decision_types: set[str] = set()
    kinds_by_dt = {dt: _intervention_kinds_for(dt) for dt in arms_by_dt}

    for dt, dt_arms in arms_by_dt.items():
        for kind in kinds_by_dt[dt]:
            for arm in dt_arms:
                located = step_index.get((dt, arm))
                if located is None:
                    continue
                run, step_idx = located
                try:
                    est = intervene(
                        dag=build_dag(run),
                        model=model,
                        step=step_idx,
                        intervention={kind: arm},
                    )
                except InvalidInterventionError:
                    continue
                label: IdentifiabilityName = est.identifiability.value
                reachable.add(label)
                if label == "identified":
                    identified_decision_types.add(dt)
    return reachable, identified_decision_types


def _identifiability_coverage(
    runs: list[Run], arms: list[ArmSupportRow]
) -> tuple[IdentifiabilityCoverage, set[str]]:
    """Probe `intervene` for every (decision_type, intervention_kind, arm) we
    can target and collect which identifiability labels came back. Also returns
    the set of decision types that produced at least one `identified` result.
    """
    classes = {binary_outcome_value(r) for r in runs}

    if len(classes) < 2:
        # Single-class corpus — outcome model can't fit. Only the unidentified
        # paths (always-replay, degenerate) are reachable through the engine.
        # We mark that explicitly so the rubric reasoning is honest.
        return _unfittable_coverage()

    model = _fit_probe_model(runs)
    if model is None:
        return _unfittable_coverage()

    # Precompute once: intervention kinds per decision type and a single-step
    # lookup index. Both were recomputed inside the triple loop before, which
    # made probing O(decision_types * arms * runs * steps).
    reachable, identified_decision_types = _probe_reachable_identifiability(
        model=model,
        arms_by_dt=_arms_by_decision_type(arms),
        step_index=_index_single_decision_steps(runs),
    )

    # Stable order for the report
    ordered: list[IdentifiabilityName] = [
        s for s in ("identified", "bounded", "unidentified") if s in reachable
    ]
    return (
        IdentifiabilityCoverage(reachable=ordered, unfittable_outcome_model=False),
        identified_decision_types,
    )


def _score_outcome_balance(
    balance: OutcomeBalance, thresholds: RubricThresholds, n_traces: int
) -> RubricCriterion:
    if n_traces == 0:
        return RubricCriterion(
            name="outcome_balance",
            passed=False,
            reason="outcome_balance: empty corpus vs at least 1 trace",
        )
    pr = balance.pass_rate
    lo = thresholds.min_pass_rate
    hi = thresholds.max_pass_rate
    if lo <= pr <= hi:
        return RubricCriterion(name="outcome_balance", passed=True, reason="outcome_balance: ok")
    return RubricCriterion(
        name="outcome_balance",
        passed=False,
        reason=f"outcome_balance: pass_rate={pr:.3f} outside [{lo:.3f}, {hi:.3f}]",
    )


def _score_arm_support(arms: list[ArmSupportRow], thresholds: RubricThresholds) -> RubricCriterion:
    by_dt: dict[str, list[ArmSupportRow]] = {}
    for row in arms:
        by_dt.setdefault(row.decision_type, []).append(row)

    best_dt: str | None = None
    best_count = 0
    for dt, rows in by_dt.items():
        count = sum(1 for r in rows if r.n >= thresholds.min_n_per_arm)
        if count > best_count:
            best_count = count
            best_dt = dt

    if best_count >= thresholds.min_arms_per_decision_type:
        return RubricCriterion(name="arm_support", passed=True, reason="arm_support: ok")

    if best_dt is None:
        return RubricCriterion(
            name="arm_support",
            passed=False,
            reason=(
                f"arm_support: no randomized decision type has any arms "
                f"with n>={thresholds.min_n_per_arm} "
                f"(need >={thresholds.min_arms_per_decision_type})"
            ),
        )

    arm_strs = [f"{r.arm}={r.n}" for r in sorted(by_dt[best_dt], key=lambda r: -r.n)]
    return RubricCriterion(
        name="arm_support",
        passed=False,
        reason=(
            f"arm_support: best decision_type={best_dt} has "
            f"{best_count} arm(s) with n>={thresholds.min_n_per_arm} "
            f"(need >={thresholds.min_arms_per_decision_type}); "
            f"arms: {', '.join(arm_strs)}"
        ),
    )


def _score_identifiability(
    coverage: IdentifiabilityCoverage,
    identified_decision_types: set[str],
    thresholds: RubricThresholds,
) -> RubricCriterion:
    have = len(identified_decision_types)
    need = thresholds.min_identified_decision_types
    if have >= need:
        return RubricCriterion(
            name="identifiability_coverage", passed=True, reason="identifiability_coverage: ok"
        )
    if coverage.unfittable_outcome_model:
        return RubricCriterion(
            name="identifiability_coverage",
            passed=False,
            reason=(
                "identifiability_coverage: outcome model unfittable "
                "(single-class corpus); no identified results possible"
            ),
        )
    reachable_str = ",".join(coverage.reachable) if coverage.reachable else "(none)"
    return RubricCriterion(
        name="identifiability_coverage",
        passed=False,
        reason=(
            f"identifiability_coverage: {have} identified decision type(s) "
            f"vs >={need}; reachable={reachable_str}"
        ),
    )


def _score_model_arm_outcome_mix(
    arms: list[ArmSupportRow], thresholds: RubricThresholds
) -> RubricCriterion:
    """Arm-agnostic check: at least 2 model_call arms with both pass and fail.

    Earlier versions hardcoded `small`/`large` from the synthetic SCM and
    silently passed for any real corpus that uses real model identifiers
    (`claude-sonnet-4-6`, `gpt-4o`, ...). The check now operates on whatever
    model_call arms the corpus actually contains: the criterion is meaningful
    if *any* two arms exist and at least two of them carry mixed outcomes.
    """
    if not thresholds.require_model_arm_outcome_mix:
        return RubricCriterion(
            name="model_arm_outcome_mix",
            passed=True,
            reason="model_arm_outcome_mix: ok",
        )

    model_arms = [row for row in arms if row.decision_type == "model_call"]
    # No model_call decisions at all: criterion does not apply (consistent with
    # the old "no recognized arms" branch).
    if not model_arms:
        return RubricCriterion(
            name="model_arm_outcome_mix",
            passed=True,
            reason="model_arm_outcome_mix: ok",
        )

    if len(model_arms) < 2:
        only = model_arms[0]
        return RubricCriterion(
            name="model_arm_outcome_mix",
            passed=False,
            reason=(
                f"model_arm_outcome_mix: only one model_call arm "
                f"({only.arm}); need >=2 arms with mixed outcomes"
            ),
        )

    mixed_arms: list[str] = []
    single_class: list[str] = []
    for row in sorted(model_arms, key=lambda r: r.arm):
        fail_count = row.n - row.pass_count
        if row.pass_count > 0 and fail_count > 0:
            mixed_arms.append(row.arm)
        else:
            single_class.append(f"{row.arm}=pass:{row.pass_count},fail:{fail_count}")

    if len(mixed_arms) >= 2:
        return RubricCriterion(
            name="model_arm_outcome_mix",
            passed=True,
            reason="model_arm_outcome_mix: ok",
        )

    return RubricCriterion(
        name="model_arm_outcome_mix",
        passed=False,
        reason="model_arm_outcome_mix: single-class arms: " + "; ".join(single_class),
    )


def analyze(
    runs: Iterable[Run],
    *,
    thresholds: RubricThresholds = DEFAULT_THRESHOLDS,
) -> CorpusReadinessReport:
    """Check whether a corpus is ready for counterfactual support questions.

    Returns a `CorpusReadinessReport`. The function is deterministic, makes
    no LLM calls, and writes no files. Empty inputs are reported (not raised).
    """
    runs_list: list[Run] = list(runs)
    balance = _outcome_balance(runs_list)
    arms = _arm_support(runs_list)
    coverage, identified_dts = _identifiability_coverage(runs_list, arms)

    criteria = [
        _score_outcome_balance(balance, thresholds, len(runs_list)),
        _score_arm_support(arms, thresholds),
        _score_identifiability(coverage, identified_dts, thresholds),
        _score_model_arm_outcome_mix(arms, thresholds),
    ]
    promote = all(c.passed for c in criteria)

    return CorpusReadinessReport(
        n_traces=len(runs_list),
        outcome_balance=balance,
        arm_support=arms,
        identifiability_coverage=coverage,
        criteria=criteria,
        promote=promote,
        thresholds=thresholds,
    )
