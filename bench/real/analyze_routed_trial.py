"""Score the engine against a randomized trial run inside the same corpus.

A corpus from `counterfact bench real --model-route-by-fixture ...` holds two
logged designs for the first `model_call`:

* `trial`: P(large) = 0.5 on every fixture. Its difference in pass rates is an
  unbiased estimate of the true effect, so it serves as ground truth.
* `fixture_router`: P(large) depends on the fixture, which also drives task
  difficulty. A naive comparison of these runs is confounded.

The report compares, on the routed runs, the naive gap and the engine's
estimate against the trial truth. It also strips the propensities from the
routed runs, which is what most production traces look like, and checks that
the engine refuses.

Usage: `python -m bench.real.analyze_routed_trial <corpus-dir>`
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

from counterfact import build_dag, fit_outcome_model, intervene
from counterfact.intervene import CausalEstimate
from counterfact.outcome.binary import binary_outcome_value
from counterfact.schema import Decision, Run
from counterfact.stats import Z_95

TARGET, BASELINE = "large", "small"


@dataclass(frozen=True)
class RoutedTrialReport:
    n_trial: int
    n_routed: int
    trial_effect: float
    trial_se: float
    naive_routed: float
    engine_routed: CausalEstimate
    engine_without_propensities: CausalEstimate


def _first_model_call(run: Run) -> Decision | None:
    return next(
        (d for step in run.steps for d in step.decisions if d.decision_type == "model_call"),
        None,
    )


def _difference_in_means(runs: list[Run], design: str) -> tuple[float, float]:
    """Per-run pass-rate gap keyed on the first model_call, with its standard error.

    The harness logs a second model_call when it retries a failed attempt;
    counting decisions instead of runs would double-weight those runs.
    """
    outcomes: dict[str, list[bool]] = {TARGET: [], BASELINE: []}
    for run in runs:
        first = _first_model_call(run)
        if first is not None and first.chosen_action in outcomes:
            outcomes[first.chosen_action].append(binary_outcome_value(run))
    missing = [arm for arm, values in outcomes.items() if not values]
    if missing:
        raise ValueError(f"the {design} runs have no {missing} model_call arm")
    rates = {arm: sum(v) / len(v) for arm, v in outcomes.items()}
    se = math.sqrt(sum(rates[a] * (1 - rates[a]) / len(outcomes[a]) for a in outcomes))
    return rates[TARGET] - rates[BASELINE], se


def _engine(runs: list[Run]) -> CausalEstimate:
    focal = runs[0]
    step = next(
        s.step_index
        for s in focal.steps
        if s.decisions and s.decisions[0].decision_type == "model_call"
    )
    return intervene(
        dag=build_dag(focal),
        model=fit_outcome_model(runs, seed=0),
        step=step,
        intervention={"model_choice": TARGET},
        baseline=BASELINE,
    )


def _without_propensities(run: Run) -> Run:
    steps = []
    for step in run.steps:
        decisions = [
            d.model_copy(
                update={
                    "policy": None,
                    "policy_params": None,
                    "valid_actions": None,
                    "propensity": None,
                    "context_features": None,
                }
            )
            for d in step.decisions
        ]
        steps.append(step.model_copy(update={"decisions": decisions}))
    return run.model_copy(update={"steps": steps})


def analyze(runs: list[Run]) -> RoutedTrialReport:
    policies = {r.run_id: getattr(_first_model_call(r), "policy", None) for r in runs}
    trial = [r for r in runs if policies[r.run_id] == "trial"]
    routed = [r for r in runs if policies[r.run_id] == "fixture_router"]
    if not trial or not routed:
        raise ValueError(
            f"corpus needs both designs; found {len(trial)} trial and {len(routed)} routed runs"
        )
    trial_effect, trial_se = _difference_in_means(trial, "trial")
    naive_routed, _ = _difference_in_means(routed, "fixture_router")
    return RoutedTrialReport(
        n_trial=len(trial),
        n_routed=len(routed),
        trial_effect=trial_effect,
        trial_se=trial_se,
        naive_routed=naive_routed,
        engine_routed=_engine(routed),
        engine_without_propensities=_engine([_without_propensities(r) for r in routed]),
    )


def to_markdown(report: RoutedTrialReport) -> str:
    lo = report.trial_effect - Z_95 * report.trial_se
    hi = report.trial_effect + Z_95 * report.trial_se
    lines = [
        f"Trial runs: {report.n_trial}. Routed runs: {report.n_routed}.",
        "",
        "| Estimate | Source | Effect of large vs small | 95% interval |",
        "| --- | --- | ---: | --- |",
        f"| Ground truth | trial, difference in means | {report.trial_effect:+.3f} "
        f"| [{lo:+.3f}, {hi:+.3f}] |",
        f"| Naive | routed, pass-rate gap | {report.naive_routed:+.3f} | — |",
    ]
    engine = report.engine_routed
    if engine.outcome_delta is not None:
        d = engine.outcome_delta
        lines.append(
            f"| Engine ({engine.identifiability.value}) | routed, AIPW | {d.point:+.3f} "
            f"| [{d.ci_low:+.3f}, {d.ci_high:+.3f}] |"
        )
    else:
        lines.append(f"| Engine ({engine.identifiability.value}) | routed | — | — |")
    refusal = report.engine_without_propensities
    bounds = refusal.bounds
    interval = f"[{bounds.lower:+.3f}, {bounds.upper:+.3f}] (Manski)" if bounds else "—"
    lines.append(
        f"| Engine ({refusal.identifiability.value}) | routed, propensities removed "
        f"| — | {interval} |"
    )
    if engine.outcome_delta is not None:
        d = engine.outcome_delta
        engine_se = (d.ci_high - d.ci_low) / (2 * Z_95)
        z = (d.point - report.trial_effect) / math.sqrt(engine_se**2 + report.trial_se**2)
        lines += [
            "",
            f"Engine minus truth: {d.point - report.trial_effect:+.3f} (z = {z:+.2f}, "
            "independent samples). Naive minus truth: "
            f"{report.naive_routed - report.trial_effect:+.3f}.",
        ]
    return "\n".join(lines) + "\n"


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print("usage: python -m bench.real.analyze_routed_trial <corpus-dir>", file=sys.stderr)
        return 2
    corpus_dir = Path(argv[0])
    runs = [Run.model_validate_json(p.read_text()) for p in sorted(corpus_dir.glob("real-*.json"))]
    try:
        report = analyze(runs)
    except ValueError as exc:
        print(f"analyze_routed_trial: {exc}", file=sys.stderr)
        return 2
    print(to_markdown(report), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
