"""Run every benchmark scenario over many seeds and score three estimators.

* naive: the pass-rate difference between arms (`pass_rate_by_arm`).
* adjust-all: the pre-identification engine. It fits one logistic model on
  every logged decision, including ones after the treatment, then applies the
  g-formula. It always returns a number.
* engine: `counterfact.intervene`. It returns a label, and a cross-fitted AIPW
  contrast only when the corpus identifies it.

Scores per scenario: bias and RMSE of each estimator against the exact truth,
the engine's label accuracy, its 95% CI coverage when it reports `identified`,
and the mean Manski bounds when it reports `bounded`. Those bounds hold
without assumptions, so they nearly always contain the truth; the table shows
how wide they are rather than a coverage rate that would flatter them.
"""

from __future__ import annotations

import json
import math
import random
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from bench.synthetic.scenarios import BASELINE, SCENARIOS, TARGET, Scenario, by_name
from counterfact import build_dag, fit_outcome_model, intervene, pass_rate_by_arm
from counterfact.errors import InsufficientOutcomeSupportError
from counterfact.outcome.model import OutcomeModel
from counterfact.schema import Run


@dataclass(frozen=True)
class SeedResult:
    scenario: str
    seed: int
    naive: float | None
    adjust_all: float | None
    label: str
    engine: float | None
    ci_low: float | None
    ci_high: float | None
    bound_low: float | None
    bound_high: float | None


@dataclass(frozen=True)
class ScenarioSummary:
    scenario: str
    breaks: str
    expected_label: str
    n_runs: int
    n_seeds: int
    true_effect: float
    naive_bias: float | None
    naive_rmse: float | None
    adjust_all_bias: float | None
    adjust_all_rmse: float | None
    label_accuracy: float
    engine_bias: float | None
    engine_rmse: float | None
    ci_coverage: float | None
    mean_ci_width: float | None
    bound_coverage: float | None
    mean_bound_low: float | None
    mean_bound_high: float | None


def _naive(runs: list[Run]) -> float | None:
    rates = {row.arm: row.pass_rate for row in pass_rate_by_arm(runs, "model_call").rows}
    if TARGET not in rates or BASELINE not in rates:
        return None
    return rates[TARGET] - rates[BASELINE]


def _adjust_all(model: OutcomeModel) -> float | None:
    index = model.feature_index
    arms = [k for k in index if k.startswith("model_call::")]
    target, baseline = f"model_call::{TARGET}", f"model_call::{BASELINE}"
    if target not in index or baseline not in index:
        return None

    def mean_under(key: str) -> float:
        X = model.train_X.copy()
        X[:, [index[k] for k in arms]] = 0.0
        X[:, index[key]] = 1.0
        return float(model.predict_proba(X).mean())

    return mean_under(target) - mean_under(baseline)


def run_seed(scenario_name: str, seed: int) -> SeedResult:
    scenario = by_name(scenario_name)
    rng = random.Random(f"{scenario.name}:{seed}")
    runs = [Run.model_validate(scenario.sample(rng, i, None)) for i in range(scenario.n)]
    empty = {"engine": None, "ci_low": None, "ci_high": None, "bound_low": None, "bound_high": None}
    try:
        model = fit_outcome_model(runs, seed=seed)
    except InsufficientOutcomeSupportError:
        return SeedResult(scenario.name, seed, _naive(runs), None, "unidentified", **empty)

    focal = runs[0]
    step = next(s.step_index for s in focal.steps if s.decisions[0].decision_type == "model_call")
    estimate = intervene(
        dag=build_dag(focal),
        model=model,
        step=step,
        intervention={"model_choice": TARGET},
        baseline=BASELINE,
    )
    delta, bounds = estimate.outcome_delta, estimate.bounds
    return SeedResult(
        scenario=scenario.name,
        seed=seed,
        naive=_naive(runs),
        adjust_all=_adjust_all(model),
        label=estimate.identifiability.value,
        engine=delta.point if delta else None,
        ci_low=delta.ci_low if delta else None,
        ci_high=delta.ci_high if delta else None,
        bound_low=bounds.lower if bounds else None,
        bound_high=bounds.upper if bounds else None,
    )


def _bias_rmse(values: list[float | None], truth: float) -> tuple[float | None, float | None]:
    errors = np.array([v - truth for v in values if v is not None])
    if errors.size == 0:
        return None, None
    return float(errors.mean()), float(math.sqrt((errors**2).mean()))


def _share(flags: list[bool]) -> float | None:
    return sum(flags) / len(flags) if flags else None


def summarize(scenario: Scenario, results: list[SeedResult], n_runs: int) -> ScenarioSummary:
    truth = scenario.true_effect
    identified = [r for r in results if r.label == "identified"]
    bounded = [r for r in results if r.label == "bounded" and r.bound_low is not None]
    naive_bias, naive_rmse = _bias_rmse([r.naive for r in results], truth)
    adjust_bias, adjust_rmse = _bias_rmse([r.adjust_all for r in results], truth)
    engine_bias, engine_rmse = _bias_rmse([r.engine for r in identified], truth)
    widths = [
        r.ci_high - r.ci_low for r in identified if r.ci_low is not None and r.ci_high is not None
    ]
    return ScenarioSummary(
        scenario=scenario.name,
        breaks=scenario.breaks,
        expected_label=scenario.expected_label,
        n_runs=n_runs,
        n_seeds=len(results),
        true_effect=truth,
        naive_bias=naive_bias,
        naive_rmse=naive_rmse,
        adjust_all_bias=adjust_bias,
        adjust_all_rmse=adjust_rmse,
        label_accuracy=sum(r.label == scenario.expected_label for r in results) / len(results),
        engine_bias=engine_bias,
        engine_rmse=engine_rmse,
        ci_coverage=_share([r.ci_low <= truth <= r.ci_high for r in identified]),  # type: ignore[operator]
        mean_ci_width=float(np.mean(widths)) if widths else None,
        bound_coverage=_share([r.bound_low <= truth <= r.bound_high for r in bounded]),  # type: ignore[operator]
        mean_bound_low=float(np.mean([r.bound_low for r in bounded])) if bounded else None,
        mean_bound_high=float(np.mean([r.bound_high for r in bounded])) if bounded else None,
    )


def run_suite(
    seeds: int,
    *,
    scenario_names: list[str] | None = None,
    workers: int | None = None,
) -> list[ScenarioSummary]:
    names = dict.fromkeys(scenario_names) if scenario_names else [s.name for s in SCENARIOS]
    scenarios = [by_name(name) for name in names]
    jobs = [(s.name, seed) for s in scenarios for seed in range(seeds)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(run_seed, *zip(*jobs, strict=True), chunksize=8))
    return [summarize(s, [r for r in results if r.scenario == s.name], s.n) for s in scenarios]


def _fmt(value: float | None, *, signed: bool = True, pct: bool = False) -> str:
    if value is None:
        return "—"
    if pct:
        return f"{value:.0%}"
    return f"{value:+.3f}" if signed else f"{value:.3f}"


def _bounds(low: float | None, high: float | None) -> str:
    if low is None or high is None:
        return "—"
    return f"[{low:+.2f}, {high:+.2f}]"


def _cell(bias: float | None, rmse: float | None) -> str:
    if bias is None:
        return "—"
    return f"{_fmt(bias)} ({_fmt(rmse, signed=False)})"


def to_markdown(summaries: list[ScenarioSummary], *, seeds: int, command: str) -> str:
    lines = [
        "| Scenario | Breaks | n | Truth | Naive bias (RMSE) | Adjust-all bias (RMSE) "
        "| Engine label (expected) | Engine bias (RMSE) | 95% CI coverage | Mean bounds |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for s in summaries:
        lines.append(
            f"| `{s.scenario}` | {s.breaks} | {s.n_runs} | {_fmt(s.true_effect)} "
            f"| {_cell(s.naive_bias, s.naive_rmse)} "
            f"| {_cell(s.adjust_all_bias, s.adjust_all_rmse)} "
            f"| {_fmt(s.label_accuracy, pct=True)} {s.expected_label} "
            f"| {_cell(s.engine_bias, s.engine_rmse)} | {_fmt(s.ci_coverage, pct=True)} "
            f"| {_bounds(s.mean_bound_low, s.mean_bound_high)} |"
        )
    header = (
        f"Generated by `{command}` over {seeds} seeds per scenario. "
        "Bias and RMSE are against the exact true effect of sonnet vs haiku. "
        "Engine bias, RMSE and CI coverage use only seeds labeled `identified`."
    )
    return header + "\n\n" + "\n".join(lines) + "\n"


def write_outputs(
    summaries: list[ScenarioSummary],
    *,
    seeds: int,
    command: str,
    markdown: Path | None,
    json_path: Path | None,
) -> None:
    if markdown is not None:
        markdown.parent.mkdir(parents=True, exist_ok=True)
        markdown.write_text(to_markdown(summaries, seeds=seeds, command=command))
    if json_path is not None:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"seeds": seeds, "command": command, "scenarios": [asdict(s) for s in summaries]}
        json_path.write_text(json.dumps(payload, indent=2) + "\n")
