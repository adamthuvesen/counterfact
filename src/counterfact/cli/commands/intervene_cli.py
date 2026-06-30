from __future__ import annotations

import argparse
import sys
from pathlib import Path

from counterfact.cli import formatters, helpers, loaders
from counterfact.errors import InvalidInterventionError
from counterfact.intervene.degenerate import degenerate_estimate, outcome_classes
from counterfact.intervene.estimate import CausalEstimate
from counterfact.schema import Decision, Run, Step

load_focal_and_corpus = loaders.load_focal_and_corpus
resolve_intervention_target = helpers.resolve_intervention_target
parse_decision_edit = helpers.parse_decision_edit
add_cli_diagnostics = helpers.add_cli_diagnostics
format_intervention_estimate = formatters.format_intervention_estimate


def _estimate_for_corpus(
    *,
    focal: Run,
    corpus: list[Run],
    step: Step,
    decision: Decision,
    intervention_kind: str,
    target_value: str,
    bootstrap: int,
    seed: int,
    use_decision_id: bool,
) -> CausalEstimate:
    from counterfact import fit_outcome_model, intervene
    from counterfact.dag import build_dag

    if len(outcome_classes(corpus)) == 1:
        return degenerate_estimate(
            corpus,
            decision_type=decision.decision_type,
            intervention_kind=intervention_kind,
            target=target_value,
        )

    model = fit_outcome_model(corpus, n_bootstrap=bootstrap, seed=seed)
    return intervene(
        dag=build_dag(focal),
        model=model,
        step=step.step_index,
        intervention={intervention_kind: target_value},
        decision_id=decision.decision_id if use_decision_id else None,
    )


def run(args: argparse.Namespace) -> int:
    from counterfact.taxonomy import is_valid_intervention

    run_path: Path = args.run_json
    loaded = load_focal_and_corpus(run_path, args.runs_dir, command="intervene")
    if loaded is None:
        return 2
    focal, corpus, _runs_dir = loaded

    target = resolve_intervention_target(args, focal)
    if target is None:
        return 2
    step, decision = target

    parsed_edit = parse_decision_edit(args.set_value)
    if parsed_edit is None:
        return 2
    intervention_kind, target_value = parsed_edit
    use_decision_id = args.decision_id is not None
    if not is_valid_intervention(decision.decision_type, intervention_kind):
        print(
            "counterfact intervene: intervention "
            f"{intervention_kind!r} is not valid on decision type "
            f"{decision.decision_type!r}",
            file=sys.stderr,
        )
        return 2

    try:
        estimate = _estimate_for_corpus(
            focal=focal,
            corpus=corpus,
            step=step,
            decision=decision,
            intervention_kind=intervention_kind,
            target_value=target_value,
            bootstrap=args.bootstrap,
            seed=args.seed,
            use_decision_id=use_decision_id,
        )
    except InvalidInterventionError as exc:
        print(f"counterfact intervene: {exc}", file=sys.stderr)
        return 2
    estimate = add_cli_diagnostics(
        estimate,
        decision=decision,
        step=step,
        targeting_mode="decision_id" if use_decision_id else "step",
    )

    estimate_json = estimate.model_dump_json(indent=2)
    if args.output is not None:
        args.output.write_text(estimate_json + "\n")
        print(str(args.output.resolve()), file=sys.stderr)

    if args.json:
        print(estimate_json)
    else:
        print(
            format_intervention_estimate(
                estimate=estimate,
                run=focal,
                decision=decision,
                step=step,
                intervention_kind=intervention_kind,
                target=target_value,
            )
        )
    return 0
