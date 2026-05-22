from __future__ import annotations

import argparse
import sys
from pathlib import Path

from counterfact.cli import formatters, helpers, loaders
from counterfact.errors import InvalidInterventionError
from counterfact.intervene.degenerate import degenerate_estimate, outcome_classes

load_run_file = loaders.load_run_file
load_corpus_dir = loaders.load_corpus_dir
require_focal_in_corpus = loaders.require_focal_in_corpus
resolve_intervention_target = helpers.resolve_intervention_target
parse_decision_edit = helpers.parse_decision_edit
add_cli_diagnostics = helpers.add_cli_diagnostics
format_intervention_estimate = formatters.format_intervention_estimate


def run(args: argparse.Namespace) -> int:
    from counterfact import fit_outcome_model, intervene
    from counterfact.dag import build_dag
    from counterfact.taxonomy import is_valid_intervention

    run_path: Path = args.run_json
    focal = load_run_file(run_path, command="intervene")
    if focal is None:
        return 2

    runs_dir: Path = args.runs_dir if args.runs_dir is not None else run_path.parent
    corpus = load_corpus_dir(runs_dir, command="intervene")
    if corpus is None:
        return 2
    if not require_focal_in_corpus(focal, corpus, runs_dir, command="intervene"):
        return 2

    target = resolve_intervention_target(args, focal)
    if target is None:
        return 2
    step, decision = target

    parsed_edit = parse_decision_edit(args.set_value)
    if parsed_edit is None:
        return 2
    intervention_kind, target_value = parsed_edit
    if not is_valid_intervention(decision.decision_type, intervention_kind):
        print(
            "counterfact intervene: intervention "
            f"{intervention_kind!r} is not valid on decision type "
            f"{decision.decision_type!r}",
            file=sys.stderr,
        )
        return 2

    try:
        if len(outcome_classes(corpus)) == 1:
            estimate = degenerate_estimate(
                corpus,
                decision_type=decision.decision_type,
                intervention_kind=intervention_kind,
                target=target_value,
            )
        else:
            model = fit_outcome_model(corpus, n_bootstrap=args.bootstrap, seed=args.seed)
            estimate = intervene(
                dag=build_dag(focal),
                model=model,
                step=step.step_index,
                intervention={intervention_kind: target_value},
                decision_id=decision.decision_id if args.decision_id is not None else None,
            )
    except InvalidInterventionError as exc:
        print(f"counterfact intervene: {exc}", file=sys.stderr)
        return 2
    estimate = add_cli_diagnostics(
        estimate,
        decision=decision,
        step=step,
        targeting_mode="decision_id" if args.decision_id is not None else "step",
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
