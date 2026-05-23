from __future__ import annotations

import argparse

from counterfact.cli import formatters, loaders

load_run_file = loaders.load_run_file
load_corpus_dir = loaders.load_corpus_dir
require_focal_in_corpus = loaders.require_focal_in_corpus
format_comparison = formatters.format_comparison


def run(args: argparse.Namespace) -> int:
    from counterfact.compare import compare_traces
    from counterfact.diagnose import build_diagnosis

    left = load_run_file(args.left_run_json, command="compare")
    if left is None:
        return 2
    right = load_run_file(args.right_run_json, command="compare")
    if right is None:
        return 2

    diagnosis = None
    if args.runs_dir is not None:
        corpus = load_corpus_dir(args.runs_dir, command="compare")
        if corpus is None:
            return 2
        focal = left if args.focal == "left" else right
        if not require_focal_in_corpus(focal, corpus, args.runs_dir, command="compare"):
            return 2
        diagnosis = build_diagnosis(
            focal,
            corpus,
            decision_type=args.decision_type,
            top_k=args.top_k,
            bootstrap=args.bootstrap,
            seed=args.seed,
            run_path=str(args.left_run_json if args.focal == "left" else args.right_run_json),
            corpus_dir=str(args.runs_dir),
        )

    comparison = compare_traces(left, right, diagnosis=diagnosis)
    if args.json:
        print(comparison.model_dump_json(indent=2, exclude_none=True))
    else:
        print(format_comparison(comparison))
    return 0
