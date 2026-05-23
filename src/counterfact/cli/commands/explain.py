from __future__ import annotations

import argparse
from pathlib import Path

from counterfact.cli import loaders

load_run_file = loaders.load_run_file
load_corpus_dir = loaders.load_corpus_dir
require_focal_in_corpus = loaders.require_focal_in_corpus


def run(args: argparse.Namespace) -> int:
    from counterfact.explain import build_report, render_html

    run_path: Path = args.run_json
    focal = load_run_file(run_path, command="explain")
    if focal is None:
        return 2

    runs_dir: Path = args.runs_dir if args.runs_dir is not None else run_path.parent
    corpus = load_corpus_dir(runs_dir, command="explain")
    if corpus is None:
        return 2
    if not require_focal_in_corpus(focal, corpus, runs_dir, command="explain"):
        return 2

    output: Path = (
        args.output
        if args.output is not None
        else (run_path.parent / f"counterfact-explain-{focal.run_id}.html")
    )

    report = build_report(
        focal,
        corpus,
        decision_type=args.decision_type,
        bootstrap=args.bootstrap,
        seed=args.seed,
        run_path=str(run_path),
        corpus_dir=str(runs_dir),
    )
    html = render_html(report)
    output.write_text(html)
    print(str(output.resolve()))
    return 0
