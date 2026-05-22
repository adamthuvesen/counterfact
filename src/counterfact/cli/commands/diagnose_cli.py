from __future__ import annotations

import argparse
import sys
from pathlib import Path

from counterfact.cli import formatters, loaders

load_run_file = loaders.load_run_file
load_corpus_dir = loaders.load_corpus_dir
require_focal_in_corpus = loaders.require_focal_in_corpus
format_diagnosis = formatters.format_diagnosis


def run(args: argparse.Namespace) -> int:
    from counterfact.diagnose import build_diagnosis_pair
    from counterfact.explain import render_html

    run_path: Path = args.run_json
    focal = load_run_file(run_path, command="diagnose")
    if focal is None:
        return 2

    runs_dir: Path = args.runs_dir if args.runs_dir is not None else run_path.parent
    corpus = load_corpus_dir(runs_dir, command="diagnose")
    if corpus is None:
        return 2
    if not require_focal_in_corpus(focal, corpus, runs_dir, command="diagnose"):
        return 2

    try:
        report, html_report = build_diagnosis_pair(
            focal,
            corpus,
            decision_type=args.decision_type,
            top_k=args.top_k,
            bootstrap=args.bootstrap,
            seed=args.seed,
            run_path=str(run_path),
            corpus_dir=str(runs_dir),
        )
    except ValueError as exc:
        print(f"counterfact diagnose: {exc}", file=sys.stderr)
        return 2

    if args.html is not None:
        try:
            args.html.write_text(render_html(html_report))
        except OSError as exc:
            print(
                f"counterfact diagnose: failed to write HTML {args.html}: {exc}",
                file=sys.stderr,
            )
            return 2
        if args.json:
            print(str(args.html.resolve()), file=sys.stderr)

    if args.json:
        print(report.model_dump_json(indent=2, exclude_none=True))
    else:
        print(format_diagnosis(report))
        if args.html is not None:
            print(f"html_report: {args.html.resolve()}")
    return 0
