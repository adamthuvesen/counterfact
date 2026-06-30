from __future__ import annotations

import argparse
import sys
from pathlib import Path

from counterfact.cli import formatters, loaders
from counterfact.explain import ExplainReport

load_focal_and_corpus = loaders.load_focal_and_corpus
format_diagnosis = formatters.format_diagnosis


def _write_html_report(path: Path, report: ExplainReport) -> bool:
    from counterfact.explain import render_html

    try:
        path.write_text(render_html(report))
    except OSError as exc:
        print(
            f"counterfact diagnose: failed to write HTML {path}: {exc}",
            file=sys.stderr,
        )
        return False
    return True


def run(args: argparse.Namespace) -> int:
    from counterfact.diagnose import build_diagnosis_pair

    run_path: Path = args.run_json
    loaded = load_focal_and_corpus(run_path, args.runs_dir, command="diagnose")
    if loaded is None:
        return 2
    focal, corpus, runs_dir = loaded

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
        if not _write_html_report(args.html, html_report):
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
