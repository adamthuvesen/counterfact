from __future__ import annotations

import argparse
import sys
from pathlib import Path

from counterfact.cli import loaders

load_corpus_dir = loaders.load_corpus_dir


def run(args: argparse.Namespace) -> int:
    from counterfact.runrecord_export import export_runrecord_parquet

    if args.to != "runrecord-parquet":
        print(f"counterfact export-runs: unsupported target {args.to!r}", file=sys.stderr)
        return 2
    corpus = load_corpus_dir(args.runs_dir, command="export-runs")
    if corpus is None:
        return 2
    output = args.output or (args.runs_dir / "runrecord.parquet")
    receipt = export_runrecord_parquet(corpus, source_corpus=args.runs_dir, output_path=output)
    if args.json:
        print(receipt.model_dump_json(indent=2))
    else:
        print(f"counterfact export-runs: wrote {Path(receipt.output_path).resolve()}")
        receipt_path = Path(receipt.output_path).with_suffix(
            Path(receipt.output_path).suffix + ".receipt.json"
        )
        print(f"receipt: {receipt_path.resolve()}")
        print(f"rows: {receipt.row_count}")
        if receipt.warnings:
            print(f"warnings: {len(receipt.warnings)}")
    return 0
