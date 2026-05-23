"""argparse wiring for counterfact CLI."""

from __future__ import annotations

import argparse
from pathlib import Path

from counterfact.cli import argparse_types
from counterfact.cli.commands import (
    analyze_corpus,
    bench_real,
    bench_synthetic,
    compare_cli,
    demo,
    diagnose_cli,
    explain,
    export_runs_cli,
    ingest_claude_agent_sdk_cli,
    ingest_generic_jsonl_cli,
    ingest_openai_agents_cli,
    intervene_cli,
)
from counterfact.cli.constants import DEFAULT_DEMO_RUNS_DIR

positive_int = argparse_types.positive_int


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="counterfact", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    demo_parser = sub.add_parser(
        "demo",
        help="Print a local naive-vs-honest causal demo without LLM calls",
    )
    demo_parser.add_argument(
        "--runs-dir",
        type=Path,
        default=DEFAULT_DEMO_RUNS_DIR,
        help="Directory of committed real traces (default: bench/real/smoke_mixed_outcome)",
    )
    demo_parser.add_argument(
        "--decision-type",
        choices=["model_call", "tool_call", "retry"],
        default="model_call",
        help="Decision type to summarize (default: model_call)",
    )
    demo_parser.add_argument("--target", default=None, help="Optional intervention arm")
    demo_parser.add_argument("--synthetic-n", type=int, default=500)
    demo_parser.add_argument("--seed", type=int, default=42)
    demo_parser.add_argument("--bootstrap", type=positive_int, default=200)
    demo_parser.add_argument(
        "--synthetic-fallback",
        action="store_true",
        help="Use a synthetic SCM corpus if --runs-dir has no trace JSON files",
    )
    demo_parser.add_argument(
        "--confound",
        action="store_true",
        help=(
            "Run the confounded synthetic showcase: generate a fresh "
            "synthetic corpus where model_choice is biased by tool_choice, "
            "and surface the naive-vs-causal contrast."
        ),
    )
    demo_parser.set_defaults(func=demo.run)

    explain_parser = sub.add_parser(
        "explain",
        help=(
            "Render a self-contained HTML report explaining one trace, grounded in CausalEstimate"
        ),
    )
    explain_parser.add_argument(
        "run_json",
        type=Path,
        help="Path to a single Run JSON file (the focal trace)",
    )
    explain_parser.add_argument(
        "--runs-dir",
        type=Path,
        default=None,
        help=(
            "Corpus directory (defaults to the parent directory of run_json). "
            "Must contain the focal run."
        ),
    )
    explain_parser.add_argument(
        "--decision-type",
        choices=["model_call", "tool_call", "retry"],
        default="model_call",
        help="Decision type to summarize (default: model_call)",
    )
    explain_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=("Output HTML path (default: <run-json-parent>/counterfact-explain-<run_id>.html)"),
    )
    explain_parser.add_argument("--bootstrap", type=positive_int, default=200)
    explain_parser.add_argument("--seed", type=int, default=42)
    explain_parser.set_defaults(func=explain.run)

    intervene_parser = sub.add_parser(
        "intervene",
        help="Estimate one decision edit on a trace and emit a CausalEstimate",
    )
    intervene_parser.add_argument(
        "run_json",
        type=Path,
        help="Path to a single Run JSON file (the focal trace)",
    )
    intervene_parser.add_argument(
        "--runs-dir",
        type=Path,
        default=None,
        help=(
            "Trace corpus directory (defaults to the parent directory of run_json). "
            "Must contain the focal run."
        ),
    )
    intervene_parser.add_argument(
        "--decision-id",
        default=None,
        help="Target a specific Decision.decision_id",
    )
    intervene_parser.add_argument(
        "--step",
        type=int,
        default=None,
        help="Target a single-decision step by step_index",
    )
    intervene_parser.add_argument(
        "--set",
        dest="set_value",
        required=True,
        help="Decision edit as key=value, e.g. model_choice=sonnet",
    )
    intervene_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit only CausalEstimate JSON to stdout",
    )
    intervene_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write the CausalEstimate JSON artifact to this path",
    )
    intervene_parser.add_argument("--bootstrap", type=positive_int, default=200)
    intervene_parser.add_argument("--seed", type=int, default=42)
    intervene_parser.set_defaults(func=intervene_cli.run)

    diagnose = sub.add_parser(
        "diagnose",
        help="Rank likely load-bearing decisions for one trace",
    )
    diagnose.add_argument("run_json", type=Path, help="Path to the focal Run JSON")
    diagnose.add_argument(
        "--runs-dir",
        type=Path,
        default=None,
        help="Trace corpus directory (defaults to the parent directory of run_json)",
    )
    diagnose.add_argument(
        "--decision-type",
        choices=["model_call", "tool_call", "retry"],
        default="model_call",
        help="Only rank decisions of this type",
    )
    diagnose.add_argument("--top-k", type=positive_int, default=3)
    diagnose.add_argument("--bootstrap", type=positive_int, default=200)
    diagnose.add_argument("--seed", type=int, default=42)
    diagnose.add_argument("--json", action="store_true", help="Emit JSON only")
    diagnose.add_argument(
        "--html",
        type=Path,
        default=None,
        help="Write a self-contained diagnosis-first HTML report to this path",
    )
    diagnose.set_defaults(func=diagnose_cli.run)

    compare = sub.add_parser(
        "compare",
        help="Compare two traces descriptively, with optional diagnosis overlay",
    )
    compare.add_argument("left_run_json", type=Path)
    compare.add_argument("right_run_json", type=Path)
    compare.add_argument(
        "--runs-dir",
        type=Path,
        default=None,
        help="Optional corpus directory for a diagnosis overlay",
    )
    compare.add_argument(
        "--focal",
        choices=["left", "right"],
        default="right",
        help="Which trace to diagnose when --runs-dir is supplied",
    )
    compare.add_argument(
        "--decision-type",
        choices=["model_call", "tool_call", "retry"],
        default="model_call",
    )
    compare.add_argument("--top-k", type=positive_int, default=3)
    compare.add_argument("--bootstrap", type=positive_int, default=200)
    compare.add_argument("--seed", type=int, default=42)
    compare.add_argument("--json", action="store_true", help="Emit JSON only")
    compare.set_defaults(func=compare_cli.run)

    ingest = sub.add_parser("ingest", help="Convert external trace data to native Run JSON")
    ingest.add_argument(
        "--list-formats",
        action="store_true",
        help="List supported source formats and exit",
    )
    ingest_sub = ingest.add_subparsers(dest="ingest_kind", required=False)
    generic_jsonl = ingest_sub.add_parser(
        "generic-jsonl",
        help="Convert JSONL records through an explicit mapping file",
    )
    generic_jsonl.add_argument("source_jsonl", type=Path)
    generic_jsonl.add_argument("--mapping", type=Path, required=True)
    generic_jsonl.add_argument("--output-dir", type=Path, required=True)
    generic_jsonl.add_argument("--json", action="store_true", help="Emit receipt JSON")
    generic_jsonl.set_defaults(func=ingest_generic_jsonl_cli.run)

    claude_sdk = ingest_sub.add_parser(
        "claude-agent-sdk",
        help=(
            "Convert a JSONL stream of Claude Agent SDK message dataclass dumps "
            "to native Run JSON (zero-config — no mapping required)"
        ),
    )
    claude_sdk.add_argument(
        "source_jsonl",
        type=Path,
        help=(
            'JSONL where each line is either {"messages": [...]} or a JSON list '
            "of message dicts captured from claude_agent_sdk.query()"
        ),
    )
    claude_sdk.add_argument("--output-dir", type=Path, required=True)
    claude_sdk.add_argument("--json", action="store_true", help="Emit receipt JSON")
    claude_sdk.set_defaults(func=ingest_claude_agent_sdk_cli.run)

    openai_agents = ingest_sub.add_parser(
        "openai-agents",
        help=(
            "Convert an OpenAI Agents SDK trace export (one JSON file with a flat "
            "spans array) to native Run JSON"
        ),
    )
    openai_agents.add_argument(
        "source_json",
        type=Path,
        help="Path to a JSON file containing one trace or a list of traces",
    )
    openai_agents.add_argument("--output-dir", type=Path, required=True)
    openai_agents.add_argument(
        "--outcome",
        choices=["pass", "fail"],
        default=None,
        help=(
            "Explicit binary outcome for traces without a counterfact.outcome "
            "marker span and without a root error. The adapter never infers "
            "outcomes from the absence of error."
        ),
    )
    openai_agents.add_argument("--json", action="store_true", help="Emit receipt JSON")
    openai_agents.set_defaults(func=ingest_openai_agents_cli.run)

    export_runs = sub.add_parser(
        "export-runs",
        help="Export native traces to another research artifact format",
    )
    export_runs.add_argument("runs_dir", type=Path, help="Directory of native trace JSON")
    export_runs.add_argument("--to", required=True, choices=["runrecord-parquet"])
    export_runs.add_argument("--output", type=Path, default=None)
    export_runs.add_argument("--json", action="store_true", help="Emit receipt JSON")
    export_runs.set_defaults(func=export_runs_cli.run)

    bench = sub.add_parser("bench", help="Generate CounterBench corpora")
    bench_sub = bench.add_subparsers(dest="bench_kind", required=True)

    syn = bench_sub.add_parser(
        "synthetic", help="Generate synthetic SCM traces (no LLM, deterministic)"
    )
    syn.add_argument("--n", type=int, required=True, help="Number of traces to generate")
    syn.add_argument("--seed", type=int, default=42, help="RNG seed (default 42)")
    syn.add_argument(
        "--output-dir",
        type=Path,
        default=Path("bench/synthetic/_out"),
        help="Where to write traces (default: bench/synthetic/_out)",
    )
    syn.set_defaults(func=bench_synthetic.run)

    real = bench_sub.add_parser("real", help="Generate real-agent traces (HUMAN GATE on first run)")
    real.add_argument("--n", type=int, required=True)
    real.add_argument("--budget-cap", type=float, default=50.0)
    real.add_argument("--output-dir", type=Path, default=Path("bench/real/pilot"))
    real.add_argument("--seed", type=int, default=0, help="Per-trace RNG seed (default: 0)")
    real.add_argument(
        "--epsilon",
        type=float,
        default=0.2,
        help="Default ε used for any decision whose --*-epsilon is not set (default: 0.2)",
    )
    real.add_argument("--tool-greedy", type=str, default="inspect_file")
    real.add_argument("--tool-epsilon", type=float, default=None)
    real.add_argument(
        "--model-greedy",
        type=str,
        default="large",
        choices=["small", "large"],
        help="Greedy arm for model_choice (default: large)",
    )
    real.add_argument(
        "--model-epsilon",
        type=float,
        default=None,
        help="ε for model_choice; falls back to --epsilon when unset",
    )
    real.add_argument(
        "--retry-greedy",
        type=str,
        default="retry_once",
        choices=["no_retry", "retry_once"],
        help="Greedy arm for retry_policy (default: retry_once)",
    )
    real.add_argument(
        "--retry-epsilon",
        type=float,
        default=None,
        help="ε for retry_policy; falls back to --epsilon when unset",
    )
    real.add_argument(
        "--fixtures",
        type=str,
        default=None,
        help=(
            "Comma-separated fixture ids to iterate over (e.g. 'csv_dedupe' "
            "or 'csv_dedupe,date_window'). Overrides --fixture-set."
        ),
    )
    real.add_argument(
        "--fixture-set",
        type=str,
        default=None,
        choices=[
            "v0",
            "easy",
            "hidden_v1",
            "hard_hidden_v1",
            "broad_calibration",
            "very_hard_hidden_v1",
            "stateful_calibration",
        ],
        help=(
            "Named fixture-set shortcut. Use 'broad_calibration' for broad "
            "date/rate-limit/version calibration, or 'stateful_calibration' "
            "for the streaming watermark fixture. Other choices are legacy "
            "harness fixtures kept for tests and historical calibration."
        ),
    )
    real.set_defaults(func=bench_real.run)

    analyze = sub.add_parser(
        "analyze",
        help="Check corpus support-readiness for counterfactual diagnosis",
    )
    analyze_sub = analyze.add_subparsers(dest="analyze_kind", required=True)
    corpus = analyze_sub.add_parser(
        "corpus",
        help="Run the corpus-readiness analyzer on a directory of trace JSON files",
    )
    corpus.add_argument("runs_dir", type=Path, help="Directory of trace JSON files")
    corpus.add_argument(
        "--min-pass-rate",
        type=float,
        default=None,
        help="Override RubricThresholds.min_pass_rate (default: 0.3)",
    )
    corpus.add_argument(
        "--max-pass-rate",
        type=float,
        default=None,
        help="Override RubricThresholds.max_pass_rate (default: 0.7)",
    )
    corpus.add_argument(
        "--min-arms",
        type=int,
        default=None,
        help="Override RubricThresholds.min_arms_per_decision_type (default: 2)",
    )
    corpus.add_argument(
        "--min-n-per-arm",
        type=int,
        default=None,
        help="Override RubricThresholds.min_n_per_arm (default: 5)",
    )
    corpus.add_argument(
        "--min-identified",
        type=int,
        default=None,
        help="Override RubricThresholds.min_identified_decision_types (default: 1)",
    )
    corpus.set_defaults(func=analyze_corpus.run)

    return p


_INGEST_FORMATS = [
    ("claude-agent-sdk", "Claude Agent SDK message JSONL (zero-config)"),
    ("openai-agents", "OpenAI Agents SDK trace JSON (requires --outcome unless derivable)"),
    ("generic-jsonl", "Any JSONL with an explicit user-supplied --mapping file"),
]
