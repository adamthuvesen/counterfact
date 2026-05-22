"""CLI constants."""

from __future__ import annotations

from pathlib import Path

DEFAULT_DEMO_RUNS_DIR = Path("bench/real/smoke_mixed_outcome")

BENCH_UNAVAILABLE_MESSAGE = (
    "counterfact bench: the bench harness is not included in the wheel. "
    "Install the development extras with `pip install counterfact[bench]` "
    'or use an editable dev install (`uv pip install -e ".[dev]"`).'
)

DEMO_CONTRAST_THRESHOLD = 0.05
DEMO_CONTRAST_TEMPLATE = (
    "naive_vs_causal_contrast: naive arm gap = {naive:+.3f}; "
    "causal arm gap (do-calculus, g-formula) = {causal:+.3f}; "
    "the marginal table overstates what the corpus supports — see "
    "DAG and assumptions."
)

INGEST_FORMATS = [
    ("claude-agent-sdk", "Claude Agent SDK message JSONL (zero-config)"),
    ("openai-agents", "OpenAI Agents SDK trace JSON (requires --outcome unless derivable)"),
    ("generic-jsonl", "Any JSONL with an explicit user-supplied --mapping file"),
]
