"""CLI constants."""

from __future__ import annotations

from pathlib import Path

DEFAULT_DEMO_RUNS_DIR = Path("bench/real/smoke_mixed_outcome")

BENCH_UNAVAILABLE_MESSAGE = (
    "counterfact bench: the bench harness is not included in the wheel. "
    "Run benchmarks from a source checkout after "
    "`uv sync --locked --extra dev --extra bench`. Installing wheel extras does not include bench/."
)

DEMO_CONTRAST_THRESHOLD = 0.05
DEMO_CONTRAST_TEMPLATE = (
    "naive_vs_causal_contrast: naive arm gap = {naive:+.3f}; "
    "causal arm gap (AIPW) = {causal:+.3f}; "
    "the marginal table overstates what the corpus supports — see "
    "DAG and assumptions."
)

INGEST_FORMATS = [
    ("claude-agent-sdk", "Claude Agent SDK message JSONL (zero-config)"),
    ("openai-agents", "OpenAI Agents SDK trace JSON (requires --outcome unless derivable)"),
    ("generic-jsonl", "Any JSONL with an explicit user-supplied --mapping file"),
]
