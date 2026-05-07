"""Adapters bridging external agent SDK trace shapes to counterfact's native Run schema.

Each module under `adapters/` parses JSON-serialized dicts only — none import the
upstream SDK at module load. Live tracing helpers that hold SDK references live
under `counterfact.tracing` instead.
"""

from __future__ import annotations

from counterfact.adapters.claude_agent_sdk import (
    ingest_claude_agent_sdk,
    run_from_messages,
)
from counterfact.adapters.openai_agents import (
    ingest_openai_agents,
    run_from_trace,
)

__all__ = [
    "ingest_claude_agent_sdk",
    "ingest_openai_agents",
    "run_from_messages",
    "run_from_trace",
]
