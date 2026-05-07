"""Live in-process tracing helpers that emit native Run JSON.

These helpers wrap external agent SDK runtimes (e.g. Claude Agent SDK,
OpenAI Agents SDK). The SDK packages are optional dependencies — the
helpers import them lazily so `counterfact.tracing` itself stays
import-safe in environments that only use the offline ingest adapters.
"""

from __future__ import annotations

from counterfact.tracing.claude_agent_tracer import ClaudeAgentTracer
from counterfact.tracing.openai_span_processor import CounterfactSpanProcessor

__all__ = ["ClaudeAgentTracer", "CounterfactSpanProcessor"]
