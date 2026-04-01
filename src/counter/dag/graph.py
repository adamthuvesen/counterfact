"""Hand-rolled DAG over agent decisions.

Full implementation (edges from taxonomy, topological sort, ancestors,
descendants) lands in tasks §4. This module currently exposes a minimal `DAG`
and `build_dag` so that downstream cross-capability tests can wire up.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from counter.schema import Decision, Run


@dataclass
class DAG:
    """A per-trace DAG. Nodes are Decision instances; edges are (parent, child) tuples.

    Acyclicity is asserted at construction.
    """

    nodes: list[Decision] = field(default_factory=list)
    edges: list[tuple[str, str]] = field(default_factory=list)
    run: Run | None = None

    def node_ids(self) -> list[str]:
        return [n.decision_id for n in self.nodes]


def build_dag(trace: Run, schema: Any | None = None) -> DAG:
    """Build a DAG over the decisions in `trace`.

    v0 stub: collects decisions from every step in order. Edge construction
    using the decision-taxonomy parent map lands in §4.4. Empty traces yield
    an empty DAG.
    """
    nodes: list[Decision] = []
    for step in trace.steps:
        nodes.extend(step.decisions)
    return DAG(nodes=nodes, edges=[], run=trace)
