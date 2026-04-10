"""Hand-rolled DAG over agent decisions.

Single-agent only, tool-call/plan-step granularity (design.md D4). Edges are
derived from the decision-taxonomy `parent_types` declarations and the
temporal order of steps: a decision at step `t` may have edges from any
declared parent type at any step `s <= t`. We attach the *most recent* parent
decision of each declared type — keeping edge density manageable and making
the graph inspectable.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

from counterfact.errors import DAGCycleError
from counterfact.schema import Decision, Run


@dataclass
class DAG:
    """A per-trace DAG. Nodes are Decision instances; edges are (parent_id, child_id)."""

    nodes: list[Decision] = field(default_factory=list)
    edges: list[tuple[str, str]] = field(default_factory=list)
    run: Run | None = None

    def __post_init__(self) -> None:
        # Acyclicity check fires on every construction (including DAG(...) directly,
        # which is what the cycle-rejection scenario uses).
        if self.nodes or self.edges:
            self.topological_sort()

    def node_ids(self) -> list[str]:
        return [n.decision_id for n in self.nodes]

    def parents_of(self, decision_id: str) -> list[str]:
        return [p for (p, c) in self.edges if c == decision_id]

    def children_of(self, decision_id: str) -> list[str]:
        return [c for (p, c) in self.edges if p == decision_id]

    def ancestors_of(self, decision_id: str) -> list[str]:
        seen: set[str] = set()
        stack = [decision_id]
        while stack:
            cur = stack.pop()
            for p in self.parents_of(cur):
                if p not in seen:
                    seen.add(p)
                    stack.append(p)
        return list(seen)

    def descendants_of(self, decision_id: str) -> list[str]:
        seen: set[str] = set()
        stack = [decision_id]
        while stack:
            cur = stack.pop()
            for c in self.children_of(cur):
                if c not in seen:
                    seen.add(c)
                    stack.append(c)
        return list(seen)

    def topological_sort(self) -> list[str]:
        """Kahn's algorithm. Raises DAGCycleError if a cycle is present."""
        indeg: dict[str, int] = defaultdict(int)
        adj: dict[str, list[str]] = defaultdict(list)
        ids = self.node_ids()
        for nid in ids:
            indeg[nid] += 0
        for parent, child in self.edges:
            adj[parent].append(child)
            indeg[child] += 1

        ready = [nid for nid in ids if indeg[nid] == 0]
        order: list[str] = []
        while ready:
            cur = ready.pop()
            order.append(cur)
            for nxt in adj[cur]:
                indeg[nxt] -= 1
                if indeg[nxt] == 0:
                    ready.append(nxt)
        if len(order) != len(ids):
            raise DAGCycleError(
                f"cycle detected in DAG: only {len(order)} of {len(ids)} nodes ordered"
            )
        return order


def build_dag(trace: Run, schema: Any | None = None) -> DAG:
    """Build a DAG over the decisions in `trace`.

    Edges connect each decision to the most recent eligible parent of each
    declared parent type, where "eligible" means: same trace, earlier or equal
    step index, and (for same-step parents) earlier list position.
    """
    # Avoid a circular import: the taxonomy module imports schema, which in
    # turn lives below this module's package; deferring keeps the surface tidy.
    from counterfact.taxonomy import parent_types

    nodes: list[Decision] = []
    flat: list[tuple[int, int, Decision]] = []  # (step_index, intra_step_pos, decision)
    for step in trace.steps:
        for pos, d in enumerate(step.decisions):
            flat.append((step.step_index, pos, d))
            nodes.append(d)
    ids = [d.decision_id for _, _, d in flat]
    duplicate_ids = sorted(
        decision_id for decision_id, count in Counter(ids).items() if count > 1
    )
    if duplicate_ids:
        raise ValueError(
            "cannot build DAG with duplicate decision_id values: "
            + ", ".join(duplicate_ids)
        )

    edges: list[tuple[str, str]] = []

    # For each decision, look back through `flat` and connect the most recent
    # decision whose type appears in this decision's parent_types.
    for i, (s_i, p_i, d) in enumerate(flat):
        wanted = set(parent_types(d.decision_type))
        if not wanted:
            continue
        # Walk backwards, picking the most recent of each parent type.
        picked: dict[str, str] = {}
        for j in range(i - 1, -1, -1):
            s_j, p_j, cand = flat[j]
            if cand.decision_type in wanted and cand.decision_type not in picked:
                # require strict precedence: earlier step OR same step with smaller pos
                if (s_j < s_i) or (s_j == s_i and p_j < p_i):
                    picked[cand.decision_type] = cand.decision_id
                    if len(picked) == len(wanted):
                        break
        for pid in picked.values():
            edges.append((pid, d.decision_id))

    return DAG(nodes=nodes, edges=edges, run=trace)
