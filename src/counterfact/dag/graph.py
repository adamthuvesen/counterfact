"""Hand-rolled DAG over agent decisions.

Single-agent only, tool-call/plan-step granularity. Edges are derived from
the decision-taxonomy `parent_types` declarations and the temporal order of
steps: a decision at step `t` may have edges from any declared parent type at
any step `s <= t`. We attach the *most recent* parent decision of each
declared type — keeping edge density manageable and making the graph
inspectable.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from counterfact.errors import DAGCycleError
from counterfact.schema import Decision, Run

FlatDecision = tuple[int, int, Decision]


@dataclass
class DAG:
    """A per-trace DAG. Nodes are Decision instances; edges are (parent_id, child_id)."""

    nodes: list[Decision] = field(default_factory=list)
    edges: list[tuple[str, str]] = field(default_factory=list)
    run: Run | None = None
    # Adjacency dicts built once at construction so query-time graph lookups
    # are O(1) rather than scanning the edge list on every call.
    _adj: dict[str, list[str]] = field(default_factory=dict, init=False, repr=False)
    _radj: dict[str, list[str]] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        ids = self.node_ids()
        self._adj = {nid: [] for nid in ids}
        self._radj = {nid: [] for nid in ids}
        for parent, child in self.edges:
            self._adj.setdefault(parent, []).append(child)
            self._radj.setdefault(child, []).append(parent)
        # Acyclicity check fires on every construction (including DAG(...) directly,
        # which is what the cycle-rejection scenario uses).
        if self.nodes or self.edges:
            self.topological_sort()

    def node_ids(self) -> list[str]:
        return [n.decision_id for n in self.nodes]

    def parents_of(self, decision_id: str) -> list[str]:
        return list(self._radj.get(decision_id, ()))

    def children_of(self, decision_id: str) -> list[str]:
        return list(self._adj.get(decision_id, ()))

    def ancestors_of(self, decision_id: str) -> list[str]:
        seen: set[str] = set()
        stack = [decision_id]
        while stack:
            cur = stack.pop()
            for p in self._radj.get(cur, ()):
                if p not in seen:
                    seen.add(p)
                    stack.append(p)
        return list(seen)

    def descendants_of(self, decision_id: str) -> list[str]:
        seen: set[str] = set()
        stack = [decision_id]
        while stack:
            cur = stack.pop()
            for c in self._adj.get(cur, ()):
                if c not in seen:
                    seen.add(c)
                    stack.append(c)
        return list(seen)

    def topological_sort(self) -> list[str]:
        """Kahn's algorithm. Raises DAGCycleError if a cycle is present."""
        indeg: dict[str, int] = defaultdict(int)
        ids = self.node_ids()
        for nid in ids:
            indeg.setdefault(nid, 0)
        for _, child in self.edges:
            indeg[child] += 1

        ready = [nid for nid in ids if indeg[nid] == 0]
        order: list[str] = []
        while ready:
            cur = ready.pop()
            order.append(cur)
            for nxt in self._adj.get(cur, ()):
                indeg[nxt] -= 1
                if indeg[nxt] == 0:
                    ready.append(nxt)
        if len(order) != len(ids):
            raise DAGCycleError(
                f"cycle detected in DAG: only {len(order)} of {len(ids)} nodes ordered"
            )
        return order


def _flatten_decisions(trace: Run) -> tuple[list[Decision], list[FlatDecision]]:
    nodes: list[Decision] = []
    flat: list[FlatDecision] = []
    for step in trace.steps:
        for pos, decision in enumerate(step.decisions):
            flat.append((step.step_index, pos, decision))
            nodes.append(decision)
    return nodes, flat


def _latest_parent_ids(
    flat: list[FlatDecision], child_index: int, wanted_types: set[str]
) -> list[str]:
    child_step, child_pos, _child = flat[child_index]
    picked: dict[str, str] = {}
    for parent_index in range(child_index - 1, -1, -1):
        parent_step, parent_pos, candidate = flat[parent_index]
        if candidate.decision_type not in wanted_types or candidate.decision_type in picked:
            continue
        # Strict precedence: earlier step OR same step with smaller position.
        if parent_step < child_step or (parent_step == child_step and parent_pos < child_pos):
            picked[candidate.decision_type] = candidate.decision_id
            if len(picked) == len(wanted_types):
                break
    return list(picked.values())


def _latest_parent_edges(flat: list[FlatDecision]) -> list[tuple[str, str]]:
    # Avoid a circular import: the taxonomy module imports schema, which in
    # turn lives below this module's package; deferring keeps the surface tidy.
    from counterfact.taxonomy import parent_types

    edges: list[tuple[str, str]] = []
    for index, (_step_index, _pos, decision) in enumerate(flat):
        wanted = set(parent_types(decision.decision_type))
        if not wanted:
            continue
        for parent_id in _latest_parent_ids(flat, index, wanted):
            edges.append((parent_id, decision.decision_id))
    return edges


def build_dag(trace: Run) -> DAG:
    """Build a DAG over the decisions in `trace`.

    Edges connect each decision to the most recent eligible parent of each
    declared parent type, where "eligible" means: same trace, earlier or equal
    step index, and (for same-step parents) earlier list position.

    Decision-ID uniqueness is enforced by `Run._decision_ids_are_unique` at
    schema-validation time; we trust that invariant here rather than re-check.
    """
    nodes, flat = _flatten_decisions(trace)
    edges = _latest_parent_edges(flat)

    return DAG(nodes=nodes, edges=edges, run=trace)
