"""The report's decision DAG section: a deterministic layered SVG."""

from __future__ import annotations

from counterfact.dag import DAG
from counterfact.explain._html import raw, section, tag
from counterfact.schema import Decision, Run

_BOX_H = 32


def _node_label(decision: Decision, run: Run | None = None) -> str:
    action = decision.chosen_action or "—"
    if (
        decision.decision_type == "termination"
        and action == "success"
        and run is not None
        and run.outcome.verifier == "pytest_hidden"
        and run.outcome.metadata.get("public_pass") is True
    ):
        action = "public_tests_passed"
    return f"{decision.decision_type}::{action}"


def _layout_positions(run: Run, *, x_step: int) -> dict[str, tuple[int, int]]:
    """x by step_index, y by position within the step."""
    return {
        decision.decision_id: (24 + step.step_index * x_step, 24 + pos * 56)
        for step in run.steps
        for pos, decision in enumerate(step.decisions)
    }


def _edge(parent_id: str, child_id: str, positions: dict[str, tuple[int, int]], box_w: int) -> str:
    px, py = positions[parent_id]
    cx, cy = positions[child_id]
    return tag(
        "line",
        {
            "class": "edge",
            "x1": str(px + box_w),
            "y1": str(py + _BOX_H // 2),
            "x2": str(cx),
            "y2": str(cy + _BOX_H // 2),
            "marker-end": "url(#cf-arrow)",
            "data-parent-id": parent_id,
            "data-child-id": child_id,
        },
    )


def _node(
    decision: Decision, label: str, position: tuple[int, int], box_w: int, focal: bool
) -> str:
    x, y = position
    rect = tag(
        "rect",
        {"x": str(x), "y": str(y), "width": str(box_w), "height": str(_BOX_H), "rx": "4"},
    )
    text = tag("text", {"x": str(x + 8), "y": str(y + 20)}, label)
    title = tag("title", None, f"{decision.decision_id} ({decision.decision_type})")
    return tag(
        "g",
        {"class": "node focal" if focal else "node", "data-decision-id": decision.decision_id},
        raw(rect + text + title),
    )


def render_dag(dag: DAG, *, focal_decision_id: str | None) -> str:
    if dag.run is None or not dag.nodes:
        return section("dag", "Decision DAG", tag("p", {"class": "placeholder"}, "(empty trace)"))

    labels = {decision.decision_id: _node_label(decision, dag.run) for decision in dag.nodes}
    box_w = max(168, max(len(label) for label in labels.values()) * 7 + 16)
    positions = _layout_positions(dag.run, x_step=box_w + 32)
    max_x = max(x for x, _ in positions.values()) + box_w + 24
    max_y = max(y for _, y in positions.values()) + _BOX_H + 24

    edges = "".join(
        _edge(parent_id, child_id, positions, box_w)
        for parent_id, child_id in dag.edges
        if parent_id in positions and child_id in positions
    )
    nodes = "".join(
        _node(
            decision,
            labels[decision.decision_id],
            positions[decision.decision_id],
            box_w,
            decision.decision_id == focal_decision_id,
        )
        for decision in dag.nodes
    )
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {max_x} {max_y}" width="{max_x}" height="{max_y}" '
        f'role="img" aria-label="decision DAG">'
        '<defs><marker id="cf-arrow" viewBox="0 0 10 10" refX="9" refY="5" '
        'markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
        '<path d="M0 0 L10 5 L0 10 z" class="arrow"/></marker></defs>'
        f'<g class="edges">{edges}</g>'
        f'<g class="nodes">{nodes}</g>'
        "</svg>"
    )
    return section(
        "dag",
        "Decision DAG",
        tag("div", {"class": "dag-wrapper"}, raw(svg)),
        subtitle=(
            "Per-trace DAG built by `counterfact.dag.build_dag`. "
            "Top-attributed decision is highlighted."
        ),
    )
