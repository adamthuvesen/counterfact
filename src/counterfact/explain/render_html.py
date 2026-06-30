"""Render an `ExplainReport` to a self-contained HTML document.

Every quantitative claim is sourced from a named `CausalEstimate` field;
nothing in this file invents a number. The single most important rule is in
`_render_estimate_card`: when `identifiability == UNIDENTIFIED`, the card
emits no numeric outcome_delta or influence — the spec requirement
"unidentified estimates suppress numeric estimates" gates this branch.
"""

from __future__ import annotations

import shlex
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import cast

from counterfact import __version__ as _COUNTERFACT_VERSION
from counterfact._fmt import outcome_label as _outcome_label
from counterfact.attribute import AttributionEntry, FailureAttribution
from counterfact.dag import DAG
from counterfact.explain._css import CSS, GLOSS
from counterfact.explain._html import Raw, raw, tag
from counterfact.explain.render_contract import shows_numeric_attribution, shows_outcome_delta
from counterfact.explain.report import ExplainReport
from counterfact.intervene.estimate import (
    CausalEstimate,
    IdentifiabilityStatus,
    SupportPayload,
)
from counterfact.schema import Decision, Run

# --------------------------------------------------------------------------
# Header / story
# --------------------------------------------------------------------------


def _render_header(report: ExplainReport, *, generated_at: datetime) -> str:
    run = report.run
    title = (
        "counterfact diagnose" if report.diagnosis_summary is not None else "counterfact explain"
    )
    items = [
        ("Run id", run.run_id),
        ("Outcome", _outcome_label(run)),
        ("Schema version", run.schema_version),
        ("Decision type", report.summary_decision_type),
        ("Corpus size", str(report.corpus_size)),
        (
            "Corpus pass rate",
            (f"{report.corpus_pass_rate:.3f}" if report.corpus_pass_rate is not None else "n/a"),
        ),
        ("Generated", generated_at.isoformat(timespec="seconds")),
    ]
    if run.metadata.agent_name:
        items.insert(2, ("Agent", run.metadata.agent_name))
    dl = tag(
        "dl",
        {"class": "meta"},
        *[raw(tag("dt", None, label) + tag("dd", None, value)) for label, value in items],
    )
    return tag(
        "header",
        {"class": "report-header"},
        raw(tag("h1", None, title)),
        raw(dl),
    )


def _top_entry(attribution: FailureAttribution) -> AttributionEntry | None:
    for entry in attribution.entries:
        if shows_numeric_attribution(entry):
            return entry
    return None


def _render_story(report: ExplainReport) -> str:
    run = report.run
    if report.degenerate_estimate is not None:
        sentence = (
            f"Run {run.run_id} ({_outcome_label(run)}). The corpus is "
            f"causally degenerate (single outcome class), so "
            f"`counterfact` declines to estimate decision-level effects "
            f"on this run; identifiability=unidentified."
        )
    else:
        top = _top_entry(report.attribution)
        if top is None:
            sentence = (
                f"Run {run.run_id} ({_outcome_label(run)}). No decision in "
                f"this trace has identifiable counterfactual support given "
                f"the supplied corpus."
            )
        else:
            sentence = (
                f"Run {run.run_id} ({_outcome_label(run)}). Top-attributed "
                f"decision: {top.decision_id} "
                f"(decision_type={top.decision_type}, "
                f"chosen_action={top.chosen_action}); "
                f"identifiability={top.identifiability.value}."
            )
    return tag("p", {"class": "story"}, sentence)


def _render_diagnosis_summary(report: ExplainReport) -> str:
    if report.diagnosis_summary is None:
        return ""
    return tag(
        "section",
        {"class": "diagnosis"},
        raw(tag("h2", None, "Diagnosis summary")),
        raw(tag("p", {"class": "diagnosis-summary"}, report.diagnosis_summary)),
    )


# --------------------------------------------------------------------------
# Trace timeline and decision cards
# --------------------------------------------------------------------------


def _render_timeline(report: ExplainReport) -> str:
    steps: list[str] = []
    for step in sorted(report.run.steps, key=lambda s: s.step_index):
        decisions: list[str] = []
        for decision in step.decisions:
            action = decision.chosen_action or "n/a"
            decisions.append(
                tag(
                    "div",
                    {"class": "decision-row"},
                    raw(
                        tag("code", {"class": "inline"}, decision.decision_id)
                        + " "
                        + tag("span", None, decision.decision_type)
                        + " "
                        + tag("span", None, f"chosen={action}")
                    ),
                )
            )
        if not decisions:
            decisions.append(tag("div", {"class": "decision-row placeholder"}, "(no decisions)"))
        step_body = tag("div", {"class": "decision-list"}, raw("".join(decisions)))
        steps.append(
            tag(
                "article",
                {"class": "timeline-step", "data-step-index": step.step_index},
                raw(
                    tag(
                        "div",
                        {"class": "step-head"},
                        raw(
                            tag("span", None, f"step {step.step_index}")
                            + tag(
                                "span",
                                {"class": "obs-count"},
                                f"observations={len(step.observations)}",
                            )
                        ),
                    )
                    + step_body
                ),
            )
        )
    if not steps:
        steps.append(tag("p", {"class": "placeholder"}, "(empty trace)"))
    return tag(
        "section",
        {"class": "timeline"},
        raw(tag("h2", None, "Trace timeline")),
        raw(
            tag(
                "p",
                {"class": "section-subtitle"},
                "Step-level decision metadata only. Raw observation and model "
                "contents are intentionally omitted.",
            )
        ),
        raw(tag("div", {"class": "timeline-list"}, raw("".join(steps)))),
    )


def _step_index_for_decision(report: ExplainReport, decision_id: str) -> int | None:
    for step in report.run.steps:
        if any(decision.decision_id == decision_id for decision in step.decisions):
            return step.step_index
    return None


def _coerce_arm_rows(value: object) -> list[str | dict[str, object]]:
    """Narrow a raw payload value to the arm-row shape `_arm_names` expects.

    `NextStep.payload` is JSON-shaped and untyped at the boundary, so the
    runtime check lives here once. Anything that isn't a list collapses to
    `[]`; element-level coercion stays in `_arm_names`.
    """
    if not isinstance(value, list):
        return []
    return cast(list[str | dict[str, object]], value)


def _arm_names(rows: list[str | dict[str, object]] | None) -> list[str]:
    # Rows are heterogeneous JSON: each element may be either a plain arm
    # name (string) or an arm dict carrying additional per-arm metadata.
    if rows is None:
        return []
    names: list[str] = []
    for row in rows:
        if isinstance(row, dict) and row.get("arm") is not None:
            names.append(str(row["arm"]))
        elif isinstance(row, str):
            names.append(row)
    return names


def _render_support_block(estimate: CausalEstimate) -> str | None:
    payload = cast(SupportPayload, estimate.next_step.payload)
    observed = _arm_names(_coerce_arm_rows(payload.get("observed_arms")))
    missing = [str(arm) for arm in payload.get("missing_arms", [])]
    missing_strata = [str(item) for item in payload.get("missing_strata", [])]
    localization = payload.get("localization_limit")
    replay_inputs = payload.get("replay_inputs_required")
    parts: list[str] = []
    if observed:
        parts.append(tag("li", None, "observed arms: " + ", ".join(observed)))
    if missing:
        parts.append(tag("li", None, "missing arms: " + ", ".join(missing)))
    if missing_strata:
        parts.append(tag("li", None, "missing strata: " + ", ".join(missing_strata)))
    if isinstance(localization, str) and localization:
        parts.append(tag("li", None, "localization limit: " + localization))
    if isinstance(replay_inputs, list) and replay_inputs:
        parts.append(
            tag(
                "li",
                None,
                "replay inputs required: " + ", ".join(str(item) for item in replay_inputs),
            )
        )
    if not parts:
        return None
    return tag(
        "div",
        {"class": "callout support"},
        raw(tag("strong", None, "support diagnostics")),
        raw(tag("ul", {"class": "flat"}, raw("".join(parts)))),
    )


def _intervene_json_command(report: ExplainReport, entry: AttributionEntry) -> str | None:
    if report.run_path is None or report.corpus_dir is None or entry.estimate is None:
        return None
    query = entry.estimate.query
    run_path = shlex.quote(report.run_path)
    corpus_dir = shlex.quote(report.corpus_dir)
    return (
        "uv run counterfact intervene "
        f"{run_path} --runs-dir {corpus_dir} "
        f"--decision-id {entry.decision_id} "
        f"--set {query.intervention_kind}={query.target} --json"
    )


def _render_decision_cards(report: ExplainReport) -> str:
    if report.degenerate_estimate is not None and not report.attribution.entries:
        return tag(
            "section",
            {"class": "decision-cards"},
            raw(tag("h2", None, "Decision cards")),
            raw(
                tag(
                    "p",
                    {"class": "placeholder"},
                    "(single-class corpus: no per-decision cards rendered)",
                )
            ),
        )

    cards: list[str] = []
    for entry in report.attribution.entries:
        step_index = _step_index_for_decision(report, entry.decision_id)
        estimate = entry.estimate
        support = _render_support_block(estimate) if estimate is not None else None
        command = _intervene_json_command(report, entry)
        if step_index is None:
            step_pill = tag(
                "span",
                {"class": "subtle", "title": "step index unresolved"},
                "step=?",
            )
        else:
            step_pill = tag("span", {"class": "subtle"}, f"step={step_index}")
        summary = tag(
            "summary",
            None,
            raw(
                tag("code", {"class": "inline"}, entry.decision_id)
                + " "
                + tag("span", None, entry.decision_type)
                + " "
                + step_pill
                + " "
                + _render_badge(entry.identifiability)
            ),
        )
        body_parts = [
            tag(
                "div",
                {"class": "decision-head"},
                raw(
                    tag("code", {"class": "inline"}, entry.decision_id)
                    + tag("span", None, entry.decision_type)
                    + step_pill
                    + _render_badge(entry.identifiability)
                ),
            ),
            tag("div", {"class": "subtle"}, f"chosen_action={entry.chosen_action}"),
        ]
        if estimate is not None:
            body_parts.append(
                tag(
                    "div",
                    {"class": "subtle"},
                    f"next_step.action={estimate.next_step.action}",
                )
            )
        if support is not None:
            body_parts.append(support)
        if command is not None:
            body_parts.append(
                tag(
                    "pre",
                    {"class": "cmd"},
                    raw(tag("code", None, command)),
                )
            )
        cards.append(
            tag(
                "details",
                {
                    "class": f"decision-card ident-{entry.identifiability.value}",
                    "open": "open" if len(cards) == 0 else None,
                },
                raw(summary + "".join(body_parts)),
            )
        )
    if not cards:
        cards.append(tag("p", {"class": "placeholder"}, "(no attribution entries available)"))
    return tag(
        "section",
        {"class": "decision-cards"},
        raw(tag("h2", None, "Decision cards")),
        raw(
            tag(
                "p",
                {"class": "section-subtitle"},
                "Decision-level trace forensics with support and replay guidance.",
            )
        ),
        raw(tag("div", {"class": "decision-card-list"}, raw("".join(cards)))),
    )


def _render_counterfactual_lookup(report: ExplainReport) -> str:
    if not report.counterfactual_lookup:
        return ""
    rows: list[str] = []
    for estimate in report.counterfactual_lookup:
        parts = [
            f"{estimate.query.decision_type}.{estimate.query.intervention_kind}",
            f"target={estimate.query.target}",
            f"identifiability={estimate.identifiability.value}",
        ]
        if shows_outcome_delta(estimate):
            delta = estimate.outcome_delta
            assert delta is not None
            parts.append(
                f"outcome_delta={delta.point:.3f} [{delta.ci_low:.3f}, {delta.ci_high:.3f}]"
            )
        else:
            parts.append(f"next_step={estimate.next_step.action}")
        rows.append(tag("div", {"class": "lookup-row"}, " | ".join(parts)))
    return tag(
        "section",
        {"class": "counterfactual-lookup"},
        raw(tag("h2", None, "Counterfactual lookup")),
        raw(
            tag(
                "p",
                {"class": "section-subtitle"},
                "Precomputed estimates only. Unidentified entries do not display numeric effects.",
            )
        ),
        raw(tag("div", {"class": "lookup-list"}, raw("".join(rows)))),
    )


# --------------------------------------------------------------------------
# Descriptive baseline
# --------------------------------------------------------------------------


def _render_descriptive(report: ExplainReport) -> str:
    table = report.pass_rate_table
    head = tag(
        "tr",
        None,
        raw(
            tag("th", None, "arm")
            + tag("th", None, "n")
            + tag("th", None, "pass")
            + tag("th", None, "rate")
            + tag("th", None, "95% CI")
        ),
    )
    if not table.rows:
        body = tag(
            "tr",
            None,
            raw(tag("td", {"colspan": 5}, "(no observed arms)")),
        )
    else:
        rows: list[str] = []
        for row in table.rows:
            rows.append(
                tag(
                    "tr",
                    None,
                    raw(
                        tag("td", None, row.arm)
                        + tag("td", None, str(row.n))
                        + tag("td", None, str(row.pass_count))
                        + tag("td", None, f"{row.pass_rate:.3f}")
                        + tag(
                            "td",
                            None,
                            f"[{row.ci_low:.3f}, {row.ci_high:.3f}]",
                        )
                    ),
                )
            )
        body = "".join(rows)
    return tag(
        "section",
        {"class": "descriptive"},
        raw(tag("h2", None, f"Descriptive baseline — {table.decision_type}")),
        raw(
            tag(
                "p",
                {"class": "section-subtitle"},
                "Marginal table — does not adjust for the agent's other "
                "randomized decisions. Use the honest verdict below for "
                "causal claims.",
            )
        ),
        raw(
            tag(
                "table",
                {"class": "pass-rate"},
                raw(tag("thead", None, raw(head))),
                raw(tag("tbody", None, raw(body))),
            )
        ),
    )


# --------------------------------------------------------------------------
# DAG (inline SVG)
# --------------------------------------------------------------------------


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


def _layout_positions(
    dag: DAG, *, x_step: int = 200, y_step: int = 56, x_pad: int = 24, y_pad: int = 24
) -> dict[str, tuple[int, int]]:
    """Deterministic layered layout: x by step_index, y by intra-step pos."""
    positions: dict[str, tuple[int, int]] = {}
    if dag.run is None:
        return positions
    for step in dag.run.steps:
        for pos, decision in enumerate(step.decisions):
            x = x_pad + step.step_index * x_step
            y = y_pad + pos * y_step
            positions[decision.decision_id] = (x, y)
    return positions


def _dag_edge_parts(
    dag: DAG,
    positions: dict[str, tuple[int, int]],
    *,
    box_w: int,
    box_h: int,
) -> list[str]:
    edge_parts: list[str] = []
    for parent_id, child_id in dag.edges:
        if parent_id not in positions or child_id not in positions:
            continue
        px, py = positions[parent_id]
        cx, cy = positions[child_id]
        edge_parts.append(
            tag(
                "line",
                {
                    "class": "edge",
                    "x1": str(px + box_w),
                    "y1": str(py + box_h // 2),
                    "x2": str(cx),
                    "y2": str(cy + box_h // 2),
                    "marker-end": "url(#cf-arrow)",
                    "data-parent-id": parent_id,
                    "data-child-id": child_id,
                },
            )
        )
    return edge_parts


def _dag_node_parts(
    dag: DAG,
    labels: dict[str, str],
    positions: dict[str, tuple[int, int]],
    *,
    box_w: int,
    box_h: int,
    focal_decision_id: str | None,
) -> list[str]:
    node_parts: list[str] = []
    for decision in dag.nodes:
        x, y = positions[decision.decision_id]
        classes = "node focal" if decision.decision_id == focal_decision_id else "node"
        rect = tag(
            "rect",
            {
                "x": str(x),
                "y": str(y),
                "width": str(box_w),
                "height": str(box_h),
                "rx": "4",
            },
        )
        text = tag(
            "text",
            {"x": str(x + 8), "y": str(y + 20)},
            labels[decision.decision_id],
        )
        title = tag(
            "title",
            None,
            f"{decision.decision_id} ({decision.decision_type})",
        )
        node_parts.append(
            tag(
                "g",
                {"class": classes, "data-decision-id": decision.decision_id},
                raw(rect + text + title),
            )
        )
    return node_parts


def _render_dag(dag: DAG, *, focal_decision_id: str | None) -> str:
    if dag.run is None or not dag.nodes:
        return tag(
            "section",
            {"class": "dag"},
            raw(tag("h2", None, "Decision DAG")),
            raw(tag("p", {"class": "placeholder"}, "(empty trace)")),
        )

    labels = {decision.decision_id: _node_label(decision, dag.run) for decision in dag.nodes}
    box_w = max(168, max(len(label) for label in labels.values()) * 7 + 16)
    box_h = 32
    positions = _layout_positions(dag, x_step=box_w + 32)
    max_x = max(x for x, _ in positions.values()) + box_w + 24
    max_y = max(y for _, y in positions.values()) + box_h + 24

    edge_parts = _dag_edge_parts(dag, positions, box_w=box_w, box_h=box_h)
    node_parts = _dag_node_parts(
        dag,
        labels,
        positions,
        box_w=box_w,
        box_h=box_h,
        focal_decision_id=focal_decision_id,
    )

    defs = (
        '<defs><marker id="cf-arrow" viewBox="0 0 10 10" refX="9" refY="5" '
        'markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
        '<path d="M0 0 L10 5 L0 10 z" class="arrow"/></marker></defs>'
    )
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {max_x} {max_y}" width="{max_x}" height="{max_y}" '
        f'role="img" aria-label="decision DAG">'
        f"{defs}"
        f'<g class="edges">{"".join(edge_parts)}</g>'
        f'<g class="nodes">{"".join(node_parts)}</g>'
        f"</svg>"
    )
    return tag(
        "section",
        {"class": "dag"},
        raw(tag("h2", None, "Decision DAG")),
        raw(
            tag(
                "p",
                {"class": "section-subtitle"},
                "Per-trace DAG built by `counterfact.dag.build_dag`. "
                "Top-attributed decision is highlighted.",
            )
        ),
        raw(tag("div", {"class": "dag-wrapper"}, raw(svg))),
    )


# --------------------------------------------------------------------------
# CausalEstimate cards (the honesty contract lives here)
# --------------------------------------------------------------------------


def _render_badge(status: IdentifiabilityStatus) -> str:
    return tag(
        "span",
        {"class": f"badge ident-{status.value}"},
        status.value,
    )


def _render_field(label: str, value: str | Raw, source: str | None = None) -> str:
    # `tag` routes children based on type: `Raw` passes through unescaped (used
    # for pre-built lists from `_render_list`), plain `str` is HTML-escaped.
    children: list[str] = [
        tag("div", {"class": "label"}, label),
        tag("div", {"class": "value"}, value),
    ]
    if source:
        children.append(tag("div", {"class": "source"}, f"source: {source}"))
    return tag("div", {"class": "field"}, raw("".join(children)))


def _render_list(items: Iterable[str]) -> Raw:
    rendered = "".join(tag("li", None, item) for item in items)
    return raw(tag("ul", {"class": "flat"}, raw(rendered)))


def _render_next_step(estimate: CausalEstimate) -> str:
    ns = estimate.next_step
    payload = cast(SupportPayload, ns.payload)
    suggested = payload.get("suggested_command")
    body_parts: list[str] = [
        tag(
            "div",
            None,
            raw(
                tag("strong", None, "next_step.action: ")
                + tag("code", {"class": "inline"}, ns.action)
            ),
        ),
        tag("p", None, ns.human_text),
    ]
    if isinstance(suggested, str) and suggested:
        body_parts.append(tag("pre", {"class": "cmd"}, raw(tag("code", None, suggested))))
    return tag(
        "div",
        {"class": "callout next-step"},
        raw("".join(body_parts)),
    )


def _render_estimate_card(
    estimate: CausalEstimate,
    *,
    title: str,
    subtitle: str | None = None,
) -> str:
    """Render a single CausalEstimate card.

    Honesty contract: when `identifiability == UNIDENTIFIED`, this function
    MUST NOT emit any quantitative point estimate, CI, or influence number
    for the entry. The branch below is the single place that gates
    `outcome_delta` rendering — keep it that way.
    """
    status = estimate.identifiability
    head_children: list[str] = [tag("span", {"class": "title"}, title)]
    if subtitle:
        head_children.append(tag("span", {"class": "sub"}, subtitle))
    head_children.append(_render_badge(status))
    head = tag("div", {"class": "card-head"}, raw("".join(head_children)))
    gloss = tag("div", {"class": "gloss"}, GLOSS[status])

    fields: list[str] = []
    if estimate.estimand:
        fields.append(_render_field("estimand", estimate.estimand, source="estimand"))
    if estimate.reason:
        fields.append(_render_field("reason", estimate.reason, source="reason"))

    if shows_outcome_delta(estimate):
        d = estimate.outcome_delta
        assert d is not None
        value = f"{d.point:.3f} [{d.ci_low:.3f}, {d.ci_high:.3f}] (n_bootstrap={d.n_bootstrap})"
        fields.append(
            _render_field(
                "outcome_delta",
                value,
                source="outcome_delta.point / ci_low / ci_high",
            )
        )

    if estimate.bounds is not None:
        b = estimate.bounds
        bound_value = f"e_value={b.e_value:.3f} (technique={b.technique})"
        if b.note:
            bound_value += f" — {b.note}"
        fields.append(_render_field("bounds", bound_value, source="bounds.e_value"))

    if estimate.adjustment_set:
        fields.append(
            _render_field(
                "adjustment_set",
                _render_list(estimate.adjustment_set),
                source="adjustment_set",
            )
        )
    if estimate.assumptions:
        fields.append(
            _render_field(
                "assumptions",
                _render_list(estimate.assumptions),
                source="assumptions",
            )
        )
    if estimate.warnings:
        fields.append(_render_field("warnings", _render_list(estimate.warnings), source="warnings"))

    next_step = _render_next_step(estimate)

    return tag(
        "div",
        {"class": f"card ident-{status.value}"},
        raw(head),
        raw(gloss),
        raw("".join(fields)),
        raw(next_step),
    )


def _render_attribution_table(attribution: FailureAttribution) -> str:
    """Compact ranked table; influence cell is `—` for unidentified entries."""
    head = tag(
        "tr",
        None,
        raw(
            tag("th", None, "rank")
            + tag("th", None, "decision_id")
            + tag("th", None, "decision_type")
            + tag("th", None, "chosen_action")
            + tag("th", None, "identifiability")
            + tag("th", None, "influence")
        ),
    )
    rows: list[str] = []
    for rank, entry in enumerate(attribution.entries, start=1):
        if not shows_numeric_attribution(entry):
            influence_cell = tag(
                "td",
                {"class": "placeholder"},
                "n/a (unidentified)",
            )
        else:
            influence_cell = tag("td", None, f"{entry.influence:.3f}")
        rows.append(
            tag(
                "tr",
                None,
                raw(
                    tag("td", None, str(rank))
                    + tag(
                        "td",
                        None,
                        raw(tag("code", {"class": "inline"}, entry.decision_id)),
                    )
                    + tag("td", None, entry.decision_type)
                    + tag("td", None, entry.chosen_action)
                    + tag(
                        "td",
                        None,
                        raw(_render_badge(entry.identifiability)),
                    )
                    + influence_cell
                ),
            )
        )
    if not rows:
        rows.append(
            tag(
                "tr",
                None,
                raw(
                    tag(
                        "td",
                        {"colspan": 6, "class": "placeholder"},
                        "(no decisions in this trace are interventionable)",
                    )
                ),
            )
        )
    return tag(
        "table",
        {"class": "attribution"},
        raw(tag("thead", None, raw(head))),
        raw(tag("tbody", None, raw("".join(rows)))),
    )


def _render_verdict(report: ExplainReport) -> str:
    if report.degenerate_estimate is not None:
        card = _render_estimate_card(
            report.degenerate_estimate,
            title="degenerate refusal — corpus is single-class",
            subtitle="No fitted model; no per-decision attribution.",
        )
        return tag(
            "section",
            {"class": "verdict"},
            raw(tag("h2", None, "Honest verdict")),
            raw(
                tag(
                    "p",
                    {"class": "section-subtitle"},
                    "Single-class corpus — `fit_outcome_model` is intentionally skipped.",
                )
            ),
            raw(tag("div", {"class": "cards"}, raw(card))),
        )

    cards = "".join(
        _render_estimate_card(
            entry.estimate,
            title=(
                f"{entry.decision_type} :: {entry.chosen_action}  (decision_id={entry.decision_id})"
            ),
            subtitle=f"rank {rank}",
        )
        for rank, entry in enumerate(report.attribution.entries, start=1)
        if entry.estimate is not None
    )
    if not cards:
        cards = tag(
            "p",
            {"class": "placeholder"},
            "(no per-decision CausalEstimates were attached by attribute_failure)",
        )
    return tag(
        "section",
        {"class": "verdict"},
        raw(tag("h2", None, "Honest verdict")),
        raw(
            tag(
                "p",
                {"class": "section-subtitle"},
                "One CausalEstimate card per ranked decision. Cards are "
                "grounded in `CausalEstimate` fields; numeric estimates are "
                "suppressed when identifiability=unidentified.",
            )
        ),
        raw(_render_attribution_table(report.attribution)),
        raw(tag("div", {"class": "cards"}, raw(cards))),
    )


# --------------------------------------------------------------------------
# Footer + entry point
# --------------------------------------------------------------------------


def _render_footer(report: ExplainReport, *, generated_at: datetime) -> str:
    bits = [
        f"counterfact v{_COUNTERFACT_VERSION}",
        f"decision_type={report.summary_decision_type}",
        f"intervention_kind={report.decision_type_intervention_kind}",
        f"bootstrap={report.bootstrap}",
        f"seed={report.seed}",
        f"generated_at={generated_at.isoformat(timespec='seconds')}",
    ]
    return tag(
        "footer",
        None,
        raw(tag("div", None, " · ".join(bits))),
        raw(
            tag(
                "div",
                None,
                "See `docs/demo-excerpt.md` for the naive-vs-honest framing.",
            )
        ),
    )


def render_html(report: ExplainReport, *, now: datetime | None = None) -> str:
    """Return a complete self-contained HTML document for the report.

    Pass `now=` (a timezone-aware datetime) to pin the generation timestamp;
    tests rely on this for byte-identical determinism.
    """
    generated_at = now if now is not None else datetime.now(tz=UTC)

    top = _top_entry(report.attribution)
    focal_id = top.decision_id if top is not None else None

    title = (
        "counterfact diagnose" if report.diagnosis_summary is not None else "counterfact explain"
    )
    head = tag(
        "head",
        None,
        raw(tag("meta", {"charset": "utf-8"})),
        raw(
            tag(
                "meta",
                {"name": "viewport", "content": "width=device-width, initial-scale=1"},
            )
        ),
        raw(tag("title", None, f"{title} — {report.run.run_id}")),
        raw(tag("style", None, raw(CSS))),
    )
    body = tag(
        "body",
        None,
        raw(
            tag(
                "main",
                {"class": "page"},
                raw(_render_header(report, generated_at=generated_at)),
                raw(_render_story(report)),
                raw(_render_diagnosis_summary(report)),
                raw(_render_timeline(report)),
                raw(_render_descriptive(report)),
                raw(_render_dag(report.dag, focal_decision_id=focal_id)),
                raw(_render_decision_cards(report)),
                raw(_render_counterfactual_lookup(report)),
                raw(_render_verdict(report)),
                raw(_render_footer(report, generated_at=generated_at)),
            )
        ),
    )
    return "<!doctype html>\n" + tag("html", {"lang": "en"}, raw(head + body))
