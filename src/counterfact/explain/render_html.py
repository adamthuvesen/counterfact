"""Render an `ExplainReport` to a self-contained HTML document.

Every quantitative claim is sourced from a named `CausalEstimate` field;
nothing in this file invents a number. The single most important rule is in
`_render_estimate_card`: when `identifiability == UNIDENTIFIED`, the card
emits no numeric outcome_delta or influence — the spec requirement
"unidentified estimates suppress numeric estimates" gates this branch.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime

from counterfact import __version__ as _COUNTERFACT_VERSION
from counterfact.attribute import AttributionEntry, FailureAttribution
from counterfact.dag import DAG
from counterfact.explain._html import Raw, raw, tag
from counterfact.explain.report import ExplainReport
from counterfact.intervene.estimate import CausalEstimate, IdentifiabilityStatus
from counterfact.schema import Decision, Run

_GLOSS = {
    IdentifiabilityStatus.IDENTIFIED: (
        "the corpus supports this counterfactual under the stated assumptions."
    ),
    IdentifiabilityStatus.BOUNDED: (
        "the corpus partially supports this counterfactual; "
        "the bound widens by the e-value."
    ),
    IdentifiabilityStatus.UNIDENTIFIED: (
        "the corpus cannot answer this question; see next step."
    ),
}

_CSS = """\
:root {
  --bg: #fafafa;
  --fg: #1a1a1a;
  --muted: #556070;
  --rule: #d8dde5;
  --card: #ffffff;
  --code-bg: #f1f3f6;
  --identified-bg: #e6f5ec;
  --identified-fg: #14532d;
  --identified-line: #14532d;
  --bounded-bg: #fef3d7;
  --bounded-fg: #7a4d00;
  --bounded-line: #b45309;
  --unidentified-bg: #fde2e2;
  --unidentified-fg: #7a1313;
  --unidentified-line: #b91c1c;
  --focal-stroke: #b91c1c;
}
* { box-sizing: border-box; }
html, body {
  background: var(--bg);
  color: var(--fg);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
    "Helvetica Neue", sans-serif;
  font-size: 14px;
  line-height: 1.5;
  margin: 0;
}
.page { max-width: 960px; margin: 0 auto; padding: 24px; }
header.report-header h1 { font-size: 20px; margin: 0 0 4px 0; }
header.report-header .meta { color: var(--muted); font-size: 13px; }
header.report-header .meta dt {
  float: left;
  clear: left;
  width: 120px;
  font-weight: 600;
}
header.report-header .meta dd { margin-left: 130px; margin-bottom: 2px; }
section { margin-top: 28px; }
section h2 { font-size: 16px; margin: 0 0 8px 0; }
section .section-subtitle {
  color: var(--muted);
  font-size: 12px;
  margin-top: 0;
  margin-bottom: 12px;
}
table {
  border-collapse: collapse;
  width: 100%;
  background: var(--card);
  border: 1px solid var(--rule);
}
th, td {
  padding: 6px 10px;
  text-align: left;
  border-bottom: 1px solid var(--rule);
  font-size: 13px;
}
th { background: #f3f5f8; font-weight: 600; }
tr:last-child td { border-bottom: none; }
.story {
  font-size: 15px;
  padding: 12px 14px;
  background: var(--card);
  border: 1px solid var(--rule);
  border-left: 4px solid var(--rule);
}
.dag-wrapper {
  background: var(--card);
  border: 1px solid var(--rule);
  padding: 8px;
  overflow: auto;
}
.dag-wrapper svg .node rect {
  fill: #f3f5f8;
  stroke: #b3bcc9;
  stroke-width: 1;
}
.dag-wrapper svg .node.focal rect {
  stroke: var(--focal-stroke);
  stroke-width: 2.5;
}
.dag-wrapper svg .node text {
  font-size: 11px;
  fill: var(--fg);
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}
.dag-wrapper svg .edge { stroke: #889; stroke-width: 1; fill: none; }
.dag-wrapper svg .arrow { fill: #889; }
.cards { display: grid; grid-template-columns: 1fr; gap: 12px; }
.card {
  background: var(--card);
  border: 1px solid var(--rule);
  padding: 14px;
  border-left-width: 4px;
}
.card.ident-identified { border-left-color: var(--identified-line); }
.card.ident-bounded { border-left-color: var(--bounded-line); }
.card.ident-unidentified { border-left-color: var(--unidentified-line); }
.card-head {
  display: flex;
  gap: 8px;
  align-items: baseline;
  flex-wrap: wrap;
}
.card-head .title { font-weight: 600; font-size: 14px; }
.card-head .sub { color: var(--muted); font-size: 12px; }
.badge {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 999px;
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.02em;
}
.badge.ident-identified {
  background: var(--identified-bg);
  color: var(--identified-fg);
}
.badge.ident-bounded {
  background: var(--bounded-bg);
  color: var(--bounded-fg);
}
.badge.ident-unidentified {
  background: var(--unidentified-bg);
  color: var(--unidentified-fg);
}
.gloss { color: var(--muted); font-size: 12px; margin-top: 4px; }
.field { margin-top: 10px; }
.field .label {
  font-weight: 600;
  font-size: 12px;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.04em;
}
.field .value { font-size: 13px; }
.field .source { color: var(--muted); font-size: 11px; }
.callout {
  margin-top: 10px;
  padding: 10px 12px;
  background: #f3f5f8;
  border-left: 3px solid #889;
}
.callout.next-step { background: #eef2f7; border-left-color: #1e40af; }
code, pre {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 12px;
}
code.inline { background: var(--code-bg); padding: 1px 4px; border-radius: 3px; }
pre.cmd {
  background: var(--code-bg);
  padding: 8px 10px;
  overflow: auto;
  margin: 6px 0 0 0;
}
ul.flat { margin: 4px 0 0 18px; padding: 0; }
ul.flat li { margin: 2px 0; font-size: 12px; }
footer {
  margin-top: 32px;
  padding-top: 12px;
  border-top: 1px solid var(--rule);
  color: var(--muted);
  font-size: 12px;
}
.placeholder { color: var(--muted); font-style: italic; }
"""


# --------------------------------------------------------------------------
# Header / story
# --------------------------------------------------------------------------


def _outcome_label(run: Run) -> str:
    if run.outcome.kind == "binary":
        return "pass" if bool(run.outcome.value) else "fail"
    return f"{run.outcome.kind}={run.outcome.value!r}"


def _render_header(report: ExplainReport, *, generated_at: datetime) -> str:
    run = report.run
    items = [
        ("Run id", run.run_id),
        ("Outcome", _outcome_label(run)),
        ("Schema version", run.schema_version),
        ("Decision type", report.summary_decision_type),
        ("Corpus size", str(report.corpus_size)),
        (
            "Corpus pass rate",
            (
                f"{report.corpus_pass_rate:.3f}"
                if report.corpus_pass_rate is not None
                else "n/a"
            ),
        ),
        ("Generated", generated_at.isoformat(timespec="seconds")),
    ]
    if run.metadata.agent_name:
        items.insert(2, ("Agent", run.metadata.agent_name))
    dl = tag(
        "dl",
        {"class": "meta"},
        *[
            raw(tag("dt", None, label) + tag("dd", None, value))
            for label, value in items
        ],
    )
    return tag(
        "header",
        {"class": "report-header"},
        raw(tag("h1", None, "counterfact explain")),
        raw(dl),
    )


def _top_entry(attribution: FailureAttribution) -> AttributionEntry | None:
    for entry in attribution.entries:
        if entry.identifiability != IdentifiabilityStatus.UNIDENTIFIED:
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


def _render_dag(
    dag: DAG, *, focal_decision_id: str | None
) -> str:
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

    edge_parts: list[str] = []
    for parent_id, child_id in dag.edges:
        if parent_id not in positions or child_id not in positions:
            continue
        px, py = positions[parent_id]
        cx, cy = positions[child_id]
        x1 = px + box_w
        y1 = py + box_h // 2
        x2 = cx
        y2 = cy + box_h // 2
        edge_parts.append(
            tag(
                "line",
                {
                    "class": "edge",
                    "x1": str(x1),
                    "y1": str(y1),
                    "x2": str(x2),
                    "y2": str(y2),
                    "marker-end": "url(#cf-arrow)",
                    "data-parent-id": parent_id,
                    "data-child-id": child_id,
                },
            )
        )

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
    children: list[str] = [tag("div", {"class": "label"}, label)]
    if isinstance(value, Raw):
        children.append(tag("div", {"class": "value"}, value))
    else:
        children.append(tag("div", {"class": "value"}, value))
    if source:
        children.append(tag("div", {"class": "source"}, f"source: {source}"))
    return tag("div", {"class": "field"}, raw("".join(children)))


def _render_list(items: Iterable[str]) -> Raw:
    rendered = "".join(tag("li", None, item) for item in items)
    return raw(tag("ul", {"class": "flat"}, raw(rendered)))


def _render_next_step(estimate: CausalEstimate) -> str:
    ns = estimate.next_step
    suggested = ns.payload.get("suggested_command")
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
        body_parts.append(
            tag("pre", {"class": "cmd"}, raw(tag("code", None, suggested)))
        )
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

    Honesty contract (see openspec/specs/explain-report/spec.md
    "unidentified estimates suppress numeric estimates"): when
    identifiability is UNIDENTIFIED, this function MUST NOT emit any
    quantitative point estimate, CI, or influence number for the entry.
    The branch below is the single place that gates `outcome_delta`
    rendering — keep it that way.
    """
    status = estimate.identifiability
    head_children: list[str] = [tag("span", {"class": "title"}, title)]
    if subtitle:
        head_children.append(tag("span", {"class": "sub"}, subtitle))
    head_children.append(_render_badge(status))
    head = tag("div", {"class": "card-head"}, raw("".join(head_children)))
    gloss = tag("div", {"class": "gloss"}, _GLOSS[status])

    fields: list[str] = []
    if estimate.estimand:
        fields.append(
            _render_field("estimand", estimate.estimand, source="estimand")
        )
    if estimate.reason:
        fields.append(_render_field("reason", estimate.reason, source="reason"))

    # Only render outcome_delta when status is not UNIDENTIFIED AND
    # outcome_delta is present. Both gates must hold.
    if (
        status != IdentifiabilityStatus.UNIDENTIFIED
        and estimate.outcome_delta is not None
    ):
        d = estimate.outcome_delta
        value = (
            f"{d.point:.3f} [{d.ci_low:.3f}, {d.ci_high:.3f}] "
            f"(n_bootstrap={d.n_bootstrap})"
        )
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
        fields.append(
            _render_field("bounds", bound_value, source="bounds.e_value")
        )

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
        fields.append(
            _render_field(
                "warnings", _render_list(estimate.warnings), source="warnings"
            )
        )

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
        if entry.identifiability == IdentifiabilityStatus.UNIDENTIFIED:
            influence_cell = tag(
                "td",
                {"class": "placeholder"},
                "n/a (unidentified)",
            )
        else:
            influence_cell = tag(
                "td", None, f"{entry.influence:.3f}"
            )
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
                    "Single-class corpus — `fit_outcome_model` is "
                    "intentionally skipped.",
                )
            ),
            raw(tag("div", {"class": "cards"}, raw(card))),
        )

    cards = "".join(
        _render_estimate_card(
            entry.estimate,
            title=(
                f"{entry.decision_type} :: {entry.chosen_action}  "
                f"(decision_id={entry.decision_id})"
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
        raw(tag("title", None, f"counterfact explain — {report.run.run_id}")),
        raw(tag("style", None, raw(_CSS))),
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
                raw(_render_descriptive(report)),
                raw(_render_dag(report.dag, focal_decision_id=focal_id)),
                raw(_render_verdict(report)),
                raw(_render_footer(report, generated_at=generated_at)),
            )
        ),
    )
    return "<!doctype html>\n" + tag("html", {"lang": "en"}, raw(head + body))
