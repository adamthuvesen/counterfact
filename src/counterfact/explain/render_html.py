"""Render an `ExplainReport` to a self-contained HTML document.

This module assembles the page and renders the header, story, timeline, and
descriptive sections. The DAG (`_dag_svg`), decision cards (`_decision_cards`),
and honest verdict (`_verdict`) have their own modules. Every quantitative
claim is sourced from a named report field; nothing here invents a number.
"""

from __future__ import annotations

from datetime import UTC, datetime

from counterfact._fmt import outcome_label as _outcome_label
from counterfact._version import __version__ as _COUNTERFACT_VERSION
from counterfact.explain._css import CSS
from counterfact.explain._dag_svg import render_dag
from counterfact.explain._decision_cards import (
    render_counterfactual_lookup,
    render_decision_cards,
)
from counterfact.explain._html import header_row, raw, section, tag
from counterfact.explain._verdict import render_verdict, top_entry
from counterfact.explain.report import ExplainReport


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
        top = top_entry(report.attribution)
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
    return section(
        "diagnosis",
        "Diagnosis summary",
        tag("p", {"class": "diagnosis-summary"}, report.diagnosis_summary),
    )


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
    return section(
        "timeline",
        "Trace timeline",
        tag("div", {"class": "timeline-list"}, raw("".join(steps))),
        subtitle=(
            "Step-level decision metadata only. Raw observation and model "
            "contents are intentionally omitted."
        ),
    )


def _render_descriptive(report: ExplainReport) -> str:
    table = report.pass_rate_table
    head = header_row("arm", "n", "pass", "rate", "95% CI")
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
    return section(
        "descriptive",
        f"Descriptive baseline — {table.decision_type}",
        tag(
            "table",
            {"class": "pass-rate"},
            raw(tag("thead", None, raw(head))),
            raw(tag("tbody", None, raw(body))),
        ),
        subtitle=(
            "Marginal table — does not adjust for the agent's other "
            "randomized decisions. Use the honest verdict below for "
            "causal claims."
        ),
    )


def _render_footer(report: ExplainReport, *, generated_at: datetime) -> str:
    bits = [
        f"counterfact v{_COUNTERFACT_VERSION}",
        f"decision_type={report.summary_decision_type}",
        f"intervention_kind={report.decision_type_intervention_kind}",
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

    top = top_entry(report.attribution)
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
                raw(render_dag(report.dag, focal_decision_id=focal_id)),
                raw(render_decision_cards(report)),
                raw(render_counterfactual_lookup(report)),
                raw(render_verdict(report)),
                raw(_render_footer(report, generated_at=generated_at)),
            )
        ),
    )
    return "<!doctype html>\n" + tag("html", {"lang": "en"}, raw(head + body))
