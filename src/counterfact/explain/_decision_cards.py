"""The report's decision-card and counterfactual-lookup sections."""

from __future__ import annotations

import shlex
from typing import cast

from counterfact.attribute import AttributionEntry
from counterfact.explain._html import raw, section, tag
from counterfact.explain._verdict import badge
from counterfact.explain.render_contract import shows_outcome_delta
from counterfact.explain.report import ExplainReport
from counterfact.intervene.estimate import CausalEstimate, SupportPayload


def _step_index_for_decision(report: ExplainReport, decision_id: str) -> int | None:
    for step in report.run.steps:
        if any(decision.decision_id == decision_id for decision in step.decisions):
            return step.step_index
    return None


def _arm_names(rows: object) -> list[str]:
    # `NextStep.payload` is untyped JSON: each row is a plain arm name or an arm dict.
    if not isinstance(rows, list):
        return []
    names: list[str] = []
    for row in cast(list[object], rows):
        if isinstance(row, dict) and row.get("arm") is not None:
            names.append(str(row["arm"]))
        elif isinstance(row, str):
            names.append(row)
    return names


def _support_block(estimate: CausalEstimate) -> str | None:
    payload = cast(SupportPayload, estimate.next_step.payload)
    localization = payload.get("localization_limit")
    replay_inputs = payload.get("replay_inputs_required")
    lines = [
        ("observed arms", _arm_names(payload.get("observed_arms"))),
        ("missing arms", [str(arm) for arm in payload.get("missing_arms", [])]),
        ("missing strata", [str(item) for item in payload.get("missing_strata", [])]),
        (
            "localization limit",
            [localization] if isinstance(localization, str) and localization else [],
        ),
        (
            "replay inputs required",
            [str(item) for item in replay_inputs] if isinstance(replay_inputs, list) else [],
        ),
    ]
    parts = "".join(
        tag("li", None, f"{label}: " + ", ".join(values)) for label, values in lines if values
    )
    if not parts:
        return None
    return tag(
        "div",
        {"class": "callout support"},
        raw(tag("strong", None, "support diagnostics")),
        raw(tag("ul", {"class": "flat"}, raw(parts))),
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


def _decision_card(report: ExplainReport, entry: AttributionEntry, *, is_open: bool) -> str:
    step_index = _step_index_for_decision(report, entry.decision_id)
    estimate = entry.estimate
    if step_index is None:
        step_pill = tag("span", {"class": "subtle", "title": "step index unresolved"}, "step=?")
    else:
        step_pill = tag("span", {"class": "subtle"}, f"step={step_index}")
    decision_id = tag("code", {"class": "inline"}, entry.decision_id)
    decision_type = tag("span", None, entry.decision_type)
    status = badge(entry.identifiability)
    summary = tag("summary", None, raw(" ".join((decision_id, decision_type, step_pill, status))))
    body = tag(
        "div", {"class": "decision-head"}, raw(decision_id + decision_type + step_pill + status)
    ) + tag("div", {"class": "subtle"}, f"chosen_action={entry.chosen_action}")
    if estimate is not None:
        body += tag("div", {"class": "subtle"}, f"next_step.action={estimate.next_step.action}")
        body += _support_block(estimate) or ""
    command = _intervene_json_command(report, entry)
    if command is not None:
        body += tag("pre", {"class": "cmd"}, raw(tag("code", None, command)))
    return tag(
        "details",
        {
            "class": f"decision-card ident-{entry.identifiability.value}",
            "open": "open" if is_open else None,
        },
        raw(summary + body),
    )


def render_decision_cards(report: ExplainReport) -> str:
    if report.degenerate_estimate is not None and not report.attribution.entries:
        return section(
            "decision-cards",
            "Decision cards",
            tag(
                "p",
                {"class": "placeholder"},
                "(single-class corpus: no per-decision cards rendered)",
            ),
        )
    cards = "".join(
        _decision_card(report, entry, is_open=index == 0)
        for index, entry in enumerate(report.attribution.entries)
    ) or tag("p", {"class": "placeholder"}, "(no attribution entries available)")
    return section(
        "decision-cards",
        "Decision cards",
        tag("div", {"class": "decision-card-list"}, raw(cards)),
        subtitle="Decision-level trace forensics with support and replay guidance.",
    )


def _lookup_row(estimate: CausalEstimate) -> str:
    parts = [
        f"{estimate.query.decision_type}.{estimate.query.intervention_kind}",
        f"target={estimate.query.target}",
        f"baseline={estimate.query.baseline}",
        f"identifiability={estimate.identifiability.value}",
    ]
    if shows_outcome_delta(estimate):
        delta = estimate.outcome_delta
        assert delta is not None
        parts.append(f"outcome_delta={delta.point:.3f} [{delta.ci_low:.3f}, {delta.ci_high:.3f}]")
    else:
        parts.append(f"next_step={estimate.next_step.action}")
    return tag("div", {"class": "lookup-row"}, " | ".join(parts))


def render_counterfactual_lookup(report: ExplainReport) -> str:
    if not report.counterfactual_lookup:
        return ""
    rows = "".join(_lookup_row(estimate) for estimate in report.counterfactual_lookup)
    return section(
        "counterfactual-lookup",
        "Counterfactual lookup",
        tag("div", {"class": "lookup-list"}, raw(rows)),
        subtitle="Precomputed estimates only. Unidentified entries do not display numeric effects.",
    )
