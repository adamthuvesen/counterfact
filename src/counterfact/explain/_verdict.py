"""The report's honest-verdict section: the attribution table and one estimate card per decision.

The single most important rule is in `_estimate_card`: when
`identifiability == UNIDENTIFIED`, the card emits no numeric outcome_delta or
influence. Every number comes from a named `CausalEstimate` field.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import cast

from counterfact.attribute import AttributionEntry, FailureAttribution
from counterfact.explain._css import GLOSS
from counterfact.explain._html import Raw, header_row, raw, section, tag
from counterfact.explain.render_contract import shows_numeric_attribution, shows_outcome_delta
from counterfact.explain.report import ExplainReport
from counterfact.intervene.estimate import CausalEstimate, IdentifiabilityStatus, SupportPayload


def badge(status: IdentifiabilityStatus) -> str:
    return tag("span", {"class": f"badge ident-{status.value}"}, status.value)


def top_entry(attribution: FailureAttribution) -> AttributionEntry | None:
    return next((entry for entry in attribution.entries if shows_numeric_attribution(entry)), None)


def _field(label: str, value: str | Raw, source: str) -> str:
    return tag(
        "div",
        {"class": "field"},
        raw(
            tag("div", {"class": "label"}, label)
            + tag("div", {"class": "value"}, value)
            + tag("div", {"class": "source"}, f"source: {source}")
        ),
    )


def _flat_list(items: Iterable[str]) -> Raw:
    return raw(tag("ul", {"class": "flat"}, raw("".join(tag("li", None, item) for item in items))))


def _next_step(estimate: CausalEstimate) -> str:
    ns = estimate.next_step
    suggested = cast(SupportPayload, ns.payload).get("suggested_command")
    body = tag(
        "div",
        None,
        raw(
            tag("strong", None, "next_step.action: ") + tag("code", {"class": "inline"}, ns.action)
        ),
    ) + tag("p", None, ns.human_text)
    if isinstance(suggested, str) and suggested:
        body += tag("pre", {"class": "cmd"}, raw(tag("code", None, suggested)))
    return tag("div", {"class": "callout next-step"}, raw(body))


def _estimate_card(estimate: CausalEstimate, *, title: str, subtitle: str) -> str:
    """Render one CausalEstimate card.

    Honesty contract: when `identifiability == UNIDENTIFIED`, this function
    MUST NOT emit any quantitative point estimate, CI, or influence number
    for the entry. `shows_outcome_delta` is the single place that gates
    `outcome_delta` rendering — keep it that way.
    """
    status = estimate.identifiability
    head = tag(
        "div",
        {"class": "card-head"},
        raw(
            tag("span", {"class": "title"}, title)
            + tag("span", {"class": "sub"}, subtitle)
            + badge(status)
        ),
    )
    fields: list[str] = []
    if estimate.estimand:
        fields.append(_field("estimand", estimate.estimand, "estimand"))
    if estimate.reason:
        fields.append(_field("reason", estimate.reason, "reason"))
    if shows_outcome_delta(estimate):
        d = estimate.outcome_delta
        assert d is not None
        value = f"{d.point:.3f} [{d.ci_low:.3f}, {d.ci_high:.3f}] ({d.ci_method})"
        fields.append(_field("outcome_delta", value, "outcome_delta.point / ci_low / ci_high"))
    if estimate.bounds is not None:
        b = estimate.bounds
        bound_value = f"[{b.lower:+.3f}, {b.upper:+.3f}] (technique={b.technique})"
        if b.note:
            bound_value += f" — {b.note}"
        fields.append(_field("bounds", bound_value, "bounds.lower / upper"))
    if estimate.diagnostics is not None:
        diag = estimate.diagnostics
        cross_check = (
            f"g-formula {diag.g_formula_delta:+.3f}, IPW {diag.ipw_delta:+.3f}; "
            f"min propensity {diag.min_propensity_target:.2f} / "
            f"{diag.min_propensity_baseline:.2f}; n={diag.n_used}"
        )
        fields.append(_field("cross-check", cross_check, "diagnostics"))
    for name in ("adjustment_set", "assumptions", "warnings"):
        items: list[str] = getattr(estimate, name)
        if items:
            fields.append(_field(name, _flat_list(items), name))
    return tag(
        "div",
        {"class": f"card ident-{status.value}"},
        raw(head),
        raw(tag("div", {"class": "gloss"}, GLOSS[status])),
        raw("".join(fields)),
        raw(_next_step(estimate)),
    )


def _attribution_row(rank: int, entry: AttributionEntry) -> str:
    if shows_numeric_attribution(entry):
        influence = tag("td", None, f"{entry.influence:.3f}")
    else:
        influence = tag("td", {"class": "placeholder"}, "n/a (unidentified)")
    return tag(
        "tr",
        None,
        raw(
            tag("td", None, str(rank))
            + tag("td", None, raw(tag("code", {"class": "inline"}, entry.decision_id)))
            + tag("td", None, entry.decision_type)
            + tag("td", None, entry.chosen_action)
            + tag("td", None, raw(badge(entry.identifiability)))
            + influence
        ),
    )


def _attribution_table(attribution: FailureAttribution) -> str:
    """Compact ranked table; the influence cell is a placeholder for unidentified entries."""
    head = header_row(
        "rank", "decision_id", "decision_type", "chosen_action", "identifiability", "influence"
    )
    rows = "".join(
        _attribution_row(rank, entry) for rank, entry in enumerate(attribution.entries, start=1)
    ) or tag(
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
    return tag(
        "table",
        {"class": "attribution"},
        raw(tag("thead", None, raw(head))),
        raw(tag("tbody", None, raw(rows))),
    )


def render_verdict(report: ExplainReport) -> str:
    if report.degenerate_estimate is not None:
        card = _estimate_card(
            report.degenerate_estimate,
            title="degenerate refusal — corpus is single-class",
            subtitle="No fitted model; no per-decision attribution.",
        )
        return section(
            "verdict",
            "Honest verdict",
            tag("div", {"class": "cards"}, raw(card)),
            subtitle="Single-class corpus — `fit_outcome_model` is intentionally skipped.",
        )

    cards = "".join(
        _estimate_card(
            entry.estimate,
            title=(
                f"{entry.decision_type} :: {entry.chosen_action}  (decision_id={entry.decision_id})"
            ),
            subtitle=f"rank {rank}",
        )
        for rank, entry in enumerate(report.attribution.entries, start=1)
        if entry.estimate is not None
    ) or tag(
        "p",
        {"class": "placeholder"},
        "(no per-decision CausalEstimates were attached by attribute_failure)",
    )
    return section(
        "verdict",
        "Honest verdict",
        _attribution_table(report.attribution),
        tag("div", {"class": "cards"}, raw(cards)),
        subtitle=(
            "One CausalEstimate card per ranked decision. Cards are "
            "grounded in `CausalEstimate` fields; numeric estimates are "
            "suppressed when identifiability=unidentified."
        ),
    )
