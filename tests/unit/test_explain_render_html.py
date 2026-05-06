"""Renderer tests grounded in CausalEstimate fields.

These tests parse the rendered HTML with stdlib (`html.parser`) plus targeted
regexes — no BeautifulSoup, no soup-style traversal. The goal is to pin the
honesty contract: identifiability badges match the source enum value,
unidentified cards never emit numeric estimates, the descriptive section is
framed and ordered before the causal one, and the document is self-contained.
"""

from __future__ import annotations

import html as html_lib
import re
import shlex
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path

import pytest

from counterfact.attribute import AttributionEntry, FailureAttribution
from counterfact.explain import build_report, render_html
from counterfact.explain.render_html import _node_label
from counterfact.explain.report import ExplainReport
from counterfact.intervene.estimate import (
    CausalEstimate,
    DistributionSummary,
    IdentifiabilityStatus,
    InterventionQuery,
    NextStep,
    SensitivityBounds,
)
from counterfact.schema import Decision, Metadata, Outcome, Run

# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

FIXED_NOW = datetime(2026, 5, 3, 12, 0, 0, tzinfo=UTC)


def _synthetic_corpus(n: int = 24, seed: int = 7) -> list[Run]:
    from bench.synthetic import generate_traces

    return [Run.model_validate(t) for t in generate_traces(n=n, seed=seed)]


def _single_class_refusal_corpus() -> list[Run]:
    return [
        Run.model_validate_json(p.read_text())
        for p in sorted(Path("bench/real/single_class_refusal").glob("*.json"))
    ]


def test_node_label__hidden_fixture_termination_success_names_public_tests() -> None:
    run = Run(
        schema_version="0.1.0",
        run_id="hidden-fail",
        outcome=Outcome(
            kind="binary",
            value=False,
            verifier="pytest_hidden",
            metadata={
                "public_pass": True,
                "hidden_pass": False,
                "generalization_gap": True,
            },
        ),
        metadata=Metadata(),
    )
    decision = Decision(
        decision_id="d-term",
        decision_type="termination",
        chosen_action="success",
    )

    assert _node_label(decision, run) == "termination::public_tests_passed"


def _identified_estimate() -> CausalEstimate:
    return CausalEstimate(
        query=InterventionQuery(
            decision_type="model_call",
            intervention_kind="model_choice",
            target="small",
            step=1,
        ),
        identifiability=IdentifiabilityStatus.IDENTIFIED,
        estimand="E[Y | do(model=small)]",
        outcome_delta=DistributionSummary(
            point=0.332, ci_low=0.179, ci_high=0.493, n_bootstrap=200
        ),
        bounds=SensitivityBounds(e_value=1.4),
        assumptions=["positivity holds for model_call"],
        warnings=[],
        next_step=NextStep(
            action="increase_n",
            payload={
                "current_n": 30,
                "estimated_required_n": 416,
                "target_ci_width": 0.1,
                "power_method": "wilson",
                "arm_breakdown": [],
                "suggested_command": (
                    "uv run counterfact bench real --n 416 "
                    "--fixture-set broad_calibration"
                ),
            },
            human_text="Tighten the CI by collecting ~416 traces.",
        ),
    )


def _unidentified_estimate() -> CausalEstimate:
    return CausalEstimate(
        query=InterventionQuery(
            decision_type="model_call",
            intervention_kind="model_choice",
            target="small",
            step=-1,
        ),
        identifiability=IdentifiabilityStatus.UNIDENTIFIED,
        reason="single-class corpus",
        outcome_delta=None,
        bounds=None,
        next_step=NextStep(
            action="broaden_arm_support",
            payload={
                "arm_name": "outcome",
                "missing_strata": ["Outcome.value=False"],
                "observed_arms": [],
                "missing_arms": [],
            },
            human_text="Collect mixed-outcome traces.",
        ),
    )


def _unidentified_support_estimate() -> CausalEstimate:
    return CausalEstimate(
        query=InterventionQuery(
            decision_type="model_call",
            intervention_kind="model_choice",
            target="opus",
            step=2,
        ),
        identifiability=IdentifiabilityStatus.UNIDENTIFIED,
        reason="missing target arm",
        next_step=NextStep(
            action="broaden_arm_support",
            payload={
                "arm_name": "model_call",
                "missing_strata": ["model_call::opus"],
                "observed_arms": [
                    {
                        "arm": "haiku",
                        "n": 10,
                        "pass_count": 4,
                        "pass_rate": 0.4,
                        "ci_low": 0.1,
                        "ci_high": 0.7,
                    }
                ],
                "missing_arms": ["opus"],
                "localization_limit": "decision-id support is absent",
            },
            human_text="Collect traces with opus support.",
        ),
    )


class _SectionLocator(HTMLParser):
    """Find the byte offsets of `<h2>` text within the rendered document."""

    def __init__(self) -> None:
        super().__init__()
        self.headings: list[tuple[int, str]] = []
        self._inside_h2 = False
        self._buf = ""
        self._start_offset: int | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "h2":
            self._inside_h2 = True
            self._buf = ""
            self._start_offset = self.getpos()[0]

    def handle_endtag(self, tag: str) -> None:
        if tag == "h2" and self._inside_h2:
            assert self._start_offset is not None
            self.headings.append((self._start_offset, self._buf))
            self._inside_h2 = False
            self._buf = ""

    def handle_data(self, data: str) -> None:
        if self._inside_h2:
            self._buf += data


class _NodeAttrCollector(HTMLParser):
    convert_charrefs = True

    def __init__(self, attr: str) -> None:
        super().__init__()
        self._attr = attr
        self.values: list[str] = []
        self.classes: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_dict = {k: (v or "") for k, v in attrs}
        if self._attr in attr_dict:
            self.values.append(attr_dict[self._attr])
            self.classes.append(attr_dict.get("class", ""))


def _collect_data_decision_ids(html: str) -> tuple[list[str], list[str]]:
    parser = _NodeAttrCollector("data-decision-id")
    parser.feed(html)
    return parser.values, parser.classes


def _count_substring(html: str, needle: str) -> int:
    return html.count(needle)


def _extract_section(html: str, h2_text: str) -> str:
    """Return the substring between `<h2>{h2_text}</h2>` and the next `<h2>`.

    Used to scope assertions to a single rendered section so substring
    matches in unrelated tables (e.g., the descriptive baseline) cannot
    silently satisfy a section-specific test.
    """
    needle = f"<h2>{h2_text}</h2>"
    start = html.find(needle)
    if start == -1:
        raise AssertionError(f"section heading not found: {h2_text!r}")
    end = html.find("<h2>", start + len(needle))
    if end == -1:
        end = len(html)
    return html[start:end]


def _build_synthetic_report(
    *, corpus_size: int = 24, seed: int = 7
) -> ExplainReport:
    corpus = _synthetic_corpus(n=corpus_size, seed=seed)
    return build_report(
        corpus[0], corpus, decision_type="model_call", bootstrap=20, seed=seed
    )


def _isolate_card(html: str, status: str) -> str:
    """Return the substring corresponding to the first card whose root
    element carries `class="card ident-<status>"`. Crude but enough for
    these regex-based assertions."""
    needle = f'class="card ident-{status}"'
    start = html.find(needle)
    assert start != -1, f"no card with status={status!r} present"
    # Find the matching closing </div> by counting nested <div> within the
    # slice. Since `tag()` always produces well-balanced output, this works.
    cursor = start
    depth = 0
    while True:
        nxt_open = html.find("<div", cursor + 1)
        nxt_close = html.find("</div>", cursor + 1)
        assert nxt_close != -1, "card never closed"
        if nxt_open != -1 and nxt_open < nxt_close:
            depth += 1
            cursor = nxt_open
            continue
        if depth == 0:
            return html[start : nxt_close + len("</div>")]
        depth -= 1
        cursor = nxt_close


def _make_synthetic_attribution(
    estimate_identified: CausalEstimate,
    estimate_unidentified: CausalEstimate,
) -> FailureAttribution:
    return FailureAttribution(
        entries=[
            AttributionEntry(
                decision_id="d-id-001",
                decision_type="model_call",
                chosen_action="small",
                influence=0.332,
                identifiability=IdentifiabilityStatus.IDENTIFIED,
                estimate=estimate_identified,
            ),
            AttributionEntry(
                decision_id="d-uid-001",
                decision_type="tool_call",
                chosen_action="inspect_file",
                influence=0.0,
                identifiability=IdentifiabilityStatus.UNIDENTIFIED,
                estimate=estimate_unidentified,
            ),
        ]
    )


def _hand_built_report(
    attribution: FailureAttribution,
    *,
    degenerate: CausalEstimate | None = None,
) -> ExplainReport:
    """Hand-build an ExplainReport from a synthetic run + corpus so tests
    can pin specific CausalEstimate combinations without going through
    `attribute_failure`."""
    corpus = _synthetic_corpus(n=12, seed=11)
    focal = corpus[0]
    base = build_report(focal, corpus, bootstrap=10, seed=11)
    return base.model_copy(
        update={
            "attribution": attribution,
            "degenerate_estimate": degenerate,
            "run_path": "runs/syn-000000.json",
            "corpus_dir": "runs",
        }
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_render__starts_with_doctype_and_lang() -> None:
    report = _build_synthetic_report()
    html = render_html(report, now=FIXED_NOW)
    assert html.lstrip()[:15].lower().startswith("<!doctype html>")
    assert '<html lang="en">' in html


def test_render__timeline_lists_steps_and_decision_metadata() -> None:
    report = _build_synthetic_report()
    html = render_html(report, now=FIXED_NOW)

    assert '<section class="timeline">' in html
    for step in report.run.steps:
        assert f'data-step-index="{step.step_index}"' in html
        for decision in step.decisions:
            assert decision.decision_id in html
            assert decision.decision_type in html
            if decision.chosen_action:
                assert decision.chosen_action in html
    first = html.find(f'data-step-index="{report.run.steps[0].step_index}"')
    last = html.find(f'data-step-index="{report.run.steps[-1].step_index}"')
    assert first < last


def test_render__timeline_omits_raw_observation_content() -> None:
    corpus = _synthetic_corpus(n=12, seed=11)
    focal = corpus[0].model_copy(
        update={
            "steps": [
                step.model_copy(
                    update={
                        "observations": [
                            {
                                "observation_id": "o-secret",
                                "content": {"stdout_tail": "RAW_OBSERVATION_SECRET"},
                            }
                        ]
                    }
                )
                if step.step_index == 1
                else step
                for step in corpus[0].steps
            ]
        }
    )
    corpus = [focal, *corpus[1:]]
    report = build_report(focal, corpus, bootstrap=10, seed=11)

    html = render_html(report, now=FIXED_NOW)
    assert "observations=1" in html
    assert "RAW_OBSERVATION_SECRET" not in html


def test_render__badge_text_matches_identifiability_value_for_each_status() -> None:
    """For every status, the badge element's text equals the enum's
    `.value`."""
    identified = _identified_estimate()
    unid = _unidentified_estimate()
    attribution = _make_synthetic_attribution(identified, unid)
    report = _hand_built_report(attribution)
    html = render_html(report, now=FIXED_NOW)

    # identified badge
    assert (
        '<span class="badge ident-identified">identified</span>' in html
    )
    # unidentified badge
    assert (
        '<span class="badge ident-unidentified">unidentified</span>' in html
    )


def test_render__identified_card_shows_outcome_delta_with_three_decimals() -> None:
    identified = _identified_estimate()
    unid = _unidentified_estimate()
    attribution = _make_synthetic_attribution(identified, unid)
    report = _hand_built_report(attribution)
    html = render_html(report, now=FIXED_NOW)
    card = _isolate_card(html, "identified")

    # Values come from outcome_delta.point / ci_low / ci_high.
    assert "0.332" in card
    assert "0.179" in card
    assert "0.493" in card
    # Source label names the field directly.
    assert "outcome_delta.point" in card


def test_render__identified_card_renders_assumptions_list() -> None:
    """Non-empty `CausalEstimate.assumptions` must surface in the rendered
    card. Each assumption is a separate list item."""
    identified = _identified_estimate()
    unid = _unidentified_estimate()
    attribution = _make_synthetic_attribution(identified, unid)
    report = _hand_built_report(attribution)
    html = render_html(report, now=FIXED_NOW)
    card = _isolate_card(html, "identified")

    # The fixture's assumption text appears verbatim inside the card.
    assert "positivity holds for model_call" in card
    # And it is rendered as a list item, not loose prose.
    assert "<li>positivity holds for model_call</li>" in card


def test_render__bounded_e_value_is_emitted_when_present_and_hidden_when_none() -> None:
    base_unid = _unidentified_estimate()

    bounded_with = CausalEstimate(
        query=InterventionQuery(
            decision_type="memory_read",
            intervention_kind="content_swap",
            target="cached_search_results",
            step=2,
        ),
        identifiability=IdentifiabilityStatus.BOUNDED,
        outcome_delta=DistributionSummary(
            point=0.21, ci_low=0.05, ci_high=0.37, n_bootstrap=200
        ),
        bounds=SensitivityBounds(e_value=1.4, note="loose bound"),
        next_step=NextStep(
            action="replay_required",
            payload={
                "intervention_target": "memory.read",
                "replay_inputs_required": ["context"],
                "note": "needs replay",
            },
            human_text="Replay required for tighter bound.",
        ),
    )
    bounded_without = bounded_with.model_copy(update={"bounds": None})

    attr_with = FailureAttribution(
        entries=[
            AttributionEntry(
                decision_id="d-bnd-1",
                decision_type="memory_read",
                chosen_action="cached",
                influence=0.21,
                identifiability=IdentifiabilityStatus.BOUNDED,
                estimate=bounded_with,
            )
        ]
    )
    attr_without = FailureAttribution(
        entries=[
            AttributionEntry(
                decision_id="d-bnd-1",
                decision_type="memory_read",
                chosen_action="cached",
                influence=0.21,
                identifiability=IdentifiabilityStatus.BOUNDED,
                estimate=bounded_without,
            )
        ]
    )
    html_with = render_html(_hand_built_report(attr_with), now=FIXED_NOW)
    html_without = render_html(_hand_built_report(attr_without), now=FIXED_NOW)

    card_with = _isolate_card(html_with, "bounded")
    card_without = _isolate_card(html_without, "bounded")

    assert "1.400" in card_with
    assert "loose bound" in card_with
    assert "bounds.e_value" in card_with
    assert "e_value" not in card_without
    _ = base_unid  # used as fixture import only; silence unused warnings


def test_render__unidentified_card_has_no_decimal_estimate() -> None:
    """Spec: unidentified card contains no `\\d+\\.\\d+\\s*\\[` substring."""
    identified = _identified_estimate()
    unid = _unidentified_estimate()
    attribution = _make_synthetic_attribution(identified, unid)
    report = _hand_built_report(attribution)
    html = render_html(report, now=FIXED_NOW)
    card = _isolate_card(html, "unidentified")

    # No decimal-followed-by-bracket pattern (the rendered shape of an
    # outcome_delta).
    assert re.search(r"\d+\.\d+\s*\[", card) is None
    # The badge still says unidentified verbatim.
    assert "unidentified" in card


def test_render__attribution_table_influence_cell_hides_numeric_for_unidentified() -> (
    None
):
    identified = _identified_estimate()
    unid = _unidentified_estimate()
    attribution = _make_synthetic_attribution(identified, unid)
    report = _hand_built_report(attribution)
    html = render_html(report, now=FIXED_NOW)

    # Influence cell for the unidentified row contains a non-numeric
    # placeholder ("n/a (unidentified)" or "—") rather than a decimal.
    table_start = html.find('<table class="attribution"')
    assert table_start != -1
    table_end = html.find("</table>", table_start)
    table = html[table_start:table_end]

    # The identified row should have a decimal-formatted influence (0.332).
    assert "0.332" in table
    # The unidentified row should carry the placeholder string.
    assert "n/a (unidentified)" in table


def test_render__suggested_command_appears_verbatim_in_code_element() -> None:
    identified = _identified_estimate()
    unid = _unidentified_estimate()
    attribution = _make_synthetic_attribution(identified, unid)
    report = _hand_built_report(attribution)
    html = render_html(report, now=FIXED_NOW)

    cmd = "uv run counterfact bench real --n 416 --fixture-set broad_calibration"
    expected = f"<code>{cmd}</code>"
    assert expected in html


def test_render__decision_card_renders_support_diagnostics() -> None:
    identified = _identified_estimate()
    unid = _unidentified_support_estimate()
    attribution = _make_synthetic_attribution(identified, unid)
    report = _hand_built_report(attribution)

    html = render_html(report, now=FIXED_NOW)
    section = _extract_section(html, "Decision cards")

    assert '<section class="decision-cards">' in html
    assert "support diagnostics" in section
    assert "observed arms: haiku" in section
    assert "missing arms: opus" in section
    assert "localization limit: decision-id support is absent" in section


def test_render__decision_card_includes_intervene_json_command() -> None:
    identified = _identified_estimate()
    unid = _unidentified_estimate()
    attribution = _make_synthetic_attribution(identified, unid)
    report = _hand_built_report(attribution)

    html = render_html(report, now=FIXED_NOW)

    assert "uv run counterfact intervene" in html
    assert "--decision-id d-id-001" in html
    assert "--set model_choice=small --json" in html


def test_render__decision_card_command_quotes_paths_with_spaces() -> None:
    identified = _identified_estimate()
    unid = _unidentified_estimate()
    attribution = _make_synthetic_attribution(identified, unid)
    base = _hand_built_report(attribution)
    report = base.model_copy(
        update={
            "run_path": "runs/My Traces/run-000.json",
            "corpus_dir": "runs/My Traces",
        }
    )

    html = render_html(report, now=FIXED_NOW)
    section = _extract_section(html, "Decision cards")

    match = re.search(r"uv run counterfact intervene[^<]+", section)
    assert match is not None, "intervene command not found in decision-cards section"
    command = html_lib.unescape(match.group(0).strip())

    parts = shlex.split(command)
    assert "runs/My Traces/run-000.json" in parts
    runs_dir_idx = parts.index("--runs-dir")
    assert parts[runs_dir_idx + 1] == "runs/My Traces"


def test_render__decision_card_unresolved_step_does_not_leak_none() -> None:
    identified = _identified_estimate()
    unid = _unidentified_estimate()
    base_attribution = _make_synthetic_attribution(identified, unid)
    orphan_entry = AttributionEntry(
        decision_id="d-not-in-run",
        decision_type="model_call",
        chosen_action="haiku",
        influence=0.0,
        identifiability=IdentifiabilityStatus.UNIDENTIFIED,
        estimate=unid,
    )
    attribution = FailureAttribution(
        entries=[*base_attribution.entries, orphan_entry]
    )
    report = _hand_built_report(attribution)

    html = render_html(report, now=FIXED_NOW)
    section = _extract_section(html, "Decision cards")

    assert "step=None" not in section
    assert "step=?" in section


def test_render__decision_card_unidentified_has_no_decimal_estimate() -> None:
    identified = _identified_estimate()
    unid = _unidentified_support_estimate()
    attribution = _make_synthetic_attribution(identified, unid)
    report = _hand_built_report(attribution)
    html = render_html(report, now=FIXED_NOW)

    start = html.find('<details class="decision-card ident-unidentified"')
    assert start != -1
    end = html.find("</details>", start)
    card = html[start:end]
    assert re.search(r"\d+\.\d+\s*\[", card) is None
    assert "next_step.action=broaden_arm_support" in card


def test_render__decision_cards_are_collapsible_details() -> None:
    identified = _identified_estimate()
    unid = _unidentified_estimate()
    attribution = _make_synthetic_attribution(identified, unid)
    report = _hand_built_report(attribution)
    html = render_html(report, now=FIXED_NOW)

    section = _extract_section(html, "Decision cards")
    assert "<details" in section
    assert "<summary>" in section
    assert "https://" not in section


def test_render__diagnosis_summary_renders_before_timeline() -> None:
    report = _build_synthetic_report().model_copy(
        update={
            "diagnosis_summary": (
                "Run syn-000000: most plausible supported failure point is d-model."
            )
        }
    )
    html = render_html(report, now=FIXED_NOW)

    diagnosis_pos = html.find("<h2>Diagnosis summary</h2>")
    timeline_pos = html.find("<h2>Trace timeline</h2>")
    assert diagnosis_pos != -1
    assert timeline_pos != -1
    assert diagnosis_pos < timeline_pos
    assert "most plausible supported failure point" in html


def test_render__counterfactual_lookup_hides_unidentified_numbers() -> None:
    identified = _identified_estimate()
    unid = _unidentified_estimate()
    report = _build_synthetic_report().model_copy(
        update={"counterfactual_lookup": [identified, unid]}
    )
    html = render_html(report, now=FIXED_NOW)
    section = _extract_section(html, "Counterfactual lookup")

    assert "outcome_delta=0.332 [0.179, 0.493]" in section
    assert "identifiability=unidentified" in section
    unidentified_row_start = section.find("identifiability=unidentified")
    unidentified_row_end = section.find("</div>", unidentified_row_start)
    unidentified_row = section[unidentified_row_start:unidentified_row_end]
    assert re.search(r"\d+\.\d+\s*\[", unidentified_row) is None


def test_render__svg_node_set_matches_dag() -> None:
    report = _build_synthetic_report()
    html = render_html(report, now=FIXED_NOW)
    expected_ids = sorted(d.decision_id for d in report.dag.nodes)
    seen_ids, _ = _collect_data_decision_ids(html)
    assert sorted(seen_ids) == expected_ids


def test_render__top_attributed_decision_node_has_focal_class() -> None:
    report = _build_synthetic_report()
    html = render_html(report, now=FIXED_NOW)

    top = next(
        (
            e
            for e in report.attribution.entries
            if e.identifiability != IdentifiabilityStatus.UNIDENTIFIED
        ),
        None,
    )
    assert top is not None, "synthetic corpus should yield identified entries"

    seen_ids, classes = _collect_data_decision_ids(html)
    idx = seen_ids.index(top.decision_id)
    assert "focal" in classes[idx]


def test_render__descriptive_heading_appears_before_honest_verdict() -> None:
    report = _build_synthetic_report()
    html = render_html(report, now=FIXED_NOW)

    locator = _SectionLocator()
    locator.feed(html)
    headings = locator.headings
    descriptive_pos = next(
        (i for i, (_, t) in enumerate(headings) if re.search(r"(?i)descriptive", t)),
        None,
    )
    verdict_pos = next(
        (
            i
            for i, (_, t) in enumerate(headings)
            if re.search(r"(?i)honest verdict|causal estimate|identifiability", t)
        ),
        None,
    )
    assert descriptive_pos is not None, "Descriptive heading missing"
    assert verdict_pos is not None, "Honest verdict heading missing"
    assert descriptive_pos < verdict_pos


def test_render__document_has_no_external_src_or_href() -> None:
    report = _build_synthetic_report()
    html = render_html(report, now=FIXED_NOW)
    assert re.search(r'(?:src|href)="https?://', html) is None


def test_render__byte_identical_for_fixed_inputs_and_clock() -> None:
    report = _build_synthetic_report()
    a = render_html(report, now=FIXED_NOW)
    b = render_html(report, now=FIXED_NOW)
    assert a == b


def test_render__single_class_corpus_produces_no_decimal_point_estimate() -> None:
    """End-to-end: the single_class_refusal single-class path must render the
    unidentified card with no card-level decimal estimate."""
    corpus = _single_class_refusal_corpus()
    focal = corpus[0]
    report = build_report(focal, corpus, bootstrap=20, seed=42)
    html = render_html(report, now=FIXED_NOW)

    card = _isolate_card(html, "unidentified")
    assert re.search(r"\d+\.\d+\s*\[", card) is None
    assert "broaden_arm_support" in card


def test_render__rejects_no_external_assets_with_strict_check() -> None:
    """Reinforce self-containment: no <script src=...> or <link href=...>
    referencing http(s) anywhere."""
    report = _build_synthetic_report()
    html = render_html(report, now=FIXED_NOW)
    assert "<script src=" not in html
    assert "<link " not in html or "http" not in html.split("<link", 1)[-1].split(">", 1)[0]


@pytest.mark.parametrize(
    "field",
    ["badge ident-identified", "badge ident-unidentified", "Descriptive baseline"],
)
def test_render__sentinel_strings_present(field: str) -> None:
    report = _build_synthetic_report()
    html = render_html(report, now=FIXED_NOW)
    if field == "badge ident-unidentified":
        # Synthetic corpus may produce some unidentified entries depending
        # on attribution selection — but the spec only guarantees the
        # status appears when a card with that status exists.
        if not any(
            e.identifiability == IdentifiabilityStatus.UNIDENTIFIED
            for e in report.attribution.entries
        ):
            pytest.skip("no unidentified entries in this synthetic corpus")
    assert field in html
