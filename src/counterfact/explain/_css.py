"""Static content for the explain HTML renderer: CSS and identifiability gloss.

These are pure strings/dicts with no rendering logic. They live here so
`render_html.py` stays focused on layout, not styling and copy.
"""

from __future__ import annotations

from counterfact.intervene.estimate import IdentifiabilityStatus

GLOSS: dict[IdentifiabilityStatus, str] = {
    IdentifiabilityStatus.IDENTIFIED: (
        "the corpus supports this counterfactual under the stated assumptions."
    ),
    IdentifiabilityStatus.BOUNDED: (
        "the corpus partially supports this counterfactual; the bound widens by the e-value."
    ),
    IdentifiabilityStatus.UNIDENTIFIED: ("the corpus cannot answer this question; see next step."),
}

CSS: str = """\
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
.timeline-list,
.decision-card-list {
  display: grid;
  grid-template-columns: 1fr;
  gap: 10px;
}
.timeline-step,
.decision-card {
  background: var(--card);
  border: 1px solid var(--rule);
  padding: 12px;
}
details.decision-card summary {
  cursor: pointer;
  list-style-position: inside;
}
.diagnosis-summary {
  font-size: 15px;
  padding: 12px 14px;
  background: var(--card);
  border: 1px solid var(--rule);
  border-left: 4px solid #1e40af;
}
.lookup-list {
  display: grid;
  grid-template-columns: 1fr;
  gap: 6px;
}
.lookup-row {
  background: var(--card);
  border: 1px solid var(--rule);
  padding: 8px 10px;
  font-size: 12px;
}
.timeline-step .step-head,
.decision-card .decision-head {
  display: flex;
  gap: 8px;
  align-items: baseline;
  flex-wrap: wrap;
  font-weight: 600;
}
.timeline-step .obs-count,
.decision-card .subtle {
  color: var(--muted);
  font-size: 12px;
}
.decision-list {
  display: grid;
  grid-template-columns: 1fr;
  gap: 4px;
  margin-top: 8px;
}
.decision-row {
  font-size: 13px;
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
.callout.support { background: #f6f7f9; border-left-color: #64748b; }
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
