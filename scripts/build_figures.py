"""Draw the benchmark figures in docs/figures/ from docs/benchmark-results.json.

Stdlib only, so the figures need no plotting dependency. Each figure is a dot
plot of estimator bias against the exact truth, one row per scenario. The SVGs
carry light and dark colors via prefers-color-scheme.

    uv run python scripts/build_figures.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "docs" / "benchmark-results.json"
OUT = ROOT / "docs" / "figures"

WIDTH = 860
LABEL_W = 320
RIGHT_PAD = 32
ROW_H = 44
TOP = 92
BOTTOM = 54

STYLE = """
<style>
  svg { --ink: #0b0b0b; --ink-2: #52514e; --ink-3: #8a8983; --grid: #e4e3df;
        --naive: #eb6834; --model: #1baf7a; --engine: #2a78d6; --bg: #fcfcfb; }
  @media (prefers-color-scheme: dark) {
    svg { --ink: #f0efec; --ink-2: #c3c2b7; --ink-3: #8f8e86; --grid: #34342f;
          --naive: #d95926; --model: #199e70; --engine: #3987e5; --bg: #1a1a19; }
  }
  text { font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
         fill: var(--ink-2); font-size: 13px; }
  .title { fill: var(--ink); font-size: 16px; font-weight: 600; }
  .sub { fill: var(--ink-2); font-size: 12.5px; }
  .row { fill: var(--ink); }
  .tick { fill: var(--ink-3); font-size: 11.5px; }
  .note { fill: var(--ink-2); font-size: 12px; font-style: italic; }
  .grid { stroke: var(--grid); stroke-width: 1; }
  .zero { stroke: var(--ink-3); stroke-width: 1.5; }
  .naive { fill: var(--naive); } .model { fill: var(--model); } .engine { fill: var(--engine); }
  .bg { fill: var(--bg); }
  circle { stroke: var(--bg); stroke-width: 2; }
</style>
"""


@dataclass(frozen=True)
class Row:
    label: str
    naive: float | None
    model: float | None
    engine: float | None
    note: str | None = None


@dataclass(frozen=True)
class Figure:
    title: str
    subtitle: str
    legend: tuple[str, str, str]
    rows: list[Row]
    x_min: float
    x_max: float
    ticks: tuple[float, ...]


def _x(value: float, fig: Figure) -> float:
    span = WIDTH - LABEL_W - RIGHT_PAD
    return LABEL_W + (value - fig.x_min) / (fig.x_max - fig.x_min) * span


def render(fig: Figure) -> str:
    height = TOP + ROW_H * len(fig.rows) + BOTTOM
    plot_bottom = TOP + ROW_H * len(fig.rows)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {height}" '
        f'width="{WIDTH}" height="{height}" role="img" aria-label="{escape(fig.title)}">',
        STYLE,
        f'<rect class="bg" width="{WIDTH}" height="{height}" rx="8"/>',
        f"<title>{escape(fig.title)}</title>",
        f'<text class="title" x="16" y="28">{escape(fig.title)}</text>',
        f'<text class="sub" x="16" y="48">{escape(fig.subtitle)}</text>',
    ]
    legend_x = 16.0
    for cls, name in zip(("naive", "model", "engine"), fig.legend, strict=True):
        parts.append(f'<circle class="{cls}" cx="{legend_x + 6}" cy="70" r="6"/>')
        parts.append(f'<text x="{legend_x + 18}" y="74">{escape(name)}</text>')
        legend_x += 18 + 6.8 * len(name) + 28
    for tick in fig.ticks:
        x = _x(tick, fig)
        cls = "zero" if tick == 0 else "grid"
        parts.append(
            f'<line class="{cls}" x1="{x:.1f}" x2="{x:.1f}" y1="{TOP - 6}" y2="{plot_bottom}"/>'
        )
        parts.append(
            f'<text class="tick" x="{x:.1f}" y="{plot_bottom + 18}" '
            f'text-anchor="middle">{tick:+.1f}</text>'.replace("+0.0", "0")
        )
    parts.append(
        f'<text class="tick" x="{_x(0, fig):.1f}" y="{plot_bottom + 38}" text-anchor="middle">'
        "mean bias against the exact truth (0 = unbiased)</text>"
    )
    for i, row in enumerate(fig.rows):
        y = TOP + ROW_H * i + ROW_H / 2
        parts.append(
            f'<text class="row" x="{LABEL_W - 14}" y="{y + 4:.1f}" '
            f'text-anchor="end">{escape(row.label)}</text>'
        )
        # Dodge the series vertically so equal biases stay visible.
        series = (("naive", row.naive, -9), ("model", row.model, 0), ("engine", row.engine, 9))
        for cls, value, dy in series:
            if value is not None:
                parts.append(
                    f'<circle class="{cls}" cx="{_x(value, fig):.1f}" cy="{y + dy:.1f}" '
                    f'r="6"><title>{value:+.3f}</title></circle>'
                )
        if row.note:
            parts.append(
                f'<text class="note" x="{_x(0, fig) - 10:.1f}" y="{y + 4:.1f}" '
                f'text-anchor="end">{escape(row.note)}</text>'
            )
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def scenario_figure(results: dict) -> Figure:
    rows = [
        Row(
            s["scenario"].replace("_", " "),
            s["naive_bias"],
            s["adjust_all_bias"],
            s["engine_bias"],
            note="engine: bounded" if s["expected_label"] == "bounded" else None,
        )
        for s in results["scenarios"]
    ]
    return Figure(
        title="Single decision: sonnet vs haiku, 8 causal models built to break estimators",
        subtitle=f"{results['seeds']} corpora per scenario. The engine refuses where the "
        "data cannot identify the effect.",
        legend=("naive pass-rate gap", "adjust-everything g-formula", "counterfact (AIPW)"),
        rows=rows,
        x_min=-0.15,
        x_max=0.35,
        ticks=(-0.1, 0.0, 0.1, 0.2, 0.3),
    )


def policy_figure(results: dict) -> Figure:
    rows = [
        Row(
            p["case"]
            .replace(", outcome model blind to tests_passed", " (blind outcome model)")
            .replace(", retry propensity not logged", " (propensity unlogged)")
            .replace("_", " "),
            p["matched_bias"],
            p["ice_bias"],
            p["dr_bias"],
            note="engine: bounded" if p["expected_label"] == "bounded" else None,
        )
        for p in results["policies"]
    ]
    return Figure(
        title="Whole policies: retry rules over a sequential agent",
        subtitle=f"{results['seeds']} corpora per case. Test results confound the retry choice.",
        legend=("filter to matching runs", "outcome model only (ICE)", "counterfact (DR)"),
        rows=rows,
        x_min=-0.25,
        x_max=0.25,
        ticks=(-0.2, -0.1, 0.0, 0.1, 0.2),
    )


def main() -> None:
    results = json.loads(RESULTS.read_text())
    OUT.mkdir(exist_ok=True)
    (OUT / "single-decision-bias.svg").write_text(render(scenario_figure(results)))
    (OUT / "policy-bias.svg").write_text(render(policy_figure(results)))


if __name__ == "__main__":
    main()
