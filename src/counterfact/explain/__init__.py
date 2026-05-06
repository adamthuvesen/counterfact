"""Per-trace narrative explanation surface.

`build_report` composes a strict `ExplainReport` from a focal `Run` plus its
corpus, reusing the existing causal engine (`attribute_failure`,
`pass_rate_by_arm`, `build_dag`) and the shared degenerate-corpus refusal so
the honesty contract is single-sourced. `render_html` turns that report into
a self-contained HTML document grounded in `CausalEstimate` fields.
"""

from counterfact.explain.render_html import render_html
from counterfact.explain.report import ExplainReport, build_report

__all__ = ["ExplainReport", "build_report", "render_html"]
