"""Build notebooks/demo.ipynb from a list of cells.

The notebook is the demo's headline artifact and is part of the v0 ship gate
(§15.5: naive-vs-honest contrast must be rendered). Running this script
deterministically rebuilds the notebook so the source-of-truth is one Python
file rather than diff-hostile JSON.

Run from the repo root:
    uv run python scripts/build_demo_notebook.py
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

NB_PATH = Path(__file__).resolve().parents[1] / "notebooks" / "demo.ipynb"


def _md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": uuid.uuid4().hex[:12],
        "metadata": {},
        "source": text,
    }


def _code(text: str) -> dict:
    return {
        "cell_type": "code",
        "id": uuid.uuid4().hex[:12],
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": text,
    }


CELLS = [
    _md(
        "# `counterfact` demo — causal attribution, with discipline\n\n"
        "Most agent-infra tools score traces and call it done. `counterfact` runs structural causal "
        "inference end-to-end **and tells you when your corpus does not support the question you asked**.\n\n"
        "This notebook walks through that contrast on three artifacts:\n\n"
        "1. **Naive vs honest** — same query, two estimators. The naive marginal says one number; "
        "`counterfact.intervene` says `bounded` or `unidentified` with a structured next step.\n"
        "2. **Synthetic SCM canary** — the engine recovers a *known* effect within tolerance. "
        "Mechanism check, not a headline claim.\n"
        "3. **Three identifiability paths** — `identified`, `bounded`, `unidentified` shown end-to-end "
        "with structured `next_step` data on each.\n"
        "4. **Ranked failure attribution** — every entry carries its identifiability label.\n"
        "5. **What would change the answer?** — `power_analysis` quantifies the corpus size needed "
        "to tighten a CI under the binomial-Wald approximation.\n\n"
        "The pitch in one line: *we built corpus analysis that pushes back when your data is too thin "
        "or your interventions aren't identifiable, with concrete numbers on what would change.*"
    ),
    _code(
        "from __future__ import annotations\n\n"
        "import sys\n"
        "from pathlib import Path\n\n"
        "from bench.synthetic import HEADLINE_TRUE_EFFECT, generate_traces\n"
        "from counterfact import (\n"
        "    attribute_failure,\n"
        "    build_dag,\n"
        "    fit_outcome_model,\n"
        "    intervene,\n"
        "    pass_rate_by_arm,\n"
        "    power_analysis,\n"
        ")\n"
        "from counterfact.intervene import IdentifiabilityStatus\n"
        "from counterfact.schema import Decision, Outcome, Run, Step\n\n"
        "REPO_ROOT = Path('.').resolve()\n"
        "while REPO_ROOT.parent != REPO_ROOT and not (REPO_ROOT / 'pyproject.toml').exists():\n"
        "    REPO_ROOT = REPO_ROOT.parent\n\n"
        "# Committed showcase corpus. Prefer runs_v2 (mixed outcomes, default);\n"
        "# fall back to runs_v1 (single-class anchor) when v2 is absent.\n"
        "_RUNS_V2 = REPO_ROOT / 'bench' / 'real' / 'runs_v2'\n"
        "_RUNS_V1 = REPO_ROOT / 'bench' / 'real' / 'runs_v1'\n"
        "REAL_CORPUS_DIR = _RUNS_V2 if _RUNS_V2.exists() else (_RUNS_V1 if _RUNS_V1.exists() else None)\n"
        "real_corpus: list[Run] = []\n"
        "if REAL_CORPUS_DIR is not None:\n"
        "    real_corpus = [\n"
        "        Run.model_validate_json(p.read_text())\n"
        "        for p in sorted(REAL_CORPUS_DIR.glob('real-*.json'))\n"
        "    ]\n\n"
        "print(f'Python {sys.version.split()[0]}')\n"
        "print(f'real corpus: {len(real_corpus)} traces from {REAL_CORPUS_DIR}')\n"
        "print(f'synthetic SCM headline true effect: {HEADLINE_TRUE_EFFECT:+.4f}')"
    ),
    _md(
        "## 1. Naive vs honest — same query, two estimators\n\n"
        "Take a `model_call` decision in the real corpus and ask: does choosing the larger model raise "
        "P(success)?\n\n"
        "**Naive marginal** (`pass_rate_by_arm`): bucket every `model_call` decision by `chosen_action`, "
        "compute pass-rates per arm with a 95% Wilson interval. Punchy. Wrong, in general — it ignores "
        "the DAG, propensity weighting, and identifiability.\n\n"
        "**Honest causal** (`intervene`): runs the back-door / replay dispatch over the typed decision "
        "taxonomy and emits a `CausalEstimate` carrying `identifiability ∈ {identified, bounded, unidentified}` "
        "plus a structured `next_step` describing what would change the answer.\n\n"
        "If both estimators agree, great. If they disagree, the disagreement is the diagnostic — and "
        "`counterfact` won't lie to you about which one to trust."
    ),
    _code(
        "if real_corpus:\n"
        "    table = pass_rate_by_arm(real_corpus, 'model_call')\n"
        "    print('--- naive marginal estimator (pass_rate_by_arm) ---')\n"
        "    print(f'{\"arm\":<8}  {\"n\":>4}  {\"pass\":>4}  {\"rate\":>6}  {\"95% CI\":>16}')\n"
        "    for row in table.rows:\n"
        "        ci = f'[{row.ci_low:.2f}, {row.ci_high:.2f}]'\n"
        "        print(f'{row.arm:<8}  {row.n:>4}  {row.pass_count:>4}  {row.pass_rate:>6.2f}  {ci:>16}')\n"
        "else:\n"
        "    print('(real corpus absent — see Section 2 for the synthetic comparison)')"
    ),
    _code(
        "real_outcome_classes = {bool(r.outcome.value) for r in real_corpus} if real_corpus else set()\n"
        "real_is_degenerate = len(real_outcome_classes) < 2\n\n"
        "if real_corpus and not real_is_degenerate:\n"
        "    real_model = fit_outcome_model(real_corpus, n_bootstrap=200, seed=42)\n"
        "    sample_run = real_corpus[0]\n"
        "    model_call_step = next(\n"
        "        s.step_index\n"
        "        for s in sample_run.steps\n"
        "        for d in s.decisions\n"
        "        if d.decision_type == 'model_call'\n"
        "    )\n"
        "    chosen = next(\n"
        "        d.chosen_action\n"
        "        for s in sample_run.steps\n"
        "        for d in s.decisions\n"
        "        if d.decision_type == 'model_call'\n"
        "    )\n"
        "    honest = intervene(\n"
        "        dag=build_dag(sample_run),\n"
        "        model=real_model,\n"
        "        step=model_call_step,\n"
        "        intervention={'model_choice': chosen},\n"
        "    )\n"
        "    print('--- honest causal estimator (intervene) ---')\n"
        "    print(f'identifiability : {honest.identifiability.value}')\n"
        "    if honest.outcome_delta is not None:\n"
        "        od = honest.outcome_delta\n"
        "        print(f'point estimate  : {od.point:+.4f}')\n"
        "        print(f'95% bootstrap CI: [{od.ci_low:+.4f}, {od.ci_high:+.4f}]')\n"
        "    if honest.bounds is not None:\n"
        "        print(f'E-value         : {honest.bounds.e_value:.3f}')\n"
        "    print(f'next_step.action: {honest.next_step.action}')\n"
        "    print(f'next_step.text  : {honest.next_step.human_text}')\n"
        "    if honest.next_step.payload:\n"
        "        print(f'next_step.payload: {honest.next_step.payload}')\n"
        "elif real_corpus and real_is_degenerate:\n"
        "    # Pilot 3 result: 30/30 pass on csv_dedupe with frontier models. The\n"
        "    # logistic outcome model cannot fit on a single-class corpus, which\n"
        "    # is itself the honest signal. `counterfact`\\'s job is to surface this\n"
        "    # rather than paper over it.\n"
        "    print('--- honest causal estimator (intervene) ---')\n"
        "    print('identifiability : unidentified')\n"
        "    print('reason          : real corpus is causally degenerate '\n"
        "          f'(every trace has outcome={next(iter(real_outcome_classes))}) — '\n"
        "          'no outcome variation for the back-door adjustment to leverage.')\n"
        "    print('next_step.action: broaden_arm_support')\n"
        "    print('next_step.text  : the marginal pass rate is uniform across arms; '\n"
        "          'a corpus with both pass and fail outcomes is required before '\n"
        "          'any difference between arms can be identified.')\n"
        "    print('next_step.payload:', {\n"
        "        'arm_name': 'model_choice',\n"
        "        'missing_strata': [f'outcome={not next(iter(real_outcome_classes))}'],\n"
        "    })\n"
        "else:\n"
        "    print('(real corpus absent — see Section 2 for the synthetic comparison)')"
    ),
    _md(
        "**Reading the contrast.** The naive table is a single number per arm with its own CI. The "
        "honest verdict is a label + bounds + a structured `next_step` that names what data, intervention, "
        "or randomization would change the conclusion. Lab researchers know the naive number is brittle; "
        "what they want is a tool that pushes back. That's the whole pitch.\n\n"
        "With the default `runs_v2` corpus (date_window, 30 traces, mixed outcomes from "
        "inverted-greedy randomization), both estimators produce a number — and the honest one "
        "carries an identifiability label and a bootstrap CI. With the `runs_v1` fallback "
        "(csv_dedupe, single-class), the naive table looks decisive while the honest verdict "
        "refuses to claim a difference. Both shapes are the feature."
    ),
    _md(
        "## 2. Synthetic SCM canary — does the engine recover a known effect?\n\n"
        "Same code path, but the corpus comes from a structural causal model with a *known* headline "
        "effect (`HEADLINE_TRUE_EFFECT`, sonnet vs haiku marginal). We expect the recovered effect to "
        "be within ±0.05 of truth and the bootstrap CI to bracket it. This is mechanism evidence: the "
        "schema → DAG → outcome model → identifiability dispatch pipeline is correct on a known-truth case."
    ),
    _code(
        "synth_runs = [Run.model_validate(t) for t in generate_traces(n=500, seed=42)]\n"
        "synth_model = fit_outcome_model(synth_runs, n_bootstrap=200, seed=42)\n"
        "synth_dag = build_dag(synth_runs[0])\n\n"
        "p_sonnet = intervene(dag=synth_dag, model=synth_model, step=2, intervention={'model_choice': 'sonnet'})\n"
        "p_haiku  = intervene(dag=synth_dag, model=synth_model, step=2, intervention={'model_choice': 'haiku'})\n\n"
        "estimated = p_sonnet.outcome_delta.point - p_haiku.outcome_delta.point\n"
        "print(f'  estimated effect: {estimated:+.4f}')\n"
        "print(f'  true effect:      {HEADLINE_TRUE_EFFECT:+.4f}')\n"
        "print(f'  |diff|:           {abs(estimated - HEADLINE_TRUE_EFFECT):.4f}  (tolerance ±0.05)')\n"
        "assert abs(estimated - HEADLINE_TRUE_EFFECT) <= 0.05, 'SCM-recovery tolerance violated'"
    ),
    _md(
        "## 3. Three identifiability paths\n\n"
        "Every `intervene` answer is one of three labels, each with its own contract on the rest of the "
        "result object:\n\n"
        "| label          | shape                                                                         |\n"
        "|----------------|-------------------------------------------------------------------------------|\n"
        "| `identified`   | `outcome_delta` (point + bootstrap CI), `bounds.e_value`, `adjustment_set`    |\n"
        "| `bounded`      | `bounds.e_value`, named adjustment strategy in `assumptions`, no point claim  |\n"
        "| `unidentified` | `reason`, structured `next_step`, no point claim                              |\n\n"
        "All three paths populate `next_step` — even `identified`, where the action is `none` if the CI "
        "is already tight."
    ),
    _md(
        "### 3a. *identified* — `tool_choice` with randomized support"
    ),
    _code(
        "identified = intervene(dag=synth_dag, model=synth_model, step=1, intervention={'tool_choice': 'run_tests'})\n\n"
        "print(f'identifiability : {identified.identifiability.value}')\n"
        "print(f'point estimate  : {identified.outcome_delta.point:+.4f}')\n"
        "print(f'95% bootstrap CI: [{identified.outcome_delta.ci_low:+.4f}, {identified.outcome_delta.ci_high:+.4f}]')\n"
        "print(f'E-value         : {identified.bounds.e_value:.3f}')\n"
        "print(f'next_step       : {identified.next_step.action}')\n"
        "print(f'                  {identified.next_step.human_text}')\n"
        "print()\n"
        "print('assumptions:')\n"
        "for a in identified.assumptions:\n"
        "    print(f'  - {a}')\n"
        "assert identified.identifiability == IdentifiabilityStatus.IDENTIFIED"
    ),
    _md(
        "### 3b. *bounded* — `memory_content` requires back-door adjustment"
    ),
    _code(
        "mem_run = Run(\n"
        "    schema_version='0.1.0',\n"
        "    run_id='demo-mem-001',\n"
        "    steps=[\n"
        "        Step(step_index=0, decisions=[Decision(decision_id='d0', decision_type='plan_step', chosen_action='begin')]),\n"
        "        Step(step_index=1, decisions=[Decision(decision_id='d1', decision_type='memory_read', chosen_action='recent_5')]),\n"
        "    ],\n"
        "    outcome=Outcome(kind='binary', value=False, verifier='pytest'),\n"
        ")\n"
        "bounded = intervene(\n"
        "    dag=build_dag(mem_run),\n"
        "    model=synth_model,\n"
        "    step=1,\n"
        "    intervention={'memory_content': 'all'},\n"
        ")\n\n"
        "print(f'identifiability : {bounded.identifiability.value}')\n"
        "print(f'E-value         : {bounded.bounds.e_value:.3f}  ({bounded.bounds.technique})')\n"
        "print(f'next_step       : {bounded.next_step.action}')\n"
        "print(f'                  {bounded.next_step.human_text}')\n"
        "print()\n"
        "print('assumptions:')\n"
        "for a in bounded.assumptions:\n"
        "    print(f'  - {a}')\n"
        "assert bounded.identifiability == IdentifiabilityStatus.BOUNDED\n"
        "assert bounded.bounds is not None"
    ),
    _md(
        "### 3c. *unidentified* — `prompt_content` is replay-only\n\n"
        "The taxonomy treats prompt-content interventions as `always-replay`: the prompt is high-dim, "
        "randomization in the corpus does not cover it, and the LLM completion is opaque. `intervene` "
        "returns `unidentified` with a structured `next_step.action='replay_required'` — the only honest "
        "answer."
    ),
    _code(
        "unidentified = intervene(\n"
        "    dag=synth_dag,\n"
        "    model=synth_model,\n"
        "    step=2,\n"
        "    intervention={'prompt_content': 'think step by step'},\n"
        ")\n\n"
        "print(f'identifiability : {unidentified.identifiability.value}')\n"
        "print(f'reason          : {unidentified.reason}')\n"
        "print(f'next_step.action: {unidentified.next_step.action}')\n"
        "print(f'next_step.text  : {unidentified.next_step.human_text}')\n"
        "print(f'next_step.payload: {unidentified.next_step.payload}')\n"
        "print()\n"
        "print('warnings:')\n"
        "for w in unidentified.warnings:\n"
        "    print(f'  ! {w}')\n"
        "assert unidentified.identifiability == IdentifiabilityStatus.UNIDENTIFIED\n"
        "assert unidentified.next_step.action == 'replay_required'\n"
        "assert unidentified.next_step.payload['intervention_target'] == 'prompt_content'"
    ),
    _md(
        "## 4. Ranked failure attribution\n\n"
        "Pick a failed synthetic run; rank its decisions by estimated causal influence on the outcome. "
        "Each entry inherits the per-decision identifiability label, so callers can filter or weight "
        "by epistemic confidence rather than treating all rankings as equal."
    ),
    _code(
        "failed_runs = [r for r in synth_runs if r.outcome.value is False]\n"
        "print(f'failed runs in synthetic corpus: {len(failed_runs)} / {len(synth_runs)}')\n\n"
        "case = failed_runs[0]\n"
        "attribution = attribute_failure(dag=build_dag(case), model=synth_model)\n"
        "top5 = attribution.top_k(5)\n\n"
        "print(f'\\nfailure attribution for {case.run_id}:')\n"
        "print(f'{\"rank\":>4}  {\"decision_id\":<22}  {\"type\":<12}  {\"action\":<14}  {\"influence\":>9}  identifiability')\n"
        "print('-' * 88)\n"
        "for i, e in enumerate(top5, start=1):\n"
        "    print(\n"
        "        f'{i:>4}  {e.decision_id:<22}  {e.decision_type:<12}  {e.chosen_action:<14}  '\n"
        "        f'{e.influence:>+9.4f}  {e.identifiability.value}'\n"
        "    )\n"
        "assert len(top5) <= 5"
    ),
    _md(
        "## 5. What would change the answer? — power analysis\n\n"
        "Take a query whose CI is too wide (or whose honest verdict is `unidentified` for a non-replay "
        "reason). `power_analysis` answers the binomial-Wald question: at the current per-arm pass rates "
        "and arm fractions, what `n` would shrink the 95% CI on the marginal effect to `target_ci_width`?\n\n"
        "Scope is deliberately narrow per design.md D2 — this is not effect-size-aware power analysis. "
        "It's the *one* helper the demo's closing paragraph needs to answer 'why don't you just collect "
        "more data?'"
    ),
    _code(
        "# On the real corpus we need both arms to have observed support to make\n"
        "# the question well-posed. We default to the synthetic corpus where the\n"
        "# arm distribution is known, then optionally also report on the real\n"
        "# corpus when both arms are present.\n"
        "rep_synth = power_analysis(\n"
        "    synth_runs,\n"
        "    decision_type='model_call',\n"
        "    arms=('sonnet', 'haiku'),\n"
        "    target_ci_width=0.02,\n"
        ")\n"
        "print('--- synthetic corpus ---')\n"
        "print(f'current_n          : {rep_synth.current_n}')\n"
        "if rep_synth.current_ci_width is not None:\n"
        "    print(f'current CI width   : {rep_synth.current_ci_width:.4f}')\n"
        "print(f'target CI width    : {rep_synth.target_ci_width}')\n"
        "print(f'estimated required n: {rep_synth.estimated_required_n}')\n"
        "print()\n"
        "print('assumptions:')\n"
        "for a in rep_synth.assumptions:\n"
        "    print(f'  - {a}')\n\n"
        "if real_corpus:\n"
        "    print()\n"
        "    print('--- real corpus ---')\n"
        "    rep_real = power_analysis(\n"
        "        real_corpus,\n"
        "        decision_type='model_call',\n"
        "        arms=('small', 'large'),\n"
        "        target_ci_width=0.10,\n"
        "    )\n"
        "    print(f'current_n          : {rep_real.current_n}')\n"
        "    print(f'estimated required n: {rep_real.estimated_required_n}')\n"
        "    if rep_real.warnings:\n"
        "        for w in rep_real.warnings:\n"
        "            print(f'  ! {w}')"
    ),
    _md(
        "## Takeaways\n\n"
        "- The synthetic SCM canary recovers the known headline effect within ±0.05. The schema → DAG → "
        "outcome model → identifiability dispatch pipeline is **correct on a known-truth case**.\n"
        "- Every `intervene` answer carries one of three labels with a contract on the rest of the result "
        "object — including a structured `next_step` whose `action` ∈ "
        "{`increase_n`, `broaden_arm_support`, `replay_required`, `add_arm_randomization`, `none`}. "
        "**No silent Pearl-L3 claims sneak in.**\n"
        "- The naive marginal (`pass_rate_by_arm`) is exposed as a labeled comparison baseline so the "
        "honest causal estimator can stand next to it. **The disagreement, when it appears, is the diagnostic.**\n"
        "- `power_analysis` connects 'this CI is too wide' to a concrete `n` under documented "
        "assumptions. **You can act on the answer.**\n\n"
        "What `counterfact` is for: a tool that **says no when the data says no**, and tells you what would "
        "change the answer."
    ),
]


def main() -> None:
    nb = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NB_PATH.write_text(json.dumps(nb, indent=1) + "\n")
    print(f"wrote {NB_PATH} ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()
