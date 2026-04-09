"""§15 final-acceptance ship gate.

Bundles every automatable acceptance criterion from `proposal.md` and `tasks.md`
§15 into one runnable test module. Tests that depend on the canonical real-agent
corpus skip gracefully when it is absent (the gate is open but unanswered);
tests that don't (synthetic determinism, forbidden-deps/imports) always run.

The shape: each test pins one ship-gate criterion. A red here means v0 is not
ready; a green here is necessary-but-not-sufficient (the demo notebook eyeball
in §15.11 is still a HUMAN GATE).

Headline intervention is `retry_policy` per design.md D9. Update
`HEADLINE_INTERVENTION_KIND` if a future amendment shifts it.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from bench.synthetic import generate_traces
from counter import fit_outcome_model
from counter.schema import Run

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_V1_DIR = REPO_ROOT / "bench" / "real" / "runs_v1"
PYPROJECT = REPO_ROOT / "pyproject.toml"
SRC_DIR = REPO_ROOT / "src"

# --- ship-gate constants (proposal.md acceptance criteria) -------------------

MIN_REAL_TRACES = 200
CLASS_BALANCE_FLOOR = 0.20  # neither pass nor fail rate below this
HEADLINE_DECISION_TYPE = "retry"
HEADLINE_INTERVENTION_KIND = "retry_policy"
HEADLINE_ARMS = ("no_retry", "retry_once")
MIN_ARM_SUPPORT = 30
CI_WIDTH_TOLERANCE = 0.25
SCM_RECOVERY_TOLERANCE = 0.05  # mirrored from design.md D10

# Forbidden surface area per design.md D13 + the proposal's explicit non-goals.
# Listed here in plain text so the scan stays unambiguous.
FORBIDDEN_DEPS = ("dowhy", "causalml", "pyro", "langchain", "langgraph", "pandas", "networkx")
FORBIDDEN_IMPORTS = ("dowhy", "causalml", "pyro", "langchain", "langgraph")


# --- fixtures ---------------------------------------------------------------


def _load_runs_v1() -> list[Run]:
    if not RUNS_V1_DIR.exists():
        return []
    return [
        Run.model_validate_json(p.read_text())
        for p in sorted(RUNS_V1_DIR.glob("*.json"))
    ]


@pytest.fixture(scope="module")
def real_corpus() -> list[Run]:
    runs = _load_runs_v1()
    if not runs:
        pytest.skip(
            f"runs_v1 corpus absent at {RUNS_V1_DIR}. Generate via "
            f"`counter bench real --n 200 --output-dir bench/real/runs_v1` "
            f"before §15 runs."
        )
    return runs


@pytest.fixture(scope="module")
def fitted_real_model(real_corpus: list[Run]) -> object:
    return fit_outcome_model(real_corpus, n_bootstrap=200, seed=42)


# --- §15.1: synthetic corpus is reproducible at ≥500 traces -----------------


def test_synthetic_corpus_is_deterministically_500_traces() -> None:
    """§15.1: synthetic generator produces 500 traces deterministically per seed."""
    a = list(generate_traces(n=500, seed=42))
    b = list(generate_traces(n=500, seed=42))
    assert len(a) == 500
    # Byte-stability under same seed (subset check — full byte equality is
    # exercised in tests/unit/test_corpus_synthetic.py)
    assert a[0] == b[0]
    assert a[-1] == b[-1]


# --- §15.2: real-corpus size + class balance --------------------------------


def test_real_corpus_meets_minimum_size(real_corpus: list[Run]) -> None:
    """§15.2: real corpus has ≥200 traces."""
    assert len(real_corpus) >= MIN_REAL_TRACES, (
        f"real corpus has {len(real_corpus)} traces; need ≥{MIN_REAL_TRACES}"
    )


def test_real_corpus_class_balance(real_corpus: list[Run]) -> None:
    """§15.2: pass/fail rates each between 20% and 80%."""
    n = len(real_corpus)
    n_pass = sum(1 for r in real_corpus if r.outcome.value)
    pass_rate = n_pass / n
    assert CLASS_BALANCE_FLOOR <= pass_rate <= 1 - CLASS_BALANCE_FLOOR, (
        f"class balance failed: pass_rate={pass_rate:.2%}; "
        f"need [{CLASS_BALANCE_FLOOR:.0%}, {1 - CLASS_BALANCE_FLOOR:.0%}]"
    )


# --- §15.2: arm support for headline intervention ---------------------------


def test_headline_intervention_arms_have_minimum_support(real_corpus: list[Run]) -> None:
    """§15.2: each arm of the headline intervention has ≥30 logged samples."""
    arm_counts: Counter[str] = Counter()
    for r in real_corpus:
        for s in r.steps:
            for d in s.decisions:
                if d.decision_type == HEADLINE_DECISION_TYPE and d.policy:
                    arm_counts[d.chosen_action] += 1
    missing = []
    for arm in HEADLINE_ARMS:
        if arm_counts[arm] < MIN_ARM_SUPPORT:
            missing.append(f"{arm}: {arm_counts[arm]}")
    assert not missing, (
        f"headline intervention {HEADLINE_INTERVENTION_KIND!r} arms below "
        f"min support ({MIN_ARM_SUPPORT}): " + ", ".join(missing)
    )


# --- §15.3: bootstrap CI width on headline effect ---------------------------


def test_bootstrap_ci_width_on_headline_within_tolerance(
    real_corpus: list[Run], fitted_real_model: object
) -> None:
    """§15.3: 95% bootstrap CI width on the headline marginal effect ≤ 0.25."""
    feat_index = fitted_real_model.feature_index
    target_keys = {arm: f"{HEADLINE_DECISION_TYPE}::{arm}" for arm in HEADLINE_ARMS}
    missing = [k for k in target_keys.values() if k not in feat_index]
    if missing:
        pytest.skip(
            f"headline arms missing from feature index (corpus has no observed "
            f"support for these arms): {missing}"
        )

    target_idx = {arm: feat_index[target_keys[arm]] for arm in HEADLINE_ARMS}
    sibling_keys = [k for k in feat_index if k.startswith(f"{HEADLINE_DECISION_TYPE}::")]
    sibling_idx = [feat_index[k] for k in sibling_keys]
    X = fitted_real_model.train_X

    def _marginal_p(coefs: np.ndarray, intercept: float, arm: str) -> float:
        Xc = X.copy()
        Xc[:, sibling_idx] = 0.0
        Xc[:, target_idx[arm]] = 1.0
        z = Xc @ coefs + intercept
        return float((1.0 / (1.0 + np.exp(-z))).mean())

    n_b = fitted_real_model.bootstrap_coefs.shape[0]
    effects = np.zeros(n_b)
    for b in range(n_b):
        coefs = fitted_real_model.bootstrap_coefs[b]
        intercept = float(fitted_real_model.bootstrap_intercepts[b])
        effects[b] = _marginal_p(coefs, intercept, HEADLINE_ARMS[1]) - _marginal_p(
            coefs, intercept, HEADLINE_ARMS[0]
        )

    ci_lo = float(np.percentile(effects, 2.5))
    ci_hi = float(np.percentile(effects, 97.5))
    width = ci_hi - ci_lo
    assert width <= CI_WIDTH_TOLERANCE, (
        f"95% CI on headline effect ({HEADLINE_INTERVENTION_KIND}): "
        f"[{ci_lo:+.4f}, {ci_hi:+.4f}], width={width:.4f}; need ≤{CI_WIDTH_TOLERANCE}"
    )


# --- §15.9: nothing on the "does not ship" list crept in --------------------


def test_no_forbidden_dependencies_in_pyproject() -> None:
    """§15.9: pyproject.toml does not list any dep that design.md D13 forbids."""
    text = PYPROJECT.read_text()
    found = []
    for dep in FORBIDDEN_DEPS:
        # Match `"<dep>` at the start of a line inside dependency blocks. The
        # leading quote prevents partial matches (e.g., `pandas` would have
        # otherwise matched `pandas-stubs` if such a thing appeared).
        if re.search(rf'(?m)^\s*"{re.escape(dep)}\b', text):
            found.append(dep)
    assert not found, (
        f"forbidden dependencies declared in pyproject.toml: {found}. "
        f"Adding any of these requires amending design.md D13 first."
    )


def test_no_forbidden_imports_in_src() -> None:
    """§15.9: src/ contains no `import dowhy/causalml/pyro/langchain/langgraph`."""
    found: list[str] = []
    for path in SRC_DIR.rglob("*.py"):
        text = path.read_text()
        for mod in FORBIDDEN_IMPORTS:
            if re.search(rf"(?m)^\s*(import|from)\s+{re.escape(mod)}\b", text):
                found.append(f"{path.relative_to(REPO_ROOT)}: imports {mod}")
    assert not found, "forbidden imports found:\n  - " + "\n  - ".join(found)


# --- §15.10/§15.11 are explicit human gates (opsx:verify + demo eyeball) ----
# Documented but not auto-tested. They're discoverable here for completeness:


def test_manual_gates_are_documented() -> None:
    """§15.10/§15.11 are explicit human gates — this test exists so the gate
    appears in pytest output as a reminder, not as a soft pass."""
    gates = [
        "§15.10: run `/opsx:verify build-counter-v0` — surface drift between specs and impl",
        "§15.11: human reads notebooks/demo.ipynb end-to-end and inspects attribution",
    ]
    # Always passes; the assertion message is the artifact.
    assert all(isinstance(g, str) for g in gates), gates
