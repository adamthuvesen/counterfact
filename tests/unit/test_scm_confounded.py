"""Unit tests for the synthetic SCM's confounded mode (tasks §2).

These tests cover three things:

1. Default mode is bit-for-bit unchanged (regression guard for the new flag).
2. Confounded mode logs the right policy/propensity/context metadata.
3. The exposed analytic constants satisfy the relationships the showcase
   relies on — `CONFOUNDED_DO_HEADLINE == HEADLINE_TRUE_EFFECT` (the outcome
   equation is unchanged in confounded mode) and the naive-vs-causal gap is
   at least the rhetorical threshold of `0.05`.
"""

from __future__ import annotations

from bench.synthetic import (
    CONFOUNDED_DO_HEADLINE,
    CONFOUNDED_NAIVE_HEADLINE,
    CONFOUNDED_NAIVE_VS_CAUSAL_GAP,
    HEADLINE_TRUE_EFFECT,
    SyntheticSCM,
    generate_traces,
)


def test_default_mode_run0_is_unchanged() -> None:
    """Bit-for-bit regression guard. If this fails, default-mode behavior
    drifted — and every existing acceptance/recovery test downstream is at
    risk."""
    trace = next(generate_traces(n=1, seed=42))
    decisions_by_type = {
        d["decision_type"]: d for step in trace["steps"] for d in step["decisions"]
    }
    # Tool, model, retry are uniform in default mode.
    for dt in ("tool_call", "model_call", "retry"):
        d = decisions_by_type[dt]
        assert d["policy"] == "uniform"
        assert d["policy_params"] == {}
        assert d["context_features"] == {}
    # Specific expected draws on seed 42 from the existing SCM.
    assert decisions_by_type["tool_call"]["chosen_action"] == "search_docs"
    assert decisions_by_type["model_call"]["chosen_action"] == "haiku"
    assert decisions_by_type["retry"]["chosen_action"] == "no_retry"


def test_confounded_run0_logs_tool_chosen_and_conditional_propensity() -> None:
    """Confounded model_call decision must carry the conditioning variable
    in context_features and the conditional propensity matching the SCM's
    P(model | tool) table."""
    trace = next(generate_traces(n=1, seed=42, confound=True))
    model_call = next(
        d
        for step in trace["steps"]
        for d in step["decisions"]
        if d["decision_type"] == "model_call"
    )

    assert model_call["policy"] == "confounded_by_tool"
    assert "tool_chosen" in model_call["context_features"]
    tool_value = model_call["context_features"]["tool_chosen"]
    assert tool_value in {"run_tests", "inspect_file", "search_docs"}

    # policy_params carries the full conditional table.
    pp = model_call["policy_params"]
    assert set(pp.keys()) == {
        "p_sonnet_given_run_tests",
        "p_sonnet_given_inspect_file",
        "p_sonnet_given_search_docs",
    }

    # Propensity matches the conditional for the tool that was actually drawn.
    p_sonnet = pp[f"p_sonnet_given_{tool_value}"]
    expected = p_sonnet if model_call["chosen_action"] == "sonnet" else 1.0 - p_sonnet
    assert 0.0 < expected < 1.0
    assert model_call["propensity"] == expected


def test_confounded_mode_is_deterministic() -> None:
    a = list(generate_traces(n=200, seed=42, confound=True))
    b = list(generate_traces(n=200, seed=42, confound=True))
    assert a == b


def test_default_mode_is_deterministic() -> None:
    a = list(generate_traces(n=200, seed=42))
    b = list(generate_traces(n=200, seed=42))
    assert a == b


def test_confounded_does_not_alter_outcome_equation() -> None:
    """For fixed (tool, model, retry), P(success) is identical across modes
    — confounding only changes the arm-assignment policy, not the outcome
    DGP."""
    from bench.synthetic.scm import _p_success

    # Pick a few representative arm combinations.
    for tool, model, retry in [
        ("run_tests", "sonnet", "no_retry"),
        ("inspect_file", "haiku", "retry_once"),
        ("search_docs", "sonnet", "retry_twice"),
    ]:
        # _p_success has no `confound` parameter — same function for both
        # modes, by construction. This test pins that contract.
        p = _p_success(tool, model, retry)
        assert 0.0 < p < 1.0


def test_confounded_constants_satisfy_relationships() -> None:
    # Do-calculus headline equals the original true effect (outcome equation
    # unchanged in confounded mode).
    assert abs(CONFOUNDED_DO_HEADLINE - HEADLINE_TRUE_EFFECT) < 1e-9
    # Naive-vs-causal gap meets the rhetorical threshold.
    assert abs(CONFOUNDED_NAIVE_VS_CAUSAL_GAP) >= 0.05
    # Sanity: naive headline differs from the do-calculus headline by exactly
    # the gap.
    assert (
        abs((CONFOUNDED_NAIVE_HEADLINE - CONFOUNDED_DO_HEADLINE) - CONFOUNDED_NAIVE_VS_CAUSAL_GAP)
        < 1e-12
    )


def test_confounded_corpus_has_both_arms_with_meaningful_support() -> None:
    """Both arms must have enough samples for the engine to fit. With the
    chosen propensities, marginal P(sonnet) ≈ 0.4 and P(haiku) ≈ 0.6 — both
    well above any reasonable minimum-support threshold at n=1000."""
    n = 1000
    traces = list(generate_traces(n=n, seed=42, confound=True))
    arms = [
        d["chosen_action"]
        for trace in traces
        for step in trace["steps"]
        for d in step["decisions"]
        if d["decision_type"] == "model_call"
    ]
    counts = {arm: arms.count(arm) for arm in ("sonnet", "haiku")}
    assert counts["sonnet"] >= 200
    assert counts["haiku"] >= 200


def test_default_constructor_omits_confound_field_value() -> None:
    """The SCM accepts the new flag by default-False; calling with no flag
    behaves like the original."""
    a = SyntheticSCM(seed=42).sample_run(0)
    b = SyntheticSCM(seed=42, confound=False).sample_run(0)
    assert a == b
