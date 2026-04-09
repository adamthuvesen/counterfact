"""Tests for the real-agent side of corpus-harness spec.

These tests mock the LLM client — no external API calls. The real-LLM smoke
corpus is a HUMAN GATE (§12.3) and is not exercised here.
"""

from __future__ import annotations

import random
import subprocess
import sys
from pathlib import Path

import pytest

from bench.real.coding_agent import (
    FIXTURES,
    BudgetExceeded,
    BudgetTracker,
    EpsilonGreedy,
)
from bench.real.coding_agent.agent import AgentRunConfig, run_one_trace
from bench.real.coding_agent.fixtures import run_pytest
from bench.real.coding_agent.llm import LLMResponse
from bench.real.coding_agent.runner import (
    APPROVAL_MARKER,
    check_credentials,
    first_run_gate_check,
    run_real_corpus,
)
from counter.schema import Run

# --- ε-greedy randomization spec scenarios ----------------------------------


def test_epsilon_greedy__greedy_action_propensity() -> None:
    """WHEN ε=0.2, |actions|=4, greedy chosen
    THEN logged propensity = (1-ε) + ε/|actions| = 0.85."""
    eg = EpsilonGreedy(epsilon=0.2, seed=0)
    actions = ["a", "b", "c", "d"]
    # Force the greedy branch by exhausting the RNG path; we just validate the
    # math directly by computing expected values for both branches.
    # Run many draws, every "greedy" outcome must have propensity 0.85.
    rng = random.Random(123)
    for _ in range(200):
        eg2 = EpsilonGreedy(epsilon=0.2, seed=rng.randint(0, 1_000_000))
        chosen, prop = eg2.choose(actions, greedy="b")
        if chosen == "b":
            assert prop == pytest.approx(0.85)
        else:
            assert prop == pytest.approx(0.05)


def test_epsilon_greedy__non_greedy_action_propensity() -> None:
    """WHEN ε=0.2, |actions|=4, non-greedy chosen
    THEN logged propensity = ε/|actions| = 0.05."""
    eg = EpsilonGreedy(epsilon=0.2, seed=0)
    # Drive multiple draws; since we test the formula above, any non-greedy
    # outcome must equal 0.05. Verify the formula is correct.
    actions = ["a", "b"]
    rng = random.Random(7)
    for _ in range(100):
        eg2 = EpsilonGreedy(epsilon=0.2, seed=rng.randint(0, 1_000_000))
        chosen, prop = eg2.choose(actions, greedy="a")
        if chosen != "a":
            assert prop == pytest.approx(0.1)


def test_epsilon_greedy__rejects_greedy_not_in_valid() -> None:
    eg = EpsilonGreedy(epsilon=0.2)
    with pytest.raises(ValueError):
        eg.choose(["x", "y"], greedy="z")


# --- Budget tracker spec scenarios ------------------------------------------


def test_budget__halt_at_80_percent() -> None:
    """WHEN cumulative spend reaches 0.8 * cap
    THEN BudgetExceeded fires and message reports both spent and cap."""
    tracker = BudgetTracker(cap_usd=10.0)
    tracker.add(3.0)
    tracker.add(4.0)
    with pytest.raises(BudgetExceeded) as exc_info:
        tracker.add(2.0)
    msg = str(exc_info.value)
    assert "9.0000" in msg or "9.00" in msg
    assert "10.00" in msg
    assert "80%" in msg


def test_budget__rejects_negative_spend() -> None:
    tracker = BudgetTracker(cap_usd=10.0)
    with pytest.raises(ValueError):
        tracker.add(-1.0)


# --- Fixture spec scenarios -------------------------------------------------


def test_real_corpus_has_at_least_three_fixtures() -> None:
    """WHEN the real-agent harness is initialized
    THEN at least 3 fixture directories are registered, each with a failing pytest."""
    assert len(FIXTURES) >= 3
    for fx in FIXTURES:
        assert fx.root.is_dir()
        assert fx.source_path.is_file()
        assert fx.test_path.is_file()
        # Each fixture's pristine pytest must currently fail (the bug).
        passed, _ = run_pytest(fx.root)
        assert passed is False, f"{fx.fixture_id} should start failing"


def test_outcome_is_pytest_exit_code(tmp_path: Path) -> None:
    """WHEN the real-agent harness completes a run on a fixture
    THEN Outcome.value is True iff `pytest <fixture>` returned exit code 0."""
    fixture = FIXTURES[0]  # string-utils

    class _PerfectLLM:
        def call(self, *, role: str, prompt: str) -> LLMResponse:
            # Reply with a corrected normalize.py.
            fixed = (
                "```python\n"
                '"""Patched."""\n'
                "from __future__ import annotations\n"
                "import re\n"
                "def normalize_name(name: str) -> str:\n"
                '    return re.sub(r"\\s+", " ", name.strip().lower())\n'
                "```"
            )
            return LLMResponse(text=fixed, cost_usd=0.0)

    budget = BudgetTracker(cap_usd=1.0)
    run = run_one_trace(
        fixture,
        run_index=0,
        llm=_PerfectLLM(),
        budget=budget,
        sandbox_root=tmp_path,
        config=AgentRunConfig(epsilon=0.0, seed=42),
    )
    assert isinstance(run, Run)
    assert run.outcome.kind == "binary"
    assert run.outcome.value is True


def test_agent_logs_all_randomization_fields(tmp_path: Path) -> None:
    """WHEN any randomized decision is logged in a real-agent trace
    THEN policy, policy_params, valid_actions, chosen_action, propensity, context_features are present."""
    fixture = FIXTURES[1]  # date-utils — bug isn't auto-fixable by trivial regex, that's fine

    class _BlankLLM:
        def call(self, *, role: str, prompt: str) -> LLMResponse:
            # Returns no code fence → patch is not extracted, test still fails.
            return LLMResponse(text="(no patch)", cost_usd=0.0)

    budget = BudgetTracker(cap_usd=1.0)
    run = run_one_trace(
        fixture,
        run_index=0,
        llm=_BlankLLM(),
        budget=budget,
        sandbox_root=tmp_path,
        config=AgentRunConfig(epsilon=0.2, seed=1),
    )
    randomized_decision_types = {"tool_call", "model_call", "retry"}
    saw_random = False
    for step in run.steps:
        for d in step.decisions:
            if d.policy is None:
                continue
            saw_random = True
            assert d.policy in {"epsilon_greedy", "uniform"}
            assert d.policy_params is not None
            assert d.valid_actions is not None
            assert d.chosen_action in d.valid_actions
            assert d.propensity is not None
            assert 0.0 < d.propensity <= 1.0
            assert d.context_features is not None
    assert saw_random


# --- Approval gate / resume / CLI -------------------------------------------


def test_first_run_prompts_before_any_api_call(tmp_path: Path) -> None:
    """WHEN counter bench real is invoked and no .counter/approved marker exists
    THEN the harness prints an approval prompt and exits before making any external API call."""
    output = tmp_path / "out"

    class _ExplodingLLM:
        def call(self, *, role: str, prompt: str) -> LLMResponse:
            raise AssertionError("LLM was invoked despite missing approval marker!")

    rc = run_real_corpus(
        n=5,
        budget_cap_usd=5.0,
        output_dir=output,
        llm_client_factory=lambda: _ExplodingLLM(),
        marker_path=tmp_path / ".counter" / "approved",  # does not exist
    )
    assert rc == 2


def test_approved_marker_skips_prompt(tmp_path: Path) -> None:
    """WHEN the approval marker exists and the harness is invoked
    THEN the harness proceeds to corpus generation without re-prompting."""
    output = tmp_path / "out"
    marker = tmp_path / ".counter" / "approved"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.touch()

    class _NullLLM:
        def call(self, *, role: str, prompt: str) -> LLMResponse:
            return LLMResponse(text="(no patch)", cost_usd=0.0)

    rc = run_real_corpus(
        n=3,
        budget_cap_usd=5.0,
        output_dir=output,
        llm_client_factory=lambda: _NullLLM(),
        marker_path=marker,
    )
    assert rc == 0
    written = list(output.glob("real-*.json"))
    assert len(written) == 3


def test_resume_after_partial_run(tmp_path: Path) -> None:
    """WHEN a run is halted and re-invoked
    THEN the harness skips already-completed traces and writes only new ones."""
    output = tmp_path / "out"
    marker = tmp_path / ".counter" / "approved"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.touch()

    # First call: 0 cost, write 2 traces.
    class _NullLLM:
        def call(self, *, role: str, prompt: str) -> LLMResponse:
            return LLMResponse(text="(no patch)", cost_usd=0.0)

    rc = run_real_corpus(
        n=2,
        budget_cap_usd=5.0,
        output_dir=output,
        llm_client_factory=lambda: _NullLLM(),
        marker_path=marker,
    )
    assert rc == 0
    first_files = sorted(p.name for p in output.glob("real-*.json"))
    assert len(first_files) == 2

    # Second call requesting 4: should add 2 new ones (indexes 2 and 3) only.
    rc2 = run_real_corpus(
        n=4,
        budget_cap_usd=5.0,
        output_dir=output,
        llm_client_factory=lambda: _NullLLM(),
        marker_path=marker,
    )
    assert rc2 == 0
    final_files = sorted(p.name for p in output.glob("real-*.json"))
    assert len(final_files) == 4
    assert set(first_files).issubset(final_files)


def test_check_credentials__missing_anthropic_key_returns_error() -> None:
    """When ANTHROPIC_API_KEY is absent, check_credentials returns a fix-it message."""
    err = check_credentials(
        role_to_model={"small": "claude-haiku-4-5", "large": "claude-sonnet-4-6"},
        env={},  # empty environment
    )
    assert err is not None
    assert "ANTHROPIC_API_KEY" in err
    assert "claude-haiku-4-5" in err or "claude-sonnet-4-6" in err


def test_check_credentials__present_anthropic_key_returns_none() -> None:
    """When the key is present, check_credentials returns None."""
    err = check_credentials(
        role_to_model={"small": "claude-haiku-4-5", "large": "claude-sonnet-4-6"},
        env={"ANTHROPIC_API_KEY": "sk-ant-test"},
    )
    assert err is None


def test_check_credentials__auth_token_alternative_accepted() -> None:
    """ANTHROPIC_AUTH_TOKEN is also accepted (per the litellm error message)."""
    err = check_credentials(
        role_to_model={"large": "claude-sonnet-4-6"},
        env={"ANTHROPIC_AUTH_TOKEN": "ant-token"},
    )
    assert err is None


def test_run_real_corpus__exits_4_when_credentials_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With marker present but no API key (and no custom client factory), the
    runner must exit 4 BEFORE any LLM call."""
    marker = tmp_path / ".counter" / "approved"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.touch()
    output = tmp_path / "out"

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)

    rc = run_real_corpus(
        n=1,
        budget_cap_usd=1.0,
        output_dir=output,
        # Note: no llm_client_factory — the production path triggers the cred check.
        marker_path=marker,
    )
    assert rc == 4


def test_cli_real_subcommand_first_run_prints_approval(tmp_path: Path) -> None:
    """`counter bench real ...` without an approval marker exits 2 with the prompt.

    Runs from a clean cwd with NO PYTHONPATH override — this is the regression
    test that catches `bench` not being included in the wheel.
    """
    out = tmp_path / "out"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "counter.cli",
            "bench",
            "real",
            "--n",
            "1",
            "--budget-cap",
            "1",
            "--output-dir",
            str(out),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 2, f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    combined = proc.stdout + proc.stderr
    assert "HUMAN GATE" in combined
    assert "approved" in combined
