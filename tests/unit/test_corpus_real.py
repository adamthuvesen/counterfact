"""Tests for the real-agent side of corpus-harness spec.

These tests mock the LLM client — no external API calls. The real-LLM smoke
corpus is a HUMAN GATE (§12.3) and is not exercised here.
"""

from __future__ import annotations

import json
import math
import random
import shutil
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
from bench.real.coding_agent.agent import (
    AgentRunConfig,
    _extract_python_block,
    run_one_trace,
)
from bench.real.coding_agent.fixtures import EASY_FIXTURES, run_pytest
from bench.real.coding_agent.llm import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_NUM_RETRIES,
    DEFAULT_REQUEST_TIMEOUT_S,
    ROLE_TO_MODEL,
    CostUnknownError,
    LiteLLMClient,
    LLMResponse,
    extract_cost,
)
from bench.real.coding_agent.runner import (
    approval_receipt_template,
    check_credentials,
    resolve_fixtures,
    run_real_corpus,
)
from counterfact.schema import Run


def _write_approval(
    marker: Path,
    *,
    n: int,
    budget_cap_usd: float,
    output_dir: Path,
    config: AgentRunConfig | None = None,
    fixture_ids: tuple[str, ...] | None = None,
    fixture_set: str | None = None,
) -> None:
    receipt = approval_receipt_template(
        n=n,
        budget_cap_usd=budget_cap_usd,
        output_dir=output_dir,
        fixtures=resolve_fixtures(fixture_ids, fixture_set),
        config=config or AgentRunConfig(),
        role_to_model=ROLE_TO_MODEL,
    )
    receipt["approved_at"] = "2026-05-10T00:00:00Z"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


# --- ε-greedy randomization spec scenarios ----------------------------------


def test_epsilon_greedy__greedy_action_propensity() -> None:
    actions = ["a", "b", "c", "d"]
    # Force the greedy branch by exhausting the RNG path; we just validate the
    # math directly by computing expected values for both branches.
    # Run many draws, every "greedy" outcome must have propensity 0.85.
    rng = random.Random(123)
    saw_greedy = False
    for _ in range(200):
        eg2 = EpsilonGreedy(epsilon=0.2, seed=rng.randint(0, 1_000_000))
        chosen, prop = eg2.choose(actions, greedy="b")
        if chosen == "b":
            saw_greedy = True
            assert prop == pytest.approx(0.85)
        else:
            assert prop == pytest.approx(0.05)
    assert saw_greedy, "greedy branch never fired across 200 draws at ε=0.2"


def test_epsilon_greedy__non_greedy_action_propensity() -> None:
    # Drive multiple draws; since we test the formula above, any non-greedy
    # outcome must equal 0.1. Verify the formula is correct.
    actions = ["a", "b"]
    rng = random.Random(7)
    saw_non_greedy = False
    for _ in range(100):
        eg2 = EpsilonGreedy(epsilon=0.2, seed=rng.randint(0, 1_000_000))
        chosen, prop = eg2.choose(actions, greedy="a")
        if chosen != "a":
            saw_non_greedy = True
            assert prop == pytest.approx(0.1)
    # Without this, an implementation that always returned the greedy action
    # would slip past — the propensity check inside the loop never runs.
    assert saw_non_greedy, "non-greedy branch never fired across 100 draws at ε=0.2"


def test_epsilon_greedy__rejects_greedy_not_in_valid() -> None:
    eg = EpsilonGreedy(epsilon=0.2)
    with pytest.raises(ValueError):
        eg.choose(["x", "y"], greedy="z")


# --- Budget tracker spec scenarios ------------------------------------------


def test_budget__halt_at_80_percent() -> None:
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


@pytest.mark.parametrize("cost", [math.nan, math.inf, -math.inf])
def test_budget__rejects_non_finite_spend(cost: float) -> None:
    tracker = BudgetTracker(cap_usd=10.0)
    with pytest.raises(ValueError, match="finite"):
        tracker.add(cost)


# --- Fixture spec scenarios -------------------------------------------------


def test_real_corpus_has_at_least_three_fixtures() -> None:
    assert len(FIXTURES) >= 3
    for fx in FIXTURES:
        assert fx.root.is_dir()
        assert fx.source_path.is_file()
        assert fx.test_path is not None
        assert fx.test_path.is_file()
        # Each fixture's pristine pytest must currently fail (the bug).
        passed, _ = run_pytest(fx.root)
        assert passed is False, f"{fx.fixture_id} should start failing"


def test_outcome_is_pytest_exit_code(tmp_path: Path) -> None:
    # Use an easy fixture (string-utils) so a known good patch makes pytest pass.
    fixture = EASY_FIXTURES[0]

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


def test_agent_logs_per_decision_policy_params_with_resolved_config(
    tmp_path: Path,
) -> None:
    """D20: when AgentRunConfig sets per-decision greedy/epsilon, every randomized
    decision logs the *resolved* values it actually used — making the trace
    self-describing about its experimental condition."""
    fixture = EASY_FIXTURES[0]

    class _BlankLLM:
        def call(self, *, role: str, prompt: str) -> LLMResponse:
            return LLMResponse(text="(no patch)", cost_usd=0.0)

    budget = BudgetTracker(cap_usd=1.0)
    cfg = AgentRunConfig(
        seed=42,
        epsilon=0.2,
        tool_greedy="inspect_file",
        tool_epsilon=None,  # falls back to 0.2
        model_greedy="small",  # explicit non-default greedy
        model_epsilon=0.4,  # explicit non-default ε
        retry_greedy="retry_once",
        retry_epsilon=None,  # falls back to 0.2
    )
    run = run_one_trace(
        fixture,
        run_index=0,
        llm=_BlankLLM(),
        budget=budget,
        sandbox_root=tmp_path,
        config=cfg,
    )

    by_type: dict[str, dict] = {}
    for s in run.steps:
        for d in s.decisions:
            if d.policy is None:
                continue
            by_type.setdefault(d.decision_type, d.policy_params or {})

    assert by_type["tool_call"] == {"epsilon": 0.2, "greedy": "inspect_file"}
    assert by_type["model_call"] == {"epsilon": 0.4, "greedy": "small"}
    assert by_type["retry"] == {"epsilon": 0.2, "greedy": "retry_once"}


def test_agent_logs_retry_policy_on_every_trace_even_when_first_attempt_passes(
    tmp_path: Path,
) -> None:
    """D18: retry_policy is decided UPFRONT. Even when the first attempt passes
    and no retry happens, the trace must log a retry decision."""
    fixture = EASY_FIXTURES[0]

    class _PerfectLLM:
        def call(self, *, role: str, prompt: str) -> LLMResponse:
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
        config=AgentRunConfig(epsilon=0.2, seed=1),
    )
    assert run.outcome.value is True  # first attempt passed
    retry_decisions = [d for s in run.steps for d in s.decisions if d.decision_type == "retry"]
    assert len(retry_decisions) == 1
    rd = retry_decisions[0]
    assert rd.policy == "epsilon_greedy"
    assert rd.chosen_action in {"no_retry", "retry_once"}
    assert rd.propensity is not None
    # The retry decision is sequenced BEFORE any model_call (D18 ordering).
    decision_types_in_order: list[str] = [d.decision_type for s in run.steps for d in s.decisions]
    first_retry = decision_types_in_order.index("retry")
    first_model = decision_types_in_order.index("model_call")
    assert first_retry < first_model, (
        f"retry_policy must be decided before model_call; got order {decision_types_in_order}"
    )


def test_agent_retry_branch_includes_failure_context_in_prompt(tmp_path: Path) -> None:
    """D18: when the retry branch fires, the second model_call's prompt must
    include the failing test output (not just the same prompt as attempt 1)."""
    fixture = EASY_FIXTURES[0]

    captured_prompts: list[str] = []

    class _BlankPatchLLM:
        """First call returns no code fence, so the patch is empty and tests fail.
        Subsequent calls also return no patch — but we just want to verify the
        retry prompt is *different* and contains the failure context."""

        def call(self, *, role: str, prompt: str) -> LLMResponse:
            captured_prompts.append(prompt)
            return LLMResponse(text="(no patch)", cost_usd=0.0)

    budget = BudgetTracker(cap_usd=1.0)
    # Force retry_once via epsilon=0 + greedy=retry_once (default greedy).
    run = run_one_trace(
        fixture,
        run_index=0,
        llm=_BlankPatchLLM(),
        budget=budget,
        sandbox_root=tmp_path,
        config=AgentRunConfig(epsilon=0.0, seed=0),
    )
    assert len(captured_prompts) == 2, (
        f"expected 2 model calls (initial + retry), got {len(captured_prompts)}"
    )
    initial, retry_prompt = captured_prompts
    assert retry_prompt != initial
    assert "previous patch" in retry_prompt.lower() or "test output" in retry_prompt.lower()
    # The trace also captures the retry decision and both model_call attempts.
    model_calls = [d for s in run.steps for d in s.decisions if d.decision_type == "model_call"]
    assert len(model_calls) == 2
    assert model_calls[0].context_features.get("attempt") == 1
    assert model_calls[1].context_features.get("attempt") == 2
    assert model_calls[1].context_features.get("informed_retry") is True


def test_agent_logs_all_randomization_fields(tmp_path: Path) -> None:
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


def _first_model_call_observation(run: Run) -> dict:
    """Return the observation produced by the first model_call decision in `run`.

    The agent loop attaches exactly one observation to each model_call step
    (see agent.run_one_trace), so `step.observations[0]` precisely identifies
    the observation generated by that decision. The earlier list-comprehension
    pattern crossed every observation with every decision in a step, which
    could pick up an unrelated observation that happened to share a step.
    """
    for step in run.steps:
        if any(d.decision_type == "model_call" for d in step.decisions):
            assert len(step.observations) == 1, (
                "model_call step should record exactly one observation; "
                f"got {len(step.observations)}"
            )
            return step.observations[0].content
    raise AssertionError("no model_call observation found")


def test_agent_observation_records_extracted_patch_text(tmp_path: Path) -> None:
    """WHEN the LLM returns a Python code block
    THEN the model_call observation's `extracted_code` field carries the
    actual patch source as a string (not a bool flag) so downstream
    consumers can inspect what the agent wrote."""
    fixture = FIXTURES[1]
    code = "def add(a, b):\n    return a + b"
    fenced_response = f"Here's the fix:\n\n```python\n{code}\n```\n"

    class _FencedLLM:
        def call(self, *, role: str, prompt: str) -> LLMResponse:
            return LLMResponse(text=fenced_response, cost_usd=0.0)

    run = run_one_trace(
        fixture,
        run_index=0,
        llm=_FencedLLM(),
        budget=BudgetTracker(cap_usd=1.0),
        sandbox_root=tmp_path,
        config=AgentRunConfig(epsilon=0.0, seed=0),
    )

    obs = _first_model_call_observation(run)
    actual_type = type(obs["extracted_code"]).__name__
    assert isinstance(obs["extracted_code"], str), (
        f"extracted_code should be the patch text (str), got {actual_type}"
    )
    assert "def add" in obs["extracted_code"]


def test_agent_observation_extracted_code_is_none_when_parse_fails(tmp_path: Path) -> None:
    """WHEN the LLM response has no code fence
    THEN `extracted_code` is None (not False), preserving the
    'parse failed' signal as a typed absence rather than a bool flag."""
    fixture = FIXTURES[1]

    class _NoFenceLLM:
        def call(self, *, role: str, prompt: str) -> LLMResponse:
            return LLMResponse(text="I don't know how to fix this.", cost_usd=0.0)

    run = run_one_trace(
        fixture,
        run_index=0,
        llm=_NoFenceLLM(),
        budget=BudgetTracker(cap_usd=1.0),
        sandbox_root=tmp_path,
        config=AgentRunConfig(epsilon=0.0, seed=0),
    )

    obs = _first_model_call_observation(run)
    assert obs["extracted_code"] is None
    assert obs["extraction_status"] == "failed"
    assert obs["extraction_failure_reason"] == "raw_response_not_python"


def test_agent_observation_records_finish_reason_and_response_chars(
    tmp_path: Path,
) -> None:
    """WHEN the LLM response includes provider finish metadata
    THEN the trace keeps it next to extraction diagnostics."""
    fixture = FIXTURES[1]

    class _StoppedLLM:
        def call(self, *, role: str, prompt: str) -> LLMResponse:
            return LLMResponse(
                text="I don't know how to fix this.",
                cost_usd=0.0,
                finish_reason="length",
            )

    run = run_one_trace(
        fixture,
        run_index=0,
        llm=_StoppedLLM(),
        budget=BudgetTracker(cap_usd=1.0),
        sandbox_root=tmp_path,
        config=AgentRunConfig(epsilon=0.0, seed=0),
    )

    obs = _first_model_call_observation(run)
    assert obs["finish_reason"] == "length"
    assert obs["response_chars"] == len("I don't know how to fix this.")
    assert "raw_response_chars" not in obs


def test_extract_python_block_prefers_fenced_code() -> None:
    text = "notes\n```python\ndef target():\n    return 1\n```\n"
    assert _extract_python_block(text, expected_function="target") == (
        "def target():\n    return 1\n"
    )


def test_extract_python_block_accepts_parseable_full_response() -> None:
    text = "from __future__ import annotations\n\n\ndef target() -> int:\n    return 1\n"
    assert _extract_python_block(text, expected_function="target") == text


def test_extract_python_block_accepts_parseable_class_full_response() -> None:
    from bench.real.coding_agent.agent import _ExpectedSymbol, _extract_python_source

    text = (
        "from __future__ import annotations\n\n\n"
        "class Target:\n"
        "    def value(self) -> int:\n"
        "        return 1\n"
    )
    result = _extract_python_source(
        text,
        expected_symbol=_ExpectedSymbol(name="Target", kind="class"),
    )
    assert result.code == text
    assert result.status == "extracted"
    assert result.reason is None


def test_extract_python_block_rejects_unparseable_commentary() -> None:
    text = "Here is the fix:\n\ndef target() -> int:\n    return 1\n"
    assert _extract_python_block(text, expected_function="target") is None


def test_extract_python_block_rejects_wrong_function() -> None:
    text = "def other() -> int:\n    return 1\n"
    assert _extract_python_block(text, expected_function="target") is None


# --- Approval gate / resume / CLI -------------------------------------------


def test_first_run_prompts_before_any_api_call(tmp_path: Path) -> None:
    output = tmp_path / "out"

    class _ExplodingLLM:
        def call(self, *, role: str, prompt: str) -> LLMResponse:
            raise AssertionError("LLM was invoked despite missing approval marker!")

    rc = run_real_corpus(
        n=5,
        budget_cap_usd=5.0,
        output_dir=output,
        llm_client_factory=lambda: _ExplodingLLM(),
        marker_path=tmp_path / ".counterfact" / "approved",  # does not exist
    )
    assert rc == 2


def test_approved_marker_skips_prompt(tmp_path: Path) -> None:
    output = tmp_path / "out"
    marker = tmp_path / ".counterfact" / "approved"
    _write_approval(marker, n=3, budget_cap_usd=5.0, output_dir=output)

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


def test_empty_approval_marker_is_rejected_before_any_api_call(tmp_path: Path) -> None:
    output = tmp_path / "out"
    marker = tmp_path / ".counterfact" / "approved"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("")

    class _ExplodingLLM:
        def call(self, *, role: str, prompt: str) -> LLMResponse:
            raise AssertionError("LLM was invoked despite invalid approval receipt")

    rc = run_real_corpus(
        n=1,
        budget_cap_usd=5.0,
        output_dir=output,
        llm_client_factory=lambda: _ExplodingLLM(),
        marker_path=marker,
    )

    assert rc == 2


def test_approval_receipt_must_match_budget_before_any_api_call(tmp_path: Path) -> None:
    output = tmp_path / "out"
    marker = tmp_path / ".counterfact" / "approved"
    _write_approval(marker, n=1, budget_cap_usd=5.0, output_dir=output)

    class _ExplodingLLM:
        def call(self, *, role: str, prompt: str) -> LLMResponse:
            raise AssertionError("LLM was invoked despite mismatched approval receipt")

    rc = run_real_corpus(
        n=1,
        budget_cap_usd=10.0,
        output_dir=output,
        llm_client_factory=lambda: _ExplodingLLM(),
        marker_path=marker,
    )

    assert rc == 2


def test_resume_after_partial_run(tmp_path: Path) -> None:
    output = tmp_path / "out"
    marker = tmp_path / ".counterfact" / "approved"
    _write_approval(marker, n=4, budget_cap_usd=5.0, output_dir=output)

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


def test_run_real_corpus__cost_unknown_exits_before_writing_trace(tmp_path: Path) -> None:
    output = tmp_path / "out"
    marker = tmp_path / ".counterfact" / "approved"
    _write_approval(marker, n=1, budget_cap_usd=5.0, output_dir=output)

    class _UnknownCostLLM:
        def call(self, *, role: str, prompt: str) -> LLMResponse:
            raise CostUnknownError("price table miss")

    rc = run_real_corpus(
        n=1,
        budget_cap_usd=5.0,
        output_dir=output,
        llm_client_factory=lambda: _UnknownCostLLM(),
        marker_path=marker,
    )
    assert rc == 5
    assert list(output.glob("real-*.json")) == []


def test_resume_counts_existing_spend_before_new_calls(tmp_path: Path) -> None:
    output = tmp_path / "out"
    marker = tmp_path / ".counterfact" / "approved"
    _write_approval(marker, n=1, budget_cap_usd=10.0, output_dir=output)

    class _CostLLM:
        def __init__(self, cost: float) -> None:
            self.cost = cost

        def call(self, *, role: str, prompt: str) -> LLMResponse:
            return LLMResponse(text="(no patch)", cost_usd=self.cost)

    rc = run_real_corpus(
        n=1,
        budget_cap_usd=10.0,
        output_dir=output,
        llm_client_factory=lambda: _CostLLM(0.6),
        marker_path=marker,
    )
    assert rc == 0
    assert len(list(output.glob("real-*.json"))) == 1

    _write_approval(marker, n=2, budget_cap_usd=1.0, output_dir=output)
    rc2 = run_real_corpus(
        n=2,
        budget_cap_usd=1.0,
        output_dir=output,
        llm_client_factory=lambda: _CostLLM(0.3),
        marker_path=marker,
    )
    assert rc2 == 3
    assert len(list(output.glob("real-*.json"))) == 1


def test_resume_halts_before_call_when_existing_spend_exceeds_threshold(
    tmp_path: Path,
) -> None:
    output = tmp_path / "out"
    marker = tmp_path / ".counterfact" / "approved"
    _write_approval(marker, n=1, budget_cap_usd=10.0, output_dir=output)

    class _CostLLM:
        def call(self, *, role: str, prompt: str) -> LLMResponse:
            return LLMResponse(text="(no patch)", cost_usd=1.0)

    rc = run_real_corpus(
        n=1,
        budget_cap_usd=10.0,
        output_dir=output,
        llm_client_factory=lambda: _CostLLM(),
        marker_path=marker,
    )
    assert rc == 0

    class _ExplodingLLM:
        def call(self, *, role: str, prompt: str) -> LLMResponse:
            raise AssertionError("resume should halt before a new LLM call")

    _write_approval(marker, n=2, budget_cap_usd=1.0, output_dir=output)
    rc2 = run_real_corpus(
        n=2,
        budget_cap_usd=1.0,
        output_dir=output,
        llm_client_factory=lambda: _ExplodingLLM(),
        marker_path=marker,
    )
    assert rc2 == 3


def test_resume_rejects_malformed_budget_ledger_before_call(tmp_path: Path) -> None:
    output = tmp_path / "out"
    marker = tmp_path / ".counterfact" / "approved"
    _write_approval(marker, n=1, budget_cap_usd=5.0, output_dir=output)
    ledger = output / ".checkpoints" / "budget_ledger.jsonl"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_text("{bad json}\n")

    class _ExplodingLLM:
        def call(self, *, role: str, prompt: str) -> LLMResponse:
            raise AssertionError("resume should reject bad ledger before an LLM call")

    rc = run_real_corpus(
        n=1,
        budget_cap_usd=5.0,
        output_dir=output,
        llm_client_factory=lambda: _ExplodingLLM(),
        marker_path=marker,
    )

    assert rc == 7


def test_resume_rejects_changed_fixture_selection(tmp_path: Path) -> None:
    output = tmp_path / "out"
    marker = tmp_path / ".counterfact" / "approved"
    _write_approval(marker, n=1, budget_cap_usd=5.0, output_dir=output)

    class _NullLLM:
        def call(self, *, role: str, prompt: str) -> LLMResponse:
            return LLMResponse(text="(no patch)", cost_usd=0.0)

    assert (
        run_real_corpus(
            n=1,
            budget_cap_usd=5.0,
            output_dir=output,
            llm_client_factory=lambda: _NullLLM(),
            marker_path=marker,
        )
        == 0
    )
    _write_approval(
        marker,
        n=2,
        budget_cap_usd=5.0,
        output_dir=output,
        fixture_ids=("csv_dedupe",),
    )
    rc = run_real_corpus(
        n=2,
        budget_cap_usd=5.0,
        output_dir=output,
        llm_client_factory=lambda: _NullLLM(),
        marker_path=marker,
        fixture_ids=("csv_dedupe",),
    )
    assert rc == 6


def test_resume_rejects_changed_randomization_config(tmp_path: Path) -> None:
    output = tmp_path / "out"
    marker = tmp_path / ".counterfact" / "approved"
    first_config = AgentRunConfig(epsilon=0.2, seed=0)
    _write_approval(
        marker,
        n=1,
        budget_cap_usd=5.0,
        output_dir=output,
        config=first_config,
    )

    class _NullLLM:
        def call(self, *, role: str, prompt: str) -> LLMResponse:
            return LLMResponse(text="(no patch)", cost_usd=0.0)

    assert (
        run_real_corpus(
            n=1,
            budget_cap_usd=5.0,
            output_dir=output,
            llm_client_factory=lambda: _NullLLM(),
            marker_path=marker,
            config=first_config,
        )
        == 0
    )
    second_config = AgentRunConfig(epsilon=0.4, seed=0)
    _write_approval(
        marker,
        n=2,
        budget_cap_usd=5.0,
        output_dir=output,
        config=second_config,
    )
    rc = run_real_corpus(
        n=2,
        budget_cap_usd=5.0,
        output_dir=output,
        llm_client_factory=lambda: _NullLLM(),
        marker_path=marker,
        config=second_config,
    )
    assert rc == 6


def test_resume_rejects_existing_traces_without_identity(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    source = next((repo_root / "bench" / "real" / "single_class_refusal").glob("real-*.json"))
    output = tmp_path / "out"
    output.mkdir()
    shutil.copy(source, output / source.name)
    marker = tmp_path / ".counterfact" / "approved"
    _write_approval(marker, n=2, budget_cap_usd=5.0, output_dir=output)

    class _ExplodingLLM:
        def call(self, *, role: str, prompt: str) -> LLMResponse:
            raise AssertionError("resume should reject before a new LLM call")

    rc = run_real_corpus(
        n=2,
        budget_cap_usd=5.0,
        output_dir=output,
        llm_client_factory=lambda: _ExplodingLLM(),
        marker_path=marker,
    )
    assert rc == 6


def test_extract_cost__prefers_response_cost_when_present() -> None:
    """If litellm populated response_cost on the response, use it directly."""
    resp = {"choices": [{"message": {"content": "x"}}], "response_cost": 0.0237}
    assert extract_cost(resp) == pytest.approx(0.0237)


@pytest.mark.parametrize("cost", [math.nan, math.inf, -math.inf])
def test_extract_cost__rejects_non_finite_response_cost(cost: float) -> None:
    resp = {"choices": [{"message": {"content": "x"}}], "response_cost": cost}
    with pytest.raises(CostUnknownError, match="non-finite"):
        extract_cost(resp)


def test_extract_cost__falls_back_to_completion_cost(monkeypatch: pytest.MonkeyPatch) -> None:
    """When response_cost is absent (None), derive cost via litellm.completion_cost."""
    import litellm  # type: ignore[import-not-found]

    resp = {"choices": [{"message": {"content": "x"}}]}
    monkeypatch.setattr(litellm, "completion_cost", lambda completion_response: 0.0123)
    assert extract_cost(resp) == pytest.approx(0.0123)


def test_extract_cost__zero_response_cost_is_returned_not_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real `response_cost == 0.0` is a valid zero, not a missing value."""
    import litellm  # type: ignore[import-not-found]

    resp = {"choices": [{"message": {"content": "x"}}], "response_cost": 0.0}

    def _should_not_be_called(**_kw: object) -> float:
        raise AssertionError("litellm fallback must not run when response_cost is 0.0")

    monkeypatch.setattr(litellm, "completion_cost", _should_not_be_called)
    assert extract_cost(resp) == 0.0


@pytest.mark.parametrize("cost", [math.nan, math.inf, -math.inf, 0.0])
def test_extract_cost__rejects_non_positive_or_non_finite_fallback(
    monkeypatch: pytest.MonkeyPatch,
    cost: float,
) -> None:
    import litellm  # type: ignore[import-not-found]

    resp = {"choices": [{"message": {"content": "x"}}]}
    monkeypatch.setattr(litellm, "completion_cost", lambda completion_response: cost)
    with pytest.raises(CostUnknownError):
        extract_cost(resp)


def test_extract_cost__raises_when_both_paths_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Production cost accounting fails closed if both cost paths are unavailable."""
    import litellm  # type: ignore[import-not-found]

    resp = {"choices": [{"message": {"content": "x"}}]}

    def _raise(**kw: object) -> float:
        raise RuntimeError("price table miss")

    monkeypatch.setattr(litellm, "completion_cost", _raise)
    with pytest.raises(CostUnknownError, match="could not determine"):
        extract_cost(resp)


def test_litellm_client__raises_when_response_cost_is_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import litellm  # type: ignore[import-not-found]

    monkeypatch.setattr(
        litellm,
        "completion",
        lambda **kw: {"choices": [{"message": {"content": "ok"}}]},
    )

    def _raise(**kw: object) -> float:
        raise RuntimeError("price table miss")

    monkeypatch.setattr(litellm, "completion_cost", _raise)
    client = LiteLLMClient(role_to_model={"small": "claude-haiku-4-5"})
    with pytest.raises(CostUnknownError):
        client.call(role="small", prompt="fix this")


def test_litellm_client__requests_large_enough_patch_budget_and_records_finish_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import litellm  # type: ignore[import-not-found]

    captured: dict[str, object] = {}

    def _completion(**kw: object) -> dict[str, object]:
        captured.update(kw)
        return {
            "choices": [
                {
                    "message": {"content": "ok"},
                    "finish_reason": "stop",
                }
            ],
            "response_cost": 0.01,
        }

    monkeypatch.setattr(litellm, "completion", _completion)
    client = LiteLLMClient(role_to_model={"small": "claude-haiku-4-5"})
    response = client.call(role="small", prompt="fix this")

    assert captured["max_tokens"] == DEFAULT_MAX_TOKENS
    assert captured["timeout"] == DEFAULT_REQUEST_TIMEOUT_S
    assert captured["num_retries"] == DEFAULT_NUM_RETRIES
    assert response.finish_reason == "stop"


def test_litellm_client__normalizes_missing_content_to_extraction_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import litellm  # type: ignore[import-not-found]

    monkeypatch.setattr(
        litellm,
        "completion",
        lambda **kw: {
            "choices": [{"message": {"content": None}, "finish_reason": "stop"}],
            "response_cost": 0.01,
        },
    )
    client = LiteLLMClient(role_to_model={"small": "claude-haiku-4-5"})

    response = client.call(role="small", prompt="fix this")

    assert response.text == ""
    assert response.finish_reason == "stop"


def test_pytest_helper_caps_output_and_scrubs_provider_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import bench.real.coding_agent.fixtures as fixtures_module

    monkeypatch.setenv("ANTHROPIC_API_KEY", "secret")
    captured_env: dict[str, str] = {}

    def _fake_run(*args, stdout, stderr, env, **kwargs):
        captured_env.update(env)
        stdout.write(b"x" * 3000)
        return subprocess.CompletedProcess(args=args, returncode=1)

    monkeypatch.setattr(fixtures_module.subprocess, "run", _fake_run)

    passed, tail = fixtures_module._run_pytest_at(tmp_path, "tests/")

    assert passed is False
    assert len(tail) == 2000
    assert set(tail) == {"x"}
    assert "ANTHROPIC_API_KEY" not in captured_env
    assert captured_env["HOME"] != str(Path.home())


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
    marker = tmp_path / ".counterfact" / "approved"
    output = tmp_path / "out"
    _write_approval(marker, n=1, budget_cap_usd=1.0, output_dir=output)

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
    """`counterfact bench real ...` without an approval marker exits 2 with the prompt.

    Runs from a clean cwd with NO PYTHONPATH override — this is the regression
    test that catches `bench` not being included in the wheel.
    """
    out = tmp_path / "out"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "counterfact.cli",
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
        timeout=30,
    )
    assert proc.returncode == 2, f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    combined = proc.stdout + proc.stderr
    assert "HUMAN GATE" in combined
    assert "approved" in combined


def test_analyze_pilot_cli_writes_notes_without_provider_calls(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    source_dir = repo_root / "bench" / "real" / "single_class_refusal"
    run_dir = tmp_path / "runs"
    run_dir.mkdir()
    for path in sorted(source_dir.glob("real-*.json"))[:3]:
        shutil.copy(path, run_dir / path.name)

    proc = subprocess.run(
        [sys.executable, "-m", "bench.real.analyze_pilot", str(run_dir)],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    assert "2x2 contingency" in proc.stdout
    assert "Extraction failures" in proc.stdout
    assert (run_dir / "PILOT_NOTES.md").is_file()


def _write_pilot_trace(
    run_dir: Path,
    name: str,
    *,
    fixture_id: str = "unicode_normalize",
    model: str = "large",
    public_pass: bool = True,
    hidden_pass: bool = False,
    extracted_code: str | None = "def dedupe_normalized(labels):\n    return labels\n",
    stdout_tail: str = "",
) -> None:
    run = {
        "steps": [
            {
                "decisions": [{"decision_type": "model_call", "chosen_action": model}],
                "observations": [
                    {
                        "content": {
                            "cost_usd": 0.0,
                            "extracted_code": extracted_code,
                        }
                    }
                ],
            },
            {
                "decisions": [{"decision_type": "tool_call", "chosen_action": "run_tests"}],
                "observations": [{"content": {"stdout_tail": stdout_tail}}],
            },
        ],
        "outcome": {
            "metadata": {
                "fixture_id": fixture_id,
                "public_pass": public_pass,
                "hidden_pass": hidden_pass,
            }
        },
    }
    (run_dir / f"real-{name}.json").write_text(json.dumps(run))


def test_analyze_pilot_counts_hidden_semantic_failures_by_model(tmp_path: Path) -> None:
    from bench.real.analyze_pilot import analyze, render

    _write_pilot_trace(tmp_path, "small", model="small")
    _write_pilot_trace(tmp_path, "large", model="large")
    _write_pilot_trace(tmp_path, "pass", model="large", hidden_pass=True)

    report = analyze(tmp_path)
    assert report["failure_modes"]["hidden_semantic_failure"] == 2
    assert report["failure_modes_by_model"]["model=small,mode=hidden_semantic_failure"] == 1
    assert report["failure_modes_by_model"]["model=large,mode=hidden_semantic_failure"] == 1
    assert report["showcase_gate_passed"] is True
    assert "Showcase composition gate" in render(report)


def test_analyze_pilot_rejects_format_dominated_failures(tmp_path: Path) -> None:
    from bench.real.analyze_pilot import analyze

    _write_pilot_trace(tmp_path, "format-small", model="small", extracted_code=None)
    _write_pilot_trace(tmp_path, "format-large", model="large", extracted_code=None)
    _write_pilot_trace(tmp_path, "semantic-large", model="large")

    report = analyze(tmp_path)
    assert report["failure_modes"]["format_failure"] == 2
    assert report["failure_modes"]["hidden_semantic_failure"] == 1
    assert report["showcase_gate_passed"] is False


def test_analyze_pilot_counts_public_failure(tmp_path: Path) -> None:
    from bench.real.analyze_pilot import analyze

    _write_pilot_trace(
        tmp_path,
        "public-failure",
        public_pass=False,
        hidden_pass=False,
        extracted_code="def dedupe_normalized(labels):\n    return []\n",
    )

    report = analyze(tmp_path)
    assert report["failure_modes"]["public_failure"] == 1
