"""Coding-agent loop for the real-agent corpus.

The loop is small and inspectable. Per design.md D16/D18, the agent decides
its retry budget *upfront* (after inspect_file, before the first model call)
so that `retry_policy` is logged on every trace — not just the ones whose
first attempt fails. When the retry branch does fire, the second model call's
prompt includes the failing test output, so the retry is informed rather than
a blind coin flip.

Randomized decision types (per the v0 commitment in design.md D5/D9):
* `model_call.model_choice` — which model role (small/large) drafts the patch
* `tool_call.tool_choice`   — which tool comes first (run_tests vs inspect_file)
* `retry.retry_policy`      — attempt budget: no_retry (1 attempt) or retry_once (2)

The loop emits a counter-native `Run` per fixture. LLM calls go through the
`LLMClient` protocol so tests can mock them.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path

from bench.real.coding_agent.budget import BudgetExceeded, BudgetTracker
from bench.real.coding_agent.fixtures import FixtureSpec, run_pytest, snapshot_fixture
from bench.real.coding_agent.llm import LLMClient
from bench.real.coding_agent.randomize import EpsilonGreedy
from counter.schema import Decision, Observation, Outcome, Run, Step

DEFAULT_MAX_STEPS = 8
TOOL_ARMS: list[str] = ["inspect_file", "run_tests", "search_docs"]
MODEL_ARMS: list[str] = ["small", "large"]
RETRY_ARMS: list[str] = ["no_retry", "retry_once"]

_FIX_PROMPT = """\
You are a small coding agent. The repository under {root} contains a Python
source file with a bug; the test below pins the expected behavior. Read both,
identify the bug, and reply with the FULL corrected source file inside a
fenced ```python``` block. Do not include any other commentary.

--- SOURCE ({source_relpath}) ---
{source}

--- TEST ({test_relpath}) ---
{test}
"""

_RETRY_PROMPT_SUFFIX = """

Your previous patch was applied but the tests still failed. Here is the test
output (last lines). Identify what your patch missed and reply with the FULL
corrected source file inside a fenced ```python``` block.

--- TEST OUTPUT (tail) ---
{test_output}
"""

_CODE_FENCE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def _extract_python_block(text: str) -> str | None:
    m = _CODE_FENCE.search(text)
    if not m:
        return None
    return m.group(1).rstrip() + "\n"


@dataclass
class AgentRunConfig:
    epsilon: float = 0.2
    max_steps: int = DEFAULT_MAX_STEPS
    seed: int = 0


def run_one_trace(
    fixture: FixtureSpec,
    *,
    run_index: int,
    llm: LLMClient,
    budget: BudgetTracker,
    sandbox_root: Path,
    config: AgentRunConfig,
) -> Run:
    """Run the agent against one fixture sandbox; return a counter-native Run."""
    rng = random.Random(config.seed ^ run_index)
    eg_tool = EpsilonGreedy(epsilon=config.epsilon, seed=rng.randint(0, 2**31 - 1))
    eg_model = EpsilonGreedy(epsilon=config.epsilon, seed=rng.randint(0, 2**31 - 1))
    eg_retry = EpsilonGreedy(epsilon=config.epsilon, seed=rng.randint(0, 2**31 - 1))

    sandbox = snapshot_fixture(fixture, sandbox_root)
    src_path = sandbox / "src" / fixture.source_relpath
    test_path = sandbox / "tests" / fixture.test_relpath

    steps: list[Step] = []
    step_index = 0

    # ----- Step 0: plan_step (deterministic) -----
    steps.append(
        Step(
            step_index=step_index,
            decisions=[
                Decision(
                    decision_id=f"d-{run_index:06d}-plan",
                    decision_type="plan_step",
                    chosen_action="begin",
                )
            ],
        )
    )
    step_index += 1

    # ----- Step 1: tool_call (ε-greedy on tool_choice; greedy = inspect_file) -----
    tool_action, tool_prop = eg_tool.choose(TOOL_ARMS, greedy="inspect_file")
    src_text = src_path.read_text()
    test_text = test_path.read_text()
    steps.append(
        Step(
            step_index=step_index,
            decisions=[
                Decision(
                    decision_id=f"d-{run_index:06d}-inspect",
                    decision_type="tool_call",
                    chosen_action=tool_action,
                    policy="epsilon_greedy",
                    policy_params={"epsilon": config.epsilon},
                    valid_actions=list(TOOL_ARMS),
                    propensity=tool_prop,
                    context_features={"first_action": True},
                )
            ],
            observations=[
                Observation(
                    observation_id=f"o-{run_index:06d}-inspect",
                    content={
                        "source_chars": len(src_text),
                        "test_chars": len(test_text),
                    },
                )
            ],
        )
    )
    step_index += 1

    # ----- Step 2: retry_policy decided UPFRONT (D18) -----
    retry_action, retry_prop = eg_retry.choose(RETRY_ARMS, greedy="retry_once")
    attempts_remaining = 1 + (1 if retry_action == "retry_once" else 0)
    steps.append(
        Step(
            step_index=step_index,
            decisions=[
                Decision(
                    decision_id=f"d-{run_index:06d}-retry",
                    decision_type="retry",
                    chosen_action=retry_action,
                    policy="epsilon_greedy",
                    policy_params={"epsilon": config.epsilon},
                    valid_actions=list(RETRY_ARMS),
                    propensity=retry_prop,
                    context_features={"step_index": step_index, "upfront": True},
                )
            ],
        )
    )
    step_index += 1

    # ----- Step 3: model_call (ε-greedy on model_choice; greedy = large) -----
    model_action, model_prop = eg_model.choose(MODEL_ARMS, greedy="large")
    prompt = _FIX_PROMPT.format(
        root=sandbox,
        source_relpath=fixture.source_relpath,
        source=src_text,
        test_relpath=fixture.test_relpath,
        test=test_text,
    )
    resp = llm.call(role=model_action, prompt=prompt)
    budget.add(resp.cost_usd)
    patched = _extract_python_block(resp.text)
    steps.append(
        Step(
            step_index=step_index,
            decisions=[
                Decision(
                    decision_id=f"d-{run_index:06d}-model-1",
                    decision_type="model_call",
                    chosen_action=model_action,
                    policy="epsilon_greedy",
                    policy_params={"epsilon": config.epsilon},
                    valid_actions=list(MODEL_ARMS),
                    propensity=model_prop,
                    context_features={"prompt_chars": len(prompt), "attempt": 1},
                )
            ],
            observations=[
                Observation(
                    observation_id=f"o-{run_index:06d}-model-1",
                    content={
                        "response_chars": len(resp.text),
                        "cost_usd": resp.cost_usd,
                        "extracted_code": patched is not None,
                    },
                )
            ],
        )
    )
    step_index += 1

    if patched is not None:
        src_path.write_text(patched)

    # ----- Step 4: tool_call run_tests (deterministic) -----
    passed, tail = run_pytest(sandbox)
    steps.append(
        Step(
            step_index=step_index,
            decisions=[
                Decision(
                    decision_id=f"d-{run_index:06d}-runtests-1",
                    decision_type="tool_call",
                    chosen_action="run_tests",
                )
            ],
            observations=[
                Observation(
                    observation_id=f"o-{run_index:06d}-runtests-1",
                    content={"passed": passed, "stdout_tail": tail[-500:]},
                )
            ],
        )
    )
    step_index += 1
    attempts_remaining -= 1

    if passed:
        steps.append(
            Step(
                step_index=step_index,
                decisions=[
                    Decision(
                        decision_id=f"d-{run_index:06d}-term",
                        decision_type="termination",
                        chosen_action="success",
                    )
                ],
            )
        )
        return _build_run(run_index, fixture, steps, success=True)

    if attempts_remaining <= 0:
        # no_retry was chosen; we ran the one allotted attempt and it failed
        steps.append(
            Step(
                step_index=step_index,
                decisions=[
                    Decision(
                        decision_id=f"d-{run_index:06d}-term",
                        decision_type="termination",
                        chosen_action="give_up",
                    )
                ],
            )
        )
        return _build_run(run_index, fixture, steps, success=False)

    # ----- Step 5: model_call retry, with failure context (D18) -----
    src_text = src_path.read_text()
    retry_prompt = _FIX_PROMPT.format(
        root=sandbox,
        source_relpath=fixture.source_relpath,
        source=src_text,
        test_relpath=fixture.test_relpath,
        test=test_text,
    ) + _RETRY_PROMPT_SUFFIX.format(test_output=tail[-1000:])
    resp2 = llm.call(role=model_action, prompt=retry_prompt)
    budget.add(resp2.cost_usd)
    patched2 = _extract_python_block(resp2.text)
    steps.append(
        Step(
            step_index=step_index,
            decisions=[
                Decision(
                    decision_id=f"d-{run_index:06d}-model-2",
                    decision_type="model_call",
                    chosen_action=model_action,
                    policy="epsilon_greedy",
                    policy_params={"epsilon": config.epsilon},
                    valid_actions=list(MODEL_ARMS),
                    propensity=model_prop,
                    context_features={
                        "prompt_chars": len(retry_prompt),
                        "attempt": 2,
                        "informed_retry": True,
                    },
                )
            ],
            observations=[
                Observation(
                    observation_id=f"o-{run_index:06d}-model-2",
                    content={
                        "response_chars": len(resp2.text),
                        "cost_usd": resp2.cost_usd,
                        "extracted_code": patched2 is not None,
                    },
                )
            ],
        )
    )
    step_index += 1

    if patched2 is not None:
        src_path.write_text(patched2)

    # ----- Step 6: tool_call run_tests (second attempt) -----
    passed2, tail2 = run_pytest(sandbox)
    steps.append(
        Step(
            step_index=step_index,
            decisions=[
                Decision(
                    decision_id=f"d-{run_index:06d}-runtests-2",
                    decision_type="tool_call",
                    chosen_action="run_tests",
                )
            ],
            observations=[
                Observation(
                    observation_id=f"o-{run_index:06d}-runtests-2",
                    content={"passed": passed2, "stdout_tail": tail2[-500:]},
                )
            ],
        )
    )
    step_index += 1

    steps.append(
        Step(
            step_index=step_index,
            decisions=[
                Decision(
                    decision_id=f"d-{run_index:06d}-term",
                    decision_type="termination",
                    chosen_action="success" if passed2 else "give_up",
                )
            ],
        )
    )
    return _build_run(run_index, fixture, steps, success=passed2)


def _build_run(
    run_index: int, fixture: FixtureSpec, steps: list[Step], *, success: bool
) -> Run:
    return Run(
        schema_version="0.1.0",
        run_id=f"real-{fixture.fixture_id}-{run_index:06d}",
        steps=steps,
        outcome=Outcome(
            kind="binary",
            value=success,
            verifier="pytest",
            metadata={"fixture_id": fixture.fixture_id},
        ),
    )


# Re-export for tests / callers that imported BudgetExceeded from this module.
__all__ = [
    "AgentRunConfig",
    "BudgetExceeded",
    "DEFAULT_MAX_STEPS",
    "MODEL_ARMS",
    "RETRY_ARMS",
    "TOOL_ARMS",
    "run_one_trace",
]
