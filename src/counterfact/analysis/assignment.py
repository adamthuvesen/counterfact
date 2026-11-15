"""Validate logged assignment evidence before using it for identification."""

import math

from counterfact.analysis.facts import DecisionFact
from counterfact.schema import Decision


def has_mass(
    decision: Decision | DecisionFact, chosen: str, propensity: float | None, arm: str
) -> bool | None:
    """Return positive mass, zero mass, or unknown for the requested action."""
    valid = decision.valid_actions or []
    if (
        propensity is None
        or not math.isfinite(propensity)
        or not 0 < propensity <= 1
        or not valid
        or len(valid) != len(set(valid))
        or chosen not in valid
        or chosen != decision.chosen_action
    ):
        return None
    params = decision.policy_params or {}
    if decision.policy == "uniform":
        if not _matches(propensity, 1 / len(valid)):
            return None
        return arm in valid
    if decision.policy == "epsilon_greedy":
        epsilon, greedy = params.get("epsilon"), params.get("greedy")
        if (
            isinstance(epsilon, bool)
            or not isinstance(epsilon, int | float)
            or not 0 <= epsilon <= 1
            or not math.isfinite(epsilon)
            or greedy not in valid
        ):
            return None
        expected = epsilon / len(valid) + (1 - epsilon if chosen == greedy else 0)
        if not _matches(propensity, expected):
            return None
        return arm in valid and (epsilon > 0 or arm == greedy)
    if arm not in valid:
        return False
    if arm == chosen:
        return True
    if len(valid) == 2:
        return propensity < 1
    return None


def _matches(logged: float, expected: float) -> bool:
    return math.isclose(logged, expected, rel_tol=1e-9, abs_tol=1e-9)
