"""ε-greedy randomization with logged propensities (design.md D8, D16)."""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class EpsilonGreedy:
    """Stateful ε-greedy sampler. Reproducible per seed."""

    epsilon: float
    seed: int = 0

    def __post_init__(self) -> None:
        if not (0.0 <= self.epsilon <= 1.0):
            raise ValueError(f"epsilon must be in [0, 1]; got {self.epsilon!r}")
        self._rng = random.Random(self.seed)

    def choose(
        self,
        valid_actions: list[str],
        greedy: str,
    ) -> tuple[str, float]:
        """Return (chosen_action, propensity).

        With probability (1-ε) pick greedy; with probability ε sample uniformly
        from valid_actions. The logged propensity reflects the *total* probability
        of having picked the chosen action under either branch.
        """
        if greedy not in valid_actions:
            raise ValueError(
                f"greedy action {greedy!r} not in valid_actions={valid_actions!r}"
            )
        n = len(valid_actions)
        if self._rng.random() < self.epsilon:
            chosen = self._rng.choice(valid_actions)
        else:
            chosen = greedy
        if chosen == greedy:
            propensity = (1.0 - self.epsilon) + self.epsilon / n
        else:
            propensity = self.epsilon / n
        return chosen, propensity


def epsilon_greedy(
    valid_actions: list[str],
    greedy: str,
    epsilon: float,
    rng: random.Random,
) -> tuple[str, float]:
    """Stateless variant for ad-hoc use. Same semantics as `EpsilonGreedy.choose`."""
    if greedy not in valid_actions:
        raise ValueError(f"greedy action {greedy!r} not in valid_actions={valid_actions!r}")
    n = len(valid_actions)
    if rng.random() < epsilon:
        chosen = rng.choice(valid_actions)
    else:
        chosen = greedy
    propensity = (1.0 - epsilon) + epsilon / n if chosen == greedy else epsilon / n
    return chosen, propensity
