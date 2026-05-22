"""USD budget tracker with 80% halt threshold (design.md D11, D17)."""

from __future__ import annotations

import math
from dataclasses import dataclass


class BudgetExceeded(Exception):
    """Raised when cumulative spend reaches the halt threshold."""

    def __init__(self, spent: float, cap: float, halt_fraction: float) -> None:
        self.spent = spent
        self.cap = cap
        self.halt_fraction = halt_fraction
        super().__init__(
            f"Budget cap {int(halt_fraction * 100)}% reached: ${spent:.4f} of ${cap:.2f}"
        )


@dataclass
class BudgetTracker:
    """Track LLM spend; halt when `halt_fraction * cap` is reached."""

    cap_usd: float
    halt_fraction: float = 0.80
    spent_usd: float = 0.0

    def __post_init__(self) -> None:
        if self.cap_usd <= 0:
            raise ValueError(f"cap_usd must be > 0; got {self.cap_usd!r}")
        if not (0.0 < self.halt_fraction <= 1.0):
            raise ValueError(f"halt_fraction must be in (0, 1]; got {self.halt_fraction!r}")

    @property
    def halt_threshold(self) -> float:
        return self.halt_fraction * self.cap_usd

    @property
    def remaining(self) -> float:
        return self.halt_threshold - self.spent_usd

    def add(self, usd: float) -> None:
        if not math.isfinite(usd):
            raise ValueError(f"spend must be finite; got {usd!r}")
        if usd < 0:
            raise ValueError(f"cannot add negative spend: {usd!r}")
        self.spent_usd += usd
        if self.spent_usd >= self.halt_threshold:
            raise BudgetExceeded(
                spent=self.spent_usd, cap=self.cap_usd, halt_fraction=self.halt_fraction
            )

    def would_halt(self, projected_usd: float) -> bool:
        return self.spent_usd + projected_usd >= self.halt_threshold
