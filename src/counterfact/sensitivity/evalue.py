"""E-value computation per VanderWeele & Ding (2017).

The E-value is the minimum strength of association (on the risk-ratio scale)
that an unmeasured confounder would need with both the intervention and the
outcome to fully explain away the observed effect.

For a risk ratio RR > 1: E = RR + sqrt(RR * (RR - 1))
For RR < 1: invert to RR' = 1/RR and apply the same formula.
"""

from __future__ import annotations

import math


def e_value(risk_ratio: float) -> float:
    """Return the E-value for the given risk ratio. RR must be > 0."""
    if risk_ratio <= 0:
        raise ValueError(f"risk_ratio must be > 0, got {risk_ratio}")
    rr = risk_ratio if risk_ratio >= 1.0 else 1.0 / risk_ratio
    return rr + math.sqrt(rr * (rr - 1.0))
