"""counter — causal attribution for agent traces."""

from counter.attribute import attribute_failure
from counter.baselines import PassRateRow, PassRateTable, pass_rate_by_arm
from counter.dag import build_dag
from counter.intervene import intervene
from counter.outcome import fit_outcome_model
from counter.power import PowerReport, power_analysis

__version__ = "0.0.0"

__all__ = [
    "PassRateRow",
    "PassRateTable",
    "PowerReport",
    "__version__",
    "attribute_failure",
    "build_dag",
    "fit_outcome_model",
    "intervene",
    "pass_rate_by_arm",
    "power_analysis",
]
