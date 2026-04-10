"""counterfact — causal attribution for agent traces."""

from counterfact.attribute import attribute_failure
from counterfact.baselines import PassRateRow, PassRateTable, pass_rate_by_arm
from counterfact.dag import build_dag
from counterfact.intervene import intervene
from counterfact.outcome import fit_outcome_model
from counterfact.power import PowerReport, power_analysis

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
