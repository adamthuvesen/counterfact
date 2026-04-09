"""counter — causal attribution for agent traces."""

from counter.attribute import attribute_failure
from counter.dag import build_dag
from counter.intervene import intervene
from counter.outcome import fit_outcome_model

__version__ = "0.0.0"

__all__ = [
    "__version__",
    "attribute_failure",
    "build_dag",
    "fit_outcome_model",
    "intervene",
]
