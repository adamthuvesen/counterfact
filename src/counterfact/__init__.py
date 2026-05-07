"""counterfact — causal attribution for agent traces."""

from importlib.metadata import PackageNotFoundError, version

from counterfact.attribute import attribute_failure
from counterfact.baselines import PassRateRow, PassRateTable, pass_rate_by_arm
from counterfact.corpus_analyzer import CorpusReadinessReport, RubricThresholds, analyze_corpus
from counterfact.dag import build_dag
from counterfact.intervene import intervene
from counterfact.outcome import fit_outcome_model
from counterfact.power import PowerReport, power_analysis

try:
    __version__ = version("counterfact")
except PackageNotFoundError:
    __version__ = "0.1.0"

__all__ = [
    "CorpusReadinessReport",
    "PassRateRow",
    "PassRateTable",
    "PowerReport",
    "RubricThresholds",
    "__version__",
    "analyze_corpus",
    "attribute_failure",
    "build_dag",
    "fit_outcome_model",
    "intervene",
    "pass_rate_by_arm",
    "power_analysis",
]
