"""Corpus-readiness analyzer.

Score a candidate corpus against the promotion rubric. No LLM calls, no file
writes. See `openspec/specs/corpus-analyzer/spec.md` for the public contract.
"""

from counterfact.corpus_analyzer.analyze import analyze
from counterfact.corpus_analyzer.report import (
    ArmSupportRow,
    CorpusReadinessReport,
    IdentifiabilityCoverage,
    OutcomeBalance,
    RubricCriterion,
)
from counterfact.corpus_analyzer.rubric import DEFAULT_THRESHOLDS, RubricThresholds

# Convenience alias matching the top-level public API.
analyze_corpus = analyze

__all__ = [
    "DEFAULT_THRESHOLDS",
    "ArmSupportRow",
    "CorpusReadinessReport",
    "IdentifiabilityCoverage",
    "OutcomeBalance",
    "RubricCriterion",
    "RubricThresholds",
    "analyze",
    "analyze_corpus",
]
