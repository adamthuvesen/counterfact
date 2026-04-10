"""Failure attribution: rank decisions by causal influence on the outcome."""

from counterfact.attribute.failure import (
    AttributionEntry,
    FailureAttribution,
    attribute_failure,
)

__all__ = ["AttributionEntry", "FailureAttribution", "attribute_failure"]
