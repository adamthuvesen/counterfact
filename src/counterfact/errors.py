"""Exception types raised by counterfact."""


class CounterError(Exception):
    """Base class for counterfact errors."""


class UnsupportedOutcomeError(CounterError):
    """Raised when an Outcome.kind is not supported by the v0 runtime."""


class InsufficientOutcomeSupportError(UnsupportedOutcomeError):
    """Raised when binary outcome data lacks both classes required for fitting."""


class InvalidInterventionError(CounterError):
    """Raised when an intervention is not valid for the targeted decision type."""


class UnknownDecisionTypeError(CounterError, KeyError):
    """Raised when a decision type is not in the registered taxonomy."""


class DAGCycleError(CounterError):
    """Raised when a constructed DAG would contain a cycle."""
