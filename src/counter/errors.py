"""Exception types raised by counter."""


class CounterError(Exception):
    """Base class for counter errors."""


class UnsupportedOutcomeError(CounterError):
    """Raised when an Outcome.kind is not supported by the v0 runtime."""


class InvalidInterventionError(CounterError):
    """Raised when an intervention is not valid for the targeted decision type."""


class UnknownDecisionTypeError(CounterError, KeyError):
    """Raised when a decision type is not in the registered taxonomy."""


class DAGCycleError(CounterError):
    """Raised when a constructed DAG would contain a cycle."""
