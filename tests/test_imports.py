"""Smoke test: package imports cleanly."""


def test_package_imports() -> None:
    import counter

    assert counter.__version__ == "0.0.0"


def test_errors_module_imports() -> None:
    from counter.errors import (
        CounterError,
        DAGCycleError,
        InvalidInterventionError,
        UnknownDecisionTypeError,
        UnsupportedOutcomeError,
    )

    assert issubclass(UnsupportedOutcomeError, CounterError)
    assert issubclass(InvalidInterventionError, CounterError)
    assert issubclass(UnknownDecisionTypeError, CounterError)
    assert issubclass(DAGCycleError, CounterError)
