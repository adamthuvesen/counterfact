"""Smoke test: package imports cleanly."""


def test_package_imports() -> None:
    from importlib.metadata import version

    import counterfact

    assert counterfact.__version__ == version("counterfact")


def test_errors_module_imports() -> None:
    from counterfact.errors import (
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
