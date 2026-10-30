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


_BLOCK_CHECKOUT_ONLY = """
import importlib, pkgutil, sys
import counterfact

class Block:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in {"bench", "scripts", "tests"}:
            raise ImportError(f"checkout-only module {name}")

sys.meta_path.insert(0, Block())
for module in pkgutil.walk_packages(counterfact.__path__, "counterfact."):
    try:
        importlib.import_module(module.name)
    except ImportError as exc:
        raise SystemExit(f"{module.name}: {exc}")
"""


def test_library_imports_without_checkout_only_modules() -> None:
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-c", _BLOCK_CHECKOUT_ONLY], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, (
        "The wheel ships src/counterfact only. Import bench/, scripts/, or tests/ inside the "
        f"function that needs them and catch ImportError: {result.stderr or result.stdout}"
    )
