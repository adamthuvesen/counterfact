"""Every acceptance test is slow, so a new one cannot drop out of `make check` unnoticed."""

from __future__ import annotations

from pathlib import Path

import pytest

HERE = Path(__file__).parent


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        if HERE in item.path.parents:
            item.add_marker(pytest.mark.slow)
