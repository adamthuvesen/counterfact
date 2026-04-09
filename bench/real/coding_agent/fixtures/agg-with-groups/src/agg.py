"""Group-by aggregations over rows that mix Decimal and float values."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from decimal import Decimal


def group_sum(rows: Iterable[tuple[str, object]]) -> dict[str, object]:
    """Sum the values per group key.

    Inputs are (key, value) pairs where each value is either a `Decimal` or a
    `float`. Mixing the two within a group is allowed; the sum should preserve
    `Decimal` precision when every input is Decimal, and degrade to `float`
    only when at least one float appears in the group.

    Examples (Decimal-only group):
        rows = [('a', Decimal('0.1')), ('a', Decimal('0.2'))]
        group_sum(rows)  -> {'a': Decimal('0.3')}

    Examples (mixed group):
        rows = [('b', Decimal('1.5')), ('b', 2.5)]
        group_sum(rows)  -> {'b': 4.0}   # demoted to float, no exception
    """
    out: dict[str, object] = defaultdict(lambda: Decimal("0"))
    for key, value in rows:
        # BUG: blindly adds whatever to a Decimal accumulator.
        # `Decimal + float` raises TypeError; `Decimal + Decimal` is fine. The
        # fix needs to detect a mixed group and either pre-convert all values
        # to float or coerce Decimals to float on the fly.
        out[key] = out[key] + value
    return dict(out)
