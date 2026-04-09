"""Row-level deduplication for the csv_dedupe fixture.

The shipped implementation is intentionally incomplete: it dedupes by exact
string equality only. The four normalization rules in spec.md are what the
agent must add (per the public/hidden test split).
"""

from __future__ import annotations


def dedupe(rows: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for row in rows:
        if row in seen:
            continue
        seen.add(row)
        out.append(row)
    return out
