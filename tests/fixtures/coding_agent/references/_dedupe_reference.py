"""Reference implementation for csv_dedupe.

This file ships alongside the buggy `dedupe.py` for two reasons:

1. The fixture's tests must be demonstrably satisfiable — the test suite
   asserts that swapping this in makes both tests_public/ and tests_hidden/
   pass.
2. Reviewers can read this file to see what `spec.md` actually demands.

It is NOT placed under any test path, NOT imported by `dedupe.py`, and the
agent never sees it (the runner snapshots only `dedupe.py` from `src/`).
"""

from __future__ import annotations

import unicodedata


def _normalize(s: str) -> str:
    # Rule 1: BOM strip (file-level marker — removed first).
    if s.startswith("﻿"):
        s = s.lstrip("﻿")
    # Rule 2: outer-whitespace strip.
    s = s.strip()
    # Rule 3: case fold.
    s = s.casefold()
    # Rule 4: Unicode NFC.
    s = unicodedata.normalize("NFC", s)
    return s


def dedupe(rows: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for row in rows:
        key = _normalize(row)
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out
