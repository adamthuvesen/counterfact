"""Unicode label normalization for the unicode_normalize fixture."""

from __future__ import annotations


def dedupe_normalized(labels: list[str]) -> list[str]:
    """Return labels deduplicated by Unicode-normalized identity.

    BUGGY ON PURPOSE: this public-passing implementation handles only exact
    duplicates after whitespace trimming and ASCII-ish lowercase comparison.
    """
    seen: set[str] = set()
    out: list[str] = []
    for label in labels:
        cleaned = label.strip()
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(cleaned)
    return out
