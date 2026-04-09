"""US-style phone-number validation."""

from __future__ import annotations

import re

# Pre-compiled pattern for a 3-3-4 digit phone number with hyphens.
# BUG: missing end anchor. `re.match` only anchors at the start, so this
# silently accepts anything that *begins* with a valid phone. The fix needs
# either an explicit `$` / `\Z` in the pattern or a switch to `re.fullmatch`.
_PHONE = re.compile(r"\d{3}-\d{3}-\d{4}")


def is_phone_number(s: str) -> bool:
    """Return True iff `s` is exactly a US phone number like 555-123-4567.

    Examples:
        is_phone_number("555-123-4567")         -> True
        is_phone_number("x555-123-4567")        -> False
        is_phone_number("555-123-4567 ext 99")  -> False
        is_phone_number("555-123-456")          -> False
        is_phone_number("")                     -> False
    """
    return bool(_PHONE.match(s))
