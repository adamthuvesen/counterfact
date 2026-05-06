"""Fixed-window rate limit for the rate_limit fixture."""

from __future__ import annotations


def allow_request(
    user_id: str,
    now_s: int,
    history: list[tuple[str, int]],
    *,
    limit: int,
    window_s: int,
) -> bool:
    """Return whether `user_id` may make a request at `now_s`.

    BUGGY ON PURPOSE: this implementation assumes history is already sorted,
    treats the lower boundary as exclusive, and skips validation.
    """
    counted = 0
    lower = now_s - window_s
    for seen_user, timestamp in history:
        if timestamp < lower:
            break
        if seen_user == user_id and lower < timestamp <= now_s:
            counted += 1
    return counted < limit
