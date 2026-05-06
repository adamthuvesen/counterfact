# rate_limit — specification

## Function under specification

```python
def allow_request(
    user_id: str,
    now_s: int,
    history: list[tuple[str, int]],
    *,
    limit: int,
    window_s: int,
) -> bool:
    """Return whether `user_id` may make a request at `now_s`."""
```

## Semantics

The rate limit is a fixed look-back window over integer seconds.

A prior request counts against the limit iff:

- it belongs to the same `user_id`
- its timestamp is in the inclusive interval `[now_s - window_s, now_s]`

The request at `now_s` is allowed when the number of counted prior requests is
strictly less than `limit`.

`history` may contain requests for other users and may be unsorted.
Each history entry represents one prior request. Duplicate entries, including
multiple requests at the same second, count separately when they match the
target user and fall inside the window.

## Input validation

The function must raise `ValueError` when:

- `limit < 1`
- `window_s < 1`
- any history timestamp is greater than `now_s`

Validation applies to the whole input before deciding allow/deny. A future
timestamp is invalid even if enough earlier entries already determine the
answer.

## Worked examples

| Input | Output | Reason |
|---|---|---|
| `user_id="u1", now_s=100, history=[("u1", 91), ("u1", 99)], limit=3, window_s=10` | `True` | Two counted requests, limit is three. |
| `user_id="u1", now_s=100, history=[("u1", 90), ("u1", 99)], limit=2, window_s=10` | `False` | Boundary timestamp `90` is inside the inclusive window. |
| `user_id="u1", now_s=100, history=[("u2", 99), ("u1", 80)], limit=1, window_s=10` | `True` | Other users and old requests do not count. |
| `user_id="u1", now_s=100, history=[("u1", 99), ("u1", 99)], limit=2, window_s=10` | `False` | Duplicate same-second requests count as two requests. |

## Out of scope

- Floating-point timestamps.
- Distributed-clock reconciliation.
- Mutating or pruning `history`.
