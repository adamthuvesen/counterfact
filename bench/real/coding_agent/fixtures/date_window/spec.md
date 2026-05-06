# date_window — specification

## Function under specification

```python
def in_any_window(target_date: str, windows: list[tuple[str, str]]) -> bool:
    """Return whether target_date falls inside any date window."""
```

## Date format

All dates are strings in strict ISO calendar-date form: `YYYY-MM-DD`.
Implementations must reject malformed dates by raising `ValueError`. Examples
of malformed dates include missing zero padding (`2024-2-09`) and impossible
calendar dates (`2023-02-29`).

## Window semantics

Each window is a `(start, end)` pair using the same date format. Window
boundaries are inclusive: a target date equal to `start` or `end` is inside the
window.

Windows may be provided in any order. The function must check all windows, not
only the first matching sort position.

Every target date and every window endpoint must be parsed and validated. If any
window endpoint is malformed, or if any window has `start > end` after date
parsing, the input is invalid and the function must raise `ValueError` even when
an earlier window would otherwise contain the target date.

Leap days are valid only in leap years. For example, `2024-02-29` is valid and
can be inside a window; `2023-02-29` is invalid and must raise `ValueError`.

## Worked examples

| Input | Output | Reason |
|---|---|---|
| `target_date="2024-03-15", windows=[("2024-03-01", "2024-03-31")]` | `True` | Target is between start and end. |
| `target_date="2024-04-01", windows=[("2024-03-01", "2024-03-31")]` | `False` | Target is after the only window. |
| `target_date="2024-03-31", windows=[("2024-03-01", "2024-03-31")]` | `True` | End boundary is inclusive. |
| `target_date="2024-05-10", windows=[("2024-06-01", "2024-06-30"), ("2024-05-01", "2024-05-31")]` | `True` | Windows are not required to be sorted. |
| `target_date="2024-02-29", windows=[("2024-02-01", "2024-02-29")]` | `True` | 2024 is a leap year and end is inclusive. |
| `target_date="2025-01-02", windows=[("2024-12-30", "2025-01-03")]` | `True` | Windows can cross month and year boundaries. |

## Out of scope

- Time zones, timestamps, times of day, locale-specific calendars.
- Open-ended windows; both `start` and `end` are always required.
- Coercing malformed strings into dates.
