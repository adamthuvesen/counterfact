"""Hidden tests — stateful stream categories named in spec.md."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from watermark_dedupe import WatermarkDeduper  # noqa: E402


def _event(event_id: str, event_time: int, payload: object | None = None) -> dict[str, object]:
    return {
        "event_id": event_id,
        "event_time": event_time,
        "payload": event_id if payload is None else payload,
    }


def test_event_time_watermark_advancement_drops_late_event() -> None:
    deduper = WatermarkDeduper(allowed_lateness=2, ttl=100)
    assert deduper.process([_event("new", 10)]) == [_event("new", 10)]

    assert deduper.process([_event("late", 8)]) == []


def test_late_event_dropping_does_not_poison_future_duplicate_state() -> None:
    deduper = WatermarkDeduper(allowed_lateness=1, ttl=100)
    deduper.process([_event("advance", 10)])

    assert deduper.process([_event("same-id", 9)]) == []
    assert deduper.process([_event("same-id", 11)]) == [_event("same-id", 11)]


def test_non_late_duplicate_still_advances_watermark() -> None:
    deduper = WatermarkDeduper(allowed_lateness=0, ttl=100)
    assert deduper.process([_event("dup", 10)]) == [_event("dup", 10)]

    assert deduper.process([_event("dup", 20)]) == []
    assert deduper.process([_event("fresh-but-now-late", 15)]) == []


def test_ttl_eviction_allows_reuse_after_retained_window() -> None:
    deduper = WatermarkDeduper(allowed_lateness=100, ttl=5)
    assert deduper.process([_event("id-1", 0)]) == [_event("id-1", 0)]
    assert deduper.process([_event("advance", 107)]) == [_event("advance", 107)]

    assert deduper.process([_event("id-1", 108)]) == [_event("id-1", 108)]


def test_checkpoint_restore_preserves_watermark_and_duplicate_state() -> None:
    original = WatermarkDeduper(allowed_lateness=2, ttl=100)
    original.process([_event("kept", 10)])
    snapshot = original.snapshot()

    restored = WatermarkDeduper()
    restored.restore(snapshot)

    assert restored.process([_event("kept", 11)]) == []
    assert restored.process([_event("late", 8)]) == []
    assert restored.process([_event("fresh", 12)]) == [_event("fresh", 12)]


def test_stable_emission_order_keeps_accepted_input_order_not_timestamp_sort() -> None:
    deduper = WatermarkDeduper(allowed_lateness=10, ttl=100)
    events = [
        _event("later", 10),
        _event("earlier", 9),
        _event("latest", 11),
    ]

    assert deduper.process(events) == events


def test_bounded_retained_state_after_watermark_advancement() -> None:
    deduper = WatermarkDeduper(allowed_lateness=50, ttl=3)
    for event_time in range(10):
        deduper.process([_event(f"id-{event_time}", event_time)])

    deduper.process([_event("advance", 60)])

    assert deduper.state_size() <= 4


def test_invalid_constructor_and_event_inputs_raise_value_error() -> None:
    with pytest.raises(ValueError):
        WatermarkDeduper(allowed_lateness=-1)
    with pytest.raises(ValueError):
        WatermarkDeduper(ttl=0)

    deduper = WatermarkDeduper()
    with pytest.raises(ValueError):
        deduper.process([{"event_id": 1, "event_time": 1, "payload": "bad"}])
    with pytest.raises(ValueError):
        deduper.process([{"event_id": "bad", "event_time": "1", "payload": "bad"}])
