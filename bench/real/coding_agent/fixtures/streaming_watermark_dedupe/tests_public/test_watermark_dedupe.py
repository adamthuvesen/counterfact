"""Public tests — in-order duplicate suppression only."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from watermark_dedupe import WatermarkDeduper  # noqa: E402


def test_in_order_duplicate_event_id_is_suppressed() -> None:
    deduper = WatermarkDeduper()
    events = [
        {"event_id": "a", "event_time": 1, "payload": {"value": "first"}},
        {"event_id": "b", "event_time": 2, "payload": {"value": "second"}},
        {"event_id": "a", "event_time": 3, "payload": {"value": "duplicate"}},
    ]

    assert deduper.process(events) == events[:2]


def test_emitted_order_matches_first_accepted_input_order() -> None:
    deduper = WatermarkDeduper()
    events = [
        {"event_id": "b", "event_time": 1, "payload": "beta"},
        {"event_id": "a", "event_time": 2, "payload": "alpha"},
        {"event_id": "c", "event_time": 3, "payload": "gamma"},
    ]

    assert deduper.process(events) == events
    assert deduper.process([events[1]]) == []
