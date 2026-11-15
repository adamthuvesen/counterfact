"""Known-good implementation for fixture fairness tests."""

from __future__ import annotations

from copy import deepcopy
from math import inf


class WatermarkDeduper:
    def __init__(self, *, allowed_lateness: int = 0, ttl: int = 100) -> None:
        if allowed_lateness < 0:
            raise ValueError("allowed_lateness must be >= 0")
        if ttl < 1:
            raise ValueError("ttl must be >= 1")
        self.allowed_lateness = allowed_lateness
        self.ttl = ttl
        self._watermark = -inf
        self._seen: dict[str, int] = {}

    def process(self, events: list[dict[str, object]]) -> list[dict[str, object]]:
        emitted: list[dict[str, object]] = []
        for event in events:
            event_id, event_time = _parse_event(event)
            if event_time <= self._watermark:
                continue
            self._advance_watermark(event_time)
            if event_id in self._seen:
                continue
            self._seen[event_id] = event_time
            emitted.append(deepcopy(event))
            self._evict()
        return emitted

    def snapshot(self) -> dict[str, object]:
        return {
            "allowed_lateness": self.allowed_lateness,
            "ttl": self.ttl,
            "watermark": self._watermark,
            "seen": dict(self._seen),
        }

    def restore(self, snapshot: dict[str, object]) -> None:
        allowed_lateness = snapshot.get("allowed_lateness")
        ttl = snapshot.get("ttl")
        watermark = snapshot.get("watermark")
        seen = snapshot.get("seen")
        if not isinstance(allowed_lateness, int) or allowed_lateness < 0:
            raise ValueError("snapshot allowed_lateness must be >= 0")
        if not isinstance(ttl, int) or ttl < 1:
            raise ValueError("snapshot ttl must be >= 1")
        if not isinstance(watermark, (int, float)):
            raise ValueError("snapshot watermark must be numeric")
        if not isinstance(seen, dict):
            raise ValueError("snapshot seen must be a mapping")
        restored: dict[str, int] = {}
        for event_id, event_time in seen.items():
            if not isinstance(event_id, str) or not isinstance(event_time, int):
                raise ValueError("snapshot seen entries are invalid")
            restored[event_id] = event_time
        self.allowed_lateness = allowed_lateness
        self.ttl = ttl
        self._watermark = watermark
        self._seen = restored
        self._evict()

    def state_size(self) -> int:
        self._evict()
        return len(self._seen)

    def _advance_watermark(self, event_time: int) -> None:
        self._watermark = max(self._watermark, event_time - self.allowed_lateness)
        self._evict()

    def _evict(self) -> None:
        cutoff = self._watermark - self.ttl
        self._seen = {
            event_id: event_time
            for event_id, event_time in self._seen.items()
            if event_time > cutoff
        }


def _parse_event(event: dict[str, object]) -> tuple[str, int]:
    event_id = event.get("event_id")
    event_time = event.get("event_time")
    if not isinstance(event_id, str):
        raise ValueError("event_id must be a string")
    if not isinstance(event_time, int):
        raise ValueError("event_time must be an integer")
    return event_id, event_time
