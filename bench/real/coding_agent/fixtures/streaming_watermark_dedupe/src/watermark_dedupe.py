"""Stateful streaming dedupe fixture."""

from __future__ import annotations

from copy import deepcopy


class WatermarkDeduper:
    """Deduplicate events by event_id.

    BUGGY ON PURPOSE: this public-passing implementation suppresses simple
    duplicates but ignores watermark advancement, late drops, TTL eviction, and
    meaningful checkpoint state.
    """

    def __init__(self, *, allowed_lateness: int = 0, ttl: int = 100) -> None:
        self.allowed_lateness = allowed_lateness
        self.ttl = ttl
        self._seen: set[str] = set()

    def process(self, events: list[dict[str, object]]) -> list[dict[str, object]]:
        emitted: list[dict[str, object]] = []
        for event in events:
            event_id = event["event_id"]
            if event_id in self._seen:
                continue
            self._seen.add(str(event_id))
            emitted.append(deepcopy(event))
        return emitted

    def snapshot(self) -> dict[str, object]:
        return {
            "allowed_lateness": self.allowed_lateness,
            "ttl": self.ttl,
            "seen": sorted(self._seen),
        }

    def restore(self, snapshot: dict[str, object]) -> None:
        self.allowed_lateness = int(snapshot["allowed_lateness"])
        self.ttl = int(snapshot["ttl"])
        self._seen = set(snapshot["seen"])

    def state_size(self) -> int:
        return len(self._seen)
