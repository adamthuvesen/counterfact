# streaming_watermark_dedupe — specification

## Class under specification

```python
class WatermarkDeduper:
    def __init__(self, *, allowed_lateness: int = 0, ttl: int = 100) -> None: ...
    def process(self, events: list[dict[str, object]]) -> list[dict[str, object]]: ...
    def snapshot(self) -> dict[str, object]: ...
    def restore(self, snapshot: dict[str, object]) -> None: ...
    def state_size(self) -> int: ...
```

## Events

Each event is a dictionary with:

- `event_id`: a string identity used for deduplication
- `event_time`: an integer event-time timestamp
- `payload`: any JSON-compatible payload value

The deduper is deterministic and must not read wall-clock time.

## Semantics

`process()` consumes events in input order and returns the newly emitted events
in that same stable order. It must not mutate input event dictionaries.

An event is emitted iff all of these are true:

- its `event_time` is not late relative to the current event-time watermark
- its `event_id` has not already been seen in retained state

The event-time watermark starts at negative infinity. After each non-late event,
advance the watermark to at least `event_time - allowed_lateness`; the watermark
must never move backward. An event with `event_time <= watermark` is late and
must be dropped without changing retained dedupe state.

Retained dedupe state is memory-bounded by `ttl`. After each watermark
advancement, event IDs last seen at or before `watermark - ttl` must be evicted.
Once an ID is evicted, a later non-late event with the same ID may be emitted
again.

`snapshot()` returns a dictionary containing enough state to continue exactly:
the current watermark, retained event IDs and their last event times, and the
constructor settings. `restore()` replaces the current object state with such a
snapshot. Restoring must preserve checkpoint behavior for duplicate suppression,
late-event dropping, TTL eviction, stable emission order, and bounded retained
state.

`state_size()` returns the number of retained event IDs after TTL eviction.

## Validation

Constructors and event dictionaries must be validated:

- `allowed_lateness` must be `>= 0`
- `ttl` must be `>= 1`
- `event_id` must be a string
- `event_time` must be an integer

Invalid inputs must raise `ValueError`.

## Out of scope

- Serialization beyond the plain `snapshot()` dictionary.
- External streaming systems, clocks, queues, databases, or threads.
- Reordering emitted events by timestamp; output order is the accepted input
  order.
