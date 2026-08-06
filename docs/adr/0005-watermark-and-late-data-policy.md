# ADR 0005, Watermark at 5 seconds on producer event time

**Status:** Accepted · **Date:** 2025-10-14

## Context

The streaming job runs windowed aggregations (6-hour blocks per building,
daily totals per site). Windowed aggregation in Structured Streaming holds
state in memory until Spark can prove no further records will arrive for a
window. Without a watermark, Spark cannot prove that, so state grows
unbounded and the job eventually dies.

The complication is that there are two distinct notions of time in this
stream and conflating them produces silently wrong results:

- `timestamp`, when the weather was **measured** (2016 dataset time).
- `weather_ts` / `event_time`, when the producer **emitted** the record.

The producer compresses five days of weather into a five-second wall-clock
tick, stamping each 24-row day block one second apart.

## Decision

Watermark on `event_time` (derived from `weather_ts`) with a **5 second**
delay threshold, matching the producer's tick interval exactly. Window the
aggregations on `timestamp`, because a "6-hour interval" is a claim about
the weather day, not about transport.

The relationship is asserted in
`tests/test_schemas_and_producer.py::test_watermark_matches_producer_cadence`
so the two cannot drift apart.

## Consequences

- State is bounded: Spark releases a window's state once the watermark
  passes it.
- Records arriving more than 5 s late are dropped. Given the producer emits
  every 5 s, a record that late belongs to a superseded batch anyway.
- The threshold is **coupled** to the tick interval. Changing one without the
  other either drops in-order records or lets state grow, hence the test.
- The 7 s and 14 s aggregation triggers are deliberately longer than the
  5 s watermark, so each window has closed before it is printed.
