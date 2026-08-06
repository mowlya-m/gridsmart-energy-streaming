# ADR 0007, Land predictions in Parquet before republishing to Kafka

**Status:** Accepted · **Date:** 2025-10-18

## Context

The dashboard consumes predictions from Kafka. The obvious topology is to
write straight from the Spark scoring job to the output topics.

That couples them. A slow or stalled dashboard consumer applies back-pressure
to the scoring job, and the scoring job is the component that must not fall
behind. It also leaves no record of what the model actually predicted, once
a message ages out of Kafka retention, it is gone, so a disputed forecast
cannot be reconstructed.

## Decision

Write every output stream to Parquet first. Then tail those Parquet
directories as a **second set of streaming reads** and republish to Kafka.

```
scored stream → Parquet (durable) → streaming read → Kafka → dashboard
```

Each query gets its own checkpoint directory under
`streamoutput/checkpoints/`.

## Consequences

- The scoring job and the dashboard are decoupled; neither can stall the
  other.
- Predictions are durable and replayable. A dashboard restarting mid-day can
  rebuild its full state from Parquet.
- Adds one hop of latency: a few seconds. Irrelevant for a dashboard whose
  own refresh is 5–14 s.
- Aggregated streams write via `foreachBatch` rather than the native Parquet
  sink, because that sink only supports `append` and would refuse an
  aggregation with open windows. The cost is at-least-once instead of
  exactly-once semantics; the `gen_ts` column lets a consumer de-duplicate.
- Every query needs its own checkpoint directory. Sharing one causes Spark
  to interleave offsets from unrelated queries, and recovery then fails in
  ways that are very hard to diagnose.
