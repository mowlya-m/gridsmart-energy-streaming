# Architecture

> The full platform diagram is in the [README](../README.md).
> This document is the written detail behind it.

Two stages, connected by one artefact: the persisted `PipelineModel`.
Everything about the design follows from making that handoff safe.

---

## Stage 1, Batch training

```
CSV → typed ingest → transform → two pipelines → evaluate → tune → PipelineModel
```

**Typed ingest.** Explicit `StructType` per table, `DecimalType` where the
metadata says decimal, 32 MB partitions. No inference.

**Transform.** Meter readings summed across the four carriers and bucketed
into 6-hour dispatch windows to form the label. Weather gaps normalised and
imputed. Peak season derived per site from observed temperature.

**Two pipelines, identical except the final estimator.** Random Forest as
baseline, GBT as candidate. Because the feature stages are shared, any
performance gap is attributable to the model rather than to preprocessing.

**Evaluate.** RMSLE primary, with RMSE / MAE / R² reported alongside. Model
choice is backed by a **paired Wilcoxon signed-rank test** over per-record
errors on the same test rows, non-parametric, because the error distribution
is heavily skewed and a t-test's normality assumption does not hold.

**Tune.** Bounded grid on the winner only, optimising RMSLE directly. The
tuned model ships **only if it beats the baseline on held-out data**; a model
that wins on validation and loses on test has overfitted the search.

---

## The handoff

The trained `PipelineModel` contains every fitted stage: imputation medians,
string-index vocabularies, one-hot layouts, and the ensemble itself.

This is the whole point. The streaming job calls `PipelineModel.load(...)`
and gets training-day state exactly. Nothing is re-fitted at inference time.

The earlier version of this project fitted an `Imputer` inside the streaming
job, which meant it computed medians from whatever five days of weather were
in the current micro-batch, silently shifting the model's input distribution
every five seconds. → [ADR 0008](adr/0008-single-source-feature-contract.md)

---

## Stage 2, Real-time inference

```
producer → Kafka → parse → watermark → join → score → aggregate → Parquet → Kafka → dashboard
```

### Producer

Deliberately plain Python, no Spark. It stands in for a field weather
station, and those are small embedded controllers with no distributed
compute; using Spark would model the problem incorrectly and hide any
back-pressure the real system would experience.

Emits 120 records every 5 s, five days of hourly readings, 24 per day, with
each day's block stamped one second apart. A file pointer advances across
ticks so replay stays chronological and wraps at EOF.

### Watermark

Two clocks, and conflating them produces silently wrong results:

| Column | Meaning | Used for |
|---|---|---|
| `timestamp` | when the weather was **measured** | windowed aggregation |
| `event_time` | when the record was **emitted** | the watermark |

5 s delay, matching the producer tick exactly. Longer and window state grows
unbounded; shorter and in-order records get dropped. Asserted by a test.
→ [ADR 0005](adr/0005-watermark-and-late-data-policy.md)

### Stream-static join

Building metadata is small (~1,400 rows) so Spark broadcasts it and no
shuffle occurs. The join fans out: one weather reading for a site becomes
one row per building at that site. That is intended, since every building
there shares the same weather and the model predicts per building.

### Three aggregations, three cadences

| Output | Grain | Trigger |
|---|---|---|
| Live predictions | per reading | 5 s |
| 6-hour totals | per building | 7 s |
| Daily totals | per site | 14 s |

All three triggers exceed the 5 s watermark, so each window has closed before
it is printed.

### Parquet as the handoff

Writing straight from Spark to the dashboard's topics would couple them: a
slow consumer applies back-pressure to the scoring job, which is the
component that must not fall behind. It also leaves no record, once a
message ages out of retention, a disputed forecast cannot be reconstructed.

So every output lands in Parquet first, and those directories are then tailed
as a **second set of streaming reads** and republished to Kafka.

Aggregated streams write via `foreachBatch` rather than the native Parquet
sink, because that sink only supports `append` and would refuse an
aggregation with open windows. The cost is at-least-once rather than
exactly-once semantics; the `gen_ts` column lets a consumer de-duplicate.

Every query gets its own checkpoint directory. Sharing one causes Spark to
interleave offsets from unrelated queries, and recovery after restart then
fails in ways that are very hard to diagnose.
→ [ADR 0007](adr/0007-parquet-as-streaming-handoff.md)

---

## Failure modes considered

| Failure | Response |
|---|---|
| Late-arriving weather | Accepted within 5 s, dropped beyond: the record belongs to a superseded batch anyway |
| Unseen `primary_use` category | `handleInvalid="keep"` routes it to a dedicated bucket instead of killing the job |
| Negative model output | Clipped to 0 before the RMSLE logarithm; a negative energy draw is physically meaningless |
| Streaming query crash | Per-query checkpoints allow independent recovery from the last committed offset |
| Dashboard stalls | Parquet decouples it from the scoring job entirely |
| Producer restart mid-replay | Pointer state resumes in order rather than jumping to the first record |
