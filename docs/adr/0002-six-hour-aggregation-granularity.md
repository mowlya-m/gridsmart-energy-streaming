# ADR 0002, Aggregate energy to 6-hour dispatch windows

**Status:** Accepted · **Date:** 2025-09-12

## Context

The meters table holds hourly readings across four meter types per building:
roughly 19 million rows, 626 MB of CSV. Two questions had to be answered
before any modelling: what granularity to predict at, and what the target
variable actually is.

Hourly prediction is tempting because that is the data's native resolution.
But a grid operator does not dispatch generation hour by hour on this
horizon: supply is planned in blocks, and a forecast finer than the
decision it feeds is precision without value.

The dataset also records four separate meters per building (electricity,
chilled water, steam, hot water). Predicting any one of them in isolation
would understate total demand, sometimes by half at sites with chilled-water
cooling.

## Decision

Aggregate to the four fixed daily blocks, 00:00–05:59, 06:00–11:59,
12:00–17:59, 18:00–23:59. And sum all four meter types into a single
`agg_value` target.

## Consequences

- **Row count drops roughly 6×**, which is the difference between a model
  that trains on a laptop and one that does not.
- The forecast matches the operator's actual decision unit.
- Sub-hourly demand spikes inside a block are averaged away. Acceptable
  here: peak-shaving is a separate, shorter-horizon problem.
- Aggregating over meter types means the model cannot attribute demand to a
  specific energy carrier. Fine for a supply-planning forecast; it would not
  be for a fuel-switching model.
- The block boundaries are hard-coded rather than configurable. They come
  from the problem statement, and making them a parameter would invite
  silent mismatches between the trainer and the streaming job.
