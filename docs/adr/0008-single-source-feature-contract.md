# ADR 0008: One feature module shared by training and inference

**Status:** Accepted · **Date:** 2026-07-28

## Context

In the first version of this project the feature logic existed twice: once
in the training notebook and once, rewritten from memory, in the streaming
job.

They drifted. The streaming copy derived `building_age` while the trained
pipeline expected `year_built`. The failure mode was the dangerous one: the
model did not error, it produced confident and completely wrong predictions.
It took a long time to find because every individual component looked
correct in isolation.

A related trap: the streaming version fitted its own `Imputer` on whatever
five days of weather were in the current micro-batch, rather than using the
medians learned at training time. That quietly shifted the model's input
distribution on every batch.

## Decision

1. All feature derivation lives in `gridsmart.features`, imported by both
   halves. The column list itself is declared once, in
   `gridsmart.config.FeatureContract`.
2. **All fitted state lives inside the `PipelineModel`**, imputation
   medians, string-index vocabularies, one-hot layouts. Nothing is fitted
   outside it. The streaming job calls `PipelineModel.load(...)` and gets
   training-day state exactly.
3. `features.engineer()` contains no `fit` and only creates columns, so it
   behaves identically on a bounded DataFrame and an unbounded streaming one.
4. `tests/test_features.py::test_engineer_produces_the_feature_contract`
   asserts the contract holds.

## Consequences

- Training/serving skew becomes structurally difficult rather than merely
  discouraged.
- Adding a feature means editing one list, not four notebooks.
- The notebooks import from `src/` instead of redefining logic inline. They
  read as narrative and analysis; the mechanics live in tested modules.
- All categorical stages use `handleInvalid="keep"`, so an unseen
  `primary_use` appearing at 03:00 routes to a dedicated bucket rather than
  killing the streaming job.
