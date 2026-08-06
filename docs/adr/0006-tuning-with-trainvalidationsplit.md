# ADR 0006: Tune with a bounded grid, on the winning model only

**Status:** Accepted · **Date:** 2025-09-22

## Context

Hyper-parameter search on Spark ML is expensive: every grid point is a full
pipeline fit over the training set. A wide grid crossed with k-fold
cross-validation multiplies out fast: a 5-parameter grid at k=3 was
projected at several hours on the available hardware.

## Decision

- Tune **only the GBT**, the model selected in ADR 0004. Tuning a model that
  will not ship is wasted compute.
- Bound the grid to 16 combinations across the three parameters that
  genuinely trade off against one another: tree capacity (`maxDepth`),
  learning rate (`stepSize`), and row sampling (`subsamplingRate`), plus
  `maxBins` for split resolution.
- Use `TrainValidationSplit` on a stratified sample for the exploratory
  sweep, then confirm the winner with `CrossValidator` on the full training
  set. TVS fits each grid point once instead of k times.
- Ship the tuned model **only if it beats the baseline on held-out RMSLE.**
  A tuned model that wins on validation and loses on test has overfitted the
  search, and shipping it would be worse than shipping the baseline.

## Consequences

- Search completes in a workable time on a laptop.
- The search optimises RMSLE directly, via `RMSLEEvaluator` (ADR 0003).
- Marginal returns beyond this grid were below the run-to-run noise floor,
  so a wider sweep would be measuring randomness.
- The explicit "only if it beats baseline" gate means a tuning run can
  legitimately conclude with no change, and that is recorded rather than
  quietly papered over.
