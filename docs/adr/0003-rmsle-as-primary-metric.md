# ADR 0003, Select models on RMSLE, not RMSE

**Status:** Accepted · **Date:** 2025-09-15

## Context

Building energy consumption in this dataset spans several orders of
magnitude. A small retail unit and a university campus appear in the same
column, differing by a factor of thousands.

RMSE penalises absolute error. On a target shaped like this, RMSE is
dominated almost entirely by the largest buildings: a model can score well
by fitting the campuses and being wildly wrong on everything else. During
early runs, exactly this happened: the RMSE-best model had visibly poor
relative accuracy across the bottom two-thirds of the estate.

Spark ML ships RMSE, MSE, MAE and R², but not RMSLE.

## Decision

Use Root Mean Squared Logarithmic Error as the primary model-selection
metric, and implement it as a proper `Evaluator` subclass
(`gridsmart.metrics.RMSLEEvaluator`) rather than computing it after the fact.

```
ε = sqrt( (1/n) · Σ (log(pᵢ + 1) − log(aᵢ + 1))² )
```

RMSE, MAE and R² are still reported alongside it, RMSE in particular,
because "how many kWh are we typically out by" is the question an operator
asks. But they do not decide which model ships.

## Consequences

- Error is measured as a **ratio**, so a 20% miss counts the same on a small
  building as on a large one.
- Because it subclasses `Evaluator`, `CrossValidator` optimises directly
  against RMSLE. Had it been computed post-hoc, hyper-parameter search would
  have been maximising a different objective than model selection: a subtle
  and very common inconsistency.
- RMSLE is **asymmetric**: under-forecasting is penalised more than
  over-forecasting by the same ratio. Operationally correct for a grid, where
  over-generating wastes fuel but under-generating risks shedding load.
- The `+1` offset means the metric is only *asymptotically* scale-invariant;
  scores converge on log(2) for a 2× error as magnitude grows. Documented and
  pinned in `tests/test_metrics.py`.
- Negative predictions must be clipped to zero before the logarithm. Tree
  ensembles can extrapolate slightly below zero on sparse leaves, and a
  single such row would otherwise turn the whole aggregate into NaN.
