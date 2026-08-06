# ADR 0004: Ship the Gradient-Boosted Tree, keep Random Forest as baseline

**Status:** Accepted · **Date:** 2025-09-20

## Context

Two candidates were trained through identical feature pipelines on an
identical 80/20 split (`seed=2025`): a Random Forest and a Gradient-Boosted
Tree. Only the final estimator differed, so any performance gap is
attributable to the model rather than to preprocessing.

Comparing on headline RMSLE alone is weak evidence: the gap could easily be
split noise.

## Decision

Ship the GBT. Retain the Random Forest as a permanent baseline rather than
deleting it.

Support the choice with a **paired Wilcoxon signed-rank test** over
per-record squared log error and absolute percentage error, scoring both
models on the *same* test rows. Paired and non-parametric, because the error
distribution is heavily skewed and a t-test's normality assumption does not
hold here.

## Consequences

- The claim "GBT is better" rests on a per-record significance test, not on
  two aggregate numbers that might differ by noise.
- Boosting fits each tree to the residuals of the last, which suits this
  problem: the bulk of the signal is a smooth diurnal cycle and the
  interesting error lives in the weather-driven deviations from it.
- GBT trains sequentially and so parallelises worse than RF. That is
  acceptable here, since training is offline and happens once.
- GBT is more sensitive to hyper-parameters, which is why it, not RF, is the
  model taken forward into tuning (ADR 0006).
- Keeping RF in the repo means any future change to features can be
  re-measured against a stable reference point.
