"""Tests for the custom RMSLE metric.

RMSLE is the metric this project selects models on, so it is the one piece
of maths worth pinning down with tests. Each case checks a property that
would otherwise fail silently in a way that still produces plausible-looking
numbers.
"""

from __future__ import annotations

import math

import pytest

from gridsmart.metrics import RMSLEEvaluator, regression_report, rmsle


def _frame(spark, pairs):
    """Build a two-column (label, prediction) DataFrame from tuples."""
    return spark.createDataFrame(pairs, "label double, prediction double")


def test_perfect_prediction_scores_zero(spark):
    """An exact fit must score exactly 0.0, not merely something small."""
    df = _frame(spark, [(100.0, 100.0), (5.0, 5.0), (0.0, 0.0)])
    assert rmsle(df) == pytest.approx(0.0, abs=1e-12)


def test_matches_hand_computed_value(spark):
    """Verify against the formula evaluated by hand in plain Python."""
    pairs = [(10.0, 12.0), (100.0, 90.0), (3.0, 4.0)]
    df = _frame(spark, pairs)

    expected = math.sqrt(sum((math.log1p(p) - math.log1p(a)) ** 2 for a, p in pairs) / len(pairs))
    assert rmsle(df) == pytest.approx(expected, rel=1e-9)


def test_zero_consumption_is_defined(spark):
    """An idle building (0 kWh) must not produce inf or NaN.

    This is the reason for the +1 inside each logarithm; log(0) would be
    undefined and a plain log-ratio metric would blow up here.
    """
    df = _frame(spark, [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0)])
    score = rmsle(df)
    assert math.isfinite(score)
    assert score > 0.0


def test_negative_predictions_are_clipped(spark):
    """A slightly negative tree output must clip to 0, not yield NaN.

    GBT can extrapolate below zero on sparse leaves. Without clipping,
    log1p(-3) is NaN and a single such row would poison the whole aggregate.
    """
    df = _frame(spark, [(10.0, -3.0), (20.0, 20.0)])
    clipped = _frame(spark, [(10.0, 0.0), (20.0, 20.0)])
    assert rmsle(df) == pytest.approx(rmsle(clipped), rel=1e-9)


def test_penalises_relative_not_absolute_error(spark):
    """The defining property: equal ratio error scores near-equally at any scale.

    A 2x over-prediction on a 10 kWh building and on a 10,000 kWh building is
    the same relative mistake. RMSE ranks the second as 1000x worse; RMSLE
    treats them as comparable. This is the entire reason RMSLE was chosen
    over RMSE for model selection on a target spanning several orders of
    magnitude.

    Note the metric is only *asymptotically* scale-invariant: the +1 offset
    inside each logarithm matters when values approach 1, so the two scores
    converge on log(2) as magnitude grows rather than matching exactly. The
    assertion below is therefore a bounded ratio, not equality.
    """
    small = rmsle(_frame(spark, [(10.0, 20.0)]))
    large = rmsle(_frame(spark, [(10_000.0, 20_000.0)]))

    # Same ratio error, magnitudes 1000x apart: scores stay within 10%.
    assert small == pytest.approx(large, rel=0.10)

    # ...and both converge on the theoretical limit of log(2).
    assert large == pytest.approx(math.log(2), rel=1e-3)


def test_asymptotic_invariance_tightens_with_magnitude(spark):
    """The larger the values, the closer two equal-ratio errors score.

    Documents the direction of the +1 offset's influence, so a future change
    to the metric that broke this would be caught.
    """
    limit = math.log(2)
    mid = rmsle(_frame(spark, [(1_000.0, 2_000.0)]))
    big = rmsle(_frame(spark, [(1_000_000.0, 2_000_000.0)]))

    assert abs(big - limit) < abs(mid - limit)


def test_under_prediction_penalised_more_than_over(spark):
    """RMSLE is asymmetric: under-forecasting hurts more.

    Operationally correct for a grid. Over-generating wastes fuel; under-
    generating risks shedding load.
    """
    under = _frame(spark, [(100.0, 50.0)])
    over = _frame(spark, [(100.0, 150.0)])
    assert rmsle(under) > rmsle(over)


def test_evaluator_minimises(spark):
    """Spark's tuning API must be told that lower is better."""
    evaluator = RMSLEEvaluator()
    assert evaluator.isLargerBetter() is False

    df = _frame(spark, [(10.0, 11.0), (20.0, 19.0)])
    assert evaluator.evaluate(df) == pytest.approx(rmsle(df), rel=1e-9)


def test_evaluator_honours_custom_column_names(spark):
    """Column names must be configurable, not hard-coded."""
    df = spark.createDataFrame([(10.0, 11.0)], "actual double, forecast double")
    evaluator = RMSLEEvaluator(labelCol="actual", predictionCol="forecast")
    assert evaluator.evaluate(df) > 0.0


def test_regression_report_returns_all_metrics(spark):
    """The report helper must return every metric the notebooks display."""
    df = _frame(spark, [(10.0, 11.0), (20.0, 18.0), (30.0, 31.0), (40.0, 44.0)])
    report = regression_report(df)

    assert set(report) == {"rmsle", "rmse", "mae", "r2"}
    assert all(math.isfinite(v) for v in report.values())
    # MAE never exceeds RMSE for the same errors (Jensen's inequality).
    assert report["mae"] <= report["rmse"] + 1e-9
