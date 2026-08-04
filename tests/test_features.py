"""Tests for the shared feature engineering layer.

These matter more than the model tests. A bug here is silent: the pipeline
still trains, still scores, and still produces confident numbers, they are
just wrong. Every test below covers a mistake that was actually made during
development.
"""

from __future__ import annotations

import datetime as dt

import pytest
from pyspark.sql import functions as F

from gridsmart.features import (
    SIX_HOUR_INTERVALS,
    add_season_flag,
    add_six_hour_interval,
    aggregate_energy,
    engineer,
    normalise_missing,
)

# --------------------------------------------------------------------------
# 6-hour bucketing
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "hour,expected",
    [
        (0, "0:00-5:59"),  # lower boundary of first block
        (5, "0:00-5:59"),  # upper boundary of first block
        (6, "6:00-11:59"),  # boundary crossing, off-by-one lives here
        (11, "6:00-11:59"),
        (12, "12:00-17:59"),
        (17, "12:00-17:59"),
        (18, "18:00-23:59"),
        (23, "18:00-23:59"),  # last hour of the day must not wrap
    ],
)
def test_interval_boundaries(spark, hour, expected):
    """Each hour lands in exactly the block the specification names."""
    df = spark.createDataFrame([(1, dt.datetime(2016, 6, 15, hour, 30))], "building_id int, ts timestamp")
    assert add_six_hour_interval(df).first()["interval"] == expected


def test_every_hour_is_labelled(spark):
    """All 24 hours map to one of the four blocks, no NULLs, no fifth label."""
    rows = [(1, dt.datetime(2016, 6, 15, h, 0)) for h in range(24)]
    df = spark.createDataFrame(rows, "building_id int, ts timestamp")
    labelled = add_six_hour_interval(df)

    assert labelled.filter(F.col("interval").isNull()).count() == 0
    assert set(r["interval"] for r in labelled.collect()) == set(SIX_HOUR_INTERVALS)


def test_aggregate_sums_all_meter_types(spark):
    """The four meter types must sum into one figure, not be counted separately.

    The target is total energy drawn by the building. Summing only the
    electricity meter would understate demand by roughly half at sites with
    chilled-water cooling.
    """
    moment = dt.datetime(2016, 6, 15, 9, 0)
    rows = [
        (1, "e", moment, 10.0),
        (1, "c", moment, 20.0),
        (1, "s", moment, 5.0),
        (1, "h", moment, 15.0),
    ]
    df = spark.createDataFrame(rows, "building_id int, meter_type string, ts timestamp, value double")

    result = aggregate_energy(df).collect()
    assert len(result) == 1
    assert result[0]["agg_value"] == pytest.approx(50.0)


def test_aggregate_keeps_buildings_separate(spark):
    """Aggregation must group by building, never sum across the estate."""
    moment = dt.datetime(2016, 6, 15, 9, 0)
    rows = [(1, "e", moment, 10.0), (2, "e", moment, 99.0)]
    df = spark.createDataFrame(rows, "building_id int, meter_type string, ts timestamp, value double")

    totals = {r["building_id"]: r["agg_value"] for r in aggregate_energy(df).collect()}
    assert totals == {1: pytest.approx(10.0), 2: pytest.approx(99.0)}


# --------------------------------------------------------------------------
# Missing-value normalisation
# --------------------------------------------------------------------------


def test_all_missing_encodings_become_null(spark):
    """Empty string, the literal 'null', and 0 all collapse to a real NULL.

    Spark's Imputer only recognises genuine NULLs. A sea level pressure of
    0 mb is an instrument failure, not a reading, left as data it drags the
    imputed mean far below the true ~1013 mb.
    """
    rows = [("1013.2",), ("",), ("null",), ("0",), (None,)]
    df = spark.createDataFrame(rows, "sea_level_pressure string")

    cleaned = normalise_missing(df, ["sea_level_pressure"])
    values = [r["sea_level_pressure"] for r in cleaned.collect()]

    assert values[0] == pytest.approx(1013.2)
    assert values[1:] == [None, None, None, None]


def test_zero_wind_direction_is_kept(spark):
    """0 degrees is due north: a valid bearing, not a sensor fault.

    Blanket-nulling every zero would delete every genuine northerly reading.
    """
    df = spark.createDataFrame([("0",), ("180",)], "wind_direction string")
    values = [r["wind_direction"] for r in normalise_missing(df, ["wind_direction"]).collect()]
    assert values == [pytest.approx(0.0), pytest.approx(180.0)]


# --------------------------------------------------------------------------
# Season labelling
# --------------------------------------------------------------------------


def test_season_flag_labels_six_months_peak(spark):
    """Exactly three hottest plus three coldest months are flagged peak."""
    rows = [
        (1, dt.datetime(2016, month, 15), float(temp))
        for month, temp in zip(range(1, 13), [2, 4, 9, 15, 20, 26, 30, 29, 22, 16, 8, 3], strict=False)
    ]
    df = spark.createDataFrame(rows, "site_id int, timestamp timestamp, air_temperature double")

    flagged = add_season_flag(df)
    peak_months = sorted(
        r["timestamp"].month for r in flagged.filter(F.col("season_flag") == "peak").collect()
    )

    # Hottest: Jul(30), Aug(29), Jun(26). Coldest: Jan(2), Dec(3), Feb(4).
    assert peak_months == [1, 2, 6, 7, 8, 12]


def test_season_flag_is_per_site_not_global(spark):
    """Two sites with different climates must get different peak months.

    This is the whole reason the flag is derived from each site's own
    observed temperatures rather than hard-coded to the calendar: the 16
    sites span several countries, and a temperate site's extremes fall in
    completely different months from an equatorial monsoon site's.
    """
    # Temperate northern site: extremes in mid-summer and mid-winter.
    temperate = [
        (1, dt.datetime(2016, m, 15), float(t))
        for m, t in zip(range(1, 13), [0, 2, 8, 14, 20, 26, 30, 29, 22, 15, 7, 1], strict=False)
    ]
    # Equatorial monsoon site: hottest before the rains, coolest during them.
    monsoon = [
        (2, dt.datetime(2016, m, 15), float(t))
        for m, t in zip(
            range(1, 13),
            [26.5, 28.0, 33.0, 35.0, 34.0, 27.0, 25.5, 25.0, 26.0, 24.0, 23.0, 25.2],
            strict=False,
        )
    ]
    df = spark.createDataFrame(
        temperate + monsoon, "site_id int, timestamp timestamp, air_temperature double"
    )

    flagged = add_season_flag(df).filter(F.col("season_flag") == "peak")
    by_site: dict[int, list[int]] = {}
    for row in flagged.collect():
        by_site.setdefault(row["site_id"], []).append(row["timestamp"].month)

    # Temperate: hottest Jul/Aug/Jun, coldest Jan/Dec/Feb.
    assert sorted(by_site[1]) == [1, 2, 6, 7, 8, 12]
    # Monsoon: hottest Apr/May/Mar, coldest Nov/Oct/Aug.
    assert sorted(by_site[2]) == [3, 4, 5, 8, 10, 11]
    assert sorted(by_site[1]) != sorted(by_site[2])


def test_tied_monthly_means_all_rank_as_peak(spark):
    """Documented behaviour: ranking uses dense_rank, so ties widen the label.

    ``dense_rank`` numbers *distinct values*, so three months sharing one
    mean temperature all occupy rank 1 and the next distinct values take
    ranks 2 and 3. Five months therefore satisfy ``cold_rank <= 3`` here
    instead of three.

    This is deliberate: an arbitrary tie-break would make the label depend
    on partition order and therefore be non-reproducible. Exact ties are
    vanishingly unlikely on real float monthly averages, but the behaviour is
    pinned here so it cannot change unnoticed.
    """
    temps = [1.0, 1.0, 1.0, 15.0, 16.0, 17.0, 30.0, 29.0, 28.0, 18.0, 19.0, 20.0]
    rows = [(1, dt.datetime(2016, m, 15), t) for m, t in zip(range(1, 13), temps, strict=False)]
    df = spark.createDataFrame(rows, "site_id int, timestamp timestamp, air_temperature double")

    peak = sorted(
        r["timestamp"].month for r in add_season_flag(df).filter(F.col("season_flag") == "peak").collect()
    )
    # Cold ranks 1-3 = {Jan,Feb,Mar} (tied), Apr, May. Hot ranks 1-3 = Jul, Aug, Sep.
    assert peak == [1, 2, 3, 4, 5, 7, 8, 9]


def test_no_ties_yields_exactly_six_peak_months(spark):
    """With distinct monthly means (the realistic case) exactly 6 are peak."""
    temps = [2.0, 4.0, 9.0, 15.0, 20.0, 26.0, 30.0, 29.0, 22.0, 16.0, 8.0, 3.0]
    rows = [(1, dt.datetime(2016, m, 15), t) for m, t in zip(range(1, 13), temps, strict=False)]
    df = spark.createDataFrame(rows, "site_id int, timestamp timestamp, air_temperature double")

    flagged = add_season_flag(df)
    assert flagged.filter(F.col("season_flag") == "peak").count() == 6
    assert flagged.filter(F.col("season_flag") == "off-peak").count() == 6


# --------------------------------------------------------------------------
# The shared engineer() step
# --------------------------------------------------------------------------


def test_engineer_produces_the_feature_contract(spark):
    """Every column the pipeline expects must exist after engineer().

    This is the guard against the training/serving skew that motivated
    extracting this module in the first place.
    """
    df = spark.createDataFrame(
        [(1, 1, dt.datetime(2016, 6, 15, 14, 0), 50_000, "Office", 20.5, 12.0, 1013.0, 3.2, 3, 1990)],
        "site_id int, building_id int, ts timestamp, square_feet int, primary_use string, "
        "air_temperature double, dew_temperature double, sea_level_pressure double, "
        "wind_speed double, floor_count int, year_built int",
    )

    result = engineer(df)
    for column in ["hour", "day_of_week", "month", "log_sqft"]:
        assert column in result.columns, f"engineer() failed to produce {column}"


def test_calendar_features_are_correct(spark):
    """Derived time columns must match the event timestamp exactly."""
    df = spark.createDataFrame(
        [(1, dt.datetime(2016, 6, 15, 14, 30), 50_000)],
        "building_id int, ts timestamp, square_feet int",
    )
    row = engineer(df).first()

    assert row["hour"] == 14
    assert row["month"] == 6
    assert row["day_of_week"] == 4  # Spark: Sunday=1, so Wednesday=4


def test_log_sqft_handles_zero_area(spark):
    """log1p must survive a zero or missing floor area without producing -inf."""
    df = spark.createDataFrame(
        [(1, dt.datetime(2016, 6, 15), 0), (2, dt.datetime(2016, 6, 15), None)],
        "building_id int, ts timestamp, square_feet int",
    )
    values = [r["log_sqft"] for r in engineer(df).collect()]

    assert values[0] == pytest.approx(0.0)  # log1p(0) == 0, not -inf
    assert values[1] is None  # NULL stays NULL for the imputer to handle
