"""Feature engineering shared by the batch trainer and the streaming scorer.

This module exists because of a specific failure mode.  In the original
two-notebook version of this project the feature logic was written twice:
once for training and once, from memory, for the streaming job.  The two
drifted: the streaming copy derived ``building_age`` while the trained
pipeline expected ``year_built``, and the model produced confident,
completely wrong predictions rather than raising an error.

Everything below is therefore written once and imported by both halves.  If a
transformation changes here, it changes everywhere.
"""

from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from .config import FEATURES

# --------------------------------------------------------------------------
# Target construction
# --------------------------------------------------------------------------

#: The four fixed daily blocks the grid operator plans against.
SIX_HOUR_INTERVALS = ["0:00-5:59", "6:00-11:59", "12:00-17:59", "18:00-23:59"]


def add_six_hour_interval(meters: DataFrame, ts_col: str = "ts") -> DataFrame:
    """Label each meter reading with the 6-hour block it falls into.

    Hourly granularity is finer than a grid operator can act on; dispatch
    decisions are made in blocks.  Bucketing to four blocks per day also cuts
    the row count by roughly 6x, which is what makes the model trainable on a
    laptop.

    Args:
    ----
        meters: Raw meter readings.
        ts_col: Name of the timestamp column.

    Returns:
    -------
        The input frame with an ``interval`` string column appended.

    """
    hour = F.hour(F.col(ts_col))
    return meters.withColumn(
        "interval",
        F.when(hour < 6, SIX_HOUR_INTERVALS[0])
        .when(hour < 12, SIX_HOUR_INTERVALS[1])
        .when(hour < 18, SIX_HOUR_INTERVALS[2])
        .otherwise(SIX_HOUR_INTERVALS[3]),
    )


def aggregate_energy(meters: DataFrame, ts_col: str = "ts") -> DataFrame:
    """Sum the four meter types into one energy figure per building-timestamp.

    The dataset records four separate meters per building (electricity,
    chilled water, steam, hot water).  The prediction target is their sum:
    total energy drawn by the building, which is what the grid must supply.

    Returns
    -------
        Columns ``building_id``, ``interval``, ``ts``, ``agg_value``.

    """
    return (
        add_six_hour_interval(meters, ts_col)
        .groupBy("building_id", "interval", ts_col)
        .agg(F.sum(F.col("value").cast("double")).alias("agg_value"))
    )


# --------------------------------------------------------------------------
# Weather cleaning
# --------------------------------------------------------------------------


def normalise_missing(weather: DataFrame, columns: list[str]) -> DataFrame:
    """Coerce the dataset's several flavours of "missing" into a true NULL.

    The raw weather CSV encodes absent readings three different ways: as an
    empty field, as the literal string ``"null"``, and, for instruments
    that failed: as ``0``. A sea level pressure of zero millibars is
    physically impossible, so treating it as data would drag the imputed mean
    far below the true value.  Spark's :class:`Imputer` only recognises
    genuine NULLs, so all three are converted here first.

    Note that ``wind_direction`` is exempt from the zero rule: 0 degrees is a
    valid bearing (due north), not a sensor fault.
    """
    zero_is_valid = {"wind_direction"}
    out = weather
    for column in columns:
        as_string = F.trim(F.col(column).cast("string"))
        cleaned = (
            F.when(F.col(column).isNull(), None)
            .when(F.lower(as_string) == "null", None)
            .when(as_string == "", None)
        )
        if column not in zero_is_valid:
            cleaned = cleaned.when(F.col(column).cast("double") == 0.0, None)
        out = out.withColumn(column, cleaned.otherwise(F.col(column).cast("double")))
    return out


def add_season_flag(weather: DataFrame, ts_col: str = "timestamp") -> DataFrame:
    """Tag each reading ``peak`` or ``off-peak`` from local temperature.

    The 16 sites span several countries and both hemispheres, so a calendar
    month means different things at different sites, July is peak cooling
    demand in one and peak heating demand in another.  Rather than hard-code
    hemispheres, the label is derived per site from that site's own observed
    temperatures: its three hottest and three coldest months are ``peak``,
    the remaining six are ``off-peak``.

    This makes the feature portable to any new site without configuration.

    Ranking uses ``dense_rank``, which numbers distinct values. If two months
    share an identical mean temperature they occupy the same rank, so a tie
    at the boundary widens the ``peak`` set beyond six months. That is
    preferred to an arbitrary tie-break, which would make the label depend on
    partition order and stop being reproducible. See
    ``tests/test_features.py::test_tied_monthly_means_all_rank_as_peak``.
    """
    monthly = (
        weather.withColumn("_month", F.month(ts_col))
        .groupBy("site_id", "_month")
        .agg(F.avg(F.col("air_temperature").cast("double")).alias("_avg_temp"))
    )

    hottest = Window.partitionBy("site_id").orderBy(F.col("_avg_temp").desc())
    coldest = Window.partitionBy("site_id").orderBy(F.col("_avg_temp").asc())

    labelled = (
        monthly.withColumn("_hot_rank", F.dense_rank().over(hottest))
        .withColumn("_cold_rank", F.dense_rank().over(coldest))
        .withColumn(
            "season_flag",
            F.when((F.col("_hot_rank") <= 3) | (F.col("_cold_rank") <= 3), "peak").otherwise("off-peak"),
        )
        .select("site_id", "_month", "season_flag")
    )

    return (
        weather.withColumn("_month", F.month(ts_col))
        .join(labelled, on=["site_id", "_month"], how="left")
        .drop("_month")
    )


def season_flag_from_month(df: DataFrame, month_col: str = "month") -> DataFrame:
    """Streaming fallback for :func:`add_season_flag`.

    The batch version needs a full year of history per site to rank months,
    which a stream does not have.  At inference time the per-site ranking is
    instead read from the lookup table persisted during training; this helper
    is the last-resort default used only when a site has never been seen
    before (a genuinely new building coming online mid-stream).
    """
    month = F.col(month_col)
    return df.withColumn(
        "season_flag",
        F.when(month.isin(12, 1, 2, 6, 7, 8), F.lit("peak")).otherwise(F.lit("off-peak")),
    )


# --------------------------------------------------------------------------
# The shared feature step
# --------------------------------------------------------------------------


def engineer(df: DataFrame, ts_col: str = "ts") -> DataFrame:
    """Derive every model input from a joined weather + building frame.

    This is the one function both the training notebook and the streaming job
    call.  It is deliberately free of any ``fit``: it only creates columns,
    so it behaves identically on a bounded DataFrame and on an unbounded
    streaming DataFrame.  All fitted state (imputation means, category
    indexes) lives inside the Spark ML pipeline instead, see
    :mod:`gridsmart.pipelines`.

    Args:
    ----
        df: A frame already joined to building metadata and weather.
        ts_col: Event-time column to derive calendar features from.

    Returns:
    -------
        The frame with every column named in ``config.FEATURES`` present.

    """
    out = (
        df
        # Calendar features. Energy demand is strongly diurnal and weekly;
        # `hour` alone carries more signal than any weather variable.
        .withColumn("hour", F.hour(ts_col))
        .withColumn("day_of_week", F.dayofweek(ts_col))
        .withColumn("month", F.month(ts_col))
        # Floor area is heavily right-skewed: a handful of campuses are
        # orders of magnitude larger than the median office. log1p keeps the
        # tail from dominating split selection while safely handling zeros.
        .withColumn("log_sqft", F.log1p(F.col("square_feet").cast("double")))
    )

    # Cast every declared numeric feature to double exactly once, here.
    for column in FEATURES.numeric:
        if column in out.columns:
            out = out.withColumn(column, F.col(column).cast("double"))

    return out


def select_model_inputs(df: DataFrame, keep: list[str] | None = None) -> DataFrame:
    """Project down to the feature contract plus optional trace columns.

    Trailing identifier columns (``building_id``, ``site_id``, ``ts``) are
    kept for joins and debugging but are never fed to the model, leaking an
    arbitrary building ID into a tree ensemble lets it memorise buildings
    instead of learning drivers.
    """
    trace = keep if keep is not None else ["building_id", "site_id", "ts"]
    wanted = [c for c in FEATURES.all_inputs + [FEATURES.label] + trace if c in df.columns]
    return df.select(*wanted)
