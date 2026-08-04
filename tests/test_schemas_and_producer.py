"""Tests for schema contracts and the Kafka producer's replay logic.

The producer tests deliberately avoid touching Kafka. The timestamp-stamping
logic is the part with real behaviour worth testing; the network send is a
one-line library call.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest
from pyspark.sql.types import DecimalType, IntegerType, StringType

from gridsmart import config
from gridsmart.producer import build_payload, iter_batches, stamp_batch
from gridsmart.schemas import (
    BUILDINGS_SCHEMA,
    METERS_SCHEMA,
    WEATHER_SCHEMA,
    WEATHER_STREAM_SCHEMA,
)

# --------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------


def test_meter_value_is_decimal_not_double():
    """The metadata specifies decimal, so the schema must say decimal.

    Inference typed this column as an integer whenever the sampled rows
    happened to be whole numbers, silently truncating every fractional
    reading downstream.
    """
    value = METERS_SCHEMA["value"].dataType
    assert isinstance(value, DecimalType)
    assert value.scale >= 3, "energy readings carry three decimal places"


def test_weather_measures_are_decimal():
    """Every instrument reading is declared with fixed precision."""
    for column in ["air_temperature", "dew_temperature", "sea_level_pressure", "wind_speed"]:
        assert isinstance(WEATHER_SCHEMA[column].dataType, DecimalType), column


def test_integer_columns_stay_integer():
    """Counts and bearings are integers; declaring them decimal wastes space."""
    assert isinstance(WEATHER_SCHEMA["cloud_coverage"].dataType, IntegerType)
    assert isinstance(WEATHER_SCHEMA["wind_direction"].dataType, IntegerType)
    assert isinstance(BUILDINGS_SCHEMA["floor_count"].dataType, IntegerType)


def test_join_keys_are_non_nullable():
    """A NULL join key silently drops rows, so the schema forbids it."""
    assert BUILDINGS_SCHEMA["building_id"].nullable is False
    assert BUILDINGS_SCHEMA["site_id"].nullable is False
    assert METERS_SCHEMA["building_id"].nullable is False


def test_stream_schema_carries_weather_ts_as_int():
    """The streaming contract requires weather_ts to arrive as an Int."""
    assert "weather_ts" in WEATHER_STREAM_SCHEMA.fieldNames()
    assert isinstance(WEATHER_STREAM_SCHEMA["weather_ts"].dataType, IntegerType)


def test_stream_timestamp_arrives_as_string():
    """JSON has no timestamp type; it arrives as a String and is cast later."""
    assert isinstance(WEATHER_STREAM_SCHEMA["timestamp"].dataType, StringType)


def test_schemas_load_real_csv_headers(spark, tmp_path):
    """A schema mismatch against the real header order corrupts every column.

    Spark applies a supplied schema positionally, so a column added or
    reordered upstream would be read into the wrong field without error.
    """
    csv = tmp_path / "weather.csv"
    csv.write_text(
        "site_id,timestamp,air_temperature,cloud_coverage,dew_temperature,"
        "sea_level_pressure,wind_direction,wind_speed\n"
        "0,2016-01-01 00:00:00,25.0,6,20.0,1019.7,0,0.0\n"
    )

    df = spark.read.option("header", True).schema(WEATHER_SCHEMA).csv(str(csv))
    row = df.first()

    assert row["site_id"] == 0
    assert float(row["air_temperature"]) == pytest.approx(25.0)
    assert float(row["sea_level_pressure"]) == pytest.approx(1019.7)


# --------------------------------------------------------------------------
# Producer replay
# --------------------------------------------------------------------------


def _weather_frame(rows: int) -> pd.DataFrame:
    start = dt.datetime(2016, 1, 1)
    return pd.DataFrame(
        {
            "site_id": [0] * rows,
            "timestamp": [start + dt.timedelta(hours=i) for i in range(rows)],
            "air_temperature": [20.0] * rows,
            "cloud_coverage": [4] * rows,
            "dew_temperature": [15.0] * rows,
            "sea_level_pressure": [1013.0] * rows,
            "wind_direction": [180] * rows,
            "wind_speed": [3.0] * rows,
        }
    )


def test_batch_is_five_days_of_hourly_readings():
    """120 records per tick: 24 hourly readings x 5 days."""
    assert config.BATCH_SIZE == 120
    assert config.BATCH_SIZE == config.ROWS_PER_DAY * config.DAYS_PER_TICK


def test_each_day_block_advances_one_second():
    """Day N is stamped at base+N-1, compressing 5 days into 5 seconds."""
    batch = _weather_frame(config.BATCH_SIZE)
    stamped = stamp_batch(batch, base_epoch=1_737_810_000)

    for day in range(config.DAYS_PER_TICK):
        block = stamped.iloc[day * config.ROWS_PER_DAY : (day + 1) * config.ROWS_PER_DAY]
        assert set(block["weather_ts"]) == {1_737_810_000 + day}


def test_stamping_does_not_mutate_the_source():
    """The replay buffer must survive being stamped repeatedly."""
    batch = _weather_frame(config.BATCH_SIZE)
    stamp_batch(batch, 1_737_810_000)
    assert "weather_ts" not in batch.columns


def test_batches_advance_without_repeating():
    """Consecutive ticks must move the pointer forward, not resend."""
    weather = _weather_frame(config.BATCH_SIZE * 3)
    batches = iter_batches(weather)

    first = next(batches)
    second = next(batches)

    assert len(first) == config.BATCH_SIZE
    assert first.index[0] == 0
    assert second.index[0] == config.BATCH_SIZE


def test_replay_wraps_at_end_of_file():
    """At EOF the pointer resets so the simulation runs indefinitely."""
    weather = _weather_frame(int(config.BATCH_SIZE * 1.5))
    batches = iter_batches(weather)

    next(batches)
    wrapped = next(batches)

    assert wrapped.index[0] == 0
    assert len(wrapped) == config.BATCH_SIZE


def test_payload_preserves_nulls():
    """A missing reading is sent as JSON null, never zero-filled.

    Zero-filling at the producer would hide the gap from the imputer inside
    the pipeline, which is the component responsible for handling it.
    """
    frame = _weather_frame(1)
    frame.loc[0, "air_temperature"] = None
    frame.loc[0, "cloud_coverage"] = None

    payload = build_payload(frame.iloc[0], weather_ts=1_737_810_000)

    assert payload["air_temperature"] is None
    assert payload["cloud_coverage"] is None
    assert payload["wind_speed"] == pytest.approx(3.0)


def test_payload_has_every_stream_schema_field():
    """The producer must emit exactly what the consumer's schema expects."""
    payload = build_payload(_weather_frame(1).iloc[0], weather_ts=1_737_810_000)
    assert set(payload) == set(WEATHER_STREAM_SCHEMA.fieldNames())


def test_watermark_matches_producer_cadence():
    """The 5s watermark is tied to the 5s tick, so they must stay in step.

    If the tick interval changes without the watermark following, late data
    is either dropped too aggressively or state grows without bound.
    """
    assert config.WATERMARK_DELAY == f"{config.TICK_SECONDS} seconds"
