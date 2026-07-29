"""Explicit Spark schemas for every dataset in the platform.

Schema inference is never used.  Two reasons:

1.  **Correctness.**  Inference samples the file and guesses.  On the meters
    table (626 MB, roughly 19 M rows) a guess made from the first few
    thousand rows silently mistyped ``value`` as an integer whenever the
    sample happened to miss the decimal readings.

2.  **Speed.**  Inference requires a full extra pass over the data before the
    real job starts.  On the meters CSV that pass alone cost roughly a
    minute per run.

The metadata supplied with the dataset describes ``value``, the latent
building features, and the weather measurements as *decimal*.  They are
declared as :class:`DecimalType` here rather than the looser
:class:`DoubleType`, so that precision is fixed by the schema instead of by
whatever the reader happened to infer.
"""

from __future__ import annotations

from pyspark.sql.types import (
    DecimalType,
    IntegerType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

# Energy readings can reach six figures and are recorded to three decimal
# places; (18, 3) leaves generous headroom without wasting storage.
ENERGY_DECIMAL = DecimalType(18, 3)

# Weather instruments report one or two decimals (temperatures) or up to two
# (sea level pressure in millibars, e.g. 1013.25).
MEASURE_DECIMAL = DecimalType(10, 2)

# Latent building features are opaque, unit-less floats.
LATENT_DECIMAL = DecimalType(18, 6)


METERS_SCHEMA = StructType(
    [
        StructField("building_id", IntegerType(), nullable=False),
        StructField("meter_type", StringType(), nullable=True),
        StructField("ts", TimestampType(), nullable=False),
        StructField("value", ENERGY_DECIMAL, nullable=True),
        StructField("row_id", IntegerType(), nullable=True),
    ]
)


BUILDINGS_SCHEMA = StructType(
    [
        StructField("site_id", IntegerType(), nullable=False),
        StructField("building_id", IntegerType(), nullable=False),
        StructField("primary_use", StringType(), nullable=True),
        StructField("square_feet", IntegerType(), nullable=True),
        StructField("floor_count", IntegerType(), nullable=True),
        StructField("row_id", IntegerType(), nullable=True),
        StructField("year_built", IntegerType(), nullable=True),
        StructField("latent_y", LATENT_DECIMAL, nullable=True),
        StructField("latent_s", LATENT_DECIMAL, nullable=True),
        StructField("latent_r", LATENT_DECIMAL, nullable=True),
    ]
)


WEATHER_SCHEMA = StructType(
    [
        StructField("site_id", IntegerType(), nullable=False),
        StructField("timestamp", TimestampType(), nullable=False),
        StructField("air_temperature", MEASURE_DECIMAL, nullable=True),
        StructField("cloud_coverage", IntegerType(), nullable=True),
        StructField("dew_temperature", MEASURE_DECIMAL, nullable=True),
        StructField("sea_level_pressure", MEASURE_DECIMAL, nullable=True),
        StructField("wind_direction", IntegerType(), nullable=True),
        StructField("wind_speed", MEASURE_DECIMAL, nullable=True),
    ]
)


# The Kafka payload is the weather schema plus the producer-stamped ingestion
# time.  Per the streaming contract, everything arrives as a String except
# `weather_ts`, which the producer writes as a Unix epoch Int.
WEATHER_STREAM_SCHEMA = StructType(
    list(WEATHER_SCHEMA.fields[:1])
    + [StructField("timestamp", StringType(), nullable=True)]
    + list(WEATHER_SCHEMA.fields[2:])
    + [StructField("weather_ts", IntegerType(), nullable=True)]
)


#: Weather columns eligible for mean imputation.
IMPUTABLE_WEATHER_COLS = [
    "air_temperature",
    "cloud_coverage",
    "dew_temperature",
    "sea_level_pressure",
    "wind_direction",
    "wind_speed",
]


def describe(schema: StructType) -> str:
    """Render a schema as a readable one-line-per-field table.

    Used in the notebooks instead of ``printSchema()`` so that the rendered
    output stays legible in the exported PDF.
    """
    width = max(len(f.name) for f in schema.fields)
    lines = [
        f"{f.name:<{width}}  {f.dataType.simpleString():<16}  nullable={f.nullable}" for f in schema.fields
    ]
    return "\n".join(lines)
