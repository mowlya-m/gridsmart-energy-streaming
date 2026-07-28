"""Central configuration for the GridSmart pipeline.

Every tunable value used by both the batch (training) and streaming
(inference) halves of the platform lives here.  Keeping them in one place
means the two halves cannot silently drift apart: a mismatch between the
feature columns used at training time and the ones produced at inference
time is the single most common cause of a silently broken ML pipeline.

Environment variables override the defaults so the same code runs unchanged
inside the Docker container, on a laptop, or in CI.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------
# Filesystem layout
# --------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.getenv("GRIDSMART_DATA_DIR", PROJECT_ROOT / "data"))
MODEL_DIR = Path(os.getenv("GRIDSMART_MODEL_DIR", PROJECT_ROOT / "models"))
STREAM_OUT_DIR = Path(os.getenv("GRIDSMART_STREAM_DIR", PROJECT_ROOT / "streamoutput"))

CHECKPOINT_DIR = STREAM_OUT_DIR / "checkpoints"
PARQUET_DIR = STREAM_OUT_DIR / "parquet_streams"

# Raw inputs (see data/README.md; these are not committed to git).
METERS_CSV = DATA_DIR / "meters.csv"
BUILDINGS_CSV = DATA_DIR / "building_information.csv"
WEATHER_CSV = DATA_DIR / "weather.csv"
NEW_METERS_CSV = DATA_DIR / "new_meters.csv"
NEW_BUILDINGS_CSV = DATA_DIR / "new_building_information.csv"

# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------

SEED = 2025
TRAIN_TEST_SPLIT = (0.8, 0.2)

# --------------------------------------------------------------------------
# Kafka topology
# --------------------------------------------------------------------------

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "kafka:9092")

TOPIC_WEATHER_IN = "weather5s"
TOPIC_PREDICTIONS = "predictions_stream"
TOPIC_SIXHOUR = "sixhour_totals_stream"
TOPIC_DAILY = "daily_totals_stream"

# --------------------------------------------------------------------------
# Streaming cadence
# --------------------------------------------------------------------------
# The producer replays five calendar days of hourly weather every five
# seconds.  24 readings/day x 5 days = 120 records per tick.  The watermark
# is deliberately pinned to the same five-second cadence: see
# docs/adr/0005-watermark-and-late-data-policy.md.

ROWS_PER_DAY = 24
DAYS_PER_TICK = 5
BATCH_SIZE = ROWS_PER_DAY * DAYS_PER_TICK
TICK_SECONDS = 5

WATERMARK_DELAY = "5 seconds"
TRIGGER_PREDICTIONS = "5 seconds"
TRIGGER_SIXHOUR = "7 seconds"
TRIGGER_DAILY = "14 seconds"

# --------------------------------------------------------------------------
# Feature contract
# --------------------------------------------------------------------------
# This is the authoritative list.  `pipelines.build_pipeline` reads it, and
# `features.engineer` guarantees to produce exactly these columns.  Adding a
# feature means editing one list, not four notebooks.


@dataclass(frozen=True)
class FeatureContract:
    """The columns a fitted pipeline expects to receive at inference time."""

    numeric: list[str] = field(
        default_factory=lambda: [
            "air_temperature",
            "dew_temperature",
            "sea_level_pressure",
            "wind_speed",
            "log_sqft",
            "floor_count",
            "year_built",
            "hour",
            "day_of_week",
            "month",
        ]
    )
    categorical: list[str] = field(
        default_factory=lambda: [
            "primary_use",
            "season_flag",
        ]
    )
    label: str = "label"

    @property
    def all_inputs(self) -> list[str]:
        """Every raw column the pipeline needs before any transformation."""
        return self.numeric + self.categorical


FEATURES = FeatureContract()

# --------------------------------------------------------------------------
# Spark tuning
# --------------------------------------------------------------------------
# 32 MB partitions keep individual tasks small enough to survive on a laptop
# JVM heap while still parallelising across all available cores.

MAX_PARTITION_BYTES = "32MB"
BATCH_MASTER = "local[*]"
STREAM_MASTER = "local[4]"
STREAM_TIMEZONE = "Australia/Melbourne"
SHUFFLE_PARTITIONS_STREAMING = 4
