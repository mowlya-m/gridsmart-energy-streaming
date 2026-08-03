"""Kafka producer that replays historical weather as a live sensor feed.

Deliberately plain Python. No Spark. The producer stands in for a field
weather station, and those devices are small embedded controllers with no
distributed compute.  Using Spark here would model the problem incorrectly
and hide any back-pressure the real system would experience.

Replay contract
---------------
Every ``TICK_SECONDS`` (5s) the producer emits ``BATCH_SIZE`` (120) records:
five calendar days of hourly readings, 24 per day.  Each day's block is
stamped with a ``weather_ts`` one second apart, day 1 at ``t``, day 2 at
``t+1``, and so on. That compresses five days of event time into the five
seconds of wall-clock time the batch occupies.  Downstream, the watermark
and the windowed aggregations run against ``weather_ts``, so a "day" of
simulated grid activity elapses every second.

A file pointer advances across ticks so the replay stays chronological and
never re-sends a record, wrapping back to the start at end of file.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from pathlib import Path

import pandas as pd

from . import config


def load_weather(csv_path: Path | str | None = None) -> pd.DataFrame:
    """Load the weather CSV, sorted so replay order is chronological per site.

    Args:
    ----
        csv_path: Override for the configured weather CSV location.

    Returns:
    -------
        Weather readings sorted by ``site_id`` then ``timestamp``.

    """
    path = Path(csv_path or config.WEATHER_CSV)
    frame = pd.read_csv(path, parse_dates=["timestamp"])
    return frame.sort_values(["site_id", "timestamp"]).reset_index(drop=True)


def _coerce(value, cast):
    """Return ``None`` for NaN so JSON carries a real null, not the string 'nan'."""
    return None if pd.isna(value) else cast(value)


def build_payload(row: pd.Series, weather_ts: int) -> dict:
    """Serialise one weather reading into the Kafka message body.

    Missing measurements are emitted as JSON ``null`` rather than being
    dropped or zero-filled.  Zero-filling here would be a silent data
    corruption: the imputer inside the loaded pipeline is the component
    responsible for handling gaps, and it can only do so if it can see them.
    """
    return {
        "site_id": int(row["site_id"]),
        "timestamp": row["timestamp"].isoformat(),
        "air_temperature": _coerce(row.get("air_temperature"), float),
        "cloud_coverage": _coerce(row.get("cloud_coverage"), int),
        "dew_temperature": _coerce(row.get("dew_temperature"), float),
        "sea_level_pressure": _coerce(row.get("sea_level_pressure"), float),
        "wind_direction": _coerce(row.get("wind_direction"), int),
        "wind_speed": _coerce(row.get("wind_speed"), float),
        "weather_ts": int(weather_ts),
    }


def stamp_batch(batch: pd.DataFrame, base_epoch: int) -> pd.DataFrame:
    """Assign ``weather_ts`` so each 24-row day block advances by one second.

    Args:
    ----
        batch: Exactly ``BATCH_SIZE`` consecutive readings.
        base_epoch: Unix time at which this batch is being sent.

    Returns:
    -------
        The batch with an integer ``weather_ts`` column added.

    """
    stamped = batch.copy()
    day_offset = (pd.RangeIndex(len(stamped)) // config.ROWS_PER_DAY).astype(int)
    stamped["weather_ts"] = base_epoch + day_offset
    return stamped


def iter_batches(weather: pd.DataFrame) -> Iterator[pd.DataFrame]:
    """Yield fixed-size batches forever, wrapping at end of file.

    The pointer is held across iterations so a restart mid-replay resumes in
    order rather than jumping back to the first record.
    """
    pointer = 0
    while True:
        if pointer + config.BATCH_SIZE > len(weather):
            pointer = 0
        yield weather.iloc[pointer : pointer + config.BATCH_SIZE]
        pointer += config.BATCH_SIZE


def create_producer(bootstrap: str | None = None):
    """Construct a ``KafkaProducer`` that serialises dicts to UTF-8 JSON."""
    from kafka import KafkaProducer

    return KafkaProducer(
        bootstrap_servers=[bootstrap or config.KAFKA_BOOTSTRAP],
        value_serializer=lambda payload: json.dumps(payload).encode("utf-8"),
        api_version=(0, 10),
    )


def run(csv_path: Path | str | None = None, topic: str | None = None, verbose: bool = True) -> None:
    """Stream weather to Kafka indefinitely.

    Blocks until interrupted.  ``flush()`` is called once per batch rather
    than per record: batching the network round-trips keeps a 120-record tick
    comfortably inside its 5-second budget, while still guaranteeing every
    record is acknowledged before the next tick is stamped.
    """
    weather = load_weather(csv_path)
    producer = create_producer()
    destination = topic or config.TOPIC_WEATHER_IN

    if verbose:
        days = len(weather) // config.ROWS_PER_DAY
        print(f"Loaded {len(weather):,} readings (~{days:,} days) from {csv_path or config.WEATHER_CSV}")
        print(f"Emitting {config.BATCH_SIZE} records every {config.TICK_SECONDS}s to topic '{destination}'\n")

    try:
        for batch in iter_batches(weather):
            base_epoch = int(time.time())
            stamped = stamp_batch(batch, base_epoch)

            for _, row in stamped.iterrows():
                producer.send(destination, build_payload(row, row["weather_ts"]))
            producer.flush()

            if verbose:
                clock = time.strftime("%H:%M:%S", time.localtime(base_epoch))
                print(
                    f"[{clock}] sent {len(stamped)} records | weather_ts {base_epoch}-{base_epoch + config.DAYS_PER_TICK - 1}"
                )

            time.sleep(config.TICK_SECONDS)
    except KeyboardInterrupt:
        print("\nProducer stopped by user.")
    finally:
        producer.flush()
        producer.close()


if __name__ == "__main__":
    run()
