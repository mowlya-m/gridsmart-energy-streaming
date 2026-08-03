"""Spark Structured Streaming: ingest, score, aggregate, and republish.

Topology
--------

    Kafka(weather5s)
        -> parse JSON against a strict schema
        -> watermark on event time (5s)
        -> join static building metadata (stream-static, broadcast)
        -> PipelineModel.transform  (feature steps + GBT, all fitted at training)
        -> three sinks:
             a) per-reading predictions                    trigger  5s
             b) 6-hour totals per building                 trigger  7s
             c) daily totals per site                      trigger 14s
        -> Parquet (durable handoff)
        -> Kafka(predictions_stream | sixhour_totals_stream | daily_totals_stream)

Parquet sits between the compute and the dashboard on purpose.  Writing
straight from Spark to the visualisation topic would couple a slow consumer
to the streaming job's back-pressure; Parquet decouples them and gives a
replayable record of what the model actually predicted.
"""

from __future__ import annotations

from pyspark.ml import PipelineModel
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from . import config
from .schemas import WEATHER_STREAM_SCHEMA
from .session import checkpoint_for

# --------------------------------------------------------------------------
# Ingest
# --------------------------------------------------------------------------


def read_weather_stream(spark: SparkSession, topic: str | None = None) -> DataFrame:
    """Subscribe to the weather topic and parse it into typed columns.

    ``startingOffsets="latest"`` means a restart picks up live traffic rather
    than replaying the whole retained log. That is correct for a dashboard,
    which cares about now rather than about history.

    Two distinct time columns are produced and they must not be confused:

    * ``timestamp``, when the weather was *measured* (2016 dataset time).
      Windowed aggregations use this, because a "6-hour interval" is a claim
      about the weather day.
    * ``event_time``: when the record was *emitted* by the producer. The
      watermark uses this, because lateness is a property of transport, not
      of the measurement.
    """
    raw = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", config.KAFKA_BOOTSTRAP)
        .option("subscribe", topic or config.TOPIC_WEATHER_IN)
        .option("startingOffsets", "latest")
        .option("failOnDataLoss", "false")
        .load()
    )

    return (
        raw.select(F.from_json(F.col("value").cast("string"), WEATHER_STREAM_SCHEMA).alias("payload"))
        .select("payload.*")
        .withColumn("timestamp", F.to_timestamp("timestamp"))
        .withColumn("event_time", F.to_timestamp(F.from_unixtime(F.col("weather_ts").cast("bigint"))))
        .withWatermark("event_time", config.WATERMARK_DELAY)
    )


def join_buildings(stream: DataFrame, buildings: DataFrame) -> DataFrame:
    """Enrich each weather reading with the buildings at that site.

    This is a stream-static join, and it fans out: one reading for a site
    becomes N rows, one per building there. That is intended: the model
    predicts per building, and every building at a site shares the same
    weather.

    The building table is small (a few thousand rows), so Spark broadcasts it
    to every executor and no shuffle occurs.
    """
    return stream.join(F.broadcast(buildings), on="site_id", how="left")


def score(stream: DataFrame, model_path: str | None = None) -> DataFrame:
    """Apply the trained pipeline to the enriched stream.

    ``PipelineModel.transform`` works unchanged on a streaming DataFrame
    because every stage is a pure transformer once fitted.  This is the
    payoff of keeping all fitted state inside the pipeline: the exact
    medians and category vocabularies from training are what score the
    stream, with no re-fitting and no drift.
    """
    model = PipelineModel.load(str(model_path or config.MODEL_DIR / "best_pipeline"))
    return model.transform(stream)


# --------------------------------------------------------------------------
# Aggregations
# --------------------------------------------------------------------------


def six_hour_totals(scored: DataFrame) -> DataFrame:
    """Total predicted energy per building, per 6-hour dispatch window."""
    return (
        scored.groupBy(F.window("timestamp", "6 hours").alias("w"), "building_id")
        .agg(F.sum("prediction").alias("total_energy_6h"))
        .select(
            F.col("w.start").alias("window_start"),
            F.col("w.end").alias("window_end"),
            "building_id",
            F.round("total_energy_6h", 3).alias("total_energy_6h"),
            F.current_timestamp().alias("gen_ts"),
        )
    )


def daily_site_totals(scored: DataFrame) -> DataFrame:
    """Total predicted energy per site, per day: the site-level supply plan."""
    return (
        scored.groupBy(F.window("timestamp", "1 day").alias("w"), "site_id")
        .agg(F.sum("prediction").alias("total_energy_day"))
        .select(
            F.col("w.start").alias("day_start"),
            F.col("w.end").alias("day_end"),
            "site_id",
            F.round("total_energy_day", 3).alias("total_energy_day"),
            F.current_timestamp().alias("gen_ts"),
        )
    )


def shortfall_by_site(predicted: DataFrame, metered: DataFrame) -> DataFrame:
    """Predicted minus actual metered energy, per site per day.

    Positive means the model over-forecast and the operator would have
    generated surplus; negative means a shortfall, which in a real grid is
    the expensive direction. It is met by spinning up peaking plant or, at
    the limit, by shedding load.

    Deliberately signed rather than absolute: the operator needs to know
    *which way* the error points, not just its size.
    """
    actual = (
        metered.withColumn("day", F.to_date("ts"))
        .groupBy("site_id", "day")
        .agg(F.sum(F.col("value").cast("double")).alias("metered_energy"))
    )
    forecast = predicted.withColumn("day", F.to_date("day_start")).select(
        "site_id", "day", F.col("total_energy_day").alias("predicted_energy")
    )
    return (
        forecast.join(actual, on=["site_id", "day"], how="inner")
        .withColumn("shortfall", F.col("predicted_energy") - F.col("metered_energy"))
        .orderBy("site_id", "day")
    )


# --------------------------------------------------------------------------
# Sinks
# --------------------------------------------------------------------------


def to_parquet(df: DataFrame, name: str, trigger: str, output_mode: str = "append"):
    """Persist a stream to Parquet with its own checkpoint directory.

    Aggregated streams use ``update`` mode via ``foreachBatch``.  Parquet's
    native sink only supports ``append``, which would refuse an aggregation
    whose windows are still open; ``foreachBatch`` writes each micro-batch as
    a normal batch write and sidesteps that restriction, at the cost of
    at-least-once rather than exactly-once semantics, acceptable for a
    dashboard, and the ``gen_ts`` column lets a consumer de-duplicate.
    """
    path = str(config.PARQUET_DIR / name)

    if output_mode == "append":
        return (
            df.writeStream.format("parquet")
            .option("path", path)
            .option("checkpointLocation", checkpoint_for(name))
            .outputMode("append")
            .trigger(processingTime=trigger)
            .queryName(name)
            .start()
        )

    def write_batch(batch_df: DataFrame, _epoch_id: int) -> None:
        batch_df.write.mode("append").parquet(path)

    return (
        df.writeStream.outputMode(output_mode)
        .foreachBatch(write_batch)
        .option("checkpointLocation", checkpoint_for(name))
        .trigger(processingTime=trigger)
        .queryName(name)
        .start()
    )


def parquet_to_kafka(spark: SparkSession, source_dir: str, topic: str, key_col: str, query_name: str):
    """Tail a Parquet directory and republish new rows to Kafka.

    The schema is bootstrapped from a static read of the directory rather
    than being hard-coded, so this helper works for all three output shapes.
    It therefore has to wait for the directory to exist: the upstream
    writer may not have flushed its first batch yet.
    """
    import os
    import time

    path = str(config.PARQUET_DIR / source_dir)
    while not os.path.exists(path):
        time.sleep(2)
    schema = spark.read.parquet(path).schema

    payload = (
        spark.readStream.format("parquet")
        .schema(schema)
        .load(path)
        .select(
            F.col(key_col).cast("string").alias("key"),
            F.to_json(F.struct("*")).alias("value"),
        )
    )

    return (
        payload.writeStream.format("kafka")
        .option("kafka.bootstrap.servers", config.KAFKA_BOOTSTRAP)
        .option("topic", topic)
        .option("checkpointLocation", checkpoint_for(query_name))
        .option("failOnDataLoss", "false")
        .outputMode("append")
        .queryName(query_name)
        .start()
    )


def stop_all(spark: SparkSession) -> None:
    """Stop every active streaming query. Use this before re-running cells."""
    for query in spark.streams.active:
        print(f"Stopping {query.name}")
        query.stop()
