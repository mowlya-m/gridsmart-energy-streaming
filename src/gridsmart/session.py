"""SparkSession factories for the batch and streaming halves of GridSmart."""

from __future__ import annotations

from pyspark import SparkConf
from pyspark.sql import SparkSession

from . import config


def batch_session(app_name: str = "GridSmart-Batch-Training") -> SparkSession:
    """Build the SparkSession used for model training.

    Two settings matter here.

    ``spark.sql.files.maxPartitionBytes`` is pinned to 32 MB.  The meters CSV
    is 626 MB; at Spark's 128 MB default it splits into five oversized
    partitions, and any single-core task holding one of those was enough to
    push the JVM into GC thrash on a laptop.  32 MB yields ~20 partitions
    that spread cleanly across cores.

    ``local[*]`` uses every available core.  On a memory-constrained machine,
    lowering this to ``local[4]`` trades wall-clock time for headroom.
    """
    conf = (
        SparkConf()
        .setAppName(app_name)
        .setMaster(config.BATCH_MASTER)
        .set("spark.sql.files.maxPartitionBytes", config.MAX_PARTITION_BYTES)
        # Spark 3.x reads pre-Gregorian timestamps written by older writers
        # as ambiguous; CORRECTED avoids a rebase exception on this dataset.
        .set("spark.sql.parquet.datetimeRebaseModeInWrite", "CORRECTED")
    )
    spark = SparkSession.builder.config(conf=conf).getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark


def streaming_session(app_name: str = "GridSmart-Streaming") -> SparkSession:
    """Build the SparkSession used for real-time inference.

    The session timezone is fixed to ``Australia/Melbourne``.  Without it
    Spark falls back to the JVM default, which differs between the Docker
    container (UTC) and a local machine. A silent timezone shift would move
    every reading into the wrong 6-hour dispatch window.

    ``shuffle.partitions`` drops from the default 200 to 4.  Each micro-batch
    carries 120 records; 200 partitions would mean 196 empty tasks per batch,
    and that scheduling overhead alone exceeded the trigger interval.
    """
    conf = (
        SparkConf()
        .setAppName(app_name)
        .setMaster(config.STREAM_MASTER)
        .set("spark.sql.session.timeZone", config.STREAM_TIMEZONE)
        .set("spark.sql.shuffle.partitions", str(config.SHUFFLE_PARTITIONS_STREAMING))
    )
    spark = SparkSession.builder.config(conf=conf).getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    config.PARQUET_DIR.mkdir(parents=True, exist_ok=True)
    return spark


def checkpoint_for(query_name: str) -> str:
    """Return a dedicated checkpoint directory for a named streaming query.

    Every query gets its own directory.  Sharing one causes Spark to
    interleave offset and state files from unrelated queries, and recovery
    after a restart then fails in ways that are extremely hard to diagnose.
    """
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in query_name)
    path = config.CHECKPOINT_DIR / safe
    path.mkdir(parents=True, exist_ok=True)
    return str(path)
