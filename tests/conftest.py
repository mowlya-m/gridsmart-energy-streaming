"""Shared pytest fixtures.

A single module-scoped local SparkSession is reused across the whole test
run. Starting a JVM costs several seconds, so a per-test session would make
the suite too slow to run on every commit.
"""

from __future__ import annotations

import pytest
from pyspark.sql import SparkSession


@pytest.fixture(scope="session")
def spark() -> SparkSession:
    """Return a minimal local SparkSession tuned for fast, small-data tests."""
    session = (
        SparkSession.builder.appName("gridsmart-tests")
        .master("local[2]")
        # Two shuffle partitions instead of 200: the test frames have a
        # handful of rows, and 198 empty tasks per shuffle dominate runtime.
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()
