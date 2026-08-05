"""Tests for pipeline construction.

These check the *shape* of the pipeline rather than its predictive accuracy.
Shape is what the streaming job depends on: it loads a fitted pipeline and
expects a specific set of stages in a specific order. A pipeline that trains
fine but has the wrong stage order produces wrong predictions silently, which
is the failure class the whole package layout exists to prevent.

Fitting a real ensemble takes minutes, so these build unfitted pipelines and
inspect them. The one test that does fit uses a handful of rows and a
single-tree forest.
"""

from __future__ import annotations

import pytest
from pyspark.ml import Pipeline
from pyspark.ml.feature import Imputer, OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml.regression import GBTRegressor, RandomForestRegressor

from gridsmart.config import FEATURES, SEED
from gridsmart.pipelines import (
    build_feature_stages,
    build_gbt,
    build_pipeline,
    build_random_forest,
)


@pytest.fixture(autouse=True)
def _active_context(spark):
    """Spark ML estimators need a live SparkContext even to be constructed.

    Every test here builds one, so the session fixture is pulled in
    automatically rather than being listed on each test signature.
    """
    return spark


# --------------------------------------------------------------------------
# Stage composition
# --------------------------------------------------------------------------


def test_stage_order_is_impute_index_encode_assemble():
    """Order is load-bearing, not cosmetic.

    The assembler consumes the encoder's output, which consumes the indexer's,
    which consumes the imputer's. Any reordering breaks the column wiring.
    """
    stages = build_feature_stages()

    assert isinstance(stages[0], Imputer)
    assert all(isinstance(s, StringIndexer) for s in stages[1 : 1 + len(FEATURES.categorical)])
    assert isinstance(stages[-2], OneHotEncoder)
    assert isinstance(stages[-1], VectorAssembler)


def test_imputer_covers_every_numeric_feature():
    """A numeric feature missing from the imputer carries NULLs into the model."""
    imputer = build_feature_stages()[0]
    assert set(imputer.getInputCols()) == set(FEATURES.numeric)


def test_imputer_uses_median_not_mean():
    """Median resists the instrument spikes present in the weather series.

    A single corrupted pressure reading of 0 or 9999 shifts a mean but barely
    moves a median. See docs/adr/0008.
    """
    assert build_feature_stages()[0].getStrategy() == "median"


def test_every_categorical_column_is_indexed():
    """A categorical left unindexed would reach the assembler as a string."""
    indexers = [s for s in build_feature_stages() if isinstance(s, StringIndexer)]
    assert {s.getInputCol() for s in indexers} == set(FEATURES.categorical)


def test_unseen_categories_are_kept_not_rejected():
    """HandleInvalid must be "keep" everywhere, or streaming dies on new data.

    An unseen `primary_use` value arriving at 03:00 has to route to a spare
    bucket. The default ("error") would kill the query instead.
    """
    for stage in build_feature_stages():
        if hasattr(stage, "getHandleInvalid"):
            assert stage.getHandleInvalid() == "keep", type(stage).__name__


def test_assembler_consumes_imputed_and_encoded_columns_only():
    """The assembler must read transformed outputs, never the raw inputs.

    Wiring it to a raw column silently bypasses imputation for that feature.
    """
    assembler = build_feature_stages()[-1]
    inputs = set(assembler.getInputCols())

    assert inputs == {f"{c}_imp" for c in FEATURES.numeric} | {f"{c}_vec" for c in FEATURES.categorical}
    assert not inputs & set(FEATURES.all_inputs), "raw columns must not reach the assembler"


def test_no_scaler_is_included():
    """Tree ensembles split on thresholds, so scaling costs a pass for nothing."""
    names = [type(s).__name__ for s in build_feature_stages()]
    assert not any("Scaler" in n for n in names), names


# --------------------------------------------------------------------------
# Estimators
# --------------------------------------------------------------------------


def test_both_models_read_the_same_label_and_features():
    """Mismatched column names would train one model on the wrong target."""
    for model in (build_random_forest(), build_gbt()):
        assert model.getLabelCol() == FEATURES.label
        assert model.getFeaturesCol() == "features"


def test_both_models_are_seeded_for_reproducibility():
    """Without a fixed seed, two runs are not comparable."""
    assert build_random_forest().getSeed() == SEED
    assert build_gbt().getSeed() == SEED


def test_overrides_reach_the_estimator():
    """Tuning passes hyper-parameters through this path."""
    assert build_gbt(maxDepth=9).getMaxDepth() == 9
    # getOrDefault reads the Param directly, avoiding the accessor-name
    # inconsistency between Spark ML estimators.
    assert build_random_forest(numTrees=13).getOrDefault("numTrees") == 13


# --------------------------------------------------------------------------
# Assembled pipelines
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name,estimator", [("rf", RandomForestRegressor), ("gbt", GBTRegressor)])
def test_build_pipeline_ends_with_the_right_estimator(name, estimator):
    """The named model must be the final stage."""
    pipeline = build_pipeline(name)
    assert isinstance(pipeline, Pipeline)
    assert isinstance(pipeline.getStages()[-1], estimator)


def test_both_pipelines_share_identical_feature_stages():
    """Identical stages are what make the RF vs GBT comparison valid.

    If the feature stages differed, a performance gap could be preprocessing
    rather than the model, and the ADR 0004 conclusion would not follow.
    """
    rf_stages = [type(s).__name__ for s in build_pipeline("rf").getStages()[:-1]]
    gbt_stages = [type(s).__name__ for s in build_pipeline("gbt").getStages()[:-1]]
    assert rf_stages == gbt_stages


def test_unknown_model_name_is_rejected():
    """A typo should fail loudly at build time, not silently pick a default."""
    with pytest.raises(ValueError, match="Unknown model"):
        build_pipeline("xgboost")


def test_pipeline_fits_and_predicts_end_to_end(spark):
    """Fit and predict end to end, as the one integration test here.

    Uses a two-tree forest over a handful of rows. This would catch a column
    wiring error that the structural tests above cannot see, since those
    inspect configuration rather than execution.
    """
    rows = [
        (
            float(20 + i),  # air_temperature
            float(10 + i),  # dew_temperature
            1013.0,  # sea_level_pressure
            3.0,  # wind_speed
            float(10 + i),  # log_sqft
            3.0,  # floor_count
            1990.0,  # year_built
            float(i % 24),  # hour
            float(i % 7 + 1),  # day_of_week
            float(i % 12 + 1),  # month
            "Office" if i % 2 else "Education",
            "peak" if i % 3 else "off-peak",
            float(100 + i * 10),  # label
        )
        for i in range(40)
    ]
    schema = (
        "air_temperature double, dew_temperature double, sea_level_pressure double, "
        "wind_speed double, log_sqft double, floor_count double, year_built double, "
        "hour double, day_of_week double, month double, "
        "primary_use string, season_flag string, label double"
    )
    df = spark.createDataFrame(rows, schema)

    model = build_pipeline("rf", numTrees=2, maxDepth=3).fit(df)
    predictions = model.transform(df)

    assert "prediction" in predictions.columns
    assert predictions.count() == len(rows)
    assert predictions.filter("prediction IS NULL").count() == 0
