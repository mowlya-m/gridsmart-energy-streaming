"""Spark ML pipeline construction for the two candidate models.

The critical design choice here is that *all* fitted state, imputation
means, string-index vocabularies, one-hot layouts, lives inside the
``PipelineModel``.  Nothing is fitted outside it.

That is what makes the streaming half of the platform possible: the
streaming job calls ``PipelineModel.load(...)`` and gets the exact
imputation means and category encodings from training day.  Had the imputer
been fitted separately in the streaming job, it would have computed means
from whatever five days of weather happened to be in the current micro-batch
and quietly shifted the model's input distribution.
"""

from __future__ import annotations

from pyspark.ml import Pipeline
from pyspark.ml.feature import Imputer, OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml.regression import GBTRegressor, RandomForestRegressor

from .config import FEATURES, SEED


def build_feature_stages() -> list:
    """Assemble the transformer stages shared by both candidate models.

    Order matters:

    1. **Impute** numeric gaps using the *median*.  Median rather than mean
       because the weather series contain instrument spikes; a single
       corrupted pressure reading of 0 or 9999 shifts a mean but barely moves
       a median.
    2. **Index** each categorical column to an integer.  ``handleInvalid="keep"``
       routes unseen categories to a dedicated bucket instead of throwing --
       essential in streaming, where a new ``primary_use`` value appearing at
       03:00 must not kill the job.
    3. **One-hot encode** the indexes so the tree ensembles do not read an
       arbitrary integer ordering as a magnitude ordering.
    4. **Assemble** everything into the single ``features`` vector Spark ML
       requires.

    No scaler is included: both candidates are tree ensembles, which split on
    thresholds and are therefore invariant to feature scale.  Adding a
    ``StandardScaler`` would cost a full pass over the data for no benefit.
    """
    imputed_cols = [f"{c}_imp" for c in FEATURES.numeric]
    imputer = Imputer(
        inputCols=FEATURES.numeric,
        outputCols=imputed_cols,
        strategy="median",
    )

    indexers = [
        StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep") for c in FEATURES.categorical
    ]

    encoder = OneHotEncoder(
        inputCols=[f"{c}_idx" for c in FEATURES.categorical],
        outputCols=[f"{c}_vec" for c in FEATURES.categorical],
        handleInvalid="keep",
    )

    assembler = VectorAssembler(
        inputCols=imputed_cols + [f"{c}_vec" for c in FEATURES.categorical],
        outputCol="features",
        handleInvalid="keep",
    )

    return [imputer, *indexers, encoder, assembler]


def build_random_forest(**overrides) -> RandomForestRegressor:
    """Random Forest baseline.

    Chosen as the baseline because it is hard to overfit and needs almost no
    tuning: a fair floor to measure the boosted model against. Defaults
    are deliberately modest (80 trees, depth 7) so a full run completes on a
    laptop rather than requiring a cluster.
    """
    params = dict(
        labelCol=FEATURES.label,
        featuresCol="features",
        numTrees=80,
        maxDepth=7,
        maxBins=64,
        subsamplingRate=0.7,
        minInstancesPerNode=10,
        seed=SEED,
    )
    params.update(overrides)
    return RandomForestRegressor(**params)


def build_gbt(**overrides) -> GBTRegressor:
    """Gradient-Boosted Tree candidate.

    Boosting fits each tree to the residuals of the last, which suits this
    problem: the bulk of the signal is a smooth diurnal cycle, and the
    interesting error lives in the weather-driven deviations from it.
    ``stepSize`` is kept low (0.1) so that many shallow trees each make a
    small correction, which generalises better than few deep ones.
    """
    params = dict(
        labelCol=FEATURES.label,
        featuresCol="features",
        maxIter=80,
        maxDepth=5,
        maxBins=64,
        stepSize=0.1,
        subsamplingRate=0.7,
        seed=SEED,
    )
    params.update(overrides)
    return GBTRegressor(**params)


def build_pipeline(model_name: str, **overrides) -> Pipeline:
    """Return an unfitted end-to-end pipeline for the named model.

    Args:
    ----
        model_name: ``"rf"`` or ``"gbt"``.
        **overrides: Estimator hyper-parameters to override the defaults.

    Raises:
    ------
        ValueError: If ``model_name`` is not a known candidate.

    """
    builders = {"rf": build_random_forest, "gbt": build_gbt}
    if model_name not in builders:
        raise ValueError(f"Unknown model {model_name!r}; expected one of {sorted(builders)}")

    return Pipeline(stages=build_feature_stages() + [builders[model_name](**overrides)])


def gbt_param_grid(pipeline: Pipeline):
    """Hyper-parameter grid for the boosted model.

    Kept to 16 combinations rather than a wide sweep: with cross-validation
    each point costs a full fit, and the marginal return past this range was
    below the run-to-run noise floor. The three parameters chosen are the
    ones that actually trade off against each other, tree capacity
    (``maxDepth``), learning rate (``stepSize``), and row sampling
    (``subsamplingRate``).
    """
    from pyspark.ml.tuning import ParamGridBuilder

    gbt = pipeline.getStages()[-1]
    return (
        ParamGridBuilder()
        .addGrid(gbt.maxDepth, [4, 5])
        .addGrid(gbt.maxBins, [32, 64])
        .addGrid(gbt.stepSize, [0.05, 0.10])
        .addGrid(gbt.subsamplingRate, [0.7, 0.9])
        .build()
    )
