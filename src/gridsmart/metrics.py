"""Evaluation metrics, including a Spark-native RMSLE evaluator.

Spark ML ships RMSE, MSE, MAE and R2 but not RMSLE.  Since RMSLE is the
metric this project selects models on, it is implemented here as a proper
:class:`~pyspark.ml.evaluation.Evaluator` subclass so it can be handed
directly to :class:`~pyspark.ml.tuning.CrossValidator`, rather than being
computed after the fact, which would leave hyper-parameter search optimising
a different objective than model selection.
"""

from __future__ import annotations

from pyspark.ml.evaluation import Evaluator
from pyspark.ml.param.shared import HasLabelCol, HasPredictionCol
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

__all__ = ["RMSLEEvaluator", "rmsle", "regression_report"]


def rmsle(
    predictions: DataFrame,
    label_col: str = "label",
    prediction_col: str = "prediction",
) -> float:
    r"""Root Mean Squared Logarithmic Error, computed entirely in Spark.

    .. math::
        \epsilon = \sqrt{\frac{1}{n}\sum_{i=1}^{n}
                   \big(\log(p_i + 1) - \log(a_i + 1)\big)^2}

    Why this metric and not RMSE: building energy consumption spans several
    orders of magnitude, from a small retail unit to a university campus.
    RMSE is dominated by the largest buildings, so a model can look excellent
    while being useless on the long tail. RMSLE measures *ratio* error, which
    weights a 20% miss on a small building the same as a 20% miss on a large
    one: which is what a grid operator actually cares about.

    The ``+1`` inside each logarithm keeps the metric defined at zero
    consumption (an idle building), and is implemented with ``log1p`` for
    numerical stability at small values.

    Negative predictions are clipped to zero before the logarithm.  A tree
    ensemble can extrapolate slightly below zero on sparse leaves; a negative
    energy draw is physically meaningless and would make the logarithm
    undefined, so clipping is the honest treatment.

    Args:
    ----
        predictions: Frame containing both the label and prediction columns.
        label_col: Name of the ground-truth column.
        prediction_col: Name of the model output column.

    Returns:
    -------
        The RMSLE as a Python float. Lower is better; 0.0 is a perfect fit.

    """
    clipped = F.greatest(F.col(prediction_col).cast("double"), F.lit(0.0))
    actual = F.greatest(F.col(label_col).cast("double"), F.lit(0.0))

    squared_log_error = (F.log1p(clipped) - F.log1p(actual)) ** 2

    result = (
        predictions.select(squared_log_error.alias("sle")).agg(F.sqrt(F.mean("sle")).alias("rmsle")).first()
    )

    return float(result["rmsle"]) if result and result["rmsle"] is not None else float("nan")


class RMSLEEvaluator(Evaluator, HasLabelCol, HasPredictionCol):
    """RMSLE wrapped as a Spark ML ``Evaluator``.

    Implementing the interface, rather than just calling :func:`rmsle` --
    means ``CrossValidator`` and ``TrainValidationSplit`` can optimise
    directly against RMSLE:

        >>> cv = CrossValidator(
        ...     estimator=pipeline,
        ...     evaluator=RMSLEEvaluator(),
        ...     estimatorParamMaps=grid,
        ... )

    ``isLargerBetter`` returns ``False`` so Spark's search correctly treats a
    lower score as an improvement.
    """

    def __init__(self, labelCol: str = "label", predictionCol: str = "prediction"):
        """Configure which columns hold the ground truth and the prediction."""
        super().__init__()
        self._set(labelCol=labelCol, predictionCol=predictionCol)

    def _evaluate(self, dataset: DataFrame) -> float:
        return rmsle(dataset, self.getLabelCol(), self.getPredictionCol())

    def isLargerBetter(self) -> bool:
        """Lower RMSLE is better, so Spark must minimise this metric."""
        return False


def regression_report(
    predictions: DataFrame,
    label_col: str = "label",
    prediction_col: str = "prediction",
) -> dict[str, float]:
    """Compute RMSLE, RMSE, MAE and R2 in a single pass over the data.

    Reported together because they answer different questions: RMSLE for
    model selection, RMSE in kWh for the operational conversation ("how many
    kilowatt-hours are we typically out by"), MAE for a median-ish sense of
    typical error, and R2 for variance explained.
    """
    label = F.col(label_col).cast("double")
    pred = F.col(prediction_col).cast("double")
    error = pred - label

    stats = predictions.agg(
        F.sqrt(F.mean(error**2)).alias("rmse"),
        F.mean(F.abs(error)).alias("mae"),
        F.var_pop(label).alias("label_var"),
        F.mean(error**2).alias("mse"),
    ).first()

    variance = float(stats["label_var"] or 0.0)
    r2 = 1.0 - (float(stats["mse"]) / variance) if variance > 0 else float("nan")

    return {
        "rmsle": rmsle(predictions, label_col, prediction_col),
        "rmse": float(stats["rmse"]),
        "mae": float(stats["mae"]),
        "r2": r2,
    }
