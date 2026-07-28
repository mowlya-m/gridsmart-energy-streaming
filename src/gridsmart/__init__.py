"""GridSmart, building energy forecasting for smart-grid operators.

A two-stage platform:

* **Batch** (:mod:`gridsmart.pipelines`) trains and persists a Spark ML
  pipeline that predicts a building's aggregate energy draw per 6-hour
  dispatch window.
* **Streaming** (:mod:`gridsmart.streaming`) loads that same fitted pipeline
  and scores a live Kafka weather feed, publishing per-building and per-site
  forecasts to downstream topics for the operator dashboard.

The two halves share :mod:`gridsmart.features`, :mod:`gridsmart.schemas` and
:mod:`gridsmart.config` so that the training-time and inference-time feature
contracts cannot drift apart.
"""

__version__ = "1.0.0"

from . import config, features, metrics, pipelines, schemas, session, streaming

__all__ = [
    "config",
    "features",
    "metrics",
    "pipelines",
    "schemas",
    "session",
    "streaming",
    "__version__",
]
