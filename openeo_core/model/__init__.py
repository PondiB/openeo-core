"""Model sub-package – STAC MLM-compatible ML model objects and openEO ML processes.

Public API
----------
* :class:`MLModel` – STAC MLM-compatible model object
* :class:`Model` – convenience factory class

openEO process functions:

* :func:`mlm_class_random_forest` – initialise RF classifier
* :func:`mlm_regr_random_forest` – initialise RF regressor
* :func:`mlm_class_xgboost` – initialise XGBoost classifier
* :func:`ml_fit` – train a model
* :func:`ml_predict` – apply a trained model
* :func:`save_ml_model` – persist model + STAC Item
* :func:`load_stac_ml` – restore model from STAC Item
"""

from openeo_core.model.mlm import MLModel, ModelInput, ModelOutput, InputStructure, ResultStructure
from openeo_core.model.base import (
    Model,
    ml_fit,
    ml_predict,
    mlm_class_random_forest,
    mlm_class_xgboost,
    mlm_regr_random_forest,
    save_ml_model,
    load_stac_ml,
)

__all__ = [
    # Core object
    "MLModel",
    "ModelInput",
    "ModelOutput",
    "InputStructure",
    "ResultStructure",
    # Factory
    "Model",
    # openEO process functions
    "mlm_class_random_forest",
    "mlm_regr_random_forest",
    "mlm_class_xgboost",
    "ml_fit",
    "ml_predict",
    "save_ml_model",
    "load_stac_ml",
]
