"""XGBoost backend – estimator builder for the MLModel layer.

This module is **internal**.  Users should use the openEO process functions
(``mlm_class_xgboost``, ``ml_fit``, ``ml_predict``) from
``openeo_core.model.base``.
"""

from __future__ import annotations

from typing import Any


def build_xgboost_estimator(
    *,
    task: str,
    learning_rate: float = 0.15,
    max_depth: int = 5,
    min_child_weight: float = 1,
    subsample: float = 0.8,
    min_split_loss: float = 1,
    seed: int | None = None,
) -> Any:
    """Build an XGBoost estimator from openEO parameters.

    Parameters
    ----------
    task : str
        ``"classification"`` or ``"regression"``.
    learning_rate
        Step size shrinkage (eta).
    max_depth
        Maximum tree depth.
    min_child_weight
        Minimum sum of instance weight needed in a child.
    subsample
        Subsample ratio of training instances.
    min_split_loss
        Minimum loss reduction (maps to XGBoost ``gamma``).
    seed
        Random seed (maps to ``random_state``).
    """
    import xgboost as xgb

    common: dict[str, Any] = {
        "learning_rate": learning_rate,
        "max_depth": max_depth,
        "min_child_weight": min_child_weight,
        "subsample": subsample,
        "gamma": min_split_loss,  # XGBoost's name for min_split_loss
        "random_state": seed,
    }

    if task == "classification":
        return xgb.XGBClassifier(**common)
    elif task == "regression":
        return xgb.XGBRegressor(**common)
    else:
        raise ValueError(f"Unknown task {task!r}")
