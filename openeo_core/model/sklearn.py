"""scikit-learn backend – estimator builders for the MLModel layer.

This module is **internal**.  Users should use the openEO process functions
(``mlm_class_random_forest``, ``mlm_regr_random_forest``, ``ml_fit``,
``ml_predict``) from ``openeo_core.model.base``.
"""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# max_variables mapping  (openEO spec → sklearn max_features)
# ---------------------------------------------------------------------------

_MAX_VARIABLES_MAP: dict[str, Any] = {
    "all": None,       # use all features
    "sqrt": "sqrt",
    "log2": "log2",
    "onethird": 1 / 3,
}


def _resolve_max_features(max_variables: int | str) -> Any:
    """Convert the openEO ``max_variables`` parameter to sklearn's ``max_features``."""
    if isinstance(max_variables, int):
        return max_variables
    if isinstance(max_variables, str):
        val = _MAX_VARIABLES_MAP.get(max_variables)
        if val is None and max_variables not in _MAX_VARIABLES_MAP:
            raise ValueError(
                f"Unknown max_variables value {max_variables!r}. "
                f"Expected an integer or one of {list(_MAX_VARIABLES_MAP)}"
            )
        return val
    raise TypeError(f"max_variables must be int or str, got {type(max_variables)}")


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def build_random_forest_estimator(
    *,
    task: str,
    max_variables: int | str = "sqrt",
    num_trees: int = 100,
    seed: int | None = None,
) -> Any:
    """Build an sklearn RandomForest estimator from openEO parameters.

    Parameters
    ----------
    task : str
        ``"classification"`` or ``"regression"``.
    max_variables
        Mapped to sklearn ``max_features``.
    num_trees
        Mapped to sklearn ``n_estimators``.
    seed
        Mapped to sklearn ``random_state``.
    """
    max_features = _resolve_max_features(max_variables)

    if task == "classification":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(
            n_estimators=num_trees,
            max_features=max_features,
            random_state=seed,
        )
    elif task == "regression":
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(
            n_estimators=num_trees,
            max_features=max_features,
            random_state=seed,
        )
    else:
        raise ValueError(f"Unknown task {task!r}")
