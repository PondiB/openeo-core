"""openEO ML process implementations and Model factory.

This module exposes the openEO-aligned process functions that match the
JSON process specs in ``docs/openeo-processes/``:

Initialization (returns untrained ``MLModel``)::

    mlm_class_random_forest(max_variables, num_trees=100, seed=None)
    mlm_regr_random_forest(max_variables, num_trees=100, seed=None)
    mlm_class_xgboost(learning_rate=0.15, max_depth=5, ...)

Training and prediction::

    ml_fit(model, training_set, target) -> MLModel
    ml_predict(data, model)             -> RasterCube

Serialisation (STAC MLM)::

    save_ml_model(data, name, options={}) -> bool
    load_stac_ml(uri, model_asset=None, input_index=0, output_index=0) -> MLModel

A convenience ``Model`` class is also available for a fluent style::

    model = Model.random_forest(max_variables="sqrt", num_trees=200)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Union

import numpy as np
import xarray as xr

from openeo_core.model.mlm import (
    MLModel,
    ModelInput,
    ModelOutput,
    InputStructure,
    ResultStructure,
)
from openeo_core.types import RasterCube, VectorCube

# =====================================================================
# Initialization processes  (create untrained MLModel objects)
# =====================================================================


def mlm_class_random_forest(
    max_variables: int | str = "sqrt",
    num_trees: int = 100,
    seed: int | None = None,
) -> MLModel:
    """Initialize a Random Forest **classification** model.

    Implements ``mlm_class_random_forest`` from the openEO process specs.
    Does **not** train — use :func:`ml_fit` afterwards.

    Parameters
    ----------
    max_variables : int | str
        Number of split variables per node.
        Accepts an integer or one of ``"all"``, ``"log2"``,
        ``"onethird"``, ``"sqrt"``.
    num_trees : int
        Number of trees (default 100).
    seed : int | None
        Random seed for reproducibility.
    """
    from openeo_core.model.sklearn import build_random_forest_estimator

    hyperparams = {"max_variables": max_variables, "num_trees": num_trees, "seed": seed}
    estimator = build_random_forest_estimator(
        task="classification",
        max_variables=max_variables,
        num_trees=num_trees,
        seed=seed,
    )

    return MLModel(
        name="Random Forest Classifier",
        architecture="Random Forest",
        tasks=["classification"],
        framework="scikit-learn",
        framework_version=_sklearn_version(),
        hyperparameters=hyperparams,
        outputs=[ModelOutput(
            name="predictions",
            tasks=["classification"],
            result=ResultStructure(shape=[-1], dim_order=["batch"], data_type="int64"),
        )],
        estimator=estimator,
        backend="sklearn",
    )


def mlm_regr_random_forest(
    max_variables: int | str = "onethird",
    num_trees: int = 100,
    seed: int | None = None,
) -> MLModel:
    """Initialize a Random Forest **regression** model.

    Implements ``mlm_regr_random_forest`` from the openEO process specs.

    Parameters are identical to :func:`mlm_class_random_forest`.
    """
    from openeo_core.model.sklearn import build_random_forest_estimator

    hyperparams = {"max_variables": max_variables, "num_trees": num_trees, "seed": seed}
    estimator = build_random_forest_estimator(
        task="regression",
        max_variables=max_variables,
        num_trees=num_trees,
        seed=seed,
    )

    return MLModel(
        name="Random Forest Regressor",
        architecture="Random Forest",
        tasks=["regression"],
        framework="scikit-learn",
        framework_version=_sklearn_version(),
        hyperparameters=hyperparams,
        outputs=[ModelOutput(
            name="predictions",
            tasks=["regression"],
            result=ResultStructure(shape=[-1], dim_order=["batch"], data_type="float64"),
        )],
        estimator=estimator,
        backend="sklearn",
    )


def mlm_class_xgboost(
    learning_rate: float = 0.15,
    max_depth: int = 5,
    min_child_weight: float = 1,
    subsample: float = 0.8,
    min_split_loss: float = 1,
    seed: int | None = None,
) -> MLModel:
    """Initialize an XGBoost **classification** model.

    Implements ``mlm_class_xgboost`` from the openEO process specs.

    Parameters
    ----------
    learning_rate : float
        Step size shrinkage (default 0.15).
    max_depth : int
        Maximum tree depth (default 5).
    min_child_weight : float
        Minimum sum of instance weight in a child (default 1).
    subsample : float
        Subsample ratio of training instances (default 0.8).
    min_split_loss : float
        Minimum loss reduction for a partition (maps to XGBoost ``gamma``,
        default 1).
    seed : int | None
        Random seed.
    """
    from openeo_core.model.xgboost_backend import build_xgboost_estimator

    hyperparams = {
        "learning_rate": learning_rate,
        "max_depth": max_depth,
        "min_child_weight": min_child_weight,
        "subsample": subsample,
        "min_split_loss": min_split_loss,
        "seed": seed,
    }
    estimator = build_xgboost_estimator(
        task="classification",
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_child_weight=min_child_weight,
        subsample=subsample,
        min_split_loss=min_split_loss,
        seed=seed,
    )

    return MLModel(
        name="XGBoost Classifier",
        architecture="XGBoost",
        tasks=["classification"],
        framework="XGBoost",
        framework_version=_xgboost_version(),
        hyperparameters=hyperparams,
        outputs=[ModelOutput(
            name="predictions",
            tasks=["classification"],
            result=ResultStructure(shape=[-1], dim_order=["batch"], data_type="int64"),
        )],
        estimator=estimator,
        backend="xgboost",
    )


# =====================================================================
# ml_fit  –  Train a machine learning model
# =====================================================================


def ml_fit(
    model: MLModel,
    training_set: VectorCube,
    target: str,
) -> MLModel:
    """Train a machine learning model.

    Implements ``ml_fit`` from the openEO process specs.

    Parameters
    ----------
    model : MLModel
        An untrained model returned by one of the ``mlm_*`` factories.
    training_set : VectorCube
        A vector data cube (GeoDataFrame) containing both features and
        the target variable.
    target : str
        Name of the target column in *training_set*.

    Returns
    -------
    MLModel
        A **new** trained model object (the original is not mutated).
    """
    import geopandas as gpd
    from openeo_core.ops.vector import to_feature_matrix

    # Materialise dask if needed
    if hasattr(training_set, "compute"):
        training_set = training_set.compute()  # type: ignore[assignment]

    if not isinstance(training_set, gpd.GeoDataFrame):
        raise TypeError(
            f"training_set must be a GeoDataFrame, got {type(training_set).__name__}"
        )

    if target not in training_set.columns:
        raise ValueError(f"Target column {target!r} not found in training_set")

    # Build X / y
    feature_cols = [
        c for c in training_set.select_dtypes(include=[np.number]).columns
        if c != target
    ]
    X = training_set[feature_cols].to_numpy()
    y = training_set[target].to_numpy()

    # Clone the model so we don't mutate the original
    trained = _clone_model(model)
    trained._estimator.fit(X, y)
    trained._trained = True
    trained._n_features = X.shape[1]
    trained._feature_names = feature_cols
    trained.pretrained = True

    # Update input metadata
    trained.inputs = [ModelInput(
        name="features",
        bands=feature_cols,
        input=InputStructure(
            shape=[-1, X.shape[1]],
            dim_order=["batch", "bands"],
            data_type=str(X.dtype),
        ),
    )]

    # Update total_parameters if available (tree models)
    trained.total_parameters = _count_parameters(trained._estimator)

    return trained


# =====================================================================
# ml_predict  –  Apply trained model to a data cube
# =====================================================================


def ml_predict(
    data: RasterCube,
    model: MLModel,
    *,
    feature_dim: str = "bands",
) -> RasterCube:
    """Apply a trained ML model to a data cube.

    Implements ``ml_predict`` from the openEO process specs.

    Parameters
    ----------
    data : RasterCube
        Input feature data cube.
    model : MLModel
        A trained model (output of :func:`ml_fit`).
    feature_dim : str
        Name of the feature / bands dimension (default ``"bands"``).

    Returns
    -------
    RasterCube
        A data cube with a ``predictions`` dimension of type ``other``.
        For single-value predictions the label is ``"0"``.
    """
    import dask.array as da_mod
    from openeo_core.ops.raster import stack_to_samples, unstack_from_samples

    if not model.trained:
        raise RuntimeError("Model has not been trained. Call ml_fit() first.")

    template = data
    stacked = stack_to_samples(data, feature_dim=feature_dim)

    if model._n_features is not None and stacked.sizes[feature_dim] != model._n_features:
        raise ValueError(
            f"Feature count mismatch: model expects {model._n_features}, "
            f"got {stacked.sizes[feature_dim]}"
        )

    raw = stacked.values if not isinstance(stacked.data, da_mod.Array) else stacked.data

    if isinstance(raw, da_mod.Array):
        def _predict_block(block: np.ndarray) -> np.ndarray:
            return model._estimator.predict(block)

        result_arr = da_mod.map_blocks(
            _predict_block,
            raw,
            dtype=np.float64,
            drop_axis=1,
        )
        result_da = xr.DataArray(result_arr, dims=["samples"])
    else:
        preds = model._estimator.predict(raw)
        result_da = xr.DataArray(preds, dims=["samples"])

    # Unstack back to spatial dims
    unstacked = unstack_from_samples(result_da, template, feature_dim=feature_dim)

    # Wrap in a "predictions" dimension per the spec
    unstacked = unstacked.expand_dims(predictions=["0"])

    return unstacked


# =====================================================================
# save_ml_model  –  Persist model + STAC Item
# =====================================================================


def save_ml_model(
    data: MLModel,
    name: str,
    options: dict[str, Any] | None = None,
) -> bool:
    """Save a machine learning model with an accompanying STAC Item.

    Implements ``save_ml_model`` from the openEO process specs.

    Parameters
    ----------
    data : MLModel
        The trained model to save.
    name : str
        A distinct name for the model (used as directory / file stem).
    options : dict
        Additional options (currently: ``"directory"`` for output path).

    Returns
    -------
    bool
        ``True`` on success, ``False`` on failure.
    """
    import joblib

    opts = options or {}
    out_dir = Path(opts.get("directory", ".")) / name
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / "model.pkl"
    stac_path = out_dir / f"{name}.stac.json"

    try:
        # Save the estimator
        joblib.dump(data._estimator, model_path)

        # Save the STAC Item
        stac_item = data.to_stac_item(model_href=str(model_path.name))
        stac_item["id"] = name
        stac_path.write_text(json.dumps(stac_item, indent=2, default=str))

        return True
    except Exception:
        return False


# =====================================================================
# load_stac_ml  –  Load model from a STAC Item
# =====================================================================


def load_stac_ml(
    uri: str,
    *,
    model_asset: str | None = None,
    input_index: int = 0,
    output_index: int = 0,
) -> MLModel:
    """Load a ML model from a STAC Item implementing the MLM extension.

    Implements ``load_stac_ml`` from the openEO process specs.

    Parameters
    ----------
    uri : str
        URL or local file path to a STAC Item JSON.
    model_asset : str | None
        Asset key with role ``mlm:model``.  Required when multiple such
        assets exist.
    input_index : int
        Index into ``mlm:input`` (default 0).
    output_index : int
        Index into ``mlm:output`` (default 0).
    """
    import joblib

    # -- Load the STAC Item JSON --
    if uri.startswith(("http://", "https://")):
        import urllib.request
        with urllib.request.urlopen(uri) as resp:
            item = json.loads(resp.read())
    else:
        item = json.loads(Path(uri).read_text())

    properties = item.get("properties", {})
    assets = item.get("assets", {})

    # -- Resolve model asset --
    model_assets = {
        k: v for k, v in assets.items()
        if "mlm:model" in v.get("roles", [])
    }
    if not model_assets:
        raise ValueError("No asset with role 'mlm:model' found in the STAC Item.")

    if model_asset is not None:
        if model_asset not in model_assets:
            raise ValueError(
                f"Asset {model_asset!r} not found among mlm:model assets: "
                f"{list(model_assets)}"
            )
        chosen = model_assets[model_asset]
    elif len(model_assets) == 1:
        chosen = next(iter(model_assets.values()))
    else:
        raise ValueError(
            f"Multiple mlm:model assets found ({list(model_assets)}). "
            f"Specify model_asset= to disambiguate."
        )

    # -- Load the estimator --
    model_href = chosen["href"]
    if not model_href.startswith(("http://", "https://")):
        # Relative to the STAC Item file
        base = Path(uri).parent if not uri.startswith(("http://", "https://")) else Path(".")
        model_href = str(base / model_href)

    estimator = joblib.load(model_href)

    # -- Determine backend from framework --
    framework = properties.get("mlm:framework", "")
    backend_map = {
        "scikit-learn": "sklearn",
        "XGBoost": "xgboost",
        "PyTorch": "torch",
    }
    backend = backend_map.get(framework)

    # -- Select input / output --
    all_inputs = properties.get("mlm:input", [])
    all_outputs = properties.get("mlm:output", [])
    selected_inputs = [all_inputs[input_index]] if input_index < len(all_inputs) else all_inputs
    selected_outputs = [all_outputs[output_index]] if output_index < len(all_outputs) else all_outputs

    properties_copy = dict(properties)
    properties_copy["mlm:input"] = selected_inputs
    properties_copy["mlm:output"] = selected_outputs

    model = MLModel.from_stac_properties(
        properties_copy,
        estimator=estimator,
        backend=backend,
    )
    model._trained = True

    # Recover feature count from the input spec
    if selected_inputs:
        inp = selected_inputs[0]
        shape = inp.get("input", {}).get("shape", [])
        if len(shape) >= 2 and shape[-1] > 0:
            model._n_features = shape[-1]
        model._feature_names = inp.get("bands", None) or None

    return model


# =====================================================================
# Model  –  convenience factory (backward-compatible fluent API)
# =====================================================================


class Model:
    """Convenience factory that delegates to the openEO process functions.

    Usage::

        model = Model.random_forest(max_variables="sqrt", num_trees=200)
        model = ml_fit(model, training_gdf, target="label")
        preds = ml_predict(raster, model)

    The returned objects are :class:`MLModel` instances.
    """

    @staticmethod
    def random_forest(
        *,
        task: str = "classification",
        max_variables: int | str = "sqrt",
        num_trees: int = 100,
        seed: int | None = None,
    ) -> MLModel:
        if task == "classification":
            return mlm_class_random_forest(max_variables=max_variables, num_trees=num_trees, seed=seed)
        elif task == "regression":
            return mlm_regr_random_forest(max_variables=max_variables, num_trees=num_trees, seed=seed)
        else:
            raise ValueError(f"Unknown task {task!r}")

    @staticmethod
    def xgboost_classifier(
        *,
        learning_rate: float = 0.15,
        max_depth: int = 5,
        min_child_weight: float = 1,
        subsample: float = 0.8,
        min_split_loss: float = 1,
        seed: int | None = None,
    ) -> MLModel:
        return mlm_class_xgboost(
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_child_weight=min_child_weight,
            subsample=subsample,
            min_split_loss=min_split_loss,
            seed=seed,
        )


# =====================================================================
# Internal helpers
# =====================================================================


def _clone_model(model: MLModel) -> MLModel:
    """Deep-copy an MLModel, cloning the estimator."""
    import copy as _copy

    new = MLModel(
        name=model.name,
        architecture=model.architecture,
        tasks=list(model.tasks),
        framework=model.framework,
        framework_version=model.framework_version,
        inputs=[_copy.deepcopy(i) for i in model.inputs],
        outputs=[_copy.deepcopy(o) for o in model.outputs],
        hyperparameters=dict(model.hyperparameters),
        pretrained=model.pretrained,
        pretrained_source=model.pretrained_source,
        total_parameters=model.total_parameters,
        memory_size=model.memory_size,
        accelerator=model.accelerator,
        accelerator_constrained=model.accelerator_constrained,
        estimator=_copy.deepcopy(model._estimator),
        backend=model._backend,
    )
    return new


def _count_parameters(estimator: Any) -> int | None:
    """Attempt to count model parameters (tree leaves, etc.)."""
    try:
        # sklearn tree ensembles
        if hasattr(estimator, "estimators_"):
            return sum(e.tree_.node_count for e in estimator.estimators_)
    except Exception:
        pass
    return None


def _sklearn_version() -> str | None:
    try:
        import sklearn
        return sklearn.__version__
    except ImportError:
        return None


def _xgboost_version() -> str | None:
    try:
        import xgboost
        return xgboost.__version__
    except Exception:
        return None
