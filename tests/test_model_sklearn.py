"""Tests for the sklearn model backend using the openEO-aligned API.

Tests cover:
* mlm_class_random_forest / mlm_regr_random_forest initialisation
* ml_fit / ml_predict flow
* MLModel STAC MLM metadata
* save_ml_model / load_stac_ml round-trip
* Model convenience factory
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
from shapely.geometry import Point

from openeo_core.model import (
    MLModel,
    Model,
    ml_fit,
    ml_predict,
    mlm_class_random_forest,
    mlm_regr_random_forest,
    save_ml_model,
    load_stac_ml,
)


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------


def _make_training_gdf():
    """GeoDataFrame with 3 numeric features + a classification label."""
    np.random.seed(42)
    n = 50
    X = np.random.rand(n, 3).astype(np.float32)
    y = (X[:, 0] > 0.5).astype(int)
    return gpd.GeoDataFrame(
        {"b1": X[:, 0], "b2": X[:, 1], "b3": X[:, 2], "label": y},
        geometry=[Point(0, 0)] * n,
    )


def _make_raster(n_bands: int = 3) -> xr.DataArray:
    np.random.seed(0)
    return xr.DataArray(
        np.random.rand(4, n_bands, 3).astype(np.float32),
        dims=["t", "bands", "x"],
        coords={
            "t": pd.date_range("2023-01-01", periods=4, freq="ME"),
            "bands": [f"b{i+1}" for i in range(n_bands)],
            "x": [0.0, 1.0, 2.0],
        },
    )


# ---------------------------------------------------------------
# Initialization tests
# ---------------------------------------------------------------


class TestMLMClassRandomForest:
    def test_creates_untrained_model(self):
        model = mlm_class_random_forest(max_variables="sqrt", num_trees=50, seed=0)
        assert isinstance(model, MLModel)
        assert not model.trained
        assert model.architecture == "Random Forest"
        assert model.tasks == ["classification"]
        assert model.framework == "scikit-learn"
        assert model.hyperparameters["num_trees"] == 50
        assert model.hyperparameters["max_variables"] == "sqrt"

    def test_max_variables_integer(self):
        model = mlm_class_random_forest(max_variables=3, num_trees=10)
        assert model.hyperparameters["max_variables"] == 3


class TestMLMRegrRandomForest:
    def test_creates_regression_model(self):
        model = mlm_regr_random_forest(max_variables="onethird", num_trees=80)
        assert model.tasks == ["regression"]
        assert model.hyperparameters["num_trees"] == 80


# ---------------------------------------------------------------
# ml_fit / ml_predict
# ---------------------------------------------------------------


class TestMLFitPredict:
    def test_fit_returns_trained_model(self):
        model = mlm_class_random_forest(max_variables="sqrt", num_trees=10, seed=0)
        gdf = _make_training_gdf()
        trained = ml_fit(model, gdf, target="label")

        assert trained.trained
        assert trained._n_features == 3
        assert trained._feature_names == ["b1", "b2", "b3"]
        # original not mutated
        assert not model.trained

    def test_fit_updates_input_metadata(self):
        model = mlm_class_random_forest(max_variables="sqrt", num_trees=10, seed=0)
        gdf = _make_training_gdf()
        trained = ml_fit(model, gdf, target="label")

        inp = trained.inputs[0]
        assert inp.bands == ["b1", "b2", "b3"]
        assert inp.input.shape == [-1, 3]

    def test_predict_returns_datacube(self):
        model = mlm_class_random_forest(max_variables="sqrt", num_trees=10, seed=0)
        gdf = _make_training_gdf()
        trained = ml_fit(model, gdf, target="label")

        raster = _make_raster(n_bands=3)
        preds = ml_predict(raster, trained)

        # Should have a "predictions" dim per openEO spec
        assert "predictions" in preds.dims
        # Spatial dims preserved
        assert "t" in preds.dims
        assert "x" in preds.dims
        # Feature dim consumed
        assert "bands" not in preds.dims

    def test_predict_untrained_raises(self):
        model = mlm_class_random_forest(max_variables="sqrt", num_trees=10)
        raster = _make_raster()
        try:
            ml_predict(raster, model)
            assert False, "Should have raised RuntimeError"
        except RuntimeError as e:
            assert "not been trained" in str(e)

    def test_feature_mismatch_raises(self):
        model = mlm_class_random_forest(max_variables="sqrt", num_trees=10, seed=0)
        gdf = _make_training_gdf()
        trained = ml_fit(model, gdf, target="label")

        raster = _make_raster(n_bands=2)
        try:
            ml_predict(raster, trained)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "mismatch" in str(e).lower()

    def test_regression_fit_predict(self):
        np.random.seed(0)
        n = 50
        X = np.random.rand(n, 2).astype(np.float32)
        y = X[:, 0] * 2 + X[:, 1]
        gdf = gpd.GeoDataFrame(
            {"f1": X[:, 0], "f2": X[:, 1], "target": y},
            geometry=[Point(0, 0)] * n,
        )

        model = mlm_regr_random_forest(max_variables="all", num_trees=10, seed=0)
        trained = ml_fit(model, gdf, target="target")

        raster = xr.DataArray(
            np.random.rand(3, 2).astype(np.float32),
            dims=["x", "bands"],
            coords={"bands": ["f1", "f2"], "x": [0, 1, 2]},
        )
        preds = ml_predict(raster, trained)
        assert "predictions" in preds.dims
        assert "x" in preds.dims


# ---------------------------------------------------------------
# STAC MLM metadata
# ---------------------------------------------------------------


class TestSTACMLMMetadata:
    def test_stac_properties(self):
        model = mlm_class_random_forest(max_variables="sqrt", num_trees=10, seed=42)
        props = model.to_stac_properties()

        assert props["mlm:name"] == "Random Forest Classifier"
        assert props["mlm:architecture"] == "Random Forest"
        assert props["mlm:tasks"] == ["classification"]
        assert props["mlm:framework"] == "scikit-learn"
        assert props["mlm:hyperparameters"]["num_trees"] == 10
        assert props["mlm:pretrained"] is False

    def test_stac_item(self):
        model = mlm_class_random_forest(max_variables="sqrt")
        item = model.to_stac_item()

        assert item["type"] == "Feature"
        assert "https://stac-extensions.github.io/mlm/v1.5.1/schema.json" in item["stac_extensions"]
        assert "model" in item["assets"]
        assert "mlm:model" in item["assets"]["model"]["roles"]

    def test_repr(self):
        model = mlm_class_random_forest(max_variables="sqrt")
        r = repr(model)
        assert "Random Forest" in r
        assert "untrained" in r


# ---------------------------------------------------------------
# save_ml_model / load_stac_ml round-trip
# ---------------------------------------------------------------


class TestSaveLoadRoundTrip:
    def test_save_and_load(self):
        model = mlm_class_random_forest(max_variables="sqrt", num_trees=10, seed=0)
        gdf = _make_training_gdf()
        trained = ml_fit(model, gdf, target="label")

        with tempfile.TemporaryDirectory() as tmpdir:
            ok = save_ml_model(trained, "my_rf", options={"directory": tmpdir})
            assert ok

            out_dir = Path(tmpdir) / "my_rf"
            assert (out_dir / "model.pkl").exists()
            stac_file = out_dir / "my_rf.stac.json"
            assert stac_file.exists()

            # Validate STAC Item structure
            stac = json.loads(stac_file.read_text())
            assert stac["type"] == "Feature"
            assert stac["properties"]["mlm:name"] == "Random Forest Classifier"

            # Load back
            loaded = load_stac_ml(str(stac_file))
            assert loaded.trained
            assert loaded.framework == "scikit-learn"
            assert loaded.architecture == "Random Forest"

            # Predict with loaded model
            raster = _make_raster(n_bands=3)
            preds = ml_predict(raster, loaded)
            assert "predictions" in preds.dims


# ---------------------------------------------------------------
# Model convenience factory
# ---------------------------------------------------------------


class TestModelFactory:
    def test_random_forest_classification(self):
        model = Model.random_forest(task="classification", max_variables="sqrt", num_trees=10)
        assert isinstance(model, MLModel)
        assert model.tasks == ["classification"]

    def test_random_forest_regression(self):
        model = Model.random_forest(task="regression", max_variables="onethird", num_trees=20)
        assert isinstance(model, MLModel)
        assert model.tasks == ["regression"]
