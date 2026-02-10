"""Tests for the XGBoost model backend using the openEO-aligned API."""

import numpy as np
import geopandas as gpd
import xarray as xr
from shapely.geometry import Point

import pytest

try:
    import xgboost  # noqa: F401
    _HAS_XGBOOST = True
except Exception:
    _HAS_XGBOOST = False

pytestmark = pytest.mark.skipif(not _HAS_XGBOOST, reason="XGBoost not available")

from openeo_core.model import (
    MLModel,
    mlm_class_xgboost,
    ml_fit,
    ml_predict,
    Model,
)


def _make_training_gdf():
    np.random.seed(42)
    n = 50
    X = np.random.rand(n, 2).astype(np.float32)
    y = (X[:, 0] > 0.5).astype(int)
    return gpd.GeoDataFrame(
        {"f1": X[:, 0], "f2": X[:, 1], "label": y},
        geometry=[Point(0, 0)] * n,
    )


class TestMLMClassXGBoost:
    def test_creates_untrained_model(self):
        model = mlm_class_xgboost(
            learning_rate=0.1,
            max_depth=3,
            min_child_weight=2,
            subsample=0.9,
            min_split_loss=0.5,
            seed=0,
        )
        assert isinstance(model, MLModel)
        assert not model.trained
        assert model.architecture == "XGBoost"
        assert model.tasks == ["classification"]
        assert model.framework == "XGBoost"
        assert model.hyperparameters["learning_rate"] == 0.1
        assert model.hyperparameters["max_depth"] == 3
        assert model.hyperparameters["min_split_loss"] == 0.5

    def test_fit_predict(self):
        model = mlm_class_xgboost(learning_rate=0.15, max_depth=3, seed=0)
        gdf = _make_training_gdf()
        trained = ml_fit(model, gdf, target="label")

        assert trained.trained

        raster = xr.DataArray(
            np.random.rand(3, 2).astype(np.float32),
            dims=["x", "bands"],
            coords={"bands": ["f1", "f2"], "x": [0, 1, 2]},
        )
        preds = ml_predict(raster, trained)
        assert "predictions" in preds.dims
        assert "x" in preds.dims
        assert "bands" not in preds.dims

    def test_stac_properties(self):
        model = mlm_class_xgboost(seed=42)
        props = model.to_stac_properties()
        assert props["mlm:framework"] == "XGBoost"
        assert props["mlm:architecture"] == "XGBoost"

    def test_model_factory(self):
        model = Model.xgboost_classifier(learning_rate=0.2, max_depth=4)
        assert isinstance(model, MLModel)
        assert model.tasks == ["classification"]
