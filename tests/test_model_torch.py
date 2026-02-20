"""Tests for the PyTorch model backends (TempCNN, LightTAE).

Tests cover:
* mlm_class_tempcnn / mlm_class_lighttae initialisation
* ml_fit / ml_predict flow
* MLModel STAC MLM metadata
* save_ml_model / load_stac_ml round-trip
* Model convenience factory
* Process registry discovery
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import pytest
import xarray as xr
from shapely.geometry import Point

from openeo_core.model import (
    MLModel,
    Model,
    ml_fit,
    ml_predict,
    mlm_class_tempcnn,
    mlm_class_lighttae,
    save_ml_model,
    load_stac_ml,
)

torch = pytest.importorskip("torch")


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------


def _make_training_gdf(n_features: int = 10, n_samples: int = 60):
    """GeoDataFrame with *n_features* numeric columns + a classification label."""
    np.random.seed(42)
    X = np.random.rand(n_samples, n_features).astype(np.float32)
    y = (X[:, 0] > 0.5).astype(int)
    data = {f"f{i}": X[:, i] for i in range(n_features)}
    data["label"] = y
    return gpd.GeoDataFrame(data, geometry=[Point(i, i) for i in range(n_samples)])


def _make_raster(n_bands: int = 10) -> xr.DataArray:
    np.random.seed(0)
    return xr.DataArray(
        np.random.rand(4, n_bands, 3).astype(np.float32),
        dims=["t", "bands", "x"],
        coords={
            "t": pd.date_range("2023-01-01", periods=4, freq="ME"),
            "bands": [f"f{i}" for i in range(n_bands)],
            "x": [0.0, 1.0, 2.0],
        },
    )


# ---------------------------------------------------------------
# TempCNN – Initialization
# ---------------------------------------------------------------


class TestMLMClassTempCNN:
    def test_creates_untrained_model(self):
        model = mlm_class_tempcnn(epochs=5, batch_size=16, seed=0)
        assert isinstance(model, MLModel)
        assert not model.trained
        assert model.architecture == "TempCNN"
        assert model.tasks == ["classification"]
        assert model.framework == "PyTorch"
        assert model.hyperparameters["epochs"] == 5

    def test_custom_cnn_layers(self):
        model = mlm_class_tempcnn(
            cnn_layers=[64, 128],
            cnn_kernels=[3, 5],
            cnn_dropout_rates=[0.1, 0.3],
            epochs=2,
        )
        assert model.hyperparameters["cnn_layers"] == [64, 128]
        assert model.hyperparameters["cnn_kernels"] == [3, 5]

    def test_default_hyperparameters(self):
        model = mlm_class_tempcnn()
        hp = model.hyperparameters
        assert hp["cnn_layers"] == [256, 256, 256]
        assert hp["cnn_kernels"] == [7, 7, 7]
        assert hp["cnn_dropout_rates"] == [0.2, 0.2, 0.2]
        assert hp["dense_layer_nodes"] == 256
        assert hp["dense_layer_dropout_rate"] == 0.5
        assert hp["optimizer"] == "adam"
        assert hp["learning_rate"] == 0.001


# ---------------------------------------------------------------
# LightTAE – Initialization
# ---------------------------------------------------------------


class TestMLMClassLightTAE:
    def test_creates_untrained_model(self):
        model = mlm_class_lighttae(epochs=5, batch_size=16, seed=0)
        assert isinstance(model, MLModel)
        assert not model.trained
        assert model.architecture == "LightTAE"
        assert model.tasks == ["classification"]
        assert model.framework == "PyTorch"
        assert model.hyperparameters["epochs"] == 5

    def test_default_hyperparameters(self):
        model = mlm_class_lighttae()
        hp = model.hyperparameters
        assert hp["epochs"] == 150
        assert hp["batch_size"] == 128
        assert hp["optimizer"] == "adam"
        assert hp["learning_rate"] == 0.0005
        assert hp["epsilon"] == 1e-8
        assert hp["weight_decay"] == 0.0007
        assert hp["lr_decay_epochs"] == 50
        assert hp["lr_decay_rate"] == 1.0

    def test_custom_hyperparameters(self):
        model = mlm_class_lighttae(
            epochs=10,
            batch_size=32,
            optimizer="radam",
            learning_rate=0.01,
            weight_decay=0.001,
            seed=123,
        )
        hp = model.hyperparameters
        assert hp["optimizer"] == "radam"
        assert hp["learning_rate"] == 0.01
        assert hp["seed"] == 123


# ---------------------------------------------------------------
# ml_fit / ml_predict – TempCNN
# ---------------------------------------------------------------


class TestTempCNNFitPredict:
    def test_fit_returns_trained_model(self):
        model = mlm_class_tempcnn(epochs=3, batch_size=16, seed=0)
        gdf = _make_training_gdf(n_features=10)
        trained = ml_fit(model, gdf, target="label")

        assert trained.trained
        assert trained._n_features == 10
        assert not model.trained  # original not mutated

    def test_fit_updates_input_metadata(self):
        model = mlm_class_tempcnn(epochs=3, batch_size=16, seed=0)
        gdf = _make_training_gdf(n_features=8)
        trained = ml_fit(model, gdf, target="label")

        inp = trained.inputs[0]
        assert len(inp.bands) == 8
        assert inp.input.shape == [-1, 8]

    def test_predict_returns_datacube(self):
        n_feat = 10
        model = mlm_class_tempcnn(epochs=3, batch_size=16, seed=0)
        gdf = _make_training_gdf(n_features=n_feat)
        trained = ml_fit(model, gdf, target="label")

        raster = _make_raster(n_bands=n_feat)
        preds = ml_predict(raster, trained)

        assert "predictions" in preds.dims
        assert "t" in preds.dims
        assert "x" in preds.dims
        assert "bands" not in preds.dims

    def test_predict_values_are_valid_labels(self):
        n_feat = 10
        model = mlm_class_tempcnn(epochs=5, batch_size=16, seed=0)
        gdf = _make_training_gdf(n_features=n_feat)
        trained = ml_fit(model, gdf, target="label")

        raster = _make_raster(n_bands=n_feat)
        preds = ml_predict(raster, trained)

        unique = np.unique(preds.values)
        assert all(v in [0, 1] for v in unique)


# ---------------------------------------------------------------
# ml_fit / ml_predict – LightTAE
# ---------------------------------------------------------------


class TestLightTAEFitPredict:
    def test_fit_returns_trained_model(self):
        model = mlm_class_lighttae(epochs=3, batch_size=16, seed=0)
        gdf = _make_training_gdf(n_features=10)
        trained = ml_fit(model, gdf, target="label")

        assert trained.trained
        assert trained._n_features == 10
        assert not model.trained

    def test_predict_returns_datacube(self):
        n_feat = 10
        model = mlm_class_lighttae(epochs=3, batch_size=16, seed=0)
        gdf = _make_training_gdf(n_features=n_feat)
        trained = ml_fit(model, gdf, target="label")

        raster = _make_raster(n_bands=n_feat)
        preds = ml_predict(raster, trained)

        assert "predictions" in preds.dims
        assert "t" in preds.dims
        assert "x" in preds.dims
        assert "bands" not in preds.dims

    def test_predict_values_are_valid_labels(self):
        n_feat = 10
        model = mlm_class_lighttae(epochs=5, batch_size=16, seed=0)
        gdf = _make_training_gdf(n_features=n_feat)
        trained = ml_fit(model, gdf, target="label")

        raster = _make_raster(n_bands=n_feat)
        preds = ml_predict(raster, trained)

        unique = np.unique(preds.values)
        assert all(v in [0, 1] for v in unique)


# ---------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------


class TestErrorHandling:
    def test_predict_untrained_raises(self):
        model = mlm_class_tempcnn(epochs=2)
        raster = _make_raster()
        with pytest.raises(RuntimeError, match="not been trained"):
            ml_predict(raster, model)

    def test_feature_mismatch_raises(self):
        model = mlm_class_tempcnn(epochs=3, batch_size=16, seed=0)
        gdf = _make_training_gdf(n_features=10)
        trained = ml_fit(model, gdf, target="label")

        raster = _make_raster(n_bands=5)
        with pytest.raises(ValueError, match="mismatch"):
            ml_predict(raster, trained)


# ---------------------------------------------------------------
# STAC MLM metadata
# ---------------------------------------------------------------


class TestSTACMLMMetadata:
    def test_tempcnn_stac_properties(self):
        model = mlm_class_tempcnn(epochs=10, seed=42)
        props = model.to_stac_properties()

        assert props["mlm:name"] == "TempCNN Classifier"
        assert props["mlm:architecture"] == "TempCNN"
        assert props["mlm:tasks"] == ["classification"]
        assert props["mlm:framework"] == "PyTorch"
        assert props["mlm:pretrained"] is False

    def test_lighttae_stac_properties(self):
        model = mlm_class_lighttae(epochs=10, seed=42)
        props = model.to_stac_properties()

        assert props["mlm:name"] == "LightTAE Classifier"
        assert props["mlm:architecture"] == "LightTAE"
        assert props["mlm:tasks"] == ["classification"]
        assert props["mlm:framework"] == "PyTorch"

    def test_stac_item_structure(self):
        model = mlm_class_tempcnn(epochs=2)
        item = model.to_stac_item()

        assert item["type"] == "Feature"
        assert "https://stac-extensions.github.io/mlm/v1.5.1/schema.json" in item["stac_extensions"]
        assert "model" in item["assets"]
        assert "mlm:model" in item["assets"]["model"]["roles"]


# ---------------------------------------------------------------
# save_ml_model / load_stac_ml round-trip
# ---------------------------------------------------------------


class TestSaveLoadRoundTrip:
    def test_tempcnn_save_and_load(self):
        model = mlm_class_tempcnn(epochs=3, batch_size=16, seed=0)
        gdf = _make_training_gdf(n_features=10)
        trained = ml_fit(model, gdf, target="label")

        with tempfile.TemporaryDirectory() as tmpdir:
            ok = save_ml_model(trained, "my_tempcnn", options={"directory": tmpdir})
            assert ok

            out_dir = Path(tmpdir) / "my_tempcnn"
            assert (out_dir / "model.pkl").exists()
            stac_file = out_dir / "my_tempcnn.stac.json"
            assert stac_file.exists()

            stac = json.loads(stac_file.read_text())
            assert stac["properties"]["mlm:name"] == "TempCNN Classifier"

            loaded = load_stac_ml(str(stac_file))
            assert loaded.trained
            assert loaded.framework == "PyTorch"

            raster = _make_raster(n_bands=10)
            preds = ml_predict(raster, loaded)
            assert "predictions" in preds.dims

    def test_lighttae_save_and_load(self):
        model = mlm_class_lighttae(epochs=3, batch_size=16, seed=0)
        gdf = _make_training_gdf(n_features=10)
        trained = ml_fit(model, gdf, target="label")

        with tempfile.TemporaryDirectory() as tmpdir:
            ok = save_ml_model(trained, "my_lighttae", options={"directory": tmpdir})
            assert ok

            stac_file = Path(tmpdir) / "my_lighttae" / "my_lighttae.stac.json"
            loaded = load_stac_ml(str(stac_file))
            assert loaded.trained
            assert loaded.architecture == "LightTAE"


# ---------------------------------------------------------------
# Model convenience factory
# ---------------------------------------------------------------


class TestModelFactory:
    def test_tempcnn(self):
        model = Model.tempcnn(epochs=5, batch_size=16, seed=0)
        assert isinstance(model, MLModel)
        assert model.tasks == ["classification"]
        assert model.architecture == "TempCNN"

    def test_lighttae(self):
        model = Model.lighttae(epochs=5, batch_size=16, seed=0)
        assert isinstance(model, MLModel)
        assert model.tasks == ["classification"]
        assert model.architecture == "LightTAE"


# ---------------------------------------------------------------
# Process registry
# ---------------------------------------------------------------


class TestProcessRegistry:
    def test_tempcnn_registered(self):
        from openeo_core.processes.registry import ProcessRegistry

        registry = ProcessRegistry()
        procs = registry.list_processes()
        assert "mlm_class_tempcnn" in procs

        spec = registry.get_process("mlm_class_tempcnn")
        assert spec["id"] == "mlm_class_tempcnn"
        assert "TempCNN" in spec["summary"]

    def test_lighttae_registered(self):
        from openeo_core.processes.registry import ProcessRegistry

        registry = ProcessRegistry()
        procs = registry.list_processes()
        assert "mlm_class_lighttae" in procs

        spec = registry.get_process("mlm_class_lighttae")
        assert spec["id"] == "mlm_class_lighttae"
        assert "LightTAE" in spec["summary"]

    def test_search_finds_torch_models(self):
        from openeo_core.processes.registry import ProcessRegistry

        registry = ProcessRegistry()
        results = registry.search("Temporal")
        ids = [r["id"] for r in results]
        assert "mlm_class_tempcnn" in ids
        assert "mlm_class_lighttae" in ids
