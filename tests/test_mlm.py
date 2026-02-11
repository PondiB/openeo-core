"""Tests for the MLModel STAC MLM-compatible object."""

from openeo_core.model.mlm import (
    InputStructure,
    MLModel,
    ModelInput,
    ModelOutput,
    ResultStructure,
)


class TestInputStructure:
    def test_roundtrip(self):
        s = InputStructure(shape=[-1, 3], dim_order=["batch", "bands"], data_type="float32")
        d = s.to_dict()
        s2 = InputStructure.from_dict(d)
        assert s2.shape == [-1, 3]
        assert s2.dim_order == ["batch", "bands"]
        assert s2.data_type == "float32"


class TestModelInput:
    def test_roundtrip(self):
        mi = ModelInput(name="features", bands=["B04", "B08"])
        d = mi.to_dict()
        mi2 = ModelInput.from_dict(d)
        assert mi2.name == "features"
        assert mi2.bands == ["B04", "B08"]


class TestModelOutput:
    def test_roundtrip(self):
        mo = ModelOutput(name="predictions", tasks=["classification"], classification_classes=[0, 1])
        d = mo.to_dict()
        assert d["classification:classes"] == [0, 1]
        mo2 = ModelOutput.from_dict(d)
        assert mo2.tasks == ["classification"]


class TestMLModel:
    def test_construction(self):
        model = MLModel(
            name="Test Model",
            architecture="Random Forest",
            tasks=["classification"],
            framework="scikit-learn",
        )
        assert model.name == "Test Model"
        assert not model.trained

    def test_stac_properties(self):
        model = MLModel(
            name="Test RF",
            architecture="Random Forest",
            tasks=["classification"],
            framework="scikit-learn",
            framework_version="1.3.0",
            hyperparameters={"n_estimators": 100},
            pretrained=False,
        )
        props = model.to_stac_properties()
        assert props["mlm:name"] == "Test RF"
        assert props["mlm:architecture"] == "Random Forest"
        assert props["mlm:tasks"] == ["classification"]
        assert props["mlm:framework"] == "scikit-learn"
        assert props["mlm:framework_version"] == "1.3.0"
        assert props["mlm:hyperparameters"] == {"n_estimators": 100}
        assert props["mlm:pretrained"] is False

    def test_stac_item_structure(self):
        model = MLModel(
            name="Test Model",
            architecture="XGBoost",
            tasks=["regression"],
            framework="XGBoost",
        )
        item = model.to_stac_item()
        assert item["type"] == "Feature"
        assert item["stac_version"] == "1.1.0"
        assert "https://stac-extensions.github.io/mlm/v1.5.1/schema.json" in item["stac_extensions"]
        assert "model" in item["assets"]
        assert "mlm:model" in item["assets"]["model"]["roles"]

    def test_from_stac_properties(self):
        props = {
            "mlm:name": "Restored Model",
            "mlm:architecture": "Random Forest",
            "mlm:tasks": ["classification"],
            "mlm:framework": "scikit-learn",
            "mlm:input": [{"name": "features", "bands": ["B04"], "input": {"shape": [-1, 1], "dim_order": ["batch", "bands"], "data_type": "float32"}}],
            "mlm:output": [{"name": "predictions", "tasks": ["classification"], "result": {"shape": [-1], "dim_order": ["batch"], "data_type": "int64"}}],
            "mlm:hyperparameters": {"n_estimators": 50},
            "mlm:pretrained": True,
        }
        model = MLModel.from_stac_properties(props, estimator="fake")
        assert model.name == "Restored Model"
        assert model.trained  # because estimator was provided
        assert model.inputs[0].bands == ["B04"]
        assert model.outputs[0].tasks == ["classification"]

    def test_repr(self):
        model = MLModel(
            name="RF",
            architecture="Random Forest",
            tasks=["classification"],
            framework="scikit-learn",
        )
        r = repr(model)
        assert "RF" in r
        assert "untrained" in r
