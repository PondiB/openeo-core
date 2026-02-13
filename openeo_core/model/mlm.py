"""STAC MLM-compatible model object.

An ``MLModel`` instance carries **STAC Machine Learning Model (MLM)**
extension metadata alongside the underlying estimator.  It is the single
object that flows through the openEO ML processes:

* ``mlm_class_random_forest`` / ``mlm_regr_random_forest`` / ``mlm_class_xgboost`` → creates an untrained ``MLModel``
* ``ml_fit(model, training_set, target)`` → returns a trained ``MLModel``
* ``ml_predict(data, model)`` → returns a prediction data cube
* ``save_ml_model(model, name)`` → persists model + STAC Item JSON
* ``load_stac_ml(uri)`` → restores an ``MLModel`` from a STAC Item
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# STAC MLM sub-structures (mirrors the JSON schema definitions)
# ---------------------------------------------------------------------------


@dataclass
class InputStructure:
    """``mlm:input[].input`` -- shape / dim_order / data_type."""

    shape: list[int] = field(default_factory=lambda: [-1, -1])
    dim_order: list[str] = field(default_factory=lambda: ["batch", "bands"])
    data_type: str = "float32"

    def to_dict(self) -> dict[str, Any]:
        return {"shape": self.shape, "dim_order": self.dim_order, "data_type": self.data_type}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "InputStructure":
        return cls(shape=d["shape"], dim_order=d["dim_order"], data_type=d["data_type"])


@dataclass
class ModelInput:
    """One entry in ``mlm:input``."""

    name: str = "features"
    bands: list[str] = field(default_factory=list)
    input: InputStructure = field(default_factory=InputStructure)
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"name": self.name, "input": self.input.to_dict()}
        if self.bands:
            d["bands"] = self.bands
        if self.description:
            d["description"] = self.description
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ModelInput":
        return cls(
            name=d.get("name", "features"),
            bands=d.get("bands", []),
            input=InputStructure.from_dict(d["input"]) if "input" in d else InputStructure(),
            description=d.get("description", ""),
        )


@dataclass
class ResultStructure:
    """``mlm:output[].result`` -- shape / dim_order / data_type."""

    shape: list[int] = field(default_factory=lambda: [-1])
    dim_order: list[str] = field(default_factory=lambda: ["batch"])
    data_type: str = "float64"

    def to_dict(self) -> dict[str, Any]:
        return {"shape": self.shape, "dim_order": self.dim_order, "data_type": self.data_type}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ResultStructure":
        return cls(shape=d["shape"], dim_order=d["dim_order"], data_type=d["data_type"])


@dataclass
class ModelOutput:
    """One entry in ``mlm:output``."""

    name: str = "predictions"
    tasks: list[str] = field(default_factory=list)
    result: ResultStructure = field(default_factory=ResultStructure)
    classification_classes: list[Any] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "name": self.name,
            "tasks": self.tasks,
            "result": self.result.to_dict(),
        }
        if self.classification_classes:
            d["classification:classes"] = self.classification_classes
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ModelOutput":
        return cls(
            name=d.get("name", "predictions"),
            tasks=d.get("tasks", []),
            result=ResultStructure.from_dict(d["result"]) if "result" in d else ResultStructure(),
            classification_classes=d.get("classification:classes", []),
        )


# ---------------------------------------------------------------------------
# MLModel – the core model object
# ---------------------------------------------------------------------------


class MLModel:
    """A STAC MLM-compatible machine-learning model object.

    This is the ``ml-model`` subtype used throughout the openEO ML
    processes.  It wraps both STAC MLM **metadata** and the underlying
    framework **estimator**.

    STAC MLM fields (``properties`` level)::

        mlm:name, mlm:architecture, mlm:tasks, mlm:framework,
        mlm:framework_version, mlm:input, mlm:output,
        mlm:hyperparameters, mlm:pretrained, mlm:pretrained_source,
        mlm:total_parameters, mlm:memory_size, mlm:accelerator
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        *,
        # STAC MLM metadata
        name: str,
        architecture: str,
        tasks: list[str],
        framework: str,
        framework_version: str | None = None,
        inputs: list[ModelInput] | None = None,
        outputs: list[ModelOutput] | None = None,
        hyperparameters: dict[str, Any] | None = None,
        pretrained: bool = False,
        pretrained_source: str | None = None,
        total_parameters: int | None = None,
        memory_size: int | None = None,
        accelerator: str | None = None,
        accelerator_constrained: bool = False,
        # Spatial context (inferred from training data)
        bbox: list[float] | None = None,
        geometry: dict[str, Any] | None = None,
        # Internal
        estimator: Any = None,
        backend: str | None = None,
    ) -> None:
        # -- STAC MLM metadata --
        self.name = name
        self.architecture = architecture
        self.tasks = list(tasks)
        self.framework = framework
        self.framework_version = framework_version
        self.inputs = list(inputs) if inputs else [ModelInput()]
        self.outputs = list(outputs) if outputs else [ModelOutput(tasks=list(tasks))]
        self.hyperparameters = dict(hyperparameters) if hyperparameters else {}
        self.pretrained = pretrained
        self.pretrained_source = pretrained_source
        self.total_parameters = total_parameters
        self.memory_size = memory_size
        self.accelerator = accelerator
        self.accelerator_constrained = accelerator_constrained

        # -- spatial context --
        self.bbox = bbox
        self.geometry = geometry

        # -- internal --
        self._estimator = estimator
        self._backend = backend
        self._trained = False
        self._n_features: int | None = None
        self._feature_names: list[str] | None = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def trained(self) -> bool:
        """Whether ``ml_fit`` has been called on this model."""
        return self._trained

    @property
    def estimator(self) -> Any:
        """Access the underlying framework estimator (internal)."""
        return self._estimator

    # ------------------------------------------------------------------
    # STAC MLM serialisation
    # ------------------------------------------------------------------

    def to_stac_properties(self) -> dict[str, Any]:
        """Return the STAC Item ``properties`` dict with ``mlm:*`` fields."""
        props: dict[str, Any] = {
            "mlm:name": self.name,
            "mlm:architecture": self.architecture,
            "mlm:tasks": self.tasks,
            "mlm:framework": self.framework,
            "mlm:input": [inp.to_dict() for inp in self.inputs],
            "mlm:output": [out.to_dict() for out in self.outputs],
            "mlm:hyperparameters": self.hyperparameters,
            "mlm:pretrained": self.pretrained,
        }
        if self.framework_version is not None:
            props["mlm:framework_version"] = self.framework_version
        if self.pretrained_source is not None:
            props["mlm:pretrained_source"] = self.pretrained_source
        if self.total_parameters is not None:
            props["mlm:total_parameters"] = self.total_parameters
        if self.memory_size is not None:
            props["mlm:memory_size"] = self.memory_size
        if self.accelerator is not None:
            props["mlm:accelerator"] = self.accelerator
            props["mlm:accelerator_constrained"] = self.accelerator_constrained
        return props

    def to_stac_item(self, *, model_href: str = "model.pkl") -> dict[str, Any]:
        """Build a minimal STAC Item dict conforming to the MLM extension.

        When spatial metadata (``bbox`` / ``geometry``) has been
        attached (e.g. inferred from training data in ``ml_fit``), the
        STAC Item will include those fields instead of ``null``.
        """
        return {
            "type": "Feature",
            "stac_version": "1.1.0",
            "stac_extensions": [
                "https://stac-extensions.github.io/mlm/v1.5.1/schema.json",
            ],
            "id": self.name,
            "geometry": self.geometry,
            "bbox": self.bbox,
            "properties": {
                **self.to_stac_properties(),
            },
            "assets": {
                "model": {
                    "href": model_href,
                    "roles": ["mlm:model"],
                    "type": "application/octet-stream",
                }
            },
            "links": [],
        }

    @classmethod
    def from_stac_properties(
        cls,
        properties: dict[str, Any],
        *,
        estimator: Any = None,
        backend: str | None = None,
        bbox: list[float] | None = None,
        geometry: dict[str, Any] | None = None,
    ) -> "MLModel":
        """Reconstruct an ``MLModel`` from STAC Item ``properties``."""
        inputs = [ModelInput.from_dict(d) for d in properties.get("mlm:input", [])]
        outputs = [ModelOutput.from_dict(d) for d in properties.get("mlm:output", [])]

        model = cls(
            name=properties["mlm:name"],
            architecture=properties["mlm:architecture"],
            tasks=properties.get("mlm:tasks", []),
            framework=properties.get("mlm:framework", ""),
            framework_version=properties.get("mlm:framework_version"),
            inputs=inputs or None,
            outputs=outputs or None,
            hyperparameters=properties.get("mlm:hyperparameters"),
            pretrained=properties.get("mlm:pretrained", False),
            pretrained_source=properties.get("mlm:pretrained_source"),
            total_parameters=properties.get("mlm:total_parameters"),
            memory_size=properties.get("mlm:memory_size"),
            accelerator=properties.get("mlm:accelerator"),
            accelerator_constrained=properties.get("mlm:accelerator_constrained", False),
            bbox=bbox,
            geometry=geometry,
            estimator=estimator,
            backend=backend,
        )
        if estimator is not None:
            model._trained = True
        return model

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        status = "trained" if self._trained else "untrained"
        return (
            f"<MLModel name={self.name!r} arch={self.architecture!r} "
            f"tasks={self.tasks} framework={self.framework!r} [{status}]>"
        )
