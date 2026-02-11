# openeo-core

A standalone Python library providing a fluent, Pythonic API for working with **raster data cubes** and **vector cubes**, implementing selected **openEO processes** locally using **xarray** and **dask**, with **STAC MLM-compatible** ML model objects.

## Installation

### Install from GitHub

```bash
# With uv
uv pip install git+https://github.com/PondiB/openeo-core.git

# With pip
pip install git+https://github.com/PondiB/openeo-core.git
```

Optional extras (ML backends, geo tools, dev):

```bash
# Geo tools (rioxarray for resample_spatial, xvec for vector data cubes)
uv pip install "openeo-core[geo] @ git+https://github.com/PondiB/openeo-core.git"

# ML backends
uv pip install "openeo-core[ml-sklearn,ml-xgboost] @ git+https://github.com/PondiB/openeo-core.git"

# Dev tools
pip install "openeo-core[dev] @ git+https://github.com/PondiB/openeo-core.git"
```

### Install from source (development)

Clone the repository and sync dependencies:

```bash
git clone https://github.com/PondiB/openeo-core.git
cd openeo-core

# Core install (xarray, dask, geopandas, pystac-client, stackstac)
uv sync

# With ML backends
uv sync --extra ml-sklearn
uv sync --extra ml-xgboost

# Everything including dev tools
uv sync --extra dev
```

## Quick Start

### Fluent DataCube API

```python
from openeo_core import DataCube

# Load from AWS Earth Search (Sentinel-2)
cube = DataCube.load_collection(
    "sentinel-2-l2a",
    spatial_extent={"west": 10.0, "south": 50.0, "east": 11.0, "north": 51.0},
    temporal_extent=("2023-06-01", "2023-06-30"),
    bands=["red", "nir"],
)

# Fluent chaining
result = (
    cube
    .filter_bbox(west=10.2, south=50.2, east=10.8, north=50.8)
    .filter_temporal(extent=("2023-06-10", "2023-06-20"))
    .ndvi(nir="nir", red="red")
    .compute()
)
```

### ML Models (openEO process-aligned, STAC MLM-compatible)

Model objects are **STAC MLM-compatible** and the API follows the openEO process specs exactly:

```python
from openeo_core.model import (
    mlm_class_random_forest,
    mlm_regr_random_forest,
    mlm_class_xgboost,
    ml_fit,
    ml_predict,
    save_ml_model,
    load_stac_ml,
)

# 1. Initialize (openEO: mlm_class_random_forest)
model = mlm_class_random_forest(
    max_variables="sqrt",  # or int, "all", "log2", "onethird"
    num_trees=200,
    seed=42,
)

# 2. Train (openEO: ml_fit)
#    training_gdf is a GeoDataFrame with feature columns + target column
trained = ml_fit(model, training_gdf, target="label")

# 3. Predict (openEO: ml_predict)
predictions = ml_predict(raster_cube, trained)
# Returns a DataArray with a "predictions" dimension

# 4. Save with STAC Item (openEO: save_ml_model)
save_ml_model(trained, name="my_rf_model")
# Creates: my_rf_model/model.pkl + my_rf_model/my_rf_model.stac.json

# 5. Load from STAC Item (openEO: load_stac_ml)
restored = load_stac_ml("my_rf_model/my_rf_model.stac.json")
predictions = ml_predict(new_raster, restored)
```

#### XGBoost classification

```python
model = mlm_class_xgboost(
    learning_rate=0.15,
    max_depth=5,
    min_child_weight=1,
    subsample=0.8,
    min_split_loss=1,
    seed=42,
)
trained = ml_fit(model, training_gdf, target="label")
```

#### STAC MLM metadata on model objects

Every model carries full STAC MLM metadata:

```python
model = mlm_class_random_forest(max_variables="sqrt", num_trees=100)
props = model.to_stac_properties()
# {
#   "mlm:name": "Random Forest Classifier",
#   "mlm:architecture": "Random Forest",
#   "mlm:tasks": ["classification"],
#   "mlm:framework": "scikit-learn",
#   "mlm:hyperparameters": {"max_variables": "sqrt", "num_trees": 100, "seed": null},
#   "mlm:input": [...],
#   "mlm:output": [...],
#   ...
# }

stac_item = model.to_stac_item()
# Full STAC Feature with MLM extension
```

#### Convenience factory (backward-compatible)

```python
from openeo_core.model import Model, ml_fit, ml_predict

model = Model.random_forest(task="classification", max_variables="sqrt", num_trees=200)
trained = ml_fit(model, gdf, target="label")
preds = ml_predict(raster, trained)
```

### Process Registry

```python
from openeo_core.processes import ProcessRegistry

registry = ProcessRegistry()
print(registry.list_processes())
ndvi_spec = registry.get_process("ndvi")
results = registry.search("vegetation")
```

### Load from STAC / GeoJSON

```python
cube = DataCube.load_stac(
    "https://earth-search.aws.element84.com/v1/collections/sentinel-2-l2a",
    assets=["red", "nir"],
)

vector = DataCube.load_geojson({"type": "FeatureCollection", "features": [...]})
```

### Vector cubes (GeoDataFrame and xvec)

Vector cubes can be GeoDataFrames or xarray DataArrays/Datasets with xvec geometry coordinates. Install the ``geo`` extra for xvec support:

```bash
uv pip install "openeo-core[geo]"
```

```python
import xarray as xr
from shapely.geometry import Point

# Create xvec-backed vector cube
da = xr.DataArray(
    [1.0, 2.0, 3.0],
    dims=["geom"],
    coords={"geom": [Point(10, 50), Point(10.5, 50.5), Point(11, 51)]},
).xvec.set_geom_indexes("geom", crs=4326)

cube = DataCube(da)
result = cube.filter_bbox(west=9, south=49, east=11, north=51)
```

## Architecture

```
openeo_core/
  __init__.py          # DataCube, type aliases
  datacube.py          # Fluent wrapper + dispatch
  types.py             # RasterCube/VectorCube/Cube aliases
  ops/
    raster.py          # xarray/dask raster operations
    vector.py          # geopandas, dask-geopandas, xvec vector operations
  io/
    collection.py      # load_collection (pystac-client + stackstac)
    stac.py            # load_stac (pystac + stackstac)
    geojson.py         # load_geojson (geopandas)
  model/
    __init__.py        # Public API exports
    mlm.py             # MLModel (STAC MLM-compatible object)
    base.py            # openEO process functions + Model factory
    sklearn.py         # scikit-learn estimator builder (internal)
    xgboost_backend.py # XGBoost estimator builder (internal)
    torch.py           # PyTorch wrapper (Phase 2 roadmap)
  processes/
    registry.py        # JSON spec registry
    resources/         # Packaged process JSON specs
```

### openEO ML Process Mapping

| openEO Process | Python Function | Description |
|---|---|---|
| `mlm_class_random_forest` | `mlm_class_random_forest()` | Init RF classifier |
| `mlm_regr_random_forest` | `mlm_regr_random_forest()` | Init RF regressor |
| `mlm_class_xgboost` | `mlm_class_xgboost()` | Init XGBoost classifier |
| `ml_fit` | `ml_fit(model, training_set, target)` | Train a model |
| `ml_predict` | `ml_predict(data, model)` | Predict with trained model |
| `save_ml_model` | `save_ml_model(data, name, options)` | Save model + STAC Item |
| `load_stac_ml` | `load_stac_ml(uri, ...)` | Load model from STAC Item |

## Running Tests

```bash
uv run pytest tests/ -v
```

## License

Apache-2.0
