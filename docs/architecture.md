# openeo-core: Software & Architecture

This document describes the software structure, design decisions, and data flow of **openeo-core** — a standalone Python library for working with raster and vector data cubes, implementing selected openEO processes locally using xarray, dask, and STAC MLM-compatible ML models.

---

## 1. Overview

### 1.1 Purpose

openeo-core provides:

- **Fluent DataCube API** — Chain operations on raster and vector cubes with a Pythonic interface
- **openEO process alignment** — Implementations follow [openEO process specifications](https://processes.openeo.org/)
- **Local execution** — Uses xarray, dask, geopandas, stackstac for in-process computation (no backend server)
- **STAC MLM compatibility** — Machine learning models carry full STAC Machine Learning Model extension metadata and can be serialized/loaded as STAC Items

### 1.2 Core Dependencies

| Package | Role |
|---------|------|
| **xarray** | Core data structure for raster cubes (DataArray) |
| **dask** | Lazy evaluation and out-of-core processing |
| **geopandas** | Vector cube representation (GeoDataFrame) |
| **xvec** | Geometry coordinates in xarray (vector cubes, zonal stats) |
| **stackstac** | Stack STAC Items into lazy xarray DataArrays |
| **pystac-client** | STAC API search (e.g. AWS Earth Search) |
| **planetary-computer** | SAS token signing for Microsoft Planetary Computer |
| **shapely** | Geometry handling |

Optional: **scikit-learn**, **xgboost** (ML), **rioxarray** (resampling), **dask-geopandas** (distributed vectors).

---

## 2. Package Structure

```
openeo_core/
├── __init__.py          # DataCube, Cube, RasterCube, VectorCube, exceptions
├── datacube.py          # Fluent wrapper + runtime dispatch (raster vs vector)
├── types.py             # Type aliases (RasterCube, VectorCube, Cube)
├── exceptions/          # openEO-aligned exception types
│   └── ndvi.py          # BandExists, DimensionAmbiguous, NirBandAmbiguous, RedBandAmbiguous
├── ops/                 # Operation implementations
│   ├── raster.py        # xarray/dask raster operations
│   └── vector.py        # geopandas, dask-geopandas, xvec vector operations
├── io/                  # Data loading
│   ├── collection.py   # load_collection (pystac-client + stackstac)
│   ├── stac.py          # load_stac (arbitrary STAC sources)
│   └── geojson.py       # load_geojson (geopandas)
├── model/               # Machine learning
│   ├── __init__.py      # Public API: mlm_*, ml_fit, ml_predict, save_ml_model, load_stac_ml
│   ├── mlm.py           # MLModel (STAC MLM-compatible object)
│   ├── base.py          # openEO process functions + Model factory
│   ├── sklearn.py       # scikit-learn estimator builder
│   ├── xgboost_backend.py # XGBoost estimator builder
│   └── torch.py         # PyTorch wrapper (roadmap)
└── processes/           # Process specification registry
    ├── registry.py      # ProcessRegistry: list, get, search
    └── resources/process_specs/current/  # Packaged JSON specs
```

---

## 3. Type System

### 3.1 Cube Types

Defined in `openeo_core/types.py`:

| Type | Representation | Notes |
|------|----------------|-------|
| **RasterCube** | `xr.DataArray` | Always xarray DataArray (numpy or dask-backed) |
| **VectorCube** | `gpd.GeoDataFrame` \| `dask_geopandas.GeoDataFrame` \| `xr.DataArray` \| `xr.Dataset` | GeoDataFrame or xarray with xvec geometry coordinates |
| **Cube** | `Union[RasterCube, VectorCube]` | Any cube type recognized by the library |

### 3.2 Raster vs Vector Detection

The `DataCube` wrapper detects cube type via:

- **Raster**: `xr.DataArray` without xvec geometry coordinates
- **Vector**: `gpd.GeoDataFrame`, `dask_geopandas.GeoDataFrame`, or `xr.DataArray`/`xr.Dataset` with `xvec.geom_coords` / `xvec.geom_coords_indexed`

---

## 4. DataCube: Fluent Wrapper

### 4.1 Design

`DataCube` is an **immutable wrapper** around a `RasterCube` or `VectorCube`. Every operation returns a **new** `DataCube` instance; the original is never mutated.

### 4.2 Runtime Dispatch

Methods dispatch to the correct implementation based on `is_raster` / `is_vector`:

| Method | Raster | Vector |
|--------|--------|--------|
| `filter_bbox` | `ops.raster.filter_bbox` | `ops.vector.filter_bbox` |
| `apply` | `ops.raster.apply` | `ops.vector.apply` |
| `ndvi`, `filter_temporal`, `aggregate_spatial`, etc. | raster-only | N/A |

### 4.3 Loaders (Classmethods)

| Classmethod | Module | Returns |
|-------------|--------|---------|
| `load_collection(collection_id, ...)` | `io.collection` | Raster `DataCube` from STAC API (Earth Search default) |
| `load_stac(source, ...)` | `io.stac` | Raster `DataCube` from arbitrary STAC source |
| `load_geojson(source, ...)` | `io.geojson` | Vector `DataCube` (GeoDataFrame) |

### 4.4 Materialisation

- `DataCube.compute()` — Materialises dask-backed data into memory; returns a new `DataCube` wrapping the computed result.
- Lazy evaluation is preserved until `.compute()` is called.

---

## 5. I/O Layer

### 5.1 load_collection

- **Default adapter**: `AWSCollectionLoader` (Element 84 Earth Search STAC API)
- **Flow**: `pystac_client.Client.open()` → `search()` with bbox, datetime, collections → `stackstac.stack()` → `xr.DataArray`
- **Protocol**: `CollectionLoader` — users can inject custom loaders via `adapter=`.

### 5.2 load_stac

- **Default adapter**: `DefaultStacLoader`
- **Accepts**: URL, file path, inline STAC JSON (Item, ItemCollection, Collection)
- **Flow**: Resolve items → filter by `temporal_extent` → `stackstac.stack()` → `xr.DataArray`
- Supports `spatial_extent` clipping.

### 5.3 load_geojson

- Uses geopandas `gpd.read_file()` (supports URL, path, dict).
- Returns vector `DataCube` (GeoDataFrame).

---

## 6. Raster Operations (`ops/raster.py`)

### 6.1 Implemented Processes

| Process | Description |
|---------|-------------|
| `ndvi` | Normalized Difference Vegetation Index: `(nir - red) / (nir + red)` |
| `filter_bbox` | Clip to bounding box (xarray `.sel`) |
| `filter_temporal` | Clip to temporal extent |
| `aggregate_spatial` | Zonal statistics over geometries (xvec `zonal_stats`) or full raster reducer |
| `aggregate_temporal` | Aggregate over calendar periods (hour, day, week, dekad, month, season, year, decade) |
| `resample_spatial` | Reproject/resample (rioxarray; optional `[geo]`) |
| `apply` | Element-wise function application |

### 6.2 NDVI Behaviour

- Nodata handling: non-positive values treated as nodata (Sentinel-2 L2A convention)
- Result clamped to `[-1, 1]`
- Optional `target_band`: append NDVI as new band or replace bands dimension

### 6.3 aggregate_spatial

- **With geometries**: GeoDataFrame or GeoJSON → xvec `zonal_stats` with CRS alignment, bounds filtering
- **Without geometries**: Full-raster reducer (mean, sum, min, max, median)
- Returns GeoDataFrame with band stats and preserved metadata when geometries are provided

### 6.4 aggregate_temporal

- Supported periods: `hour`, `day`, `week`, `dekad`, `month`, `season`, `tropical-season`, `year`, `decade`, `decade-ad`
- Reducers: `mean`, `sum`, `min`, `max`, `median`
- Uses pandas `Grouper` with appropriate `freq` for period labelling

---

## 7. Vector Operations (`ops/vector.py`)

| Process | Description |
|---------|-------------|
| `filter_bbox` | Clip geometries to bounding box |
| `apply` | Apply function to vector cube |
| `to_feature_matrix` | Convert vector cube to feature matrix for ML |

---

## 8. Machine Learning Model System

### 8.1 openEO Process Flow

```
mlm_class_random_forest() / mlm_regr_random_forest() / mlm_class_xgboost()
        ↓
    (untrained MLModel)
        ↓
ml_fit(model, training_set, target)
        ↓
    (trained MLModel)
        ↓
ml_predict(raster_cube, model)
        ↓
    (RasterCube with "predictions" dimension)
```

Serialisation:

```
save_ml_model(model, name)  →  name/model.pkl + name/name.stac.json
load_stac_ml(uri)           →  MLModel
```

### 8.2 MLModel Object

`MLModel` (in `model/mlm.py`) carries:

- **Estimator** — underlying scikit-learn/XGBoost object
- **STAC MLM metadata** — name, architecture, tasks, framework, hyperparameters, inputs, outputs
- **Methods**: `to_stac_properties()`, `to_stac_item()` for STAC Item generation

### 8.3 Backends

| Backend | Module | Use |
|---------|--------|-----|
| scikit-learn | `model.sklearn` | Random Forest (classification, regression) |
| XGBoost | `model.xgboost_backend` | XGBoost classification |
| PyTorch | `model.torch` | Roadmap |

### 8.4 ml_predict Data Flow

1. Raster cube: `(bands, y, x)` or `(time, bands, y, x)` — flattened to `(samples, bands)`
2. Model predicts on feature matrix
3. Output reshaped to match spatial (and optionally temporal) structure with a `predictions` dimension/coordinate

---

## 9. Process Registry

`ProcessRegistry` (`processes/registry.py`):

- Loads JSON specs from `openeo_core.processes.resources.process_specs.current`
- `list_processes()` — sorted process IDs
- `get_process(id)` — full JSON spec
- `search(text)` — case-insensitive search in id and description

Specs mirror the openEO process definitions and are used for discovery and documentation.

---

## 10. Data Flow Summary

```
STAC API / GeoJSON
       ↓
  io.collection / io.stac / io.geojson
       ↓
  DataCube(raster | vector)
       ↓
  ops.raster / ops.vector (immutable chaining)
       ↓
  .compute() → materialised result
```

ML path:

```
GeoDataFrame (training) + mlm_*()  →  ml_fit  →  trained MLModel
RasterCube + trained MLModel       →  ml_predict  →  RasterCube with predictions
```

---

## 11. File Layout (docs/)

```
docs/
├── architecture.md       # This document
├── openeo-processes/     # openEO process JSON specs (reference)
└── stac-mlm/             # STAC MLM extension schema
```

---

## 12. Extension Points

| Extension | Mechanism |
|-----------|-----------|
| Custom collection loader | `DataCube.load_collection(..., adapter=MyCollectionLoader)` |
| Custom STAC loader | `DataCube.load_stac(..., adapter=MyStacLoader)` |
| New ML backend | Implement estimator builder; register in `model/base.py` |
| New process | Add to `ops/raster.py` or `ops/vector.py`; add `DataCube` method; add JSON spec to `process_specs/current/` |
