"""Core type aliases for openeo-core data cubes."""

from __future__ import annotations

from typing import Union

import geopandas as gpd
import xarray as xr

try:
    import dask_geopandas
except ImportError:  # pragma: no cover
    dask_geopandas = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Public type aliases
# ---------------------------------------------------------------------------

RasterCube = xr.DataArray
"""A raster data cube – always an xarray DataArray (numpy or dask-backed)."""

VectorCube = Union[gpd.GeoDataFrame, "dask_geopandas.GeoDataFrame", xr.Dataset]
"""A vector data cube – GeoDataFrame, dask GeoDataFrame, or xarray Dataset."""

Cube = Union[RasterCube, VectorCube]
"""Any data cube type recognised by the library."""
