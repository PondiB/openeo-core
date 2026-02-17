"""Core type aliases for openeo-core data cubes."""

from __future__ import annotations

from typing import Union

import geopandas as gpd
import xarray as xr
import dask_geopandas

# ---------------------------------------------------------------------------
# Public type aliases
# ---------------------------------------------------------------------------

RasterCube = xr.DataArray
"""A raster data cube – always an xarray DataArray (numpy or dask-backed)."""

VectorCube = Union[
    gpd.GeoDataFrame,
    dask_geopandas.GeoDataFrame,
    xr.DataArray,
    xr.Dataset,
]
"""A vector data cube – GeoDataFrame, dask GeoDataFrame, or xarray DataArray/Dataset
with geometry coordinates (xvec format). xarray types require the ``xvec`` package
and must have geometry-backed dimensions."""

Cube = Union[RasterCube, VectorCube]
"""Any data cube type recognised by the library."""
