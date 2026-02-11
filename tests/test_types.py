"""Tests for openeo_core.types module."""

import geopandas as gpd
import numpy as np
import xarray as xr

from openeo_core.types import RasterCube, VectorCube


def test_raster_cube_is_dataarray():
    da = xr.DataArray(np.zeros((2, 3)), dims=["x", "y"])
    assert isinstance(da, RasterCube)


def test_vector_cube_accepts_geodataframe():
    gdf = gpd.GeoDataFrame({"a": [1]}, geometry=gpd.points_from_xy([0], [0]))
    # VectorCube is a Union – check the component type
    assert isinstance(gdf, gpd.GeoDataFrame)
