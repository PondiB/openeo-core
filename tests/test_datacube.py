"""Tests for the DataCube fluent wrapper."""

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from shapely.geometry import Point

from openeo_core.datacube import DataCube


def _make_raster_da() -> xr.DataArray:
    np.random.seed(0)
    return xr.DataArray(
        np.random.rand(2, 2, 4, 4).astype(np.float32),
        dims=["t", "bands", "y", "x"],
        coords={
            "t": pd.date_range("2023-01-01", periods=2, freq="ME"),
            "bands": ["red", "nir"],
            "y": np.linspace(50, 51, 4),
            "x": np.linspace(10, 11, 4),
        },
    )


def _make_vector_gdf() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {"val": [1, 2]},
        geometry=[Point(0, 0), Point(1, 1)],
        crs="EPSG:4326",
    )


class TestDataCubeRaster:
    def test_is_raster(self):
        cube = DataCube(_make_raster_da())
        assert cube.is_raster
        assert not cube.is_vector

    def test_ndvi_fluent(self):
        cube = DataCube(_make_raster_da())
        result = cube.ndvi(nir="nir", red="red")
        assert isinstance(result, DataCube)
        assert result.is_raster
        assert "bands" not in result.data.dims

    def test_filter_bbox_fluent(self):
        cube = DataCube(_make_raster_da())
        result = cube.filter_bbox(west=10, south=50, east=10.5, north=50.5)
        assert result.data.sizes["x"] <= 4

    def test_filter_temporal_fluent(self):
        cube = DataCube(_make_raster_da())
        result = cube.filter_temporal(extent=("2023-01-01", "2023-01-31"))
        assert result.data.sizes["t"] == 1

    def test_apply_fluent(self):
        cube = DataCube(_make_raster_da())
        result = cube.apply(lambda x: x * 10)
        np.testing.assert_allclose(result.data.values, cube.data.values * 10)

    def test_aggregate_spatial_fluent(self):
        cube = DataCube(_make_raster_da())
        result = cube.aggregate_spatial(reducer="mean")
        assert "x" not in result.data.dims

    def test_aggregate_temporal_fluent(self):
        cube = DataCube(_make_raster_da())
        result = cube.aggregate_temporal(period="year", reducer="mean")
        assert result.data.sizes["t"] <= 2

    def test_chain(self):
        """Test fluent chaining of multiple operations."""
        cube = DataCube(_make_raster_da())
        result = (
            cube.filter_bbox(west=10, south=50, east=10.5, north=50.5)
            .filter_temporal(extent=("2023-01-01", "2023-01-31"))
            .ndvi()
        )
        assert result.is_raster
        assert "bands" not in result.data.dims

    def test_compute_noop_for_numpy(self):
        cube = DataCube(_make_raster_da())
        computed = cube.compute()
        # Non-dask data should pass through without error;
        # the underlying DataArray values should be identical.
        import numpy as np
        np.testing.assert_array_equal(computed.data.values, cube.data.values)


class TestDataCubeVector:
    def test_is_vector(self):
        cube = DataCube(_make_vector_gdf())
        assert cube.is_vector
        assert not cube.is_raster

    def test_filter_bbox_vector(self):
        cube = DataCube(_make_vector_gdf())
        result = cube.filter_bbox(west=-1, south=-1, east=2, north=2)
        assert result.is_vector

    def test_repr(self):
        r = repr(DataCube(_make_raster_da()))
        assert "Raster" in r
        v = repr(DataCube(_make_vector_gdf()))
        assert "Vector" in v
