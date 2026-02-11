"""Tests for the DataCube fluent wrapper."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import geopandas as gpd
from shapely.geometry import Point

from openeo_core.datacube import DataCube
import importlib.util


def _make_raster_da() -> xr.DataArray:
    np.random.seed(0)
    return xr.DataArray(
        np.random.rand(2, 2, 4, 4).astype(np.float32),
        dims=["time", "bands", "latitude", "longitude"],
        coords={
            "time": pd.date_range("2023-01-01", periods=2, freq="ME"),
            "bands": ["red", "nir"],
            "latitude": np.linspace(50, 51, 4),
            "longitude": np.linspace(10, 11, 4),
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
        assert result.data.sizes["longitude"] <= 4

    def test_filter_temporal_fluent(self):
        cube = DataCube(_make_raster_da())
        result = cube.filter_temporal(extent=("2023-01-01", "2023-01-31"))
        assert result.data.sizes["time"] == 1

    def test_apply_fluent(self):
        cube = DataCube(_make_raster_da())
        result = cube.apply(lambda x: x * 10)
        np.testing.assert_allclose(result.data.values, cube.data.values * 10)

    def test_aggregate_spatial_fluent(self):
        cube = DataCube(_make_raster_da())
        result = cube.aggregate_spatial(reducer="mean")
        assert "longitude" not in result.data.dims

    def test_aggregate_temporal_fluent(self):
        cube = DataCube(_make_raster_da())
        result = cube.aggregate_temporal(period="year", reducer="mean")
        assert result.data.sizes["time"] <= 2

    def test_aggregate_temporal_period_alias(self):
        """aggregate_temporal_period should be an alias for aggregate_temporal."""
        cube = DataCube(_make_raster_da())
        result = cube.aggregate_temporal_period(period="month", reducer="mean")
        assert isinstance(result, DataCube)
        assert result.is_raster

    def test_aggregate_temporal_dimension_kwarg(self):
        """The dimension= parameter should override t_dim."""
        cube = DataCube(_make_raster_da())
        result = cube.aggregate_temporal(period="month", reducer="mean", dimension="time")
        assert isinstance(result, DataCube)

    def test_aggregate_temporal_dekad(self):
        cube = DataCube(_make_raster_da())
        result = cube.aggregate_temporal(period="dekad", reducer="mean")
        assert isinstance(result, DataCube)

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


_HAS_RIOXARRAY = importlib.util.find_spec("rioxarray") is not None


@pytest.mark.skipif(not _HAS_RIOXARRAY, reason="rioxarray not installed")
class TestDataCubeResampleSpatial:
    def _make_geo_cube(self) -> DataCube:
        np.random.seed(7)
        da = xr.DataArray(
            np.random.rand(2, 10, 10).astype(np.float32),
            dims=["bands", "latitude", "longitude"],
            coords={
                "bands": ["red", "nir"],
                "latitude": np.linspace(51, 50, 10),
                "longitude": np.linspace(10, 11, 10),
            },
        )
        import rioxarray  # noqa: F811
        da = da.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude")
        da = da.rio.write_crs("EPSG:4326")
        return DataCube(da)

    def test_resample_spatial_reproject(self):
        cube = self._make_geo_cube()
        result = cube.resample_spatial(projection=3857)
        assert isinstance(result, DataCube)
        assert result.data.rio.crs.to_epsg() == 3857

    def test_resample_spatial_resolution(self):
        cube = self._make_geo_cube()
        result = cube.resample_spatial(resolution=0.5)
        assert result.data.sizes["longitude"] < cube.data.sizes["longitude"]

    def test_resample_spatial_requires_raster(self):
        gdf = gpd.GeoDataFrame(
            {"val": [1]}, geometry=[Point(0, 0)], crs="EPSG:4326"
        )
        cube = DataCube(gdf)
        with pytest.raises(TypeError, match="resample_spatial.*requires a raster"):
            cube.resample_spatial(projection=3857)


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
