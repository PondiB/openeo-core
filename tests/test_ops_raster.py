"""Tests for openeo_core.ops.raster module."""

import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr

from openeo_core.ops.raster import (
    aggregate_spatial,
    aggregate_temporal,
    apply,
    filter_bbox,
    filter_temporal,
    ndvi,
    stack_to_samples,
    unstack_from_samples,
)


def _make_raster(dask_backed: bool = False) -> xr.DataArray:
    """Create a small test raster cube with (t, bands, y, x) dims."""
    np.random.seed(42)
    data = np.random.rand(2, 2, 4, 4).astype(np.float32)
    if dask_backed:
        data = da.from_array(data, chunks=(1, 2, 4, 4))  # type: ignore[assignment]
    return xr.DataArray(
        data,
        dims=["t", "bands", "y", "x"],
        coords={
            "t": pd.date_range("2023-01-01", periods=2, freq="ME"),
            "bands": ["red", "nir"],
            "y": np.linspace(50, 51, 4),
            "x": np.linspace(10, 11, 4),
        },
    )


# ---------------------------------------------------------------
# NDVI
# ---------------------------------------------------------------


class TestNDVI:
    def test_ndvi_drops_bands_dim(self):
        cube = _make_raster()
        result = ndvi(cube)
        assert "bands" not in result.dims

    def test_ndvi_target_band(self):
        cube = _make_raster()
        result = ndvi(cube, target_band="ndvi")
        assert "bands" in result.dims
        assert "ndvi" in result.coords["bands"].values

    def test_ndvi_values(self):
        cube = _make_raster()
        result = ndvi(cube)
        r = cube.sel(bands="red")
        n = cube.sel(bands="nir")
        expected = (n - r) / (n + r)
        np.testing.assert_allclose(result.values, expected.values)


# ---------------------------------------------------------------
# filter_bbox
# ---------------------------------------------------------------


class TestFilterBbox:
    def test_filter_bbox(self):
        cube = _make_raster()
        result = filter_bbox(cube, west=10.0, south=50.0, east=10.5, north=50.5)
        assert result.sizes["x"] <= cube.sizes["x"]
        assert result.sizes["y"] <= cube.sizes["y"]


# ---------------------------------------------------------------
# filter_temporal
# ---------------------------------------------------------------


class TestFilterTemporal:
    def test_filter_temporal(self):
        cube = _make_raster()
        result = filter_temporal(cube, extent=("2023-01-01", "2023-01-31"))
        assert result.sizes["t"] == 1


# ---------------------------------------------------------------
# aggregate_spatial
# ---------------------------------------------------------------


class TestAggregateSpatial:
    def test_aggregate_spatial_mean(self):
        cube = _make_raster()
        result = aggregate_spatial(cube, None, reducer="mean")
        assert "x" not in result.dims
        assert "y" not in result.dims


# ---------------------------------------------------------------
# aggregate_temporal
# ---------------------------------------------------------------


class TestAggregateTemporal:
    def test_aggregate_temporal_year(self):
        cube = _make_raster()
        result = aggregate_temporal(cube, period="year", reducer="mean")
        assert result.sizes["t"] <= cube.sizes["t"]


# ---------------------------------------------------------------
# apply
# ---------------------------------------------------------------


class TestApply:
    def test_apply_multiply(self):
        cube = _make_raster()
        result = apply(cube, lambda x: x * 2)
        np.testing.assert_allclose(result.values, cube.values * 2)


# ---------------------------------------------------------------
# stack / unstack utilities
# ---------------------------------------------------------------


class TestStackUnstack:
    def test_roundtrip(self):
        cube = _make_raster()
        stacked = stack_to_samples(cube, feature_dim="bands")
        assert stacked.dims == ("samples", "bands")
        assert stacked.sizes["bands"] == 2

    def test_dask_stays_lazy(self):
        cube = _make_raster(dask_backed=True)
        stacked = stack_to_samples(cube, feature_dim="bands")
        assert isinstance(stacked.data, da.Array), "Expected dask array to remain lazy"
