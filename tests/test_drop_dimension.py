"""Tests for the drop_dimension process."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from openeo_core import DataCube
from openeo_core.exceptions import DimensionLabelCountMismatch, DimensionNotAvailable
from openeo_core.ops.raster import drop_dimension


def _make_raster() -> xr.DataArray:
    """Create a small test raster cube with (time, bands, latitude, longitude)."""
    np.random.seed(42)
    data = np.random.rand(2, 3, 4, 4).astype(np.float32)
    return xr.DataArray(
        data,
        dims=["time", "bands", "latitude", "longitude"],
        coords={
            "time": pd.date_range("2023-01-01", periods=2, freq="ME"),
            "bands": ["red", "green", "nir"],
            "latitude": np.linspace(50, 51, 4),
            "longitude": np.linspace(10, 11, 4),
        },
    )


def _single_time_raster() -> xr.DataArray:
    """Raster with a single time label (time dimension length 1)."""
    np.random.seed(99)
    data = np.random.rand(1, 3, 4, 4).astype(np.float32)
    return xr.DataArray(
        data,
        dims=["time", "bands", "latitude", "longitude"],
        coords={
            "time": pd.date_range("2023-01-01", periods=1, freq="ME"),
            "bands": ["red", "green", "nir"],
            "latitude": np.linspace(50, 51, 4),
            "longitude": np.linspace(10, 11, 4),
        },
    )


class TestDropDimension:
    def test_drop_single_label_dimension(self):
        cube = _single_time_raster()
        result = drop_dimension(cube, name="time")
        assert "time" not in result.dims
        assert set(result.dims) == {"bands", "latitude", "longitude"}
        np.testing.assert_allclose(result.values, cube.isel(time=0).values)

    def test_datacube_method(self):
        cube = DataCube(_single_time_raster())
        out = cube.drop_dimension(name="time")
        assert isinstance(out, DataCube)
        assert "time" not in out.data.dims

    def test_dimension_not_available(self):
        cube = _make_raster()
        with pytest.raises(DimensionNotAvailable, match="does not exist"):
            drop_dimension(cube, name="extras")

    def test_label_count_mismatch_multi_label(self):
        cube = _make_raster()
        with pytest.raises(
            DimensionLabelCountMismatch,
            match="exceeds one",
        ):
            drop_dimension(cube, name="time")
