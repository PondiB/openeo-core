"""Tests for reduce_dimension and apply_kernel processes."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from openeo_core.exceptions import (
    DimensionNotAvailable,
    KernelDimensionsUneven,
)
from openeo_core.ops.raster import apply_kernel, reduce_dimension


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# reduce_dimension
# ---------------------------------------------------------------------------


class TestReduceDimension:
    def test_reduce_bands_with_mean(self):
        cube = _make_raster()
        result = reduce_dimension(cube, np.mean, dimension="bands")
        assert "bands" not in result.dims
        assert set(result.dims) == {"time", "latitude", "longitude"}

    def test_reduce_time_with_sum(self):
        cube = _make_raster()
        result = reduce_dimension(cube, np.sum, dimension="time")
        assert "time" not in result.dims
        assert set(result.dims) == {"bands", "latitude", "longitude"}

    def test_reduce_values_correct(self):
        """Verify the reduced values match a manual computation."""
        cube = _make_raster()
        result = reduce_dimension(cube, np.mean, dimension="bands")
        expected = cube.values.mean(axis=1)  # bands is axis 1
        np.testing.assert_allclose(result.values, expected, rtol=1e-5)

    def test_reduce_with_max(self):
        cube = _make_raster()
        result = reduce_dimension(cube, np.max, dimension="bands")
        expected = cube.values.max(axis=1)
        np.testing.assert_allclose(result.values, expected)

    def test_reduce_with_min(self):
        cube = _make_raster()
        result = reduce_dimension(cube, np.min, dimension="time")
        expected = cube.values.min(axis=0)
        np.testing.assert_allclose(result.values, expected)

    def test_reduce_with_median(self):
        cube = _make_raster()
        result = reduce_dimension(cube, np.median, dimension="bands")
        expected = np.median(cube.values, axis=1)
        np.testing.assert_allclose(result.values, expected)

    def test_dimension_not_available(self):
        cube = _make_raster()
        with pytest.raises(DimensionNotAvailable, match="does not exist"):
            reduce_dimension(cube, np.mean, dimension="nonexistent")

    def test_preserves_other_coords(self):
        cube = _make_raster()
        result = reduce_dimension(cube, np.mean, dimension="bands")
        assert "time" in result.coords
        assert "latitude" in result.coords
        assert "longitude" in result.coords

    # --- String-based reducers ---

    def test_string_mean(self):
        cube = _make_raster()
        result = reduce_dimension(cube, "mean", dimension="bands")
        expected = cube.values.mean(axis=1)
        assert "bands" not in result.dims
        np.testing.assert_allclose(result.values, expected, rtol=1e-5)

    def test_string_sum(self):
        cube = _make_raster()
        result = reduce_dimension(cube, "sum", dimension="time")
        expected = cube.values.sum(axis=0)
        assert "time" not in result.dims
        np.testing.assert_allclose(result.values, expected, rtol=1e-5)

    def test_string_min(self):
        cube = _make_raster()
        result = reduce_dimension(cube, "min", dimension="bands")
        expected = cube.values.min(axis=1)
        np.testing.assert_allclose(result.values, expected)

    def test_string_max(self):
        cube = _make_raster()
        result = reduce_dimension(cube, "max", dimension="bands")
        expected = cube.values.max(axis=1)
        np.testing.assert_allclose(result.values, expected)

    def test_string_median(self):
        cube = _make_raster()
        result = reduce_dimension(cube, "median", dimension="bands")
        expected = np.median(cube.values, axis=1)
        np.testing.assert_allclose(result.values, expected)

    def test_string_std(self):
        cube = _make_raster()
        result = reduce_dimension(cube, "std", dimension="bands")
        expected = cube.values.std(axis=1)
        np.testing.assert_allclose(result.values, expected, rtol=1e-5)

    def test_string_var(self):
        cube = _make_raster()
        result = reduce_dimension(cube, "var", dimension="bands")
        expected = cube.values.var(axis=1)
        np.testing.assert_allclose(result.values, expected, rtol=1e-5)

    def test_string_count(self):
        cube = _make_raster()
        result = reduce_dimension(cube, "count", dimension="bands")
        # All values are finite, so count == number of bands (3)
        assert np.all(result.values == 3)

    def test_string_average_alias(self):
        cube = _make_raster()
        result_avg = reduce_dimension(cube, "average", dimension="bands")
        result_mean = reduce_dimension(cube, "mean", dimension="bands")
        np.testing.assert_allclose(result_avg.values, result_mean.values)

    def test_dotted_path_numpy_nanmean(self):
        cube = _make_raster()
        result = reduce_dimension(cube, "numpy.nanmean", dimension="bands")
        expected = np.nanmean(cube.values, axis=1)
        np.testing.assert_allclose(result.values, expected, rtol=1e-5)

    def test_dotted_path_numpy_nansum(self):
        cube = _make_raster()
        result = reduce_dimension(cube, "numpy.nansum", dimension="time")
        expected = np.nansum(cube.values, axis=0)
        np.testing.assert_allclose(result.values, expected, rtol=1e-5)

    def test_unknown_string_raises(self):
        cube = _make_raster()
        with pytest.raises(ValueError, match="Unknown reducer"):
            reduce_dimension(cube, "not_a_reducer", dimension="bands")

    def test_invalid_dotted_path_raises(self):
        cube = _make_raster()
        with pytest.raises(ValueError, match="Unknown reducer"):
            reduce_dimension(cube, "fake.module.func", dimension="bands")


# ---------------------------------------------------------------------------
# apply_kernel
# ---------------------------------------------------------------------------


class TestApplyKernel:
    def test_identity_kernel(self):
        """A kernel that only passes through the center value."""
        cube = _make_raster()
        # Identity kernel: only center pixel weight = 1
        kernel = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
        result = apply_kernel(cube, kernel=kernel)
        np.testing.assert_allclose(result.values, cube.values, atol=1e-6)

    def test_factor(self):
        """Factor should scale the convolved result."""
        cube = _make_raster()
        kernel = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
        result = apply_kernel(cube, kernel=kernel, factor=2.0)
        np.testing.assert_allclose(result.values, cube.values * 2.0, atol=1e-5)

    def test_uniform_kernel(self):
        """A 3x3 box blur should produce smoothed output."""
        cube = _make_raster()
        kernel = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
        result = apply_kernel(cube, kernel=kernel, factor=1.0 / 9.0)
        # Result should be smoother — just verify shape is preserved
        assert result.shape == cube.shape

    def test_preserves_dims(self):
        cube = _make_raster()
        kernel = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
        result = apply_kernel(cube, kernel=kernel)
        assert result.dims == cube.dims

    def test_kernel_even_dimensions_raises(self):
        cube = _make_raster()
        with pytest.raises(KernelDimensionsUneven, match="uneven"):
            apply_kernel(cube, kernel=[[1, 1], [1, 1]])

    def test_kernel_even_rows_raises(self):
        cube = _make_raster()
        with pytest.raises(KernelDimensionsUneven, match="uneven"):
            apply_kernel(cube, kernel=[[1, 1, 1], [1, 1, 1]])

    def test_kernel_even_cols_raises(self):
        cube = _make_raster()
        with pytest.raises(KernelDimensionsUneven, match="uneven"):
            apply_kernel(cube, kernel=[[1, 1], [1, 1], [1, 1]])

    def test_spatial_dim_not_available(self):
        cube = _make_raster()
        with pytest.raises(DimensionNotAvailable, match="does not exist"):
            apply_kernel(cube, kernel=[[0, 0, 0], [0, 1, 0], [0, 0, 0]], x_dim="x")

    def test_border_replicate(self):
        """Border mode 'replicate' should not raise."""
        cube = _make_raster()
        kernel = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
        result = apply_kernel(cube, kernel=kernel, border="replicate")
        assert result.shape == cube.shape

    def test_border_reflect(self):
        cube = _make_raster()
        kernel = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
        result = apply_kernel(cube, kernel=kernel, border="reflect")
        assert result.shape == cube.shape

    def test_border_numeric(self):
        cube = _make_raster()
        kernel = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
        result = apply_kernel(cube, kernel=kernel, border=99.0)
        assert result.shape == cube.shape

    def test_replace_invalid_nan(self):
        """NaN values should be replaced before convolution."""
        cube = _make_raster()
        cube_with_nan = cube.copy()
        cube_with_nan.values[0, 0, 0, 0] = np.nan
        kernel = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
        result = apply_kernel(cube_with_nan, kernel=kernel, replace_invalid=0.0)
        # The NaN pixel should now be 0 after replace_invalid
        assert np.isfinite(result.values[0, 0, 0, 0])

    def test_gaussian_blur_kernel(self):
        """Test with a realistic Gaussian-like kernel."""
        cube = _make_raster()
        kernel = [
            [1, 2, 1],
            [2, 4, 2],
            [1, 2, 1],
        ]
        result = apply_kernel(cube, kernel=kernel, factor=1.0 / 16.0)
        assert result.shape == cube.shape
        # Gaussian blur should not produce NaN or Inf
        assert np.all(np.isfinite(result.values))
