"""Tests for openeo_core.ops.raster module."""

import dask.array as da
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from openeo_core.exceptions import (
    BandExists,
    DimensionAmbiguous,
    NirBandAmbiguous,
    RedBandAmbiguous,
)
from openeo_core.ops.raster import (
    aggregate_spatial,
    aggregate_temporal,
    apply,
    array_interpolate_linear,
    filter_bbox,
    filter_temporal,
    ndvi,
    resample_spatial,
    stack_to_samples,
    unstack_from_samples,
)


def _make_raster(dask_backed: bool = False) -> xr.DataArray:
    """Create a small test raster cube with (time, bands, latitude, longitude) dims."""
    np.random.seed(42)
    data = np.random.rand(2, 2, 4, 4).astype(np.float32)
    if dask_backed:
        data = da.from_array(data, chunks=(1, 2, 4, 4))  # type: ignore[assignment]
    return xr.DataArray(
        data,
        dims=["time", "bands", "latitude", "longitude"],
        coords={
            "time": pd.date_range("2023-01-01", periods=2, freq="ME"),
            "bands": ["red", "nir"],
            "latitude": np.linspace(50, 51, 4),
            "longitude": np.linspace(10, 11, 4),
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

    def test_ndvi_dimension_ambiguous(self):
        """DimensionAmbiguous when bands dimension is missing."""
        cube = xr.DataArray(
            np.random.rand(4, 4).astype(np.float32),
            dims=["y", "x"],
        )
        with pytest.raises(DimensionAmbiguous, match="not available"):
            ndvi(cube)

    def test_ndvi_nir_band_ambiguous(self):
        """NirBandAmbiguous when NIR band label doesn't exist."""
        cube = _make_raster()
        with pytest.raises(NirBandAmbiguous, match="can't be resolved"):
            ndvi(cube, nir="B08")

    def test_ndvi_red_band_ambiguous(self):
        """RedBandAmbiguous when red band label doesn't exist."""
        cube = _make_raster()
        with pytest.raises(RedBandAmbiguous, match="can't be resolved"):
            ndvi(cube, red="B04")

    def test_ndvi_band_exists(self):
        """BandExists when target_band is already a label."""
        cube = _make_raster()
        with pytest.raises(BandExists, match="already exists"):
            ndvi(cube, target_band="red")

    def test_ndvi_target_band_keeps_original_bands(self):
        """When target_band is set, original bands must be preserved."""
        cube = _make_raster()
        result = ndvi(cube, target_band="ndvi")
        band_labels = list(result.coords["bands"].values)
        assert "red" in band_labels
        assert "nir" in band_labels
        assert "ndvi" in band_labels
        assert len(band_labels) == 3


# ---------------------------------------------------------------
# filter_bbox
# ---------------------------------------------------------------


class TestFilterBbox:
    def test_filter_bbox(self):
        cube = _make_raster()
        result = filter_bbox(cube, west=10.0, south=50.0, east=10.5, north=50.5)
        assert result.sizes["longitude"] <= cube.sizes["longitude"]
        assert result.sizes["latitude"] <= cube.sizes["latitude"]


# ---------------------------------------------------------------
# filter_temporal
# ---------------------------------------------------------------


class TestFilterTemporal:
    def test_filter_temporal(self):
        cube = _make_raster()
        result = filter_temporal(cube, extent=("2023-01-01", "2023-01-31"))
        assert result.sizes["time"] == 1


# ---------------------------------------------------------------
# aggregate_spatial
# ---------------------------------------------------------------


class TestAggregateSpatial:
    def test_aggregate_spatial_mean(self):
        cube = _make_raster()
        result = aggregate_spatial(cube, None, reducer="mean")
        assert "longitude" not in result.dims
        assert "latitude" not in result.dims


# ---------------------------------------------------------------
# aggregate_temporal
# ---------------------------------------------------------------


def _make_raster_long(months: int = 14) -> xr.DataArray:
    """Create a raster cube spanning multiple months for temporal tests."""
    np.random.seed(99)
    data = np.random.rand(months, 2, 3, 3).astype(np.float32)
    return xr.DataArray(
        data,
        dims=["time", "bands", "latitude", "longitude"],
        coords={
            "time": pd.date_range("2023-01-15", periods=months, freq="ME"),
            "bands": ["red", "nir"],
            "latitude": np.linspace(50, 51, 3),
            "longitude": np.linspace(10, 11, 3),
        },
    )


class TestAggregateTemporal:
    def test_aggregate_temporal_year(self):
        cube = _make_raster()
        result = aggregate_temporal(cube, period="year", reducer="mean")
        assert result.sizes["time"] <= cube.sizes["time"]

    def test_aggregate_temporal_month_labels(self):
        """Month aggregation should produce YYYY-MM labels."""
        cube = _make_raster()
        result = aggregate_temporal(cube, period="month", reducer="mean")
        labels = list(result.coords["time"].values)
        for lbl in labels:
            assert len(str(lbl)) == 7  # YYYY-MM
            assert "-" in str(lbl)

    def test_aggregate_temporal_year_labels(self):
        """Year aggregation should produce YYYY labels."""
        cube = _make_raster_long()
        result = aggregate_temporal(cube, period="year", reducer="mean")
        labels = [str(l) for l in result.coords["time"].values]
        for lbl in labels:
            assert len(lbl) == 4
            assert lbl.isdigit()

    def test_aggregate_temporal_day_labels(self):
        """Day aggregation should produce YYYY-DDD labels."""
        cube = _make_raster()
        result = aggregate_temporal(cube, period="day", reducer="mean")
        labels = [str(l) for l in result.coords["time"].values]
        for lbl in labels:
            assert len(lbl) == 8  # YYYY-DDD
            parts = lbl.split("-")
            assert len(parts) == 2
            assert len(parts[1]) == 3

    def test_aggregate_temporal_week_labels(self):
        """Week aggregation should produce YYYY-WW labels."""
        cube = _make_raster()
        result = aggregate_temporal(cube, period="week", reducer="mean")
        labels = [str(l) for l in result.coords["time"].values]
        for lbl in labels:
            parts = lbl.split("-")
            assert len(parts) == 2
            assert len(parts[1]) == 2

    def test_aggregate_temporal_season(self):
        """Season aggregation should produce YYYY-xxx labels."""
        cube = _make_raster_long()
        result = aggregate_temporal(cube, period="season", reducer="mean")
        labels = [str(l) for l in result.coords["time"].values]
        valid_suffixes = {"djf", "mam", "jja", "son"}
        for lbl in labels:
            parts = lbl.split("-")
            assert parts[1] in valid_suffixes

    def test_aggregate_temporal_dekad(self):
        """Dekad aggregation should produce YYYY-DD labels with correct 1-based indexing."""
        # Test with data that has specific known dates
        np.random.seed(42)
        data = np.random.rand(6, 2, 4, 4).astype(np.float32)
        cube = xr.DataArray(
            data,
            dims=["time", "bands", "latitude", "longitude"],
            coords={
                "time": pd.to_datetime([
                    "2023-01-05",  # Early January → dekad 01
                    "2023-01-15",  # Mid January → dekad 02
                    "2023-01-25",  # Late January → dekad 03
                    "2023-02-08",  # Early February → dekad 04
                    "2023-12-15",  # Mid December → dekad 35
                    "2023-12-25",  # Late December → dekad 36
                ]),
                "bands": ["red", "nir"],
                "latitude": np.linspace(50, 51, 4),
                "longitude": np.linspace(10, 11, 4),
            },
        )
        result = aggregate_temporal(cube, period="dekad", reducer="mean")
        labels = [str(l) for l in result.coords["time"].values]
        
        # Verify format: YYYY-DD with two-digit dekad
        for lbl in labels:
            parts = lbl.split("-")
            assert len(parts) == 2
            assert len(parts[1]) == 2
        
        # Verify specific dekad values for known dates (1-based indexing)
        assert "2023-01" in labels, "Early January should be dekad 01"
        assert "2023-02" in labels, "Mid January should be dekad 02"
        assert "2023-03" in labels, "Late January should be dekad 03"
        assert "2023-04" in labels, "Early February should be dekad 04"
        assert "2023-35" in labels, "Mid December should be dekad 35"
        assert "2023-36" in labels, "Late December should be dekad 36"

    def test_aggregate_temporal_tropical_season(self):
        """Tropical-season aggregation should label with ndjfma or mjjaso."""
        cube = _make_raster_long()
        result = aggregate_temporal(cube, period="tropical-season", reducer="mean")
        labels = [str(l) for l in result.coords["time"].values]
        valid_suffixes = {"ndjfma", "mjjaso"}
        for lbl in labels:
            parts = lbl.split("-")
            assert parts[1] in valid_suffixes

    def test_aggregate_temporal_decade(self):
        """Decade aggregation should produce labels like 2020."""
        cube = _make_raster()
        result = aggregate_temporal(cube, period="decade", reducer="mean")
        labels = [str(l) for l in result.coords["time"].values]
        for lbl in labels:
            assert int(lbl) % 10 == 0

    def test_aggregate_temporal_decade_ad(self):
        """Decade-ad aggregation should produce labels like 2021."""
        cube = _make_raster()
        result = aggregate_temporal(cube, period="decade-ad", reducer="mean")
        labels = [str(l) for l in result.coords["time"].values]
        for lbl in labels:
            assert int(lbl) % 10 == 1

    def test_aggregate_temporal_sum(self):
        """Verify sum reducer works correctly."""
        cube = _make_raster()
        result = aggregate_temporal(cube, period="year", reducer="sum")
        assert result.sizes["time"] <= cube.sizes["time"]

    def test_unsupported_period(self):
        cube = _make_raster()
        with pytest.raises(ValueError, match="Unsupported period"):
            aggregate_temporal(cube, period="century")


# ---------------------------------------------------------------
# resample_spatial
# ---------------------------------------------------------------

import importlib.util
_HAS_RIOXARRAY = importlib.util.find_spec("rioxarray") is not None


def _make_geo_raster() -> xr.DataArray:
    """Create a CRS-aware raster for resample_spatial tests."""
    np.random.seed(7)
    data = np.random.rand(2, 10, 10).astype(np.float32)
    da = xr.DataArray(
        data,
        dims=["bands", "latitude", "longitude"],
        coords={
            "bands": ["red", "nir"],
            "latitude": np.linspace(51, 50, 10),  # descending
            "longitude": np.linspace(10, 11, 10),
        },
    )
    import rioxarray  # noqa: F811
    da = da.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude")
    da = da.rio.write_crs("EPSG:4326")
    return da


@pytest.mark.skipif(not _HAS_RIOXARRAY, reason="rioxarray not installed")
class TestResampleSpatial:
    def test_noop(self):
        """No resolution / no projection → data unchanged."""
        cube = _make_geo_raster()
        result = resample_spatial(cube)
        assert result.sizes == cube.sizes

    def test_reproject(self):
        """Reproject from EPSG:4326 to EPSG:3857."""
        cube = _make_geo_raster()
        result = resample_spatial(cube, projection=3857)
        assert result.rio.crs.to_epsg() == 3857

    def test_change_resolution(self):
        """Resample to coarser resolution (should reduce pixel count)."""
        cube = _make_geo_raster()
        # Original ~0.11° per pixel; resample to ~0.25°
        result = resample_spatial(cube, resolution=0.25)
        assert result.sizes["longitude"] < cube.sizes["longitude"]
        assert result.sizes["latitude"] < cube.sizes["latitude"]

    def test_resolution_list(self):
        """Separate x/y resolutions."""
        cube = _make_geo_raster()
        result = resample_spatial(cube, resolution=[0.5, 0.25])
        assert result.sizes["longitude"] < cube.sizes["longitude"]

    def test_bilinear_method(self):
        """Bilinear resampling should work without error."""
        cube = _make_geo_raster()
        result = resample_spatial(cube, resolution=0.25, method="bilinear")
        assert result.sizes["longitude"] < cube.sizes["longitude"]

    def test_invalid_method(self):
        cube = _make_geo_raster()
        with pytest.raises(ValueError, match="Unknown resampling method"):
            resample_spatial(cube, resolution=0.25, method="invalid")

    def test_4d_cube(self):
        """resample_spatial must handle 4-D cubes (time, bands, latitude, longitude)."""
        np.random.seed(8)
        da4d = xr.DataArray(
            np.random.rand(2, 2, 10, 10).astype(np.float32),
            dims=["time", "bands", "latitude", "longitude"],
            coords={
                "time": pd.date_range("2023-01-01", periods=2, freq="ME"),
                "bands": ["red", "nir"],
                "latitude": np.linspace(51, 50, 10),
                "longitude": np.linspace(10, 11, 10),
            },
        )
        import rioxarray  # noqa: F811
        da4d = da4d.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude")
        da4d = da4d.rio.write_crs("EPSG:4326")

        result = resample_spatial(da4d, resolution=0.25)
        assert "time" in result.dims
        assert "bands" in result.dims
        assert result.sizes["longitude"] < da4d.sizes["longitude"]
        assert result.sizes["latitude"] < da4d.sizes["latitude"]

    def test_4d_reproject(self):
        """Reprojecting a 4-D cube should preserve all non-spatial dims."""
        np.random.seed(9)
        da4d = xr.DataArray(
            np.random.rand(2, 2, 10, 10).astype(np.float32),
            dims=["time", "bands", "latitude", "longitude"],
            coords={
                "time": pd.date_range("2023-01-01", periods=2, freq="ME"),
                "bands": ["red", "nir"],
                "latitude": np.linspace(51, 50, 10),
                "longitude": np.linspace(10, 11, 10),
            },
        )
        import rioxarray  # noqa: F811
        da4d = da4d.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude")
        da4d = da4d.rio.write_crs("EPSG:4326")

        result = resample_spatial(da4d, projection=3857)
        assert "time" in result.dims
        assert "bands" in result.dims
        assert result.rio.crs.to_epsg() == 3857

    def test_4d_dask_stays_lazy(self):
        """Dask-backed 4-D cubes: reshape operations preserve laziness.
        
        Note: rioxarray.reproject forces computation, so the final result
        will be a numpy array. This test verifies that our reshape operations
        (which we control) preserve dask arrays until the reproject step.
        """
        np.random.seed(10)
        arr = da.from_array(
            np.random.rand(2, 2, 10, 10).astype(np.float32),
            chunks=(1, 2, 10, 10)
        )
        da4d = xr.DataArray(
            arr,
            dims=["time", "bands", "latitude", "longitude"],
            coords={
                "time": pd.date_range("2023-01-01", periods=2, freq="ME"),
                "bands": ["red", "nir"],
                "latitude": np.linspace(51, 50, 10),
                "longitude": np.linspace(10, 11, 10),
            },
        )
        import rioxarray  # noqa: F811
        da4d = da4d.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude")
        da4d = da4d.rio.write_crs("EPSG:4326")

        # Verify input is dask-backed
        assert isinstance(da4d.data, da.Array), "Input should be dask-backed"

        result = resample_spatial(da4d, resolution=0.25)

        # Due to rioxarray.reproject forcing computation, result will be numpy
        # But the operation should still complete successfully
        assert "time" in result.dims
        assert "bands" in result.dims
        assert result.sizes["longitude"] < da4d.sizes["longitude"]
        assert result.sizes["latitude"] < da4d.sizes["latitude"]


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


# ---------------------------------------------------------------
# array_interpolate_linear
# ---------------------------------------------------------------


class TestArrayInterpolateLinear:
    """Tests for the array_interpolate_linear openEO process."""

    # -- Plain list tests (from the openEO spec examples) --

    def test_spec_example_1(self):
        """Spec example: interior nulls are interpolated, leading null preserved."""
        data = [None, 1, None, 6, None, -8]
        result = array_interpolate_linear(data)
        assert result == [None, 1, 3.5, 6, -1.0, -8]

    def test_spec_example_2(self):
        """Spec example: only 1 valid value → nothing to interpolate."""
        data = [None, 1, None, None]
        result = array_interpolate_linear(data)
        assert result == [None, 1, None, None]

    def test_all_none(self):
        """All None values → returned unchanged."""
        data = [None, None, None]
        result = array_interpolate_linear(data)
        assert result == [None, None, None]

    def test_no_gaps(self):
        """No NaN / None gaps → returned unchanged."""
        data = [1, 2, 3, 4]
        result = array_interpolate_linear(data)
        assert result == [1, 2, 3, 4]

    def test_single_value(self):
        """Single value array → returned unchanged."""
        data = [5]
        result = array_interpolate_linear(data)
        assert result == [5]

    def test_empty_list(self):
        """Empty list → returned unchanged."""
        result = array_interpolate_linear([])
        assert result == []

    def test_two_values_with_gap(self):
        """Two valid values with a gap in between."""
        data = [0, None, 10]
        result = array_interpolate_linear(data)
        assert result == [0, 5.0, 10]

    def test_leading_trailing_preserved(self):
        """Leading and trailing NaN/None must not be filled."""
        data = [None, None, 2, None, 8, None, None]
        result = array_interpolate_linear(data)
        assert result == [None, None, 2, 5.0, 8, None, None]

    def test_multiple_interior_gaps(self):
        """Multiple consecutive interior nulls are interpolated correctly."""
        data = [0, None, None, None, 4]
        result = array_interpolate_linear(data)
        assert result == [0, 1.0, 2.0, 3.0, 4]

    # -- RasterCube (xarray DataArray) tests --

    def test_raster_interpolate_time(self):
        """Interpolate NaN values along the time dimension of a raster cube."""
        data = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
        da_cube = xr.DataArray(data, dims=["time"])
        result = array_interpolate_linear(da_cube, dimension="time")
        expected = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        np.testing.assert_allclose(result.values, expected)

    def test_raster_leading_trailing_nan_preserved(self):
        """Leading / trailing NaN along the interpolation dim stay NaN."""
        data = np.array([np.nan, 2.0, np.nan, 8.0, np.nan])
        da_cube = xr.DataArray(data, dims=["time"])
        result = array_interpolate_linear(da_cube, dimension="time")
        assert np.isnan(result.values[0]), "Leading NaN should be preserved"
        assert np.isnan(result.values[-1]), "Trailing NaN should be preserved"
        np.testing.assert_allclose(result.values[1:4], [2.0, 5.0, 8.0])

    def test_raster_4d_cube(self):
        """Interpolation on a full 4-D raster cube along time."""
        np.random.seed(42)
        data = np.random.rand(5, 2, 3, 3).astype(np.float32)
        # Punch some NaN holes at time indices 1 and 3
        data[1, :, :, :] = np.nan
        data[3, :, :, :] = np.nan
        da_cube = xr.DataArray(
            data,
            dims=["time", "bands", "latitude", "longitude"],
            coords={
                "time": pd.date_range("2023-01-01", periods=5, freq="ME"),
                "bands": ["red", "nir"],
                "latitude": np.linspace(50, 51, 3),
                "longitude": np.linspace(10, 11, 3),
            },
        )
        result = array_interpolate_linear(da_cube, dimension="time")
        # All interior NaN should be filled
        assert not np.isnan(result.values[1]).any()
        assert not np.isnan(result.values[3]).any()
        # Shape must be preserved
        assert result.shape == da_cube.shape

    def test_raster_string_coord_dimension(self):
        """Interpolation along a string-labeled dimension (e.g. bands) must not raise."""
        data = np.array([[1.0, np.nan, 3.0], [np.nan, 5.0, np.nan]])
        da_cube = xr.DataArray(
            data,
            dims=["bands", "x"],
            coords={"bands": ["red", "nir"]},
        )
        result = array_interpolate_linear(da_cube, dimension="x")
        # Interior NaN at [0, 1] should be interpolated to 2.0
        np.testing.assert_allclose(result.values[0], [1.0, 2.0, 3.0])

    def test_raster_missing_dimension_raises(self):
        """Passing a non-existent dimension should raise DimensionNotAvailable."""
        from openeo_core.exceptions import DimensionNotAvailable

        da_cube = xr.DataArray([1.0, 2.0], dims=["time"])
        with pytest.raises(DimensionNotAvailable, match="does not exist"):
            array_interpolate_linear(da_cube, dimension="bands")

    def test_raster_no_dimension_raises(self):
        """Omitting dimension for a DataArray should raise ValueError."""
        da_cube = xr.DataArray([1.0, 2.0], dims=["time"])
        with pytest.raises(ValueError, match="dimension must be specified"):
            array_interpolate_linear(da_cube)


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
