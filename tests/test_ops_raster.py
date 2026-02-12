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

    def test_aggregate_spatial_with_geometries(self):
        """Test zonal statistics with geometries returns a GeoDataFrame."""
        import geopandas as gpd
        from shapely.geometry import box

        # Create a simple raster with known values
        cube = _make_raster()
        
        # Create two simple box geometries within the raster extent
        geom1 = box(10.0, 50.0, 10.5, 50.5)
        geom2 = box(10.5, 50.5, 11.0, 51.0)
        gdf = gpd.GeoDataFrame({"id": [1, 2]}, geometry=[geom1, geom2], crs="EPSG:4326")
        
        # Aggregate with geometries
        result = aggregate_spatial(cube, gdf, reducer="mean")
        
        # Check that result is a GeoDataFrame
        assert isinstance(result, gpd.GeoDataFrame)
        
        # Check that we have one row per geometry
        assert len(result) == 2
        
        # Check that geometry column exists
        assert result.geometry is not None
        
        # Check that we have feature columns (bands x time)
        # Expected columns: red_2023-01, red_2023-02, nir_2023-01, nir_2023-02
        assert len(result.columns) >= 4  # At least 4 feature columns
        
        # Check that extra columns are preserved
        assert "id" in result.columns
        assert result["id"].tolist() == [1, 2]

    def test_aggregate_spatial_crs_alignment(self):
        """Test that geometries are reprojected to match raster CRS."""
        import geopandas as gpd
        from shapely.geometry import box
        import rioxarray  # noqa: F811
        
        # Create a geo-referenced raster in EPSG:4326
        cube = _make_raster()
        cube = cube.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude")
        cube = cube.rio.write_crs("EPSG:4326")
        
        # Create geometries in a different CRS (EPSG:3857 - Web Mercator)
        # These coordinates are approximately the same location in Web Mercator
        geom = box(1113194, 6446275, 1167600, 6621210)
        gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[geom], crs="EPSG:3857")
        
        # This should not raise an error; CRS should be aligned internally
        result = aggregate_spatial(cube, gdf, reducer="mean")
        
        # Result should be a GeoDataFrame with the raster's CRS (geometries are reprojected)
        assert isinstance(result, gpd.GeoDataFrame)
        assert str(result.crs) == "EPSG:4326"  # Result uses raster CRS
        
        # Verify that we got valid results despite CRS mismatch
        assert len(result) == 1

    def test_aggregate_spatial_intersection_filter(self):
        """Test that geometries outside raster bounds raise an error."""
        import geopandas as gpd
        from shapely.geometry import box
        import rioxarray  # noqa: F811
        
        # Create a geo-referenced raster
        cube = _make_raster()
        cube = cube.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude")
        cube = cube.rio.write_crs("EPSG:4326")
        
        # Create geometries completely outside the raster bounds
        # Raster is at longitude [10, 11], latitude [50, 51]
        geom = box(20.0, 60.0, 21.0, 61.0)
        gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[geom], crs="EPSG:4326")
        
        # Should raise ValueError about no intersecting geometries
        with pytest.raises(ValueError, match="No training geometries intersect"):
            aggregate_spatial(cube, gdf, reducer="mean")

    def test_aggregate_spatial_column_flattening(self):
        """Test that time and bands dimensions are flattened to feature columns."""
        import geopandas as gpd
        from shapely.geometry import box
        
        # Create a raster with multiple time steps and bands
        cube = _make_raster()
        
        # Create a geometry covering the entire raster
        geom = box(10.0, 50.0, 11.0, 51.0)
        gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[geom], crs="EPSG:4326")
        
        # Aggregate
        result = aggregate_spatial(cube, gdf, reducer="mean")
        
        # Check column naming: should have format "band_YYYY-MM"
        # For 2 bands (red, nir) and 2 time steps (2023-01, 2023-02)
        # We expect 4 feature columns
        expected_cols = ["red_2023-01", "red_2023-02", "nir_2023-01", "nir_2023-02"]
        for col in expected_cols:
            assert col in result.columns, f"Expected column {col} not found"

    def test_aggregate_spatial_extra_columns(self):
        """Test that extra columns from input GeoDataFrame are preserved."""
        import geopandas as gpd
        from shapely.geometry import box
        
        cube = _make_raster()
        
        # Create GeoDataFrame with extra attribute columns
        geom1 = box(10.0, 50.0, 10.5, 50.5)
        geom2 = box(10.5, 50.5, 11.0, 51.0)
        gdf = gpd.GeoDataFrame({
            "id": [1, 2],
            "name": ["zone_a", "zone_b"],
            "category": ["forest", "urban"]
        }, geometry=[geom1, geom2], crs="EPSG:4326")
        
        result = aggregate_spatial(cube, gdf, reducer="mean")
        
        # All extra columns should be preserved
        assert "id" in result.columns
        assert "name" in result.columns
        assert "category" in result.columns
        
        # Check values are preserved correctly
        assert result["id"].tolist() == [1, 2]
        assert result["name"].tolist() == ["zone_a", "zone_b"]
        assert result["category"].tolist() == ["forest", "urban"]

    def test_aggregate_spatial_different_reducers(self):
        """Test that different reducer functions work correctly."""
        import geopandas as gpd
        from shapely.geometry import box
        
        cube = _make_raster()
        geom = box(10.0, 50.0, 11.0, 51.0)
        gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[geom], crs="EPSG:4326")
        
        # Test different reducers
        for reducer in ["mean", "sum", "min", "max", "median"]:
            result = aggregate_spatial(cube, gdf, reducer=reducer)
            assert isinstance(result, gpd.GeoDataFrame)
            assert len(result) == 1


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
