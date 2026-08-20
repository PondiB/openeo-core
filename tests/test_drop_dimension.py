"""Tests for the drop_dimension process."""

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from shapely.geometry import Point

# Import xvec for its side effect of registering the `.xvec` accessor on xarray objects.
import xvec
# Explicitly reference xvec so static analysis tools see it as used.
_ = xvec

from openeo_core import DataCube
from openeo_core.exceptions import DimensionLabelCountMismatch, DimensionNotAvailable
from openeo_core.ops.raster import drop_dimension
from openeo_core.ops.vector import drop_dimension as drop_dimension_vector


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

    def test_label_count_mismatch_no_labels(self):
        cube = _make_raster().isel(time=slice(0, 0))
        with pytest.raises(DimensionLabelCountMismatch, match="no labels"):
            drop_dimension(cube, name="time")


class TestDropScalarCoordinate:
    """A dimension already collapsed to a scalar coordinate is still droppable."""

    def test_drop_scalar_coordinate(self):
        # isel without drop=True leaves `time` behind as a scalar coordinate.
        cube = _make_raster().isel(time=0)
        assert "time" not in cube.dims and "time" in cube.coords

        result = drop_dimension(cube, name="time")
        assert "time" not in result.coords
        assert set(result.dims) == {"bands", "latitude", "longitude"}
        np.testing.assert_allclose(result.values, cube.values)

    def test_drop_scalar_coordinate_via_datacube(self):
        cube = DataCube(_make_raster().isel(time=0))
        out = cube.drop_dimension(name="time")
        assert "time" not in out.data.coords

    def test_non_dimension_coordinate_is_not_a_dimension(self):
        # A 1-D non-dimension coordinate is not a cube dimension.
        cube = _make_raster().assign_coords(
            label=("bands", ["a", "b", "c"]),
        )
        with pytest.raises(DimensionNotAvailable, match="does not exist"):
            drop_dimension(cube, name="label")


def _xvec_cube(n_geometries: int = 1) -> xr.DataArray:
    """xvec-backed vector cube with a `geom` dimension and a `vars` dimension."""
    geoms = [Point(10 + i, 50 + i) for i in range(n_geometries)]
    da = xr.DataArray(
        np.arange(n_geometries * 2, dtype=np.float64).reshape(n_geometries, 2),
        dims=["geom", "vars"],
        coords={"geom": geoms, "vars": ["ndvi", "evi"]},
    )
    return da.xvec.set_geom_indexes("geom", crs=4326)


def _geo_frame(n_rows: int = 3, n_properties: int = 2) -> gpd.GeoDataFrame:
    columns = {
        f"prop{i}": np.arange(n_rows, dtype=np.float64) + i
        for i in range(n_properties)
    }
    return gpd.GeoDataFrame(
        columns,
        geometry=[Point(10 + i, 50 + i) for i in range(n_rows)],
        crs="EPSG:4326",
    )


class TestDropDimensionXvec:
    def test_drop_single_geometry_dimension(self):
        cube = _xvec_cube(n_geometries=1)
        result = drop_dimension_vector(cube, name="geom")
        assert "geom" not in result.dims
        assert "geom" not in result.coords
        assert set(result.dims) == {"vars"}

    def test_drop_non_geometry_dimension(self):
        cube = _xvec_cube(n_geometries=3).isel(vars=slice(0, 1))
        result = drop_dimension_vector(cube, name="vars")
        assert set(result.dims) == {"geom"}
        # The cube keeps its geometries.
        assert list(result.coords["geom"].values) == list(cube.coords["geom"].values)

    def test_multi_label_geometry_dimension(self):
        cube = _xvec_cube(n_geometries=3)
        with pytest.raises(DimensionLabelCountMismatch, match="exceeds one"):
            drop_dimension_vector(cube, name="geom")

    def test_dimension_not_available(self):
        cube = _xvec_cube(n_geometries=1)
        with pytest.raises(DimensionNotAvailable, match="does not exist"):
            drop_dimension_vector(cube, name="time")

    def test_datacube_dispatches_to_vector(self):
        cube = DataCube(_xvec_cube(n_geometries=1))
        assert cube.is_vector
        out = cube.drop_dimension(name="geom")
        assert isinstance(out, DataCube)
        assert "geom" not in out.data.dims

    def test_xvec_dataset(self):
        ds = _xvec_cube(n_geometries=1).to_dataset(name="values")
        result = drop_dimension_vector(ds, name="geom")
        assert "geom" not in result.dims


class TestDropDimensionGeoDataFrame:
    def test_drop_properties_dimension(self):
        gdf = _geo_frame(n_rows=3, n_properties=1)
        result = drop_dimension_vector(gdf, name="properties")
        assert isinstance(result, gpd.GeoDataFrame)
        assert list(result.columns) == ["geometry"]
        assert len(result) == 3

    def test_drop_geometry_dimension(self):
        gdf = _geo_frame(n_rows=1, n_properties=2)
        result = drop_dimension_vector(gdf, name="geometry")
        # Without geometries the result is a plain DataFrame.
        assert not isinstance(result, gpd.GeoDataFrame)
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["prop0", "prop1"]

    def test_geometry_dimension_multi_label(self):
        gdf = _geo_frame(n_rows=3, n_properties=1)
        with pytest.raises(DimensionLabelCountMismatch, match="exceeds one"):
            drop_dimension_vector(gdf, name="geometry")

    def test_properties_dimension_multi_label(self):
        gdf = _geo_frame(n_rows=1, n_properties=2)
        with pytest.raises(DimensionLabelCountMismatch, match="exceeds one"):
            drop_dimension_vector(gdf, name="properties")

    def test_properties_dimension_no_labels(self):
        gdf = _geo_frame(n_rows=1, n_properties=0)
        with pytest.raises(DimensionLabelCountMismatch, match="no labels"):
            drop_dimension_vector(gdf, name="properties")

    def test_geometry_dimension_no_labels(self):
        gdf = _geo_frame(n_rows=0, n_properties=1)
        with pytest.raises(DimensionLabelCountMismatch, match="no labels"):
            drop_dimension_vector(gdf, name="geometry")

    def test_renamed_geometry_column(self):
        gdf = _geo_frame(n_rows=1, n_properties=1).rename_geometry("geom")
        result = drop_dimension_vector(gdf, name="geom")
        assert list(result.columns) == ["prop0"]

    def test_dimension_not_available(self):
        gdf = _geo_frame(n_rows=1, n_properties=1)
        with pytest.raises(DimensionNotAvailable, match="does not exist"):
            drop_dimension_vector(gdf, name="time")

    def test_datacube_dispatches_to_vector(self):
        cube = DataCube(_geo_frame(n_rows=3, n_properties=1))
        out = cube.drop_dimension(name="properties")
        assert isinstance(out, DataCube)
        assert list(out.data.columns) == ["geometry"]

    def test_dask_geo_frame(self):
        dask_geopandas = pytest.importorskip("dask_geopandas")
        ddf = dask_geopandas.from_geopandas(
            _geo_frame(n_rows=3, n_properties=1), npartitions=2
        )
        result = drop_dimension_vector(ddf, name="properties")
        assert list(result.columns) == ["geometry"]
        with pytest.raises(DimensionLabelCountMismatch, match="exceeds one"):
            drop_dimension_vector(ddf, name="geometry")

    def test_unsupported_type(self):
        with pytest.raises(TypeError, match="drop_dimension expects"):
            drop_dimension_vector([1, 2, 3], name="geometry")
