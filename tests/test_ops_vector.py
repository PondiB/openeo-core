"""Tests for openeo_core.ops.vector module."""

import numpy as np
import xarray as xr

import geopandas as gpd
from shapely.geometry import Point

from openeo_core.ops.vector import filter_bbox, to_feature_matrix

# Import xvec for its side effect of registering the `.xvec` accessor on xarray objects.
import xvec
# Explicitly reference xvec so static analysis tools see it as used.
_ = xvec


def _make_gdf() -> gpd.GeoDataFrame:
    """Small test GeoDataFrame."""
    return gpd.GeoDataFrame(
        {"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0], "label": [0, 1, 0]},
        geometry=[Point(10, 50), Point(10.5, 50.5), Point(11, 51)],
        crs="EPSG:4326",
    )


class TestFilterBboxVector:
    def test_filter_bbox_keeps_inside(self):
        gdf = _make_gdf()
        result = filter_bbox(gdf, west=9, south=49, east=11, north=51)
        # Points (10,50) and (10.5,50.5) are inside; (11,51) is on boundary
        assert len(result) >= 2


class TestToFeatureMatrix:
    def test_auto_features(self):
        gdf = _make_gdf()
        X, y = to_feature_matrix(gdf, target_column="label")
        assert X.shape == (3, 2)  # columns a, b
        assert y is not None
        np.testing.assert_array_equal(y, [0, 1, 0])

    def test_explicit_features(self):
        gdf = _make_gdf()
        X, y = to_feature_matrix(gdf, feature_columns=["a"], target_column="label")
        assert X.shape == (3, 1)


class TestFilterBboxXvec:
    def test_filter_bbox_xvec(self):
        """filter_bbox works on xvec-backed DataArray."""
        from shapely.geometry import Point

        da = xr.DataArray(
            [1.0, 2.0, 3.0],
            dims=["geom"],
            coords={"geom": [Point(10, 50), Point(10.5, 50.5), Point(11, 51)]},
        )
        da = da.xvec.set_geom_indexes("geom", crs=4326)

        result = filter_bbox(da, west=9, south=49, east=11, north=51)
        assert isinstance(result, xr.DataArray)
        assert result.sizes["geom"] >= 2

    def test_filter_bbox_xvec_dataset(self):
        """filter_bbox works on xvec-backed Dataset."""
        from shapely.geometry import Point

        ds = xr.Dataset(
            {"val": (["geom"], [1.0, 2.0, 3.0])},
            coords={"geom": [Point(10, 50), Point(10.5, 50.5), Point(11, 51)]},
        )
        ds = ds.xvec.set_geom_indexes("geom", crs=4326)

        result = filter_bbox(ds, west=9, south=49, east=11, north=51)
        assert isinstance(result, xr.Dataset)
        assert "val" in result.data_vars


class TestToFeatureMatrixXvec:
    def test_to_feature_matrix_xvec(self):
        """to_feature_matrix converts xvec DataArray via GeoDataFrame."""
        da = xr.DataArray(
            [[1.0, 4.0, 0], [2.0, 5.0, 1], [3.0, 6.0, 0]],
            dims=["geom", "vars"],
            coords={
                "geom": [Point(0, 0), Point(1, 1), Point(2, 2)],
                "vars": ["a", "b", "label"],
            },
        )
        da = da.xvec.set_geom_indexes("geom", crs=4326)

        # to_geopandas produces columns from vars; we need explicit feature/target
        gdf = da.xvec.to_geopandas()
        X, y = to_feature_matrix(gdf, feature_columns=["a", "b"], target_column="label")
        assert X.shape == (3, 2)
        assert y is not None
        np.testing.assert_array_equal(y, [0, 1, 0])
