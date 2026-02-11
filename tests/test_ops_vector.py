"""Tests for openeo_core.ops.vector module."""

import geopandas as gpd
import numpy as np
from shapely.geometry import Point

from openeo_core.ops.vector import filter_bbox, to_feature_matrix


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
