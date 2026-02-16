"""Tests for vector_buffer and vector_reproject processes."""

import pytest
import geopandas as gpd
from shapely.geometry import Point, Polygon

from openeo_core.exceptions import DimensionNotAvailable, UnitMismatch
from openeo_core.ops.vector import vector_buffer, vector_reproject


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_projected_gdf() -> gpd.GeoDataFrame:
    """GeoDataFrame in EPSG:3857 (metre-based CRS)."""
    return gpd.GeoDataFrame(
        {"name": ["a", "b", "c"]},
        geometry=[Point(0, 0), Point(100, 200), Point(500, 500)],
        crs="EPSG:3857",
    )


def _make_geographic_gdf() -> gpd.GeoDataFrame:
    """GeoDataFrame in EPSG:4326 (degree-based CRS)."""
    return gpd.GeoDataFrame(
        {"name": ["a", "b"]},
        geometry=[Point(10, 50), Point(11, 51)],
        crs="EPSG:4326",
    )


# ---------------------------------------------------------------------------
# vector_buffer
# ---------------------------------------------------------------------------


class TestVectorBuffer:
    def test_positive_buffer_expands(self):
        gdf = _make_projected_gdf()
        result = vector_buffer(gdf, distance=10.0)
        # Points become polygons after buffering
        for geom in result.geometry:
            assert geom.geom_type == "Polygon"
            assert geom.area > 0

    def test_negative_buffer_shrinks(self):
        # Create a large polygon to shrink
        poly = Polygon([(0, 0), (1000, 0), (1000, 1000), (0, 1000)])
        gdf = gpd.GeoDataFrame(
            {"name": ["big"]}, geometry=[poly], crs="EPSG:3857"
        )
        original_area = gdf.geometry.iloc[0].area
        result = vector_buffer(gdf, distance=-50.0)
        assert result.geometry.iloc[0].area < original_area

    def test_properties_preserved(self):
        gdf = _make_projected_gdf()
        result = vector_buffer(gdf, distance=5.0)
        assert list(result["name"]) == ["a", "b", "c"]

    def test_unit_mismatch_geographic_crs(self):
        gdf = _make_geographic_gdf()
        with pytest.raises(UnitMismatch, match="not metres"):
            vector_buffer(gdf, distance=100.0)

    def test_zero_distance_raises(self):
        gdf = _make_projected_gdf()
        with pytest.raises(ValueError, match="must not be 0"):
            vector_buffer(gdf, distance=0)

    def test_no_crs_raises(self):
        gdf = gpd.GeoDataFrame(
            {"name": ["a"]}, geometry=[Point(0, 0)]
        )
        with pytest.raises(UnitMismatch, match="no CRS"):
            vector_buffer(gdf, distance=10.0)


# ---------------------------------------------------------------------------
# vector_reproject
# ---------------------------------------------------------------------------


class TestVectorReproject:
    def test_reproject_epsg(self):
        gdf = _make_geographic_gdf()
        result = vector_reproject(gdf, projection=3857)
        assert result.crs.to_epsg() == 3857

    def test_reproject_preserves_rows(self):
        gdf = _make_geographic_gdf()
        result = vector_reproject(gdf, projection=3857)
        assert len(result) == len(gdf)

    def test_reproject_preserves_properties(self):
        gdf = _make_geographic_gdf()
        result = vector_reproject(gdf, projection=3857)
        assert list(result["name"]) == list(gdf["name"])

    def test_dimension_not_available(self):
        gdf = _make_geographic_gdf()
        with pytest.raises(DimensionNotAvailable, match="does not exist"):
            vector_reproject(gdf, projection=3857, dimension="nonexistent")

    def test_reproject_with_wkt_string(self):
        gdf = _make_geographic_gdf()
        # Use an EPSG code as WKT2 alternative (projecting to UTM zone 32N)
        result = vector_reproject(gdf, projection=32632)
        assert result.crs.to_epsg() == 32632

    def test_roundtrip_reproject(self):
        """Reproject 4326→3857→4326 should approximately preserve coordinates."""
        gdf = _make_geographic_gdf()
        step1 = vector_reproject(gdf, projection=3857)
        step2 = vector_reproject(step1, projection=4326)
        for orig, back in zip(gdf.geometry, step2.geometry):
            assert abs(orig.x - back.x) < 1e-6
            assert abs(orig.y - back.y) < 1e-6
