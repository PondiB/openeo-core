"""Tests for the GeoJSON loader."""

import geopandas as gpd

from openeo_core.io.geojson import load_geojson


class TestLoadGeoJson:
    def test_load_from_dict(self):
        geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [10, 50]},
                    "properties": {"name": "A"},
                },
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [11, 51]},
                    "properties": {"name": "B"},
                },
            ],
        }
        gdf = load_geojson(geojson)
        assert isinstance(gdf, gpd.GeoDataFrame)
        assert len(gdf) == 2
        assert gdf.crs is not None

    def test_load_with_crs_override(self):
        geojson = {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [0, 0]},
            "properties": {},
        }
        gdf = load_geojson(geojson, crs="EPSG:3857")
        assert gdf.crs.to_epsg() == 3857
