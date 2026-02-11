"""I/O sub-package – data loaders (collection, STAC, GeoJSON)."""

from openeo_core.io.geojson import GeoJsonLoader, load_geojson
from openeo_core.io.collection import CollectionLoader, load_collection
from openeo_core.io.stac import StacLoader, load_stac

__all__ = [
    "CollectionLoader",
    "GeoJsonLoader",
    "StacLoader",
    "load_collection",
    "load_geojson",
    "load_stac",
]
