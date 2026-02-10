"""GeoJSON loader – load GeoJSON into a VectorCube."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import geopandas as gpd


@runtime_checkable
class GeoJsonLoader(Protocol):
    """Protocol for GeoJSON loaders."""

    def load_geojson(
        self,
        source: str | dict,
        *,
        crs: str | None = None,
        **kwargs: Any,
    ) -> gpd.GeoDataFrame:
        ...


class DefaultGeoJsonLoader:
    """Default GeoJSON loader using geopandas."""

    def load_geojson(
        self,
        source: str | dict,
        *,
        crs: str | None = None,
        **kwargs: Any,
    ) -> gpd.GeoDataFrame:
        """Load GeoJSON from a file path, URL, or inline dict.

        Parameters
        ----------
        source : str | dict
            File path / URL to a GeoJSON file, or an inline GeoJSON dict.
        crs : str | None
            Optional CRS to assign (e.g. ``"EPSG:4326"``).  If *None* the
            CRS is inferred from the data (GeoJSON is always WGS 84).
        """
        if isinstance(source, dict):
            features = source.get("features", [source])
            gdf = gpd.GeoDataFrame.from_features(features)
        else:
            gdf = gpd.read_file(source)

        if crs is not None:
            gdf = gdf.set_crs(crs, allow_override=True)
        elif gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")

        return gdf


# Module-level convenience function using the default loader.
_default = DefaultGeoJsonLoader()


def load_geojson(
    source: str | dict,
    *,
    crs: str | None = None,
    **kwargs: Any,
) -> gpd.GeoDataFrame:
    """Load GeoJSON into a :class:`~geopandas.GeoDataFrame`.

    Delegates to :class:`DefaultGeoJsonLoader`.
    """
    return _default.load_geojson(source, crs=crs, **kwargs)
