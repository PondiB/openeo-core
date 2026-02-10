"""DataCube – fluent wrapper with runtime dispatch to raster / vector ops.

Usage::

    from openeo_core import DataCube

    cube = DataCube.load_collection("sentinel-2-l2a", spatial_extent=..., bands=[...])
    result = cube.filter_bbox(west=..., south=..., east=..., north=...) \
                 .filter_temporal(extent=("2023-01-01", "2023-06-30")) \
                 .ndvi(nir="B08", red="B04") \
                 .compute()
"""

from __future__ import annotations

from typing import Any, Callable, Union

import geopandas as gpd
import xarray as xr

from openeo_core.types import Cube, RasterCube, VectorCube


class DataCube:
    """Immutable wrapper around a raster or vector data cube.

    Methods return **new** ``DataCube`` instances so that the original is
    never mutated.
    """

    def __init__(self, data: Cube) -> None:
        self._data = data

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def data(self) -> Cube:
        """Access the underlying xarray DataArray or GeoDataFrame."""
        return self._data

    @property
    def is_raster(self) -> bool:
        return isinstance(self._data, xr.DataArray)

    @property
    def is_vector(self) -> bool:
        return isinstance(self._data, (gpd.GeoDataFrame,)) or (
            _has_dask_geopandas() and isinstance(self._data, _dask_geopandas().GeoDataFrame)
        )

    # ------------------------------------------------------------------
    # Loaders (classmethods)
    # ------------------------------------------------------------------

    @classmethod
    def load_collection(
        cls,
        collection_id: str,
        *,
        adapter: Any | None = None,
        spatial_extent: dict | None = None,
        temporal_extent: tuple[str, str] | None = None,
        bands: list[str] | None = None,
        properties: dict | None = None,
        **kwargs: Any,
    ) -> "DataCube":
        """Load a named EO collection into a raster DataCube.

        Parameters
        ----------
        collection_id : str
            STAC collection identifier (e.g. ``"sentinel-2-l2a"``).
        adapter : CollectionLoader | None
            Custom loader.  Uses the default AWS Earth Search adapter when ``None``.
        spatial_extent : dict | None
            Bounding box ``{west, south, east, north}``.
        temporal_extent : tuple[str, str] | None
            Temporal range ``("start", "end")`` in ISO-8601.
        bands : list[str] | None
            Band / asset names to include.
        properties : dict | None
            Extra STAC query parameters.
        """
        from openeo_core.io.collection import load_collection as _load

        da = _load(
            collection_id,
            spatial_extent=spatial_extent,
            temporal_extent=temporal_extent,
            bands=bands,
            properties=properties,
            adapter=adapter,
            **kwargs,
        )
        return cls(da)

    @classmethod
    def load_stac(
        cls,
        source: str | dict,
        *,
        adapter: Any | None = None,
        assets: list[str] | None = None,
        spatial_extent: dict | None = None,
        temporal_extent: tuple[str, str] | None = None,
        **kwargs: Any,
    ) -> "DataCube":
        """Load data from a STAC Item / Collection / API endpoint.

        Parameters
        ----------
        source : str | dict
            URL, file path, or inline STAC JSON dict.
        adapter : StacLoader | None
            Custom STAC loader.
        """
        from openeo_core.io.stac import load_stac as _load

        result = _load(
            source,
            assets=assets,
            spatial_extent=spatial_extent,
            temporal_extent=temporal_extent,
            adapter=adapter,
            **kwargs,
        )
        return cls(result)

    @classmethod
    def load_geojson(
        cls,
        source: str | dict,
        *,
        crs: str | None = None,
        **kwargs: Any,
    ) -> "DataCube":
        """Load GeoJSON into a vector DataCube.

        Parameters
        ----------
        source : str | dict
            File path, URL, or inline GeoJSON dict.
        crs : str | None
            CRS to assign.
        """
        from openeo_core.io.geojson import load_geojson as _load

        gdf = _load(source, crs=crs, **kwargs)
        return cls(gdf)

    # ------------------------------------------------------------------
    # Raster operations
    # ------------------------------------------------------------------

    def ndvi(
        self,
        *,
        nir: str = "nir",
        red: str = "red",
        target_band: str | None = None,
        bands_dim: str = "bands",
    ) -> "DataCube":
        """Compute NDVI on a raster cube."""
        self._assert_raster("ndvi")
        from openeo_core.ops.raster import ndvi as _ndvi

        return DataCube(
            _ndvi(self._data, nir=nir, red=red, target_band=target_band, bands_dim=bands_dim)  # type: ignore[arg-type]
        )

    def filter_bbox(
        self,
        *,
        west: float,
        south: float,
        east: float,
        north: float,
        x_dim: str = "x",
        y_dim: str = "y",
    ) -> "DataCube":
        """Filter data to a bounding box."""
        if self.is_raster:
            from openeo_core.ops.raster import filter_bbox as _fb

            return DataCube(
                _fb(self._data, west=west, south=south, east=east, north=north, x_dim=x_dim, y_dim=y_dim)  # type: ignore[arg-type]
            )
        else:
            from openeo_core.ops.vector import filter_bbox as _fb_v

            return DataCube(
                _fb_v(self._data, west=west, south=south, east=east, north=north)  # type: ignore[arg-type]
            )

    def filter_temporal(
        self,
        *,
        extent: tuple[str, str],
        t_dim: str = "t",
    ) -> "DataCube":
        """Filter a raster cube to a temporal interval."""
        self._assert_raster("filter_temporal")
        from openeo_core.ops.raster import filter_temporal as _ft

        return DataCube(_ft(self._data, extent=extent, t_dim=t_dim))  # type: ignore[arg-type]

    def aggregate_spatial(
        self,
        geometries: Any = None,
        *,
        reducer: str = "mean",
        x_dim: str = "x",
        y_dim: str = "y",
    ) -> "DataCube":
        """Aggregate raster values over spatial geometries."""
        self._assert_raster("aggregate_spatial")
        from openeo_core.ops.raster import aggregate_spatial as _agg

        return DataCube(
            _agg(self._data, geometries, reducer=reducer, x_dim=x_dim, y_dim=y_dim)  # type: ignore[arg-type]
        )

    def aggregate_temporal(
        self,
        *,
        period: str = "month",
        reducer: str = "mean",
        t_dim: str = "t",
    ) -> "DataCube":
        """Aggregate raster values over calendar periods."""
        self._assert_raster("aggregate_temporal")
        from openeo_core.ops.raster import aggregate_temporal as _agg

        return DataCube(_agg(self._data, period=period, reducer=reducer, t_dim=t_dim))  # type: ignore[arg-type]

    def apply(
        self,
        process: Callable[..., Any],
        *,
        context: Any = None,
    ) -> "DataCube":
        """Apply a function element-wise to every value in the cube."""
        if self.is_raster:
            from openeo_core.ops.raster import apply as _apply_r

            return DataCube(_apply_r(self._data, process, context=context))  # type: ignore[arg-type]
        else:
            from openeo_core.ops.vector import apply as _apply_v

            return DataCube(_apply_v(self._data, process, context=context))  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # Materialisation
    # ------------------------------------------------------------------

    def compute(self) -> "DataCube":
        """Materialise dask-backed data into memory.

        Returns a new ``DataCube`` wrapping the computed result.
        """
        if hasattr(self._data, "compute"):
            return DataCube(self._data.compute())
        return self

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        kind = "Raster" if self.is_raster else "Vector"
        return f"<DataCube({kind}) {self._data.__class__.__name__}>"

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _assert_raster(self, method: str) -> None:
        if not self.is_raster:
            raise TypeError(f"{method}() requires a raster cube, got {type(self._data).__name__}")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _has_dask_geopandas() -> bool:
    try:
        import dask_geopandas  # noqa: F401

        return True
    except ImportError:
        return False


def _dask_geopandas():
    import dask_geopandas

    return dask_geopandas
