"""DataCube – fluent wrapper with runtime dispatch to raster / vector ops.

Usage::

    from openeo_core import DataCube

    cube = DataCube.load_collection("sentinel-2-l2a", spatial_extent=..., bands=[...])
    result = cube.filter_bbox(west=..., south=..., east=..., north=...) \
                 .filter_temporal(extent=("2023-01-01", "2023-06-30")) \
                 .ndvi(nir="nir", red="red") \
                 .compute()
"""

from __future__ import annotations

from typing import Any, Callable

import geopandas as gpd
import xarray as xr

from openeo_core.types import Cube


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
        if isinstance(self._data, xr.DataArray):
            return not _is_xvec_vector(self._data)
        return False

    @property
    def is_vector(self) -> bool:
        if isinstance(self._data, (gpd.GeoDataFrame,)):
            return True
        if _has_dask_geopandas() and isinstance(self._data, _dask_geopandas().GeoDataFrame):
            return True
        return _is_xvec_vector(self._data)

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
            Bounding box ``{west, south, east, north}``.  May include an
            optional ``crs`` key (EPSG code as int, or WKT2 string;
            defaults to ``4326``).
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
        x_dim: str = "longitude",
        y_dim: str = "latitude",
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
        t_dim: str = "time",
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
        x_dim: str = "longitude",
        y_dim: str = "latitude",
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
        dimension: str | None = None,
        t_dim: str = "time",
    ) -> "DataCube":
        """Aggregate raster values over calendar periods.

        Parameters
        ----------
        period : str
            Calendar period: ``"hour"``, ``"day"``, ``"week"``, ``"dekad"``,
            ``"month"``, ``"season"``, ``"tropical-season"``, ``"year"``,
            ``"decade"``, ``"decade-ad"``.
        reducer : str
            Named reducer (``"mean"``, ``"sum"``, ``"min"``, ``"max"``,
            ``"median"``).
        dimension : str | None
            Explicit temporal dimension name.  When ``None`` (default),
            falls back to *t_dim*.
        t_dim : str
            Legacy alias for the temporal dimension name.
        """
        self._assert_raster("aggregate_temporal")
        from openeo_core.ops.raster import aggregate_temporal as _agg

        effective_dim = dimension if dimension is not None else t_dim
        return DataCube(_agg(self._data, period=period, reducer=reducer, t_dim=effective_dim))  # type: ignore[arg-type]

    # Alias matching the openEO process id
    aggregate_temporal_period = aggregate_temporal

    def resample_spatial(
        self,
        *,
        resolution: float | list[float] = 0,
        projection: int | str | None = None,
        method: str = "near",
        align: str = "upper-left",
        x_dim: str = "longitude",
        y_dim: str = "latitude",
    ) -> "DataCube":
        """Resample and/or reproject spatial dimensions.

        Requires the ``rioxarray`` optional dependency (``uv sync --extra geo``).

        Parameters
        ----------
        resolution : float | list[float]
            Target resolution.  A single number for both axes, or
            ``[x_res, y_res]``.  ``0`` keeps the original resolution.
        projection : int | str | None
            Target CRS as EPSG code or WKT2 string.  ``None`` keeps current.
        method : str
            Resampling method (``"near"``, ``"bilinear"``, ``"cubic"``, etc.).
        align : str
            Corner alignment.
        """
        self._assert_raster("resample_spatial")
        from openeo_core.ops.raster import resample_spatial as _rs

        return DataCube(
            _rs(
                self._data,  # type: ignore[arg-type]
                resolution=resolution,
                projection=projection,
                method=method,
                align=align,
                x_dim=x_dim,
                y_dim=y_dim,
            )
        )

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

    def reduce_dimension(
        self,
        reducer: str | Callable[..., Any],
        *,
        dimension: str,
        context: Any = None,
    ) -> "DataCube":
        """Reduce a dimension by applying a reducer function.

        The specified dimension is collapsed and dropped from the result.

        Parameters
        ----------
        reducer : str | callable
            A string name (``"mean"``, ``"sum"``, ``"min"``, ``"max"``,
            ``"median"``, ``"std"``, ``"var"``, ``"prod"``, ``"count"``,
            ``"any"``, ``"all"``), a dotted Python path
            (e.g. ``"numpy.nanmean"``), or a callable.
        dimension : str
            Dimension to reduce over.
        context
            Optional extra data forwarded to *reducer*.
        """
        self._assert_raster("reduce_dimension")
        from openeo_core.ops.raster import reduce_dimension as _rd

        return DataCube(
            _rd(self._data, reducer, dimension=dimension, context=context)  # type: ignore[arg-type]
        )

    def apply_kernel(
        self,
        *,
        kernel: list[list[float]],
        factor: float = 1.0,
        border: float | str = 0,
        replace_invalid: float = 0.0,
        x_dim: str = "longitude",
        y_dim: str = "latitude",
    ) -> "DataCube":
        """Apply a 2-D spatial convolution kernel.

        Parameters
        ----------
        kernel : list[list[float]]
            2-D array of convolution weights (odd dimensions required).
        factor : float
            Multiplicative factor applied after convolution.
        border : float | str
            Border handling strategy.
        replace_invalid : float
            Value to substitute for NaN / Inf before convolution.
        """
        self._assert_raster("apply_kernel")
        from openeo_core.ops.raster import apply_kernel as _ak

        return DataCube(
            _ak(
                self._data,  # type: ignore[arg-type]
                kernel=kernel,
                factor=factor,
                border=border,
                replace_invalid=replace_invalid,
                x_dim=x_dim,
                y_dim=y_dim,
            )
        )

    def array_interpolate_linear(
        self,
        *,
        dimension: str | None = None,
    ) -> "DataCube":
        """One-dimensional linear interpolation over NaN / no-data values.

        Fills interior NaN / no-data gaps using linear interpolation along
        *dimension*, leaving leading and trailing NaN values untouched.

        Implements the ``array_interpolate_linear`` openEO process.

        Parameters
        ----------
        dimension : str | None
            Dimension along which to interpolate (e.g. ``"time"``).
            Required for raster cubes.
        """
        self._assert_raster("array_interpolate_linear")
        from openeo_core.ops.raster import array_interpolate_linear as _ail

        return DataCube(
            _ail(self._data, dimension=dimension)  # type: ignore[arg-type]
        )

    # ------------------------------------------------------------------
    # Vector operations
    # ------------------------------------------------------------------

    def vector_buffer(
        self,
        *,
        distance: float,
    ) -> "DataCube":
        """Buffer each geometry by *distance* metres.

        Parameters
        ----------
        distance : float
            Buffer distance in metres.  Positive expands, negative shrinks.
        """
        self._assert_vector("vector_buffer")
        from openeo_core.ops.vector import vector_buffer as _vb

        return DataCube(_vb(self._data, distance=distance))  # type: ignore[arg-type]

    def vector_reproject(
        self,
        *,
        projection: int | str,
        dimension: str | None = None,
    ) -> "DataCube":
        """Reproject geometries to a different CRS.

        Parameters
        ----------
        projection : int | str
            Target CRS as EPSG code (int) or WKT2 string.
        dimension : str | None
            Geometry dimension to reproject.  ``None`` reprojects all.
        """
        self._assert_vector("vector_reproject")
        from openeo_core.ops.vector import vector_reproject as _vr

        return DataCube(
            _vr(self._data, projection=projection, dimension=dimension)  # type: ignore[arg-type]
        )

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
    # Plotting
    # ------------------------------------------------------------------

    def plot(self, *args: Any, **kwargs: Any) -> Any:
        """Delegate to the underlying data object's ``.plot()`` method.

        For raster cubes this calls :meth:`xarray.DataArray.plot`;
        for vector cubes from GeoDataFrame calls :meth:`geopandas.GeoDataFrame.plot`;
        for xvec-backed vectors calls :meth:`xarray.DataArray.xvec.plot`.

        All positional and keyword arguments are forwarded as-is.
        """
        if self.is_vector and _is_xvec_vector(self._data):
            return self._data.xvec.plot(*args, **kwargs)
        return self._data.plot(*args, **kwargs)

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

    def _assert_vector(self, method: str) -> None:
        if not self.is_vector:
            raise TypeError(f"{method}() requires a vector cube, got {type(self._data).__name__}")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _is_xvec_vector(obj: Any) -> bool:
    """Return True if *obj* is an xarray object with xvec geometry coordinates."""
    if not isinstance(obj, (xr.DataArray, xr.Dataset)):
        return False
    try:
        import xvec  # noqa: F401
    except ImportError:
        return False
    if not hasattr(obj, "xvec"):
        return False
    try:
        geom_coords = obj.xvec.geom_coords
        geom_indexed = obj.xvec.geom_coords_indexed
    except AttributeError:
        return False
    return bool(geom_coords) or bool(geom_indexed)


def _has_dask_geopandas() -> bool:
    try:
        import dask_geopandas  # noqa: F401

        return True
    except ImportError:
        return False


def _dask_geopandas():
    import dask_geopandas

    return dask_geopandas
