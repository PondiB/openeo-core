"""Vector operations – geopandas, dask-geopandas, and xvec implementations."""

from __future__ import annotations

from typing import Any, Callable

import geopandas as gpd
import numpy as np
import pyproj
import xarray as xr

from openeo_core.exceptions import DimensionNotAvailable, UnitMismatch
from openeo_core.types import VectorCube

try:
    import dask_geopandas
except ImportError:  # pragma: no cover
    dask_geopandas = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# vector_buffer
# ---------------------------------------------------------------------------


def vector_buffer(
    geometries: VectorCube,
    *,
    distance: float,
) -> VectorCube:
    """Buffer each geometry by *distance* metres.

    Implements the ``vector_buffer`` openEO process.  A positive *distance*
    dilates (expands) geometries; a negative *distance* erodes (shrinks) them.
    Results are always polygons.

    Parameters
    ----------
    geometries : VectorCube
        Input vector cube.  Feature properties are preserved.
    distance : float
        Buffer distance in the units of the geometry's CRS.  The CRS must
        use metre-based units; otherwise a ``UnitMismatch`` exception is raised.

    Raises
    ------
    UnitMismatch
        If the CRS of the geometries is not metre-based (e.g. EPSG:4326 uses
        degrees).  Use ``vector_reproject()`` first to convert to a projected CRS.
    """
    if distance == 0:
        raise ValueError("distance must not be 0")

    # --- xvec-backed xarray ---
    if isinstance(geometries, (xr.DataArray, xr.Dataset)) and _has_xvec_geometry(geometries):
        coord_name = _first_geom_coord_name(geometries)
        if coord_name is None:
            raise ValueError("xvec data has no geometry coordinate to buffer")
        # Determine CRS from the xvec geometry index
        crs = _xvec_crs(geometries, coord_name)
        _assert_metre_crs(crs)
        geom_values = geometries.coords[coord_name].values
        buffered = [g.buffer(distance) if not g.is_empty else g for g in geom_values]
        return geometries.assign_coords({coord_name: buffered})

    # --- xarray without xvec ---
    if isinstance(geometries, (xr.DataArray, xr.Dataset)):
        raise TypeError(
            "vector_buffer only supports xarray objects with xvec geometry coordinates."
        )

    # --- GeoDataFrame / dask GeoDataFrame ---
    if isinstance(geometries, gpd.GeoDataFrame) or (
        dask_geopandas is not None
        and isinstance(geometries, dask_geopandas.GeoDataFrame)
    ):
        _assert_metre_crs(geometries.crs)
        result = geometries.copy()
        result["geometry"] = result.geometry.buffer(distance)
        return result

    raise TypeError(
        f"vector_buffer expects a GeoDataFrame, dask GeoDataFrame, or xvec-backed "
        f"xarray; got {type(geometries)!r}."
    )


# ---------------------------------------------------------------------------
# vector_reproject
# ---------------------------------------------------------------------------


def vector_reproject(
    data: VectorCube,
    *,
    projection: int | str,
    dimension: str | None = None,
) -> VectorCube:
    """Reproject geometries to a different coordinate reference system.

    Implements the ``vector_reproject`` openEO process.

    Parameters
    ----------
    data : VectorCube
        Input vector cube.
    projection : int | str
        Target CRS as an EPSG code (int) or WKT2 string.
    dimension : str | None
        Name of the geometry dimension to reproject.  If ``None``, all
        geometry dimensions are reprojected.

    Raises
    ------
    DimensionNotAvailable
        If the specified *dimension* does not exist.
    """
    target_crs = f"EPSG:{projection}" if isinstance(projection, int) else projection

    # --- xvec-backed xarray ---
    if isinstance(data, (xr.DataArray, xr.Dataset)) and _has_xvec_geometry(data):
        if dimension is not None:
            if dimension not in data.dims and dimension not in data.coords:
                raise DimensionNotAvailable(
                    f"A dimension with the specified name '{dimension}' does not exist."
                )
            coord_names = [dimension]
        else:
            coord_names = list(
                data.xvec.geom_coords_indexed
                if data.xvec.geom_coords_indexed
                else data.xvec.geom_coords
            )
        for coord_name in coord_names:
            src_crs = _xvec_crs(data, coord_name)
            geom_values = data.coords[coord_name].values
            transformer = pyproj.Transformer.from_crs(
                src_crs, target_crs, always_xy=True
            )
            from shapely.ops import transform as shapely_transform

            reprojected = [shapely_transform(transformer.transform, g) for g in geom_values]
            data = data.assign_coords({coord_name: reprojected})
        return data

    # --- xarray without xvec ---
    if isinstance(data, (xr.DataArray, xr.Dataset)):
        raise TypeError(
            "vector_reproject only supports xarray objects with xvec geometry coordinates."
        )

    # --- GeoDataFrame / dask GeoDataFrame ---
    if isinstance(data, gpd.GeoDataFrame) or (
        dask_geopandas is not None
        and isinstance(data, dask_geopandas.GeoDataFrame)
    ):
        if dimension is not None and dimension != "geometry":
            raise DimensionNotAvailable(
                f"A dimension with the specified name '{dimension}' does not exist."
            )
        return data.to_crs(target_crs)

    raise TypeError(
        f"vector_reproject expects a GeoDataFrame, dask GeoDataFrame, or xvec-backed "
        f"xarray; got {type(data)!r}."
    )


# ---------------------------------------------------------------------------
# filter_bbox (vector)
# ---------------------------------------------------------------------------


def filter_bbox(
    data: VectorCube,
    *,
    west: float,
    south: float,
    east: float,
    north: float,
) -> VectorCube:
    """Keep geometries whose bounding box is fully inside the extent.

    Supports GeoDataFrame, dask GeoDataFrame, and xarray DataArray/Dataset
    with xvec geometry coordinates.
    """
    from shapely.geometry import box

    bbox = box(west, south, east, north)

    # xvec-backed xarray
    if isinstance(data, (xr.DataArray, xr.Dataset)) and _has_xvec_geometry(data):
        coord_name = _first_geom_coord_name(data)
        if coord_name is None:
            raise ValueError("xvec data has no geometry coordinate to query")
        # Default query uses bbox intersection; predicate="contains" filters to
        # geometries inside bbox (bbox.contains(geom))
        return data.xvec.query(coord_name, bbox, predicate="contains")

    # xarray without xvec geometry: explicitly unsupported
    if isinstance(data, (xr.DataArray, xr.Dataset)):
        raise TypeError(
            "filter_bbox only supports xarray DataArray/Dataset objects with xvec "
            "geometry coordinates; got an xarray object without xvec geometry."
        )

    # GeoDataFrame / dask GeoDataFrame
    if isinstance(data, gpd.GeoDataFrame) or (
        dask_geopandas is not None
        and isinstance(data, dask_geopandas.GeoDataFrame)
    ):
        mask = data.geometry.within(bbox)
        return data.loc[mask].copy()

    raise TypeError(
        f"filter_bbox only supports GeoDataFrame, dask GeoDataFrame, or xarray "
        f"DataArray/Dataset with xvec geometry; got {type(data)!r}."
    )

# ---------------------------------------------------------------------------
# apply (vector – row-wise operation)
# ---------------------------------------------------------------------------


def apply(
    data: VectorCube,
    process: Callable[..., Any],
    *,
    context: Any = None,
) -> VectorCube:
    """Apply a function to each row / partition of a vector cube.

    For GeoDataFrame / dask GeoDataFrame, applies directly.
    For xvec-backed xarray, converts to GeoDataFrame, applies, returns GeoDataFrame.
    """
    # xvec-backed xarray: convert to GeoDataFrame, apply, return GeoDataFrame
    if isinstance(data, (xr.DataArray, xr.Dataset)) and _has_xvec_geometry(data):
        gdf = data.xvec.to_geopandas()
        result = process(gdf, context=context) if context is not None else process(gdf)
        return result

    if dask_geopandas is not None and isinstance(data, dask_geopandas.GeoDataFrame):
        return data.map_partitions(process, context=context)
    return process(data, context=context)


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------


def to_feature_matrix(
    gdf: VectorCube,
    *,
    feature_columns: list[str] | None = None,
    target_column: str | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Convert a vector cube into a feature matrix X and optional target y.

    Supports GeoDataFrame and xvec-backed xarray (converted to GeoDataFrame first).

    Parameters
    ----------
    gdf : VectorCube
        Input vector cube.
    feature_columns : list[str] | None
        Columns to use as features.  If *None*, all numeric columns
        (excluding *target_column*) are used.
    target_column : str | None
        Column name for the target variable.

    Returns
    -------
    X : np.ndarray  (n_samples, n_features)
    y : np.ndarray | None  (n_samples,)
    """
    # Convert supported xarray inputs (with xvec geometry) to GeoDataFrame
    if isinstance(gdf, (xr.DataArray, xr.Dataset)):
        if not _has_xvec_geometry(gdf):
            raise TypeError(
                "to_feature_matrix only supports xarray DataArray/Dataset inputs "
                "that are backed by xvec geometry coordinates."
            )
        gdf = gdf.xvec.to_geopandas()

    # At this point we require a GeoDataFrame to proceed
    if not isinstance(gdf, gpd.GeoDataFrame):
        raise TypeError(
            f"to_feature_matrix expects a GeoDataFrame or xvec-backed xarray input, "
            f"got {type(gdf)!r}."
        )
    if feature_columns is None:
        numeric = gdf.select_dtypes(include=[np.number])
        if target_column and target_column in numeric.columns:
            numeric = numeric.drop(columns=[target_column])
        feature_columns = list(numeric.columns)

    X = gdf[feature_columns].to_numpy()
    y = gdf[target_column].to_numpy() if target_column else None
    return X, y


# ---------------------------------------------------------------------------
# xvec helpers
# ---------------------------------------------------------------------------


def _has_xvec_geometry(obj: xr.DataArray | xr.Dataset) -> bool:
    """Return True if *obj* has xvec geometry coordinates."""
    try:
        import xvec  # noqa: F401
    except ImportError:
        return False
    if not hasattr(obj, "xvec"):
        return False
    try:
        return bool(obj.xvec.geom_coords) or bool(obj.xvec.geom_coords_indexed)
    except AttributeError:
        return False


def _first_geom_coord_name(obj: xr.DataArray | xr.Dataset) -> str | None:
    """Return the first geometry coordinate name, or None."""
    try:
        indexed = obj.xvec.geom_coords_indexed
        if indexed:
            return next(iter(indexed))
        coords = obj.xvec.geom_coords
        if coords:
            return next(iter(coords))
    except (AttributeError, StopIteration):
        pass
    return None


def _xvec_crs(obj: xr.DataArray | xr.Dataset, coord_name: str) -> pyproj.CRS | None:
    """Extract the CRS from an xvec geometry coordinate."""
    try:
        idx = obj.indexes[coord_name]
        if hasattr(idx, "crs"):
            return idx.crs
    except (KeyError, AttributeError):
        pass
    return None


def _assert_metre_crs(crs: Any) -> None:
    """Raise ``UnitMismatch`` if *crs* is not metre-based."""
    if crs is None:
        raise UnitMismatch(
            "The geometries have no CRS assigned.  Assign a metre-based CRS or "
            "use vector_reproject() first."
        )
    resolved = pyproj.CRS(crs)
    axis_info = resolved.axis_info
    # Check whether the CRS uses metre-based units on any axis
    units = {a.unit_name for a in axis_info}
    if "metre" not in units and "meter" not in units:
        raise UnitMismatch(
            "The unit of the spatial reference system is not metres, but the "
            "given distance is in metres.  Use vector_reproject() to convert "
            "the geometries to a suitable spatial reference system."
        )
