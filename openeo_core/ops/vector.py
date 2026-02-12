"""Vector operations – geopandas, dask-geopandas, and xvec implementations."""

from __future__ import annotations

from typing import Any, Callable, Union

import geopandas as gpd
import numpy as np
import xarray as xr

try:
    import dask_geopandas
except ImportError:  # pragma: no cover
    dask_geopandas = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# filter_bbox (vector)
# ---------------------------------------------------------------------------


def filter_bbox(
    data: Union[gpd.GeoDataFrame, xr.DataArray, xr.Dataset],
    *,
    west: float,
    south: float,
    east: float,
    north: float,
) -> Union[gpd.GeoDataFrame, xr.DataArray, xr.Dataset]:
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
    data: Union[gpd.GeoDataFrame, xr.DataArray, xr.Dataset],
    process: Callable[..., Any],
    *,
    context: Any = None,
) -> Union[gpd.GeoDataFrame, xr.DataArray, xr.Dataset]:
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
    gdf: Union[gpd.GeoDataFrame, xr.DataArray, xr.Dataset],
    *,
    feature_columns: list[str] | None = None,
    target_column: str | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Convert a vector cube into a feature matrix X and optional target y.

    Supports GeoDataFrame and xvec-backed xarray (converted to GeoDataFrame first).

    Parameters
    ----------
    gdf : GeoDataFrame | DataArray | Dataset
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
    if isinstance(gdf, (xr.DataArray, xr.Dataset)) and _has_xvec_geometry(gdf):
        gdf = gdf.xvec.to_geopandas()

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
