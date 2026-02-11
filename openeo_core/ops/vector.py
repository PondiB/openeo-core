"""Vector operations – geopandas / dask-geopandas implementations."""

from __future__ import annotations

from typing import Any, Callable

import geopandas as gpd
import numpy as np

try:
    import dask_geopandas
except ImportError:  # pragma: no cover
    dask_geopandas = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# filter_bbox (vector)
# ---------------------------------------------------------------------------


def filter_bbox(
    data: gpd.GeoDataFrame,
    *,
    west: float,
    south: float,
    east: float,
    north: float,
) -> gpd.GeoDataFrame:
    """Keep geometries whose bounding box is fully inside the extent."""
    from shapely.geometry import box

    bbox = box(west, south, east, north)
    mask = data.geometry.within(bbox)
    return data.loc[mask].copy()


# ---------------------------------------------------------------------------
# apply (vector – row-wise operation)
# ---------------------------------------------------------------------------


def apply(
    data: gpd.GeoDataFrame,
    process: Callable[..., Any],
    *,
    context: Any = None,
) -> gpd.GeoDataFrame:
    """Apply a function to each row / partition of a (dask-)GeoDataFrame."""
    if dask_geopandas is not None and isinstance(data, dask_geopandas.GeoDataFrame):
        return data.map_partitions(process, context=context)
    return process(data, context=context)


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------


def to_feature_matrix(
    gdf: gpd.GeoDataFrame,
    *,
    feature_columns: list[str] | None = None,
    target_column: str | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Convert a GeoDataFrame into a feature matrix X and optional target y.

    Parameters
    ----------
    gdf : GeoDataFrame
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
    if feature_columns is None:
        numeric = gdf.select_dtypes(include=[np.number])
        if target_column and target_column in numeric.columns:
            numeric = numeric.drop(columns=[target_column])
        feature_columns = list(numeric.columns)

    X = gdf[feature_columns].to_numpy()
    y = gdf[target_column].to_numpy() if target_column else None
    return X, y
