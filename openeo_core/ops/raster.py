"""Raster operations – xarray / dask implementations of openEO processes."""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np
import xarray as xr

# ---------------------------------------------------------------------------
# NDVI
# ---------------------------------------------------------------------------


def ndvi(
    data: xr.DataArray,
    *,
    nir: str = "nir",
    red: str = "red",
    target_band: str | None = None,
    bands_dim: str = "bands",
) -> xr.DataArray:
    """Compute the Normalized Difference Vegetation Index.

    Parameters
    ----------
    data : xr.DataArray
        Raster cube with a *bands* dimension containing at least *nir* and *red*.
    nir, red : str
        Band labels for the near-infrared and red channels.
    target_band : str | None
        If given, the NDVI is appended as a new band with this name;
        otherwise the *bands* dimension is dropped.
    bands_dim : str
        Name of the bands dimension (default ``"bands"``).
    """
    nir_data = data.sel({bands_dim: nir})
    red_data = data.sel({bands_dim: red})

    result = (nir_data - red_data) / (nir_data + red_data)

    if target_band is not None:
        result = result.expand_dims({bands_dim: [target_band]})
        result = xr.concat([data, result], dim=bands_dim)
    else:
        result.name = "ndvi"

    return result


# ---------------------------------------------------------------------------
# filter_bbox
# ---------------------------------------------------------------------------


def filter_bbox(
    data: xr.DataArray,
    *,
    west: float,
    south: float,
    east: float,
    north: float,
    x_dim: str = "x",
    y_dim: str = "y",
) -> xr.DataArray:
    """Restrict a raster cube to a bounding box.

    Uses xarray label-based selection; assumes monotonic spatial coords.
    """
    # Handle both ascending and descending y-axes
    y_coords = data.coords[y_dim].values
    if y_coords[0] < y_coords[-1]:
        y_slice = slice(south, north)
    else:
        y_slice = slice(north, south)

    return data.sel({x_dim: slice(west, east), y_dim: y_slice})


# ---------------------------------------------------------------------------
# filter_temporal
# ---------------------------------------------------------------------------


def filter_temporal(
    data: xr.DataArray,
    *,
    extent: tuple[str, str],
    t_dim: str = "t",
) -> xr.DataArray:
    """Filter a raster cube to a temporal interval ``[start, end]``."""
    start, end = extent
    return data.sel({t_dim: slice(start, end)})


# ---------------------------------------------------------------------------
# aggregate_spatial
# ---------------------------------------------------------------------------


def aggregate_spatial(
    data: xr.DataArray,
    geometries: Any,
    *,
    reducer: str = "mean",
    x_dim: str = "x",
    y_dim: str = "y",
) -> xr.DataArray:
    """Aggregate raster values over spatial geometries.

    This is a simplified implementation that applies a named reducer
    over the spatial dimensions.

    Parameters
    ----------
    data : xr.DataArray
        Raster cube.
    geometries
        Not used in the simplified path – aggregates over the full spatial extent.
        A future version will clip/mask per geometry.
    reducer : str
        One of ``"mean"``, ``"sum"``, ``"min"``, ``"max"``, ``"median"``.
    """
    return getattr(data, reducer)(dim=[x_dim, y_dim])


# ---------------------------------------------------------------------------
# aggregate_temporal
# ---------------------------------------------------------------------------


def aggregate_temporal(
    data: xr.DataArray,
    *,
    period: str = "month",
    reducer: str = "mean",
    t_dim: str = "t",
) -> xr.DataArray:
    """Aggregate raster values over calendar periods.

    Parameters
    ----------
    period : str
        One of ``"year"``, ``"month"``, ``"day"``, ``"season"``, etc.
        Maps to xarray resample frequency codes.
    reducer : str
        Named reducer applied within each period.
    """
    freq_map = {
        "hour": "H",
        "day": "D",
        "week": "W",
        "month": "ME",
        "season": "QS-DEC",
        "year": "YE",
    }
    freq = freq_map.get(period)
    if freq is None:
        raise ValueError(
            f"Unsupported period {period!r}. Choose from {list(freq_map)}"
        )

    resampled = data.resample({t_dim: freq})
    return getattr(resampled, reducer)()


# ---------------------------------------------------------------------------
# apply (generic per-element operation)
# ---------------------------------------------------------------------------


def apply(
    data: xr.DataArray,
    process: Callable[..., Any],
    *,
    context: Any = None,
) -> xr.DataArray:
    """Apply a unary function to every value in the data cube.

    Uses :func:`xarray.apply_ufunc` so that the operation is dask-safe.

    Parameters
    ----------
    data : xr.DataArray
        Input raster cube.
    process : callable
        Function ``f(x, context=...) -> x`` applied element-wise.
    context
        Optional extra data forwarded to *process*.
    """
    return xr.apply_ufunc(
        process,
        data,
        kwargs={"context": context} if context is not None else {},
        dask="parallelized",
        output_dtypes=[data.dtype],
    )


# ---------------------------------------------------------------------------
# Raster utilities for ML
# ---------------------------------------------------------------------------


def stack_to_samples(
    data: xr.DataArray,
    feature_dim: str = "bands",
) -> xr.DataArray:
    """Stack all non-feature dims into a ``samples`` dim.

    Returns a 2-D DataArray with shape ``(samples, features)``.
    """
    non_feature = [d for d in data.dims if d != feature_dim]
    stacked = data.stack(samples=non_feature)
    return stacked.transpose("samples", feature_dim)


def unstack_from_samples(
    result: xr.DataArray,
    template: xr.DataArray,
    feature_dim: str = "bands",
) -> xr.DataArray:
    """Reverse :func:`stack_to_samples` using *template*'s multi-index."""
    non_feature = [d for d in template.dims if d != feature_dim]
    stacked_template = template.stack(samples=non_feature)
    result = result.assign_coords(samples=stacked_template.coords["samples"])
    return result.unstack("samples")
