"""Raster operations – xarray / dask implementations of openEO processes."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import xarray as xr

from openeo_core.exceptions import (
    BandExists,
    DimensionAmbiguous,
    DimensionNotAvailable,
    KernelDimensionsUneven,
    NirBandAmbiguous,
    RedBandAmbiguous,
)
from openeo_core.types import RasterCube, VectorCube

# ---------------------------------------------------------------------------
# NDVI
# ---------------------------------------------------------------------------


def ndvi(
    data: RasterCube,
    *,
    nir: str = "nir",
    red: str = "red",
    target_band: str | None = None,
    bands_dim: str = "bands",
) -> RasterCube:
    """Compute the Normalized Difference Vegetation Index.

    Implements the ``ndvi`` openEO process.  The formula is
    ``(nir - red) / (nir + red)``.

    Parameters
    ----------
    data : RasterCube
        Raster cube with a dimension of type *bands* containing at least
        the *nir* and *red* bands.
    nir, red : str
        Band labels for the near-infrared and red channels.  Defaults to
        the common names ``"nir"`` and ``"red"``.
    target_band : str | None
        If given, the NDVI is appended as a new band with this name and
        the *bands* dimension is kept.  If ``None`` (default) the *bands*
        dimension is dropped.

    Raises
    ------
    DimensionAmbiguous
        If *bands_dim* is not present in the data cube.
    NirBandAmbiguous
        If the NIR band cannot be found.
    RedBandAmbiguous
        If the red band cannot be found.
    BandExists
        If *target_band* already exists as a label in the bands dimension.
    """
    # --- Validate bands dimension exists ---
    if bands_dim not in data.dims:
        raise DimensionAmbiguous(
            f"Dimension of type 'bands' ('{bands_dim}') is not available. "
            f"Available dimensions: {list(data.dims)}"
        )

    band_labels = list(data.coords[bands_dim].values)

    # --- Validate NIR band ---
    if nir not in band_labels:
        raise NirBandAmbiguous(
            f"The NIR band '{nir}' can't be resolved. "
            f"Available bands: {band_labels}. "
            f"Please specify the NIR band name."
        )

    # --- Validate red band ---
    if red not in band_labels:
        raise RedBandAmbiguous(
            f"The red band '{red}' can't be resolved. "
            f"Available bands: {band_labels}. "
            f"Please specify the red band name."
        )

    # --- Validate target_band doesn't already exist ---
    if target_band is not None and target_band in band_labels:
        raise BandExists(
            f"A band with the name '{target_band}' already exists."
        )

    nir_data = data.sel({bands_dim: nir}).astype(np.float32)
    red_data = data.sel({bands_dim: red}).astype(np.float32)

    # Mask nodata: Sentinel-2 L2A uses 0 as nodata, and newer
    # processing baselines (04.00+) apply a -1000 offset so valid
    # reflectance should be > 0.  Treat non-positive values as nodata.
    nir_data = xr.where(nir_data > 0, nir_data, np.nan)
    red_data = xr.where(red_data > 0, red_data, np.nan)

    # Safe division: return NaN where (nir + red) == 0 to avoid
    # RuntimeWarning: divide by zero encountered in divide.
    denominator = nir_data + red_data
    result = xr.where(denominator != 0, (nir_data - red_data) / denominator, np.nan)

    # NDVI is physically bounded to [-1, 1].  Clamp to catch any
    # residual artifacts from nodata/fill values.
    result = result.clip(-1, 1)
    result = result.astype(np.float32)

    if target_band is not None:
        result = result.expand_dims({bands_dim: [target_band]})
        # Use coords="minimal" + join="override" so that extra
        # metadata coordinates (e.g. STAC 'title' on bands) that
        # exist on `data` but not on the computed `result` don't
        # cause a concat error.
        result = xr.concat(
            [data, result],
            dim=bands_dim,
            coords="minimal",
            join="override",
        )
    else:
        result.name = "ndvi"

    return result


# ---------------------------------------------------------------------------
# filter_bbox
# ---------------------------------------------------------------------------


def filter_bbox(
    data: RasterCube,
    *,
    west: float,
    south: float,
    east: float,
    north: float,
    x_dim: str = "longitude",
    y_dim: str = "latitude",
) -> RasterCube:
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
    data: RasterCube,
    *,
    extent: tuple[str, str],
    t_dim: str = "time",
) -> RasterCube:
    """Filter a raster cube to a temporal interval ``[start, end]``."""
    start, end = extent
    return data.sel({t_dim: slice(start, end)})


# ---------------------------------------------------------------------------
# aggregate_spatial
# ---------------------------------------------------------------------------


def aggregate_spatial(
    data: RasterCube,
    geometries: Any,
    *,
    reducer: str = "mean",
    x_dim: str = "longitude",
    y_dim: str = "latitude",
    t_dim: str = "time",
    bands_dim: str = "bands",
) -> RasterCube | VectorCube:
    """Aggregate raster values over spatial geometries.

    When *geometries* is provided, computes zonal statistics (one value per
    geometry per band) and returns a GeoDataFrame (vector cube).

    When *geometries* is ``None``, applies the reducer over the full spatial
    extent and returns a scalar DataArray.

    Parameters
    ----------
    data : RasterCube
        Raster cube.
    geometries : str | geopandas.GeoDataFrame | geopandas.GeoSeries | None
        Path to a GeoJSON file, or a GeoDataFrame/GeoSeries of polygons.
        If ``None``, aggregates over the full extent.
    reducer : str
        One of ``"mean"``, ``"sum"``, ``"min"``, ``"max"``, ``"median"``.
    x_dim, y_dim : str
        Spatial dimension names.
    t_dim, bands_dim : str
        Temporal and bands dimension names (for flattening to feature table).
    """
    import geopandas as gpd

    if geometries is None:
        return getattr(data, reducer)(dim=[x_dim, y_dim])

    # Load geometries
    if isinstance(geometries, (str, Path)):
        gdf = gpd.read_file(geometries)
        geoms = gdf.geometry
        extra_cols = gdf.drop(columns=["geometry"])
    else:
        geoms = geometries.geometry if isinstance(geometries, gpd.GeoDataFrame) else geometries
        extra_cols = (
            geometries.drop(columns=["geometry"])
            if isinstance(geometries, gpd.GeoDataFrame)
            else None
        )

    import xvec  # noqa: F401

    # Align CRS: geometries must match raster CRS for zonal_stats
    data_crs = None
    if hasattr(data, "rio") and data.rio.crs is not None:
        data_crs = str(data.rio.crs)
    if data_crs is None:
        try:
            import stackstac
            data_epsg = stackstac.array_epsg(data, default=None)
            if data_epsg is not None:
                data_crs = f"EPSG:{data_epsg}"
        except Exception:
            # Best-effort CRS detection via optional stackstac; on any failure,
            # fall back to leaving ``data_crs`` as None and continue without it.
            pass
    if data_crs is not None and geoms.crs is not None:
        geoms_crs = str(geoms.crs)
        if geoms_crs != data_crs:
            geoms = geoms.to_crs(data_crs)
        # Filter to geometries that intersect raster bounds (avoids xvec IndexError when none overlap)
        x_min = float(data.coords[x_dim].min())
        x_max = float(data.coords[x_dim].max())
        y_min = float(data.coords[y_dim].min())
        y_max = float(data.coords[y_dim].max())
        from shapely.geometry import box
        raster_box = box(x_min, y_min, x_max, y_max)
        mask = geoms.intersects(raster_box)
        if not mask.any():
            raise ValueError(
                "No training geometries intersect the raster extent. "
                "Check that geometries and raster cover the same area (e.g. same bbox, CRS alignment)."
            )
        geoms = geoms[mask]
        if extra_cols is not None:
            extra_cols = extra_cols.loc[mask]

    # xvec.zonal_stats: stats="mean"|"median"|"sum"|"min"|"max"
    stats = reducer if reducer in ("mean", "median", "sum", "min", "max") else "mean"
    zonal = data.xvec.zonal_stats(
        geoms,
        x_coords=x_dim,
        y_coords=y_dim,
        stats=stats,
    )
    # Only trigger eager computation when zonal is chunked (e.g., dask-backed).
    if getattr(zonal, "chunks", None):
        zonal = zonal.compute()

    # Flatten to (geometry, features) for ML: collapse time and bands into columns
    non_geom_dims = [d for d in zonal.dims if d != "geometry"]
    if len(non_geom_dims) > 1:
        stacked = zonal.stack(_flat=tuple(non_geom_dims), create_index=False).transpose(
            "geometry", "_flat"
        )
        if bands_dim in zonal.dims and t_dim in zonal.dims:
            labels = [
                f"{b}_{str(t)[:7]}"
                for t in zonal.coords[t_dim].values
                for b in zonal.coords[bands_dim].values
            ]
        elif bands_dim in zonal.dims:
            labels = list(zonal.coords[bands_dim].values)
        else:
            labels = [str(i) for i in range(stacked.sizes["_flat"])]
        geom_index = stacked.coords["geometry"].values
        wide = pd.DataFrame(stacked.values, index=geom_index, columns=labels)
    else:
        # For 0 or 1 non-geometry dimensions, `to_dataframe()` may produce an index
        # that includes the non-geometry dimension (e.g. MultiIndex (geometry, bands)).
        # Extract the actual geometry objects from the index into a column and use that
        # as the GeoDataFrame geometry.
        wide = zonal.to_dataframe()
        # If geometry is stored as an index level, move it to a column.
        if isinstance(wide.index, pd.MultiIndex) and "geometry" in wide.index.names:
            wide = wide.reset_index("geometry")
        if "geometry" in wide.columns:
            # Use the geometry column values as the geometry array
            geom_index = wide["geometry"].values
            # Drop the geometry column from attributes and reset remaining index
            wide = wide.drop(columns=["geometry"]).reset_index(drop=True)
        else:
            # Fallback: use geometries from the input GeoDataFrame, aligned by row
            geom_index = geoms.geometry.values
    gdf_out = gpd.GeoDataFrame(wide, geometry=geom_index, crs=geoms.crs)
    if extra_cols is not None and len(extra_cols.columns) > 0:
        for c in extra_cols.columns:
            gdf_out[c] = extra_cols[c].values
    return gdf_out


# ---------------------------------------------------------------------------
# aggregate_temporal
# ---------------------------------------------------------------------------


def aggregate_temporal(
    data: RasterCube,
    *,
    period: str = "month",
    reducer: str = "mean",
    t_dim: str = "time",
) -> RasterCube:
    """Aggregate raster values over calendar periods.

    Implements ``aggregate_temporal_period`` from the openEO process specs.

    Parameters
    ----------
    period : str
        One of ``"hour"``, ``"day"``, ``"week"``, ``"dekad"``,
        ``"month"``, ``"season"``, ``"tropical-season"``, ``"year"``,
        ``"decade"``, ``"decade-ad"``.
    reducer : str
        Named reducer applied within each period (e.g. ``"mean"``,
        ``"sum"``, ``"min"``, ``"max"``, ``"median"``).
    t_dim : str
        Name of the temporal dimension.
    """

    # Periods that map directly to xarray resample frequencies
    _SIMPLE_FREQ: dict[str, str] = {
        "hour": "h",
        "day": "D",
        "week": "W",
        "month": "ME",
        "season": "QS-DEC",
        "year": "YE",
    }

    freq = _SIMPLE_FREQ.get(period)

    if freq is not None:
        resampled = data.resample({t_dim: freq})
        result = getattr(resampled, reducer)()
        # Relabel the temporal coordinates per the openEO spec
        result = _relabel_temporal(result, period, t_dim)
        return result

    # -- Periods that need custom groupby logic --

    if period == "dekad":
        return _aggregate_by_group(data, _dekad_label, reducer, t_dim)

    if period == "tropical-season":
        return _aggregate_by_group(data, _tropical_season_label, reducer, t_dim)

    if period == "decade":
        return _aggregate_by_group(data, _decade_label, reducer, t_dim)

    if period == "decade-ad":
        return _aggregate_by_group(data, _decade_ad_label, reducer, t_dim)

    raise ValueError(
        f"Unsupported period {period!r}. Choose from: "
        f"hour, day, week, dekad, month, season, tropical-season, "
        f"year, decade, decade-ad"
    )


# -- temporal label helpers ---------------------------------------------------

def _relabel_temporal(
    data: RasterCube, period: str, t_dim: str
) -> RasterCube:
    """Relabel the temporal dimension to match the openEO spec format."""

    coords = data.coords[t_dim].values
    fmt_map = {
        "hour": lambda dt: dt.strftime("%Y-%m-%d-%H"),
        "day": lambda dt: f"{dt.year}-{dt.timetuple().tm_yday:03d}",
        "week": lambda dt: f"{dt.isocalendar()[0]}-{dt.isocalendar()[1]:02d}",
        "month": lambda dt: dt.strftime("%Y-%m"),
        "season": _season_label_from_dt,
        "year": lambda dt: str(dt.year),
    }
    fmt = fmt_map.get(period)
    if fmt is None:
        return data

    new_labels = [fmt(pd.Timestamp(t)) for t in coords]
    return data.assign_coords({t_dim: new_labels})


def _season_label_from_dt(dt) -> str:
    dt = pd.Timestamp(dt)
    m = dt.month
    if m in (12, 1, 2):
        year = dt.year + 1 if m == 12 else dt.year
        return f"{year}-djf"
    elif m in (3, 4, 5):
        return f"{dt.year}-mam"
    elif m in (6, 7, 8):
        return f"{dt.year}-jja"
    else:
        return f"{dt.year}-son"


def _dekad_label(dt) -> str:
    """Dekad label: ``YYYY-DD`` where DD is the 1-based dekad index (01..36)."""
    import pandas as pd
    dt = pd.Timestamp(dt)
    day = dt.day
    if day <= 10:
        dekad_in_month = 1
    elif day <= 20:
        dekad_in_month = 2
    else:
        dekad_in_month = 3
    dekad = (dt.month - 1) * 3 + dekad_in_month
    return f"{dt.year}-{dekad:02d}"


def _tropical_season_label(dt) -> str:
    """Tropical season: Nov-Apr → ``YYYY-ndjfma``, May-Oct → ``YYYY-mjjaso``."""
    import pandas as pd
    dt = pd.Timestamp(dt)
    if dt.month in (11, 12, 1, 2, 3, 4):
        # Use the season's starting year: Nov–Dec in their calendar year,
        # Jan–Apr assigned to the previous year so the whole Nov–Apr block
        # shares the same season label.
        year = dt.year if dt.month in (11, 12) else dt.year - 1
        return f"{year}-ndjfma"
    else:
        return f"{dt.year}-mjjaso"


def _decade_label(dt) -> str:
    """0-to-9 decade: year ending in 0 to next year ending in 9."""
    dt = pd.Timestamp(dt)
    return f"{(dt.year // 10) * 10}"


def _decade_ad_label(dt) -> str:
    """1-to-0 decade (AD aligned): year ending in 1 to next year ending in 0."""
    dt = pd.Timestamp(dt)
    return f"{((dt.year - 1) // 10) * 10 + 1}"


def _aggregate_by_group(
    data: RasterCube,
    label_fn,
    reducer: str,
    t_dim: str,
) -> RasterCube:
    """Group by a custom label function and apply a reducer."""

    labels = [label_fn(t) for t in data.coords[t_dim].values]
    grouped = data.groupby(xr.DataArray(labels, dims=[t_dim], name="__period__"))
    result = getattr(grouped, reducer)()
    # Rename the groupby coordinate back to the temporal dim name
    if "__period__" in result.dims:
        result = result.rename({"__period__": t_dim})
    return result


# ---------------------------------------------------------------------------
# resample_spatial
# ---------------------------------------------------------------------------


def resample_spatial(
    data: RasterCube,
    *,
    resolution: float | list[float] = 0,
    projection: int | str | None = None,
    method: str = "near",
    align: str = "upper-left",
    x_dim: str = "longitude",
    y_dim: str = "latitude",
) -> RasterCube:
    """Resample and/or reproject the spatial dimensions of a raster cube.

    Implements ``resample_spatial`` from the openEO process specs.
    Requires ``rioxarray`` (optional ``geo`` extra).

    Parameters
    ----------
    data : RasterCube
        Raster cube with spatial dimensions.
    resolution : float | list[float]
        Target resolution in units of *projection*.  A single number
        applies to both axes; a two-element list sets ``[x_res, y_res]``.
        ``0`` (default) keeps the original resolution.
    projection : int | str | None
        Target CRS as an EPSG code (int) or WKT2 string.
        ``None`` keeps the current projection.
    method : str
        Resampling method aligned with gdalwarp: ``"near"``, ``"bilinear"``,
        ``"cubic"``, ``"cubicspline"``, ``"lanczos"``, ``"average"``,
        ``"mode"``, ``"max"``, ``"min"``, ``"med"``, ``"q1"``, ``"q3"``,
        ``"sum"``, ``"rms"``.
    align : str
        Corner alignment (``"upper-left"``, ``"lower-left"``,
        ``"upper-right"``, ``"lower-right"``).  Currently informational.
    """
    try:
        import rioxarray  # noqa: F401
    except ImportError:
        raise ImportError(
            "resample_spatial requires rioxarray.  "
            "Install it with:  pip install rioxarray"
        )

    from rasterio.enums import Resampling

    if resolution == 0 and projection is None:
        return data  # nothing to do

    # Map openEO method names to rasterio Resampling enum
    _METHOD_MAP: dict[str, Resampling] = {
        "near": Resampling.nearest,
        "bilinear": Resampling.bilinear,
        "cubic": Resampling.cubic,
        "cubicspline": Resampling.cubic_spline,
        "lanczos": Resampling.lanczos,
        "average": Resampling.average,
        "mode": Resampling.mode,
        "max": Resampling.max,
        "min": Resampling.min,
        "med": Resampling.med,
        "q1": Resampling.q1,
        "q3": Resampling.q3,
        "sum": Resampling.sum,
        "rms": Resampling.rms,
    }
    resampling = _METHOD_MAP.get(method)
    if resampling is None:
        raise ValueError(
            f"Unknown resampling method {method!r}. "
            f"Choose from {list(_METHOD_MAP)}"
        )

    # Ensure rioxarray spatial dims are set
    data = data.rio.set_spatial_dims(x_dim=x_dim, y_dim=y_dim)

    # If data doesn't have a CRS yet, assume EPSG:4326
    if data.rio.crs is None:
        data = data.rio.write_crs("EPSG:4326")

    # Resolve target resolution
    if isinstance(resolution, (int, float)) and resolution > 0:
        res_x = res_y = float(resolution)
    elif isinstance(resolution, (list, tuple)) and len(resolution) == 2:
        res_x, res_y = float(resolution[0]), float(resolution[1])
    else:
        res_x = res_y = 0.0

    # rioxarray only supports 2-D (y, x) or 3-D (band, y, x) arrays.
    # When the cube has >3 dims (e.g. t, bands, y, x) we flatten
    # all non-spatial dims into one, reproject, then reshape back.
    extra_dims = [d for d in data.dims if d not in (x_dim, y_dim)]
    needs_reshape = len(extra_dims) > 1

    if needs_reshape:
        data = _resample_nd(
            data, extra_dims, x_dim, y_dim,
            projection=projection,
            res_x=res_x, res_y=res_y,
            resampling=resampling,
        )
        return data

    # --- Simple 2-D / 3-D path (no reshape needed) ---

    # Reproject if projection is specified
    if projection is not None:
        dst_crs = f"EPSG:{projection}" if isinstance(projection, int) else projection
        data = data.rio.reproject(dst_crs, resampling=resampling)

    # Resample if resolution is specified and > 0
    if res_x > 0 and res_y > 0:
        crs = data.rio.crs or "EPSG:4326"
        data = data.rio.reproject(
            crs,
            resolution=(res_x, res_y),
            resampling=resampling,
        )

    # rioxarray.reproject() renames spatial dims to 'y'/'x'.
    # Restore the caller's original dim names.
    data = _restore_spatial_dim_names(data, x_dim, y_dim)

    return data


def _resample_nd(
    data: RasterCube,
    extra_dims: list[str],
    x_dim: str,
    y_dim: str,
    *,
    projection: int | str | None,
    res_x: float,
    res_y: float,
    resampling,
) -> RasterCube:
    """Handle resample_spatial for arrays with >3 dimensions.

    Flattens extra non-spatial dims into a single dimension using
    dask-compatible reshape, reprojects the 3-D array, then reshapes back.

    .. warning::
       This function uses rioxarray's reproject operation, which currently
       forces computation of dask-backed arrays. If the input has a dask-backed
       array, the result will be computed and returned as a numpy array.
       See: https://github.com/corteva/rioxarray/issues/481
    """
    # Save extra-dim metadata (sizes, coords)
    extra_sizes = [data.sizes[d] for d in extra_dims]
    extra_coords = {d: data.coords[d] for d in extra_dims if d in data.coords}

    # Transpose to (extra..., y, x) then flatten extra dims (dask-safe)
    data = data.transpose(*extra_dims, y_dim, x_dim)
    vals = data.data  # shape: (*extra_sizes, ny, nx) - keeps dask arrays lazy
    orig_shape = vals.shape
    ny, nx = orig_shape[-2], orig_shape[-1]
    flat_n = int(np.prod(extra_sizes))
    flat_vals = vals.reshape(flat_n, ny, nx)

    # Build a 3-D DataArray that rioxarray can handle
    flat_da = xr.DataArray(
        flat_vals,
        dims=["__flat__", y_dim, x_dim],
        coords={
            y_dim: data.coords[y_dim],
            x_dim: data.coords[x_dim],
        },
    )
    flat_da = flat_da.rio.set_spatial_dims(x_dim=x_dim, y_dim=y_dim)
    if data.rio.crs is not None:
        flat_da = flat_da.rio.write_crs(data.rio.crs)
    else:
        flat_da = flat_da.rio.write_crs("EPSG:4326")

    # Reproject / resample
    if projection is not None:
        dst_crs = f"EPSG:{projection}" if isinstance(projection, int) else projection
        flat_da = flat_da.rio.reproject(dst_crs, resampling=resampling)

    if res_x > 0 and res_y > 0:
        crs = flat_da.rio.crs or "EPSG:4326"
        flat_da = flat_da.rio.reproject(
            crs,
            resolution=(res_x, res_y),
            resampling=resampling,
        )

    # rioxarray renames spatial dims to 'y'/'x' — restore original names
    flat_da = _restore_spatial_dim_names(flat_da, x_dim, y_dim)

    # Reshape back to original extra dims (dask-safe)
    new_ny = flat_da.sizes[y_dim]
    new_nx = flat_da.sizes[x_dim]
    result_vals = flat_da.data.reshape(*extra_sizes, new_ny, new_nx)

    # Rebuild DataArray with original extra-dim coordinates
    result = xr.DataArray(
        result_vals,
        dims=[*extra_dims, y_dim, x_dim],
        coords={
            **extra_coords,
            y_dim: flat_da.coords[y_dim],
            x_dim: flat_da.coords[x_dim],
        },
    )
    result = result.rio.set_spatial_dims(x_dim=x_dim, y_dim=y_dim)
    result = result.rio.write_crs(flat_da.rio.crs)

    return result


def _restore_spatial_dim_names(
    data: RasterCube, x_dim: str, y_dim: str
) -> RasterCube:
    """Rename rioxarray's default ``y``/``x`` dims back to the caller's names."""
    rename: dict[str, str] = {}
    if x_dim != "x" and "x" in data.dims and x_dim not in data.dims:
        rename["x"] = x_dim
    if y_dim != "y" and "y" in data.dims and y_dim not in data.dims:
        rename["y"] = y_dim
    if rename:
        data = data.rename(rename)
    return data


# ---------------------------------------------------------------------------
# apply (generic per-element operation)
# ---------------------------------------------------------------------------


def apply(
    data: RasterCube,
    process: Callable[..., Any],
    *,
    context: Any = None,
) -> RasterCube:
    """Apply a unary function to every value in the data cube.

    Uses :func:`xarray.apply_ufunc` so that the operation is dask-safe.

    Parameters
    ----------
    data : RasterCube
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
    data: RasterCube,
    feature_dim: str = "bands",
) -> RasterCube:
    """Stack all non-feature dims into a ``samples`` dim.

    Returns a 2-D DataArray with shape ``(samples, features)``.
    """
    non_feature = [d for d in data.dims if d != feature_dim]
    stacked = data.stack(samples=non_feature)
    return stacked.transpose("samples", feature_dim)


def unstack_from_samples(
    result: RasterCube,
    template: RasterCube,
    feature_dim: str = "bands",
) -> RasterCube:
    """Reverse :func:`stack_to_samples` using *template*'s multi-index."""
    non_feature = [d for d in template.dims if d != feature_dim]
    stacked_template = template.stack(samples=non_feature)
    result = result.assign_coords(samples=stacked_template.coords["samples"])
    return result.unstack("samples")


# ---------------------------------------------------------------------------
# reduce_dimension
# ---------------------------------------------------------------------------


# Built-in reducer names mapped to numpy functions.
_BUILTIN_REDUCERS: dict[str, Callable[..., Any]] = {
    "mean": np.mean,
    "average": np.mean,
    "sum": np.sum,
    "min": np.min,
    "max": np.max,
    "median": np.median,
    "std": np.std,
    "var": np.var,
    "prod": np.prod,
    "any": np.any,
    "all": np.all,
    "count": lambda x, axis=None: np.sum(np.isfinite(x), axis=axis),
}


def _resolve_reducer(reducer: str | Callable[..., Any]) -> Callable[..., Any]:
    """Resolve *reducer* to a callable.

    Accepts:
    * A callable – returned as-is.
    * A built-in name (``"mean"``, ``"sum"``, …) – looked up from the map.
    * A dotted Python path (e.g. ``"numpy.nanmean"``) – imported dynamically.

    Raises
    ------
    ValueError
        If the string cannot be resolved to a callable.
    """
    if callable(reducer):
        return reducer

    if not isinstance(reducer, str):
        raise TypeError(
            f"reducer must be a callable or a string, got {type(reducer)!r}"
        )

    # Try built-in name first
    builtin = _BUILTIN_REDUCERS.get(reducer)
    if builtin is not None:
        return builtin

    # Try dotted import path (e.g. "numpy.nanmean", "scipy.stats.gmean")
    if "." in reducer:
        parts = reducer.rsplit(".", 1)
        if len(parts) == 2:
            module_path, func_name = parts
            try:
                import importlib

                mod = importlib.import_module(module_path)
                func = getattr(mod, func_name, None)
                if func is not None and callable(func):
                    return func
            except (ImportError, ModuleNotFoundError):
                pass

    raise ValueError(
        f"Unknown reducer {reducer!r}. Use a callable, a built-in name "
        f"({', '.join(sorted(_BUILTIN_REDUCERS))}), or a dotted Python "
        f"path like 'numpy.nanmean'."
    )


def reduce_dimension(
    data: RasterCube,
    reducer: str | Callable[..., Any],
    *,
    dimension: str,
    context: Any = None,
) -> RasterCube:
    """Collapse a dimension by applying a reducer function.

    Implements the ``reduce_dimension`` openEO process.  The *reducer*
    receives a 1-D array of values along *dimension* for each pixel and
    must return a single scalar value.  The specified dimension is dropped
    from the result.

    Parameters
    ----------
    data : RasterCube
        Input raster cube.
    reducer : str | callable
        Either a string name or a callable.

        **String names** – built-in reducers mapped to numpy functions:
        ``"mean"``, ``"sum"``, ``"min"``, ``"max"``, ``"median"``,
        ``"std"``, ``"var"``, ``"prod"``, ``"any"``, ``"all"``,
        ``"count"`` (counts finite values), ``"average"`` (alias for mean).

        **Dotted Python path** – any importable function, e.g.
        ``"numpy.nanmean"`` or ``"numpy.nansum"``.

        **Callable** – a function ``f(values, axis=...) -> array``
        applied along *dimension*.
    dimension : str
        Name of the dimension to reduce over.
    context
        Optional extra data forwarded to *reducer* (only when *reducer*
        is a callable that accepts a ``context`` keyword argument).

    Raises
    ------
    DimensionNotAvailable
        If the specified *dimension* does not exist.
    ValueError
        If a string *reducer* cannot be resolved to a callable.
    """
    if dimension not in data.dims:
        raise DimensionNotAvailable(
            f"A dimension with the specified name '{dimension}' does not exist. "
            f"Available dimensions: {list(data.dims)}"
        )

    reduce_fn = _resolve_reducer(reducer)

    if context is not None:
        # Wrap the reducer to accept `axis` and forward context
        def _wrapper(values: np.ndarray, axis: int | None = None) -> np.ndarray:
            return reduce_fn(values, axis=axis, context=context)

        result = data.reduce(_wrapper, dim=dimension)
    else:
        result = data.reduce(reduce_fn, dim=dimension)

    return result


# ---------------------------------------------------------------------------
# apply_kernel
# ---------------------------------------------------------------------------

# Map openEO border mode names to scipy.ndimage mode names
_BORDER_MODE_MAP: dict[str, str] = {
    "replicate": "nearest",
    "reflect": "reflect",
    "reflect_pixel": "mirror",
    "wrap": "wrap",
}


def apply_kernel(
    data: RasterCube,
    *,
    kernel: list[list[float]],
    factor: float = 1.0,
    border: float | str = 0,
    replace_invalid: float = 0.0,
    x_dim: str = "longitude",
    y_dim: str = "latitude",
) -> RasterCube:
    """Apply a 2-D spatial convolution kernel to a raster cube.

    Implements the ``apply_kernel`` openEO process.  The kernel is applied
    to the ``(y, x)`` spatial dimensions of each slice independently.

    Parameters
    ----------
    data : RasterCube
        Raster cube with spatial dimensions.
    kernel : list[list[float]]
        2-D array of convolution weights.  Each dimension must have an
        uneven (odd) number of elements.
    factor : float
        Multiplicative factor applied to each convolved value (default 1).
    border : float | str
        How to handle borders.  A numeric value fills borders with that
        constant; string values ``"replicate"``, ``"reflect"``,
        ``"reflect_pixel"``, or ``"wrap"`` use the corresponding strategy.
    replace_invalid : float
        Value to substitute for NaN / Inf / non-numeric entries before
        convolution (default 0).
    x_dim, y_dim : str
        Names of the spatial dimensions.

    Raises
    ------
    KernelDimensionsUneven
        If either kernel dimension has an even number of elements.
    DimensionNotAvailable
        If the spatial dimensions are not found.
    """
    from scipy.ndimage import convolve

    kernel_arr = np.asarray(kernel, dtype=np.float64)

    # Validate kernel dimensions are odd
    if kernel_arr.ndim != 2:
        raise KernelDimensionsUneven(
            "The kernel must be a two-dimensional array."
        )
    if kernel_arr.shape[0] % 2 == 0 or kernel_arr.shape[1] % 2 == 0:
        raise KernelDimensionsUneven(
            "Each dimension of the kernel must have an uneven number of elements."
        )

    # Validate spatial dimensions exist
    for dim in (y_dim, x_dim):
        if dim not in data.dims:
            raise DimensionNotAvailable(
                f"A dimension with the specified name '{dim}' does not exist. "
                f"Available dimensions: {list(data.dims)}"
            )

    # Resolve border mode for scipy.ndimage.convolve
    if isinstance(border, str):
        scipy_mode = _BORDER_MODE_MAP.get(border)
        if scipy_mode is None:
            raise ValueError(
                f"Unknown border mode {border!r}. Choose from: "
                f"{list(_BORDER_MODE_MAP)} or a numeric constant."
            )
        cval = 0.0
    else:
        scipy_mode = "constant"
        cval = float(border)

    def _convolve_slice(arr: np.ndarray) -> np.ndarray:
        """Convolve a single 2-D spatial slice."""
        # Replace invalid values
        arr = np.where(np.isfinite(arr), arr, replace_invalid)
        # Apply convolution and multiply by factor
        return convolve(arr, kernel_arr, mode=scipy_mode, cval=cval) * factor

    # Determine the axes for the spatial dims
    spatial_dims = [y_dim, x_dim]

    # Use apply_ufunc with vectorize=True so that _convolve_slice
    # receives individual 2-D (y, x) slices rather than the full N-D array.
    result = xr.apply_ufunc(
        _convolve_slice,
        data,
        input_core_dims=[spatial_dims],
        output_core_dims=[spatial_dims],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[data.dtype],
    )

    return result
