"""STAC loader – load STAC Items / Collections into data cubes.

The default implementation uses **pystac-client** for API searches and
**stackstac** to materialise raster items as lazy xarray DataArrays.
"""

from __future__ import annotations

from typing import Any, Protocol, Union, runtime_checkable

import geopandas as gpd
import xarray as xr

# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class StacLoader(Protocol):
    """Protocol for STAC loaders (``load_stac``)."""

    def load_stac(
        self,
        source: str | dict,
        *,
        assets: list[str] | None = None,
        spatial_extent: dict | None = None,
        temporal_extent: tuple[str, str] | None = None,
        **kwargs: Any,
    ) -> Union[xr.DataArray, gpd.GeoDataFrame]:
        ...


# ---------------------------------------------------------------------------
# Default implementation
# ---------------------------------------------------------------------------


class DefaultStacLoader:
    """Load STAC resources using pystac + stackstac.

    Supports:
    * Single STAC Item URL / file path / inline dict  -> raster DataArray
    * STAC ItemCollection / API search URL            -> raster DataArray
    """

    def load_stac(
        self,
        source: str | dict,
        *,
        assets: list[str] | None = None,
        spatial_extent: dict | None = None,
        temporal_extent: tuple[str, str] | None = None,
        **kwargs: Any,
    ) -> Union[xr.DataArray, gpd.GeoDataFrame]:
        import pystac
        import stackstac

        items = self._resolve_items(source)

        # Filter by temporal extent if provided
        if temporal_extent is not None:
            import dateutil.parser

            t_start = dateutil.parser.isoparse(temporal_extent[0])
            t_end = dateutil.parser.isoparse(temporal_extent[1])
            items = [
                it
                for it in items
                if it.datetime is not None
                and t_start <= it.datetime <= t_end
            ]

        if len(items) == 0:
            raise ValueError("No STAC items matched the given filters.")

        stack_kwargs: dict[str, Any] = {}
        if assets is not None:
            stack_kwargs["assets"] = assets
        if spatial_extent is not None:
            stack_kwargs["bounds_latlon"] = [
                spatial_extent["west"],
                spatial_extent["south"],
                spatial_extent["east"],
                spatial_extent["north"],
            ]
        stack_kwargs.update(kwargs)

        da: xr.DataArray = stackstac.stack(items, **stack_kwargs)

        # Normalise dimension names
        rename: dict[str, str] = {}
        if "time" in da.dims:
            rename["time"] = "t"
        if "band" in da.dims:
            rename["band"] = "bands"
        if rename:
            da = da.rename(rename)

        return da

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_items(source: str | dict) -> list:
        """Resolve *source* to a list of :class:`pystac.Item` objects."""
        import pystac
        import pystac_client

        if isinstance(source, dict):
            # Inline STAC JSON
            stac_type = source.get("type", "")
            if stac_type == "Feature":
                return [pystac.Item.from_dict(source)]
            if stac_type == "FeatureCollection":
                return [
                    pystac.Item.from_dict(f)
                    for f in source.get("features", [])
                ]
            # Try as catalog / collection
            catalog = pystac.Catalog.from_dict(source)
            return list(catalog.get_all_items())

        # String source – URL or file path
        if source.startswith(("http://", "https://")):
            # Try as a STAC API search endpoint first
            try:
                client = pystac_client.Client.open(source)
                return list(client.get_collection(client.id).get_all_items())  # type: ignore[arg-type]
            except Exception:
                pass

            # Try as a direct STAC Item / Collection JSON URL
            obj = pystac.read_file(source)
            if isinstance(obj, pystac.Item):
                return [obj]
            if isinstance(obj, (pystac.Collection, pystac.Catalog)):
                return list(obj.get_all_items())
            raise ValueError(f"Cannot interpret STAC resource at {source!r}")

        # Local file
        obj = pystac.read_file(source)
        if isinstance(obj, pystac.Item):
            return [obj]
        if isinstance(obj, (pystac.Collection, pystac.Catalog)):
            return list(obj.get_all_items())
        raise ValueError(f"Cannot interpret STAC resource at {source!r}")


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

_default = DefaultStacLoader()


def load_stac(
    source: str | dict,
    *,
    assets: list[str] | None = None,
    spatial_extent: dict | None = None,
    temporal_extent: tuple[str, str] | None = None,
    adapter: StacLoader | None = None,
    **kwargs: Any,
) -> Union[xr.DataArray, gpd.GeoDataFrame]:
    """Load STAC items into a data cube.

    Parameters
    ----------
    adapter : StacLoader | None
        Custom STAC loader.  Uses :class:`DefaultStacLoader` when *None*.
    """
    loader = adapter or _default
    return loader.load_stac(
        source,
        assets=assets,
        spatial_extent=spatial_extent,
        temporal_extent=temporal_extent,
        **kwargs,
    )
