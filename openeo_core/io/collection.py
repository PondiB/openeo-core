"""Collection loader – load named EO collections into RasterCubes.

The default implementation uses **pystac-client** to search the
`Earth Search <https://earth-search.aws.element84.com/v1>`_ STAC API
(Sentinel-2 L2A on AWS) and **stackstac** to build a lazy xarray DataArray.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import xarray as xr

# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class CollectionLoader(Protocol):
    """Protocol for collection loaders (``load_collection``)."""

    def load_collection(
        self,
        collection_id: str,
        *,
        spatial_extent: dict | None = None,
        temporal_extent: tuple[str, str] | None = None,
        bands: list[str] | None = None,
        properties: dict | None = None,
        **kwargs: Any,
    ) -> xr.DataArray:
        ...


# ---------------------------------------------------------------------------
# Default implementation – AWS Earth Search via pystac-client + stackstac
# ---------------------------------------------------------------------------


class AWSCollectionLoader:
    """Load collections from the Element 84 Earth Search STAC API on AWS.

    This is a convenience default; users can inject any
    :class:`CollectionLoader`-compatible adapter.

    Parameters
    ----------
    api_url : str
        STAC API endpoint. Defaults to Earth Search v1.
    """

    DEFAULT_API_URL = "https://earth-search.aws.element84.com/v1"

    def __init__(self, api_url: str | None = None) -> None:
        self.api_url = api_url or self.DEFAULT_API_URL

    def load_collection(
        self,
        collection_id: str,
        *,
        spatial_extent: dict | None = None,
        temporal_extent: tuple[str, str] | None = None,
        bands: list[str] | None = None,
        properties: dict | None = None,
        **kwargs: Any,
    ) -> xr.DataArray:
        """Search the STAC API and return a dask-backed DataArray.

        Parameters
        ----------
        collection_id : str
            STAC collection identifier (e.g. ``"sentinel-2-l2a"``).
        spatial_extent : dict | None
            Bounding box as ``{west, south, east, north}`` (WGS 84).
        temporal_extent : tuple[str, str] | None
            ``(start_datetime, end_datetime)`` ISO-8601 strings.
        bands : list[str] | None
            Asset / band names to include.  ``None`` loads all.
        properties : dict | None
            Extra STAC query parameters (e.g. cloud cover filter).
        """
        import pystac_client
        import stackstac

        catalog = pystac_client.Client.open(self.api_url)

        search_kwargs: dict[str, Any] = {
            "collections": [collection_id],
            "max_items": kwargs.pop("max_items", 100),
        }

        if spatial_extent is not None:
            search_kwargs["bbox"] = [
                spatial_extent["west"],
                spatial_extent["south"],
                spatial_extent["east"],
                spatial_extent["north"],
            ]

        if temporal_extent is not None:
            search_kwargs["datetime"] = "/".join(temporal_extent)

        if properties:
            search_kwargs["query"] = properties

        items = catalog.search(**search_kwargs).item_collection()

        if len(items) == 0:
            raise ValueError(
                f"No items found for collection {collection_id!r} with the "
                f"given filters."
            )

        # Build lazy DataArray via stackstac
        stack_kwargs: dict[str, Any] = {}
        if bands is not None:
            stack_kwargs["assets"] = bands
        if spatial_extent is not None:
            stack_kwargs["bounds_latlon"] = [
                spatial_extent["west"],
                spatial_extent["south"],
                spatial_extent["east"],
                spatial_extent["north"],
            ]

        stack_kwargs.update(kwargs)

        da: xr.DataArray = stackstac.stack(items, **stack_kwargs)

        # Rename dimensions to the library's conventional names
        rename_map: dict[str, str] = {}
        if "time" in da.dims:
            rename_map["time"] = "t"
        if "band" in da.dims:
            rename_map["band"] = "bands"
        if rename_map:
            da = da.rename(rename_map)

        return da


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

_default = AWSCollectionLoader()


def load_collection(
    collection_id: str,
    *,
    spatial_extent: dict | None = None,
    temporal_extent: tuple[str, str] | None = None,
    bands: list[str] | None = None,
    properties: dict | None = None,
    adapter: CollectionLoader | None = None,
    **kwargs: Any,
) -> xr.DataArray:
    """Load a named EO collection into a raster DataArray.

    Parameters
    ----------
    adapter : CollectionLoader | None
        Custom loader.  Uses the default AWS Earth Search adapter when *None*.
    """
    loader = adapter or _default
    return loader.load_collection(
        collection_id,
        spatial_extent=spatial_extent,
        temporal_extent=temporal_extent,
        bands=bands,
        properties=properties,
        **kwargs,
    )
