"""Collection loader – load named EO collections into RasterCubes.

The default implementation uses **pystac-client** to search the
`Earth Search <https://earth-search.aws.element84.com/v1>`_ STAC API
(Sentinel-2 L2A on AWS) and **stackstac** to build a lazy xarray DataArray.

A Microsoft `Planetary Computer <https://planetarycomputer.microsoft.com/>`_
loader is also provided, which uses the **planetary-computer** package for
SAS token signing.
"""

from __future__ import annotations

from abc import ABC
from typing import Any, Protocol, runtime_checkable

import xarray as xr
import planetary_computer
import pystac_client
import stackstac

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
# Base implementation with shared logic
# ---------------------------------------------------------------------------


class BaseCollectionLoader(ABC):
    """Base class for STAC collection loaders with shared logic.

    This is an abstract base class. Subclasses must define ``DEFAULT_API_URL``
    and may override ``_open_catalog`` if they need custom catalog
    initialization (e.g., with authentication modifiers).

    Parameters
    ----------
    api_url : str, optional
        STAC API endpoint. If not provided, uses the subclass's
        ``DEFAULT_API_URL``.
    """

    DEFAULT_API_URL: str = ""

    def __init__(self, api_url: str | None = None) -> None:
        if not hasattr(self.__class__, 'DEFAULT_API_URL') or not self.__class__.DEFAULT_API_URL:
            raise ValueError(
                f"{self.__class__.__name__} must define a non-empty DEFAULT_API_URL"
            )
        self.api_url = api_url or self.DEFAULT_API_URL

    def _open_catalog(self) -> pystac_client.Client:
        """Open the STAC catalog client.

        Subclasses can override this method to customize catalog initialization,
        for example by adding authentication or signing modifiers.

        Returns
        -------
        pystac_client.Client
            The opened STAC catalog client.
        """
        return pystac_client.Client.open(self.api_url)

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
            Bounding box as ``{west, south, east, north}``.  May include
            an optional ``crs`` key (EPSG code as int, or WKT2 string).
            Defaults to ``4326`` (WGS 84) when omitted.
        temporal_extent : tuple[str, str] | None
            ``(start_datetime, end_datetime)`` ISO-8601 strings.
        bands : list[str] | None
            Asset / band names to include.  ``None`` loads all.
        properties : dict | None
            Extra STAC query parameters (e.g. cloud cover filter).
        """

        catalog = self._open_catalog()

        search_kwargs: dict[str, Any] = {
            "collections": [collection_id],
            "max_items": kwargs.pop("max_items", 100),
        }

        # Extract the optional CRS from spatial_extent (openEO spec default: 4326).
        # Track whether the user explicitly provided a CRS so we can honour it
        # even when it equals the default (4326).
        extent_crs: int | str = 4326
        user_specified_crs = False
        if spatial_extent is not None and "crs" in spatial_extent:
            extent_crs = spatial_extent["crs"]
            user_specified_crs = True

        if spatial_extent is not None:
            bbox_coords = [
                spatial_extent["west"],
                spatial_extent["south"],
                spatial_extent["east"],
                spatial_extent["north"],
            ]
            # pystac-client search always expects WGS 84 bbox
            if _is_epsg_4326(extent_crs):
                search_kwargs["bbox"] = bbox_coords
            else:
                search_kwargs["bbox"] = _reproject_bbox(bbox_coords, extent_crs, 4326)

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

        # Determine the output EPSG for stacking.
        # If the user explicitly specified a CRS via spatial_extent, honour it
        # (even when it's 4326); otherwise auto-detect from STAC item properties.
        if user_specified_crs and isinstance(extent_crs, int):
            stack_kwargs.setdefault("epsg", extent_crs)
        else:
            detected_epsg = _detect_common_epsg(items)
            stack_kwargs.setdefault("epsg", detected_epsg)

        if spatial_extent is not None:
            bbox_coords = [
                spatial_extent["west"],
                spatial_extent["south"],
                spatial_extent["east"],
                spatial_extent["north"],
            ]
            if _is_epsg_4326(extent_crs):
                stack_kwargs.setdefault("bounds_latlon", bbox_coords)
            else:
                stack_kwargs.setdefault("bounds", bbox_coords)

        stack_kwargs.update(kwargs)

        da: xr.DataArray = stackstac.stack(items, **stack_kwargs)

        # Rename dimensions to the library's conventional names.
        # stackstac produces: (time, band, y, x)
        # We normalise to: (time, bands, latitude, longitude)
        rename_map: dict[str, str] = {}
        if "band" in da.dims:
            rename_map["band"] = "bands"
        if "y" in da.dims:
            rename_map["y"] = "latitude"
        if "x" in da.dims:
            rename_map["x"] = "longitude"
        if rename_map:
            da = da.rename(rename_map)

        return da


# ---------------------------------------------------------------------------
# Default implementation – AWS Earth Search via pystac-client + stackstac
# ---------------------------------------------------------------------------


class AWSCollectionLoader(BaseCollectionLoader):
    """Load collections from the Element 84 Earth Search STAC API on AWS.

    This is a convenience default; users can inject any
    :class:`CollectionLoader`-compatible adapter.

    Inherits the initialization and load_collection behavior from
    :class:`BaseCollectionLoader`.

    Parameters
    ----------
    api_url : str, optional
        STAC API endpoint. Defaults to Earth Search v1.
    """

    DEFAULT_API_URL = "https://earth-search.aws.element84.com/v1"


# ---------------------------------------------------------------------------
# Microsoft Planetary Computer via pystac-client + stackstac
# ---------------------------------------------------------------------------


class MicrosoftPlanetaryComputerLoader(BaseCollectionLoader):
    """Load collections from the Microsoft Planetary Computer STAC API.

    Assets hosted on Azure Blob Storage require SAS-token signing which
    is handled transparently by the **planetary-computer** package
    (``pip install planetary-computer``).

    Inherits the initialization and load_collection behavior from
    :class:`BaseCollectionLoader`, and overrides ``_open_catalog`` to add
    SAS token signing.

    See https://planetarycomputer.microsoft.com/docs/quickstarts/reading-stac/

    Parameters
    ----------
    api_url : str, optional
        STAC API endpoint.  Defaults to the Planetary Computer v1 API.
    """

    DEFAULT_API_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

    def _open_catalog(self) -> pystac_client.Client:
        """Open the Planetary Computer STAC catalog with SAS token signing.

        Returns
        -------
        pystac_client.Client
            The opened STAC catalog client with planetary_computer modifier.
        """
        return pystac_client.Client.open(
            self.api_url,
            modifier=planetary_computer.sign_inplace,
        )


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


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _is_epsg_4326(crs: int | str) -> bool:
    """Return True if *crs* represents EPSG:4326 (WGS 84)."""
    if isinstance(crs, int):
        return crs == 4326
    if isinstance(crs, str):
        return crs.upper() in ("EPSG:4326", "4326")
    return False


def _reproject_bbox(
    bbox: list[float],
    src_crs: int | str,
    dst_crs: int | str,
) -> list[float]:
    """Reproject a ``[west, south, east, north]`` bbox between CRS.

    Uses pyproj's ``Transformer``.  Handles axis-order differences by
    using the ``always_xy=True`` flag.
    """
    from pyproj import Transformer

    src = f"EPSG:{src_crs}" if isinstance(src_crs, int) else src_crs
    dst = f"EPSG:{dst_crs}" if isinstance(dst_crs, int) else dst_crs
    transformer = Transformer.from_crs(src, dst, always_xy=True)

    west, south, east, north = bbox
    # Transform corners and take envelope
    xs, ys = transformer.transform([west, east, west, east], [south, south, north, north])
    return [min(xs), min(ys), max(xs), max(ys)]


def _detect_common_epsg(items: list) -> int:
    """Try to find a common EPSG code across STAC items.

    Looks at ``proj:code`` / ``proj:epsg`` in item properties.
    If all items share the same CRS, return that EPSG code;
    otherwise fall back to ``4326`` (WGS 84).
    """
    codes: set[int] = set()
    for item in items:
        props = item.properties if hasattr(item, "properties") else {}
        # proj:code is e.g. "EPSG:32632"
        code_str = props.get("proj:code", "")
        if code_str and code_str.upper().startswith("EPSG:"):
            try:
                codes.add(int(code_str.split(":")[1]))
            except (ValueError, IndexError):
                # Ignore malformed proj:code values; rely on proj:epsg or fallback CRS.
                pass
        # proj:epsg is a direct integer
        epsg = props.get("proj:epsg")
        if isinstance(epsg, int):
            codes.add(epsg)

    if len(codes) == 1:
        return codes.pop()
    # Multiple CRS zones or none detected → use WGS 84
    return 4326
