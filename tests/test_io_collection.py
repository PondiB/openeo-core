"""Tests for STAC collection loading helpers."""

from __future__ import annotations

import pytest

from openeo_core.io.collection import _resolve_band_assets_for_stackstac


class _FakeAssets(dict):
    """Minimal stand-in for pystac.Item.assets (mapping of id -> asset)."""


class _FakeItem:
    def __init__(self, asset_ids: list[str]) -> None:
        self.assets = _FakeAssets((k, object()) for k in asset_ids)


def test_resolve_pc_style_assets_unchanged() -> None:
    item = _FakeItem(["B04", "B08", "SCL"])
    assert _resolve_band_assets_for_stackstac(item, ["B04", "B08"]) == ["B04", "B08"]


def test_resolve_common_names_to_pc_assets() -> None:
    item = _FakeItem(["B04", "B08", "SCL"])
    assert _resolve_band_assets_for_stackstac(item, ["red", "nir"]) == ["B04", "B08"]


def test_resolve_earth_search_style_assets_unchanged() -> None:
    item = _FakeItem(["red", "nir", "scl"])
    assert _resolve_band_assets_for_stackstac(item, ["red", "nir"]) == ["red", "nir"]


def test_resolve_unknown_band_raises() -> None:
    item = _FakeItem(["B04", "B08"])
    with pytest.raises(ValueError, match="No STAC asset for band 'foo'"):
        _resolve_band_assets_for_stackstac(item, ["foo"])
