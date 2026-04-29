"""Tests for the ProcessRegistry."""

import pytest

from openeo_core.processes.registry import ProcessRegistry


@pytest.fixture
def registry():
    return ProcessRegistry()


class TestProcessRegistry:
    def test_list_processes_not_empty(self, registry):
        procs = registry.list_processes()
        assert len(procs) > 0

    def test_list_processes_sorted(self, registry):
        procs = registry.list_processes()
        assert procs == sorted(procs)

    def test_get_process_known(self, registry):
        spec = registry.get_process("ndvi")
        assert spec["id"] == "ndvi"
        assert "parameters" in spec

    def test_get_process_unknown_raises(self, registry):
        with pytest.raises(KeyError, match="not_a_real_process"):
            registry.get_process("not_a_real_process")

    def test_search(self, registry):
        results = registry.search("vegetation")
        assert any(r["id"] == "ndvi" for r in results)

    def test_load_collection_spec(self, registry):
        spec = registry.get_process("load_collection")
        assert "parameters" in spec

    def test_load_stac_spec(self, registry):
        spec = registry.get_process("load_stac")
        assert "parameters" in spec

    def test_filter_bbox_spec(self, registry):
        spec = registry.get_process("filter_bbox")
        assert "parameters" in spec

    def test_eq_spec(self, registry):
        spec = registry.get_process("eq")
        assert spec["id"] == "eq"
        assert spec["categories"] == ["texts", "comparison"]

    def test_or_spec(self, registry):
        spec = registry.get_process("or")
        assert spec["id"] == "or"
        assert spec["categories"] == ["logic"]
        assert "process_graph" in spec

    def test_median_spec(self, registry):
        spec = registry.get_process("median")
        assert spec["id"] == "median"
        assert spec["categories"] == ["math > statistics", "reducer"]
        assert "process_graph" in spec

    def test_array_element_spec(self, registry):
        spec = registry.get_process("array_element")
        assert spec["id"] == "array_element"
        assert spec["categories"] == ["arrays", "reducer"]
        assert "exceptions" in spec

    def test_reduce_dimension_spec(self, registry):
        spec = registry.get_process("reduce_dimension")
        assert spec["id"] == "reduce_dimension"
        assert spec["categories"] == ["cubes", "reducer"]
        assert "links" in spec
        assert "exceptions" in spec

    def test_drop_dimension_spec(self, registry):
        spec = registry.get_process("drop_dimension")
        assert spec["id"] == "drop_dimension"
        assert spec["categories"] == ["cubes"]
        assert spec["exceptions"]["DimensionLabelCountMismatch"]["message"].startswith(
            "The number of dimension labels exceeds one"
        )
